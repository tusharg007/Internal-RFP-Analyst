"""Streamlit dashboard for the Internal RFP Analyst app."""

import gc
import time

import streamlit as st

from agent import (
    KB_NOT_READY_MESSAGE,
    LLM_CONFIG_WARNING,
    NO_SCOPE_DOCUMENTS_MESSAGE,
    _get_provider_name,
    create_agent,
    format_llm_error,
    is_debug_mode_enabled,
    prepare_query,
    query_agent_stream,
)
from config import (
    APP_SUBTITLE,
    APP_TITLE,
    MAX_UPLOAD_SIZE_MB,
    OFFLINE_SMOKE_EVAL_RESULTS_PATH,
    REAL_KB_EVAL_RESULTS_PATH,
    SAMPLE_DOCS_DIR,
    SAMPLE_QUESTIONS,
    UPLOADS_DIR,
    VECTORSTORE_DIR,
    get_api_keys,
)
from document_generator import generate_all_documents
from rag_engine import (
    VECTORSTORE_LOCKED_MESSAGE,
    get_vectorstore_stats,
    ingest_documents,
)
from rfp_analyst.evals import load_eval_snapshot
from rfp_analyst.exceptions import (
    IngestionError,
    KnowledgeBaseNotReadyError,
    LLMProviderNotConfiguredError,
    NoDocumentsFoundError,
    RetrievalError,
    UnsupportedFileError,
)
from rfp_analyst.health import get_app_health
from rfp_analyst.ui import get_chat_avatar
from rfp_analyst.uploads import is_uploaded_pdf_unchanged, persist_uploaded_pdf

SCOPE_LABELS = {
    "upload": "Uploaded documents only",
    "sample": "Sample documents only",
    "all": "All documents",
}
LABEL_TO_SCOPE = {label: scope for scope, label in SCOPE_LABELS.items()}
PENDING_UPLOADS_MESSAGE = (
    "Uploaded files are pending indexing. Click Ingest Documents before asking about them."
)

st.set_page_config(
    page_title="Internal RFP Analyst",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #1a1d29 0%, #2d2f3e 100%);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #a0a0a0;
        margin-top: 0.3rem;
    }
    .reasoning-box {
        background: rgba(108, 99, 255, 0.08);
        border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(0, 200, 100, 0.15);
        color: #00c864;
    }
    .status-pending {
        background: rgba(255, 180, 0, 0.15);
        color: #ffb400;
    }
    .provider-badge {
        background: rgba(108, 99, 255, 0.12);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
        color: #9D97FF;
        text-align: center;
        margin-top: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "scope" not in st.session_state:
    st.session_state.scope = "all"
if "upload_notice" not in st.session_state:
    st.session_state.upload_notice = ""
if "ingestion_in_progress" not in st.session_state:
    st.session_state.ingestion_in_progress = False
if "last_ingestion_error" not in st.session_state:
    st.session_state.last_ingestion_error = ""
if "pending_uploads" not in st.session_state:
    st.session_state.pending_uploads = False
if "setup_attempted" not in st.session_state:
    st.session_state.setup_attempted = False
if "status_message" not in st.session_state:
    st.session_state.status_message = ""


def clear_runtime_objects() -> None:
    """Release cached app objects before rebuilding the vector store."""
    st.session_state.agent = None
    for key in ("retriever", "vectorstore", "chroma_client"):
        st.session_state.pop(key, None)
    for cache_name in ("cache_resource", "cache_data"):
        cache = getattr(st, cache_name, None)
        if cache and hasattr(cache, "clear"):
            cache.clear()
    gc.collect()


def build_health_snapshot() -> tuple[dict, dict]:
    """Return vector store stats and central health checks."""
    groq_api_key, google_api_key = get_api_keys()
    stats = get_vectorstore_stats(
        sample_dir=SAMPLE_DOCS_DIR,
        uploads_dir=UPLOADS_DIR,
        persist_dir=VECTORSTORE_DIR,
    )
    health = get_app_health(
        vectorstore_stats=stats,
        data_dir=SAMPLE_DOCS_DIR,
        vectorstore_dir=VECTORSTORE_DIR,
        uploads_dir=UPLOADS_DIR,
        groq_api_key=groq_api_key,
        google_api_key=google_api_key,
    )
    st.session_state.pending_uploads = health.get("pending_upload_count", 0) > 0
    return stats, health


def available_scope_options(health: dict) -> list[str]:
    """Return the scope labels that make sense for the current indexed KB."""
    options = []
    if health.get("indexed_upload_document_count", 0) > 0:
        options.append(SCOPE_LABELS["upload"])
    if health.get("indexed_sample_document_count", 0) > 0:
        options.append(SCOPE_LABELS["sample"])
    if health.get("indexed_document_count", 0) > 0:
        options.append(SCOPE_LABELS["all"])
    return options or [SCOPE_LABELS["all"]]


def resolve_scope(health: dict) -> str:
    """Keep the current scope valid and default uploads when available."""
    available_scopes = [LABEL_TO_SCOPE[label] for label in available_scope_options(health)]
    current_scope = st.session_state.scope
    if current_scope in available_scopes:
        return current_scope
    if health.get("indexed_upload_document_count", 0) > 0:
        st.session_state.scope = "upload"
    elif "all" in available_scopes:
        st.session_state.scope = "all"
    else:
        st.session_state.scope = available_scopes[0]
    return st.session_state.scope


def render_reasoning(reasoning_trace, show_reasoning: bool):
    """Render source trace details after a response."""
    if not reasoning_trace or not show_reasoning:
        return

    with st.expander("Sources Used", expanded=False):
        for step in reasoning_trace:
            if "tool" in step:
                tool_label = step.get("tool", "tool")
                input_summary = step.get("input_summary") or step.get("input", {}).get("query", "")
                output_summary = step.get("output_summary", "")
                output_block = f"<br>{output_summary}" if output_summary else ""
                st.markdown(
                    (
                        '<div class="reasoning-box">'
                        f"<strong>{tool_label}</strong><br>"
                        f"<em>{input_summary}</em>"
                        f"{output_block}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            elif "tool_response" in step:
                st.markdown(
                    (
                        '<div class="reasoning-box">'
                        f'<strong>{step["tool_response"]}</strong><br>'
                        f'<em>{step["snippet"][:150]}...</em>'
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )


def add_assistant_message(message: str, reasoning=None):
    """Persist an assistant message to session state."""
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": message,
            "reasoning": reasoning or [],
        }
    )


def show_generation_warning(message: str):
    """Display a friendly warning and preserve it in chat history."""
    st.warning(message)
    add_assistant_message(message)


def handle_uploaded_files(uploaded_files) -> None:
    """Validate and save uploaded PDF files without crashing the app."""
    saved_files = []
    for uploaded_file in uploaded_files:
        try:
            if is_uploaded_pdf_unchanged(uploaded_file, uploads_dir=UPLOADS_DIR):
                # Streamlit retains file-uploader selections across reruns. An
                # unchanged file is already persisted and must not become
                # "pending" again immediately after successful ingestion.
                continue
            save_path = persist_uploaded_pdf(uploaded_file, uploads_dir=UPLOADS_DIR)
            saved_files.append(save_path.name)
        except UnsupportedFileError as exc:
            st.warning(str(exc))

    if saved_files:
        st.session_state.pending_uploads = True
        st.session_state.setup_attempted = True
        st.session_state.upload_notice = (
            f"Uploaded {len(saved_files)} file(s). Click Ingest Documents to index them."
        )
        st.session_state.last_ingestion_error = ""
        clear_runtime_objects()


def finish_ingestion(success: bool, message: str = "") -> None:
    """Reset ingestion flags after a manual ingestion attempt."""
    st.session_state.ingestion_in_progress = False
    if success:
        st.session_state.last_ingestion_error = ""
        st.session_state.pending_uploads = False
        st.session_state.upload_notice = ""
        st.session_state.status_message = message
    else:
        st.session_state.last_ingestion_error = message
        st.session_state.status_message = ""


def run_manual_ingestion() -> bool:
    """Run ingestion only when the user explicitly clicks the button."""
    st.session_state.ingestion_in_progress = True
    st.session_state.last_ingestion_error = ""
    st.session_state.setup_attempted = True
    clear_runtime_objects()

    try:
        with st.spinner("Processing documents..."):
            ingest_documents(
                sample_dir=SAMPLE_DOCS_DIR,
                uploads_dir=UPLOADS_DIR,
                persist_dir=VECTORSTORE_DIR,
            )
    except (NoDocumentsFoundError, KnowledgeBaseNotReadyError) as exc:
        finish_ingestion(False, str(exc))
        return False
    except IngestionError as exc:
        finish_ingestion(False, str(exc))
        return False
    except Exception as exc:
        error_text = str(exc)
        if "locked" in error_text.lower() or "winerror 5" in error_text.lower():
            finish_ingestion(False, VECTORSTORE_LOCKED_MESSAGE)
        else:
            finish_ingestion(False, f"Document ingestion failed: {exc}")
        return False

    finish_ingestion(True, "Documents ingested successfully.")
    return True


def render_eval_snapshot():
    """Show offline smoke and real KB evaluation snapshots separately."""
    st.markdown("### Evaluation Snapshot")

    st.caption("Offline Smoke Evaluation")
    offline_snapshot = load_eval_snapshot(
        OFFLINE_SMOKE_EVAL_RESULTS_PATH,
        missing_message="No offline smoke evaluation run found",
    )
    if offline_snapshot["status"] != "ready":
        st.caption(offline_snapshot["message"])
    else:
        payload = offline_snapshot["payload"]
        st.caption(f"Latency: {offline_snapshot['latency_display']}")
        for key in ("score", "pass_rate", "notes"):
            if key in payload:
                st.caption(f"{key.replace('_', ' ').title()}: {payload[key]}")

    st.caption("Real KB Evaluation")
    real_snapshot = load_eval_snapshot(
        REAL_KB_EVAL_RESULTS_PATH,
        missing_message="No real KB evaluation run found",
    )
    if real_snapshot["status"] != "ready":
        st.caption(real_snapshot["message"])
        return

    payload = real_snapshot["payload"]
    st.caption(f"Latency: {real_snapshot['latency_display']}")
    for key in ("score", "pass_rate", "mode", "notes"):
        if key in payload:
            st.caption(f"{key.replace('_', ' ').title()}: {payload[key]}")


def process_query(user_query: str, show_reasoning: bool, retrieval_scope: str):
    """Process a user query with safe health checks and scoped streaming output."""
    stats, health = build_health_snapshot()
    if st.session_state.ingestion_in_progress:
        show_generation_warning("Document ingestion is in progress. Please wait for it to finish.")
        return
    if not health["llm_provider_configured"]:
        show_generation_warning(LLM_CONFIG_WARNING)
        return
    if not health["vectorstore_ready"]:
        show_generation_warning(KB_NOT_READY_MESSAGE)
        return
    if st.session_state.pending_uploads or health.get("pending_upload_count", 0) > 0:
        show_generation_warning(PENDING_UPLOADS_MESSAGE)
        return
    if retrieval_scope in {"sample", "upload"} and int(stats.get("scope_chunk_counts", {}).get(retrieval_scope, 0) or 0) == 0:
        show_generation_warning(NO_SCOPE_DOCUMENTS_MESSAGE)
        return

    try:
        if st.session_state.agent is None:
            st.session_state.agent = create_agent()

        prompt, reasoning_trace = prepare_query(
            user_query,
            chat_history=st.session_state.messages,
            retrieval_scope=retrieval_scope,
            vectorstore_stats=stats,
            llm=st.session_state.agent,
        )

        with st.chat_message("assistant", avatar=get_chat_avatar("assistant")):
            full_response = st.write_stream(
                query_agent_stream(st.session_state.agent, prompt)
            )
            render_reasoning(reasoning_trace, show_reasoning)

        add_assistant_message(full_response, reasoning_trace)
    except (LLMProviderNotConfiguredError, KnowledgeBaseNotReadyError) as exc:
        show_generation_warning(str(exc))
    except RetrievalError as exc:
        show_generation_warning(str(exc))
    except Exception as exc:
        safe_llm_error = format_llm_error(exc, debug=is_debug_mode_enabled())
        if safe_llm_error:
            show_generation_warning(safe_llm_error)
            return

        error_text = str(exc)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
            show_generation_warning(
                "Rate limit reached. Please wait a moment and try again. Consider adding a GROQ_API_KEY for faster, more reliable responses."
            )
        else:
            show_generation_warning(f"Request failed cleanly: {error_text}")


SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
stats, health = build_health_snapshot()
current_scope = resolve_scope(health)
interaction_locked = st.session_state.ingestion_in_progress
uploads_pending = st.session_state.pending_uploads or health.get("pending_upload_count", 0) > 0
chat_disabled = interaction_locked or not (health["llm_provider_configured"] and health["vectorstore_ready"]) or uploads_pending
scope_options = available_scope_options(health)
current_scope_label = SCOPE_LABELS[current_scope] if SCOPE_LABELS[current_scope] in scope_options else scope_options[0]

with st.sidebar:
    st.markdown("### Knowledge Base")
    st.markdown("---")

    if health["vectorstore_ready"]:
        st.markdown(
            '<span class="status-badge status-ready">Ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-pending">Not ready</span>',
            unsafe_allow_html=True,
        )
        st.caption(KB_NOT_READY_MESSAGE)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{health["document_count"]}</div><div class="stat-label">Documents</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{health["chunk_count"]}</div><div class="stat-label">Chunks</div></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.status_message:
        st.success(st.session_state.status_message)
    if st.session_state.upload_notice:
        st.success(st.session_state.upload_notice)
    if st.session_state.last_ingestion_error:
        st.warning(st.session_state.last_ingestion_error)
    if uploads_pending:
        st.warning(PENDING_UPLOADS_MESSAGE)

    st.markdown("---")
    st.markdown("### Document Scope")
    selected_scope_label = st.selectbox(
        "Document scope",
        options=scope_options,
        index=scope_options.index(current_scope_label),
        label_visibility="collapsed",
        disabled=interaction_locked,
    )
    st.session_state.scope = LABEL_TO_SCOPE[selected_scope_label]

    st.markdown("---")
    st.markdown("### LLM Provider")
    st.markdown(
        f'<div class="provider-badge">{_get_provider_name()}</div>',
        unsafe_allow_html=True,
    )
    if not health["llm_provider_configured"]:
        st.warning(
            "Add GROQ_API_KEY or GOOGLE_API_KEY in .env locally, or in Streamlit secrets on deployment, to enable chat answers."
        )

    st.markdown("---")
    st.markdown("### App Health")
    st.caption(f"Vectorstore ready: {'Yes' if health['vectorstore_ready'] else 'No'}")
    st.caption(f"Sample docs available: {health['sample_document_count']}")
    st.caption(f"Uploaded docs available: {health['upload_document_count']}")
    st.caption(f"Indexed uploaded docs: {health['indexed_upload_document_count']}")
    st.caption(
        f"Provider configured: {'Yes' if health['llm_provider_configured'] else 'No'}"
    )
    for directory_name, exists in health["required_directories"].items():
        st.caption(f"{directory_name.title()} dir: {'OK' if exists else 'Missing'}")

    st.markdown("---")
    st.markdown("### Document Ingestion")
    if st.button("Generate Sample PDFs", use_container_width=True, disabled=interaction_locked):
        with st.spinner("Generating sample PDFs..."):
            generate_all_documents()
        st.session_state.setup_attempted = True
        st.session_state.status_message = "Sample PDFs generated."
        st.session_state.last_ingestion_error = ""
        time.sleep(0.2)
        st.rerun()

    if st.button(
        "Ingest Documents",
        use_container_width=True,
        type="primary",
        disabled=interaction_locked,
    ):
        if run_manual_ingestion():
            time.sleep(0.2)
            st.rerun()

    st.markdown("---")
    st.markdown("### Upload Custom PDFs")
    uploaded_files = st.file_uploader(
        f"Upload PDFs up to {MAX_UPLOAD_SIZE_MB} MB each",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        disabled=interaction_locked,
    )
    if uploaded_files and not interaction_locked:
        handle_uploaded_files(uploaded_files)

    st.markdown("---")
    show_reasoning = st.toggle("Show Source Traces", value=True, disabled=interaction_locked)

    st.markdown("---")
    render_eval_snapshot()

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True, disabled=interaction_locked):
        st.session_state.messages = []
        clear_runtime_objects()
        st.rerun()

st.markdown(
    f"""
    <div class="main-header">
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.last_ingestion_error:
    st.warning(st.session_state.last_ingestion_error)

if not health["llm_provider_configured"]:
    st.warning(
        "Chat is disabled until you add GROQ_API_KEY or GOOGLE_API_KEY in .env locally, or in Streamlit secrets on deployment. Document upload and ingestion still work without an LLM."
    )
if uploads_pending:
    st.warning(PENDING_UPLOADS_MESSAGE)
if not health["vectorstore_ready"]:
    st.info(KB_NOT_READY_MESSAGE)

if not st.session_state.messages:
    st.markdown("#### Try asking")
    columns = st.columns(2)
    for index, question in enumerate(SAMPLE_QUESTIONS[:6]):
        with columns[index % 2]:
            if st.button(
                question,
                key=f"sample_{index}",
                use_container_width=True,
                disabled=chat_disabled,
            ):
                st.session_state.pending_query = question
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=get_chat_avatar(message["role"])):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_reasoning(message.get("reasoning", []), show_reasoning)

pending_query = st.session_state.pending_query
if pending_query and not interaction_locked:
    st.session_state.pending_query = None
    process_query(pending_query, show_reasoning, st.session_state.scope)

user_input = st.chat_input(
    "Ask about past projects, tech stacks, proposals...",
    disabled=chat_disabled,
)
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=get_chat_avatar("user")):
        st.markdown(user_input)
    process_query(user_input, show_reasoning, st.session_state.scope)
