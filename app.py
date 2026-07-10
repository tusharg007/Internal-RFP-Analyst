import json
import time
from pathlib import Path

import streamlit as st

# Must be first Streamlit command
st.set_page_config(
    page_title="Internal RFP Analyst",
    page_icon="?",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agent import (
    _get_provider_name,
    create_agent,
    is_llm_provider_configured,
    prepare_query,
    query_agent_stream,
)
from config import APP_TITLE, APP_SUBTITLE, DATA_DIR, SAMPLE_QUESTIONS
from rag_engine import get_vectorstore_stats, ingest_documents
from rfp_analyst.exceptions import (
    IngestionError,
    KnowledgeBaseNotReadyError,
    LLMProviderNotConfiguredError,
    RFPAnalystError,
    UnsupportedFileError,
)
from rfp_analyst.health import get_app_health
from rfp_analyst.ui.helpers import format_latency_display, get_chat_avatar
from rfp_analyst.uploads import validate_uploaded_pdf

APP_ROOT = Path(__file__).resolve().parent
EVAL_RESULTS_PATH = APP_ROOT / "evals" / "results.json"
LLM_CONFIGURATION_WARNING = (
    "No LLM provider is configured. Add GROQ_API_KEY or GOOGLE_API_KEY in your .env file locally, "
    "or in Streamlit secrets on deployment."
)
KNOWLEDGE_BASE_NOT_READY_MESSAGE = (
    "Knowledge base is not ready. Generate or upload PDFs and click Ingest Documents."
)

st.markdown("""
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
        transition: transform 0.2s ease;
    }
    .stat-card:hover { transform: translateY(-2px); }
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

    .evaluation-box {
        background: rgba(0, 180, 255, 0.08);
        border: 1px solid rgba(0, 180, 255, 0.25);
        border-radius: 12px;
        padding: 0.9rem;
        margin-top: 0.5rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #151823 100%);
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
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def _load_evaluation_snapshot():
    if not EVAL_RESULTS_PATH.exists():
        return None
    try:
        return json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _render_reasoning_trace(reasoning_trace: list):
    with st.expander("Sources Used", expanded=False):
        for step in reasoning_trace:
            if "tool" in step:
                details = ", ".join(
                    f"{key}: {value}" for key, value in step.get("input", {}).items()
                )
                st.markdown(
                    f'<div class="reasoning-box"><strong>Tool:</strong> {step["tool"]}<br><em>{details}</em></div>',
                    unsafe_allow_html=True,
                )
            elif "tool_response" in step:
                st.markdown(
                    f'<div class="reasoning-box"><strong>{step["tool_response"]}</strong><br><em>{step["snippet"][:150]}...</em></div>',
                    unsafe_allow_html=True,
                )


def _handle_uploads(uploaded_files):
    if not uploaded_files:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    uploaded_count = 0
    for uploaded_file in uploaded_files:
        try:
            safe_name = validate_uploaded_pdf(uploaded_file)
            save_path = DATA_DIR / safe_name
            with open(save_path, "wb") as handle:
                handle.write(uploaded_file.getbuffer())
            uploaded_count += 1
        except UnsupportedFileError as error:
            st.warning(str(error))
    if uploaded_count:
        st.success(f"Uploaded {uploaded_count} file(s). Click 'Ingest Documents' to index.")


provider_name = _get_provider_name()
health = get_app_health(provider_name)
llm_configured = health["llm_provider_configured"]
kb_ready = health["vectorstore_ready"]

with st.sidebar:
    st.markdown("### Knowledge Base")
    st.markdown("---")

    stats = get_vectorstore_stats()

    if stats["status"] == "ready":
        st.markdown('<span class="status-badge status-ready">Ready</span>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-number">{stats["total_documents"]}</div>'
                f'<div class="stat-label">Documents</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-number">{stats["total_chunks"]}</div>'
                f'<div class="stat-label">Chunks</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<span class="status-badge status-pending">Not Initialized</span>', unsafe_allow_html=True)
        st.info(KNOWLEDGE_BASE_NOT_READY_MESSAGE)

    st.markdown("---")
    st.markdown("### LLM Provider")
    st.markdown(f'<div class="provider-badge">{provider_name}</div>', unsafe_allow_html=True)
    if not llm_configured:
        st.warning(LLM_CONFIGURATION_WARNING)

    st.markdown("---")
    st.markdown("### App Health")
    st.caption(f"Vectorstore ready: {'Yes' if health['vectorstore_ready'] else 'No'}")
    st.caption(f"Required directories ready: {'Yes' if all(item['exists'] for item in health['required_directories'].values()) else 'No'}")

    evaluation_snapshot = _load_evaluation_snapshot()
    st.markdown("---")
    st.markdown("### Evaluation Snapshot")
    if evaluation_snapshot:
        metrics = evaluation_snapshot.get("metrics", {})
        st.markdown(
            (
                '<div class="evaluation-box">'
                f"Pass Rate: {evaluation_snapshot.get('passed_questions', 0)}/{evaluation_snapshot.get('total_questions', 0)}<br>"
                f"Retrieval Hit Rate: {metrics.get('retrieval_hit_rate', 0):.2f}<br>"
                f"Citation Coverage: {metrics.get('citation_coverage', 0):.2f}<br>"
                f"Grounded Answer Score: {metrics.get('grounded_answer_score', 0):.2f}<br>"
                f"Average Latency: {format_latency_display(metrics)}<br>"
                f"Tool Call Count: {metrics.get('tool_call_count', 0):.2f}<br>"
                f"Failure Rate: {metrics.get('failure_rate', 0):.2f}"
                '</div>'
            ),
            unsafe_allow_html=True,
        )
    else:
        st.info("No evaluation run found")

    st.markdown("---")
    st.markdown("### Document Ingestion")

    if st.button("Ingest Documents", use_container_width=True, type="primary"):
        with st.spinner("Processing documents..."):
            try:
                ingest_documents()
                st.success("Documents ingested successfully.")
                st.session_state.agent = None
                time.sleep(1)
                st.rerun()
            except (IngestionError, RFPAnalystError) as error:
                st.warning(str(error))
            except Exception as error:
                st.warning(f"Ingestion failed: {error}")

    st.markdown("---")
    st.markdown("### Upload Custom PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    _handle_uploads(uploaded_files)

    st.markdown("---")
    st.markdown("### Settings")
    show_reasoning = st.toggle("Show Source Traces", value=True)

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = None
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

if not kb_ready:
    st.info(KNOWLEDGE_BASE_NOT_READY_MESSAGE)
if not llm_configured:
    st.warning(LLM_CONFIGURATION_WARNING)

if not st.session_state.messages:
    st.markdown("#### Try asking:")
    cols = st.columns(2)
    for index, question in enumerate(SAMPLE_QUESTIONS[:6]):
        with cols[index % 2]:
            if st.button(
                question,
                key=f"sample_{index}",
                use_container_width=True,
                disabled=(not llm_configured or not kb_ready),
            ):
                st.session_state.pending_query = question
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=get_chat_avatar(message["role"])):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("reasoning") and show_reasoning:
            _render_reasoning_trace(message["reasoning"])


def _process_query(user_query: str):
    try:
        if not llm_configured:
            raise LLMProviderNotConfiguredError(LLM_CONFIGURATION_WARNING)
        if not kb_ready:
            raise KnowledgeBaseNotReadyError(KNOWLEDGE_BASE_NOT_READY_MESSAGE)

        if st.session_state.agent is None:
            st.session_state.agent = create_agent()

        prompt, reasoning_trace = prepare_query(user_query, chat_history=st.session_state.messages)

        with st.chat_message("assistant", avatar=get_chat_avatar("assistant")):
            full_response = st.write_stream(query_agent_stream(st.session_state.agent, prompt))
            if reasoning_trace and show_reasoning:
                _render_reasoning_trace(reasoning_trace)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response, "reasoning": reasoning_trace}
        )
    except (LLMProviderNotConfiguredError, KnowledgeBaseNotReadyError, RFPAnalystError) as error:
        friendly_message = str(error)
        st.warning(friendly_message)
        st.session_state.messages.append({"role": "assistant", "content": friendly_message, "reasoning": []})
    except Exception as error:
        error_text = str(error)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
            friendly_message = (
                "Rate limit reached. Please wait a moment and try again. "
                "Consider adding a GROQ_API_KEY for faster, more reliable responses."
            )
        else:
            friendly_message = f"Error: {error_text}"
        st.warning(friendly_message)
        st.session_state.messages.append({"role": "assistant", "content": friendly_message, "reasoning": []})


pending_query = st.session_state.pending_query
if pending_query:
    st.session_state.pending_query = None
    _process_query(pending_query)

if user_input := st.chat_input(
    "Ask about past projects, tech stacks, proposals...",
    disabled=(not llm_configured or not kb_ready),
):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=get_chat_avatar("user")):
        st.markdown(user_input)
    _process_query(user_input)
