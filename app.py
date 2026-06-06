"""
Streamlit Dashboard — Internal RFP Analyst Chatbot.
Professional chatbot UI with streaming responses, citations, and source traces.
"""

import streamlit as st
import time
from pathlib import Path

# Must be first Streamlit command
st.set_page_config(
    page_title="Internal RFP Analyst",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import APP_TITLE, APP_SUBTITLE, SAMPLE_QUESTIONS, DATA_DIR
from rag_engine import ingest_documents, get_vectorstore_stats
from agent import create_agent, prepare_query, query_agent_stream, _get_provider_name

# ─── Custom CSS ───────────────────────────────────────────────────────────────

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

    .citation-tag {
        display: inline-block;
        background: rgba(108, 99, 255, 0.15);
        color: #9D97FF;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 2px;
    }

    .stChatMessage { border-radius: 12px; }

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


# ─── Session State Initialization ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ─── Auto-Setup (for Streamlit Cloud: generate PDFs & ingest if needed) ───────

if "auto_setup_done" not in st.session_state:
    stats = get_vectorstore_stats()
    if stats["status"] == "not_initialized":
        with st.spinner("🔧 First-time setup: Generating documents & building knowledge base..."):
            from document_generator import generate_all_documents
            generate_all_documents()
            ingest_documents()
    st.session_state.auto_setup_done = True


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Knowledge Base")
    st.markdown("---")

    # Stats
    stats = get_vectorstore_stats()

    if stats["status"] == "ready":
        st.markdown(
            '<span class="status-badge status-ready">● Ready</span>',
            unsafe_allow_html=True,
        )
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
        st.markdown(
            '<span class="status-badge status-pending">● Not Initialized</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # LLM Provider info
    st.markdown("### 🤖 LLM Provider")
    provider_name = _get_provider_name()
    st.markdown(f'<div class="provider-badge">⚡ {provider_name}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Ingestion
    st.markdown("### 📥 Document Ingestion")

    if st.button("🚀 Ingest Documents", use_container_width=True, type="primary"):
        with st.spinner("Processing documents..."):
            try:
                ingest_documents()
                st.success("✅ Documents ingested successfully!")
                st.session_state.agent = None
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Upload custom PDFs
    st.markdown("---")
    st.markdown("### 📄 Upload Custom PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for uf in uploaded_files:
            save_path = DATA_DIR / uf.name
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} file(s). Click 'Ingest Documents' to index.")

    # Settings
    st.markdown("---")
    st.markdown("### 🎛️ Settings")
    show_reasoning = st.toggle("Show Source Traces", value=True)

    # Reset
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = None
        st.rerun()


# ─── Main Area ────────────────────────────────────────────────────────────────

# Header
st.markdown(
    f"""
    <div class="main-header">
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sample questions (only when no messages)
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS[:6]):
        with cols[i % 2]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_query = q
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

        # Show reasoning trace if available
        if msg["role"] == "assistant" and "reasoning" in msg and msg["reasoning"] and show_reasoning:
            with st.expander("📚 Sources Used", expanded=False):
                for step in msg["reasoning"]:
                    if "tool" in step:
                        st.markdown(
                            f'<div class="reasoning-box">'
                            f'🔧 <strong>Retrieval Query:</strong> {step["input"].get("query", "")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    elif "tool_response" in step:
                        st.markdown(
                            f'<div class="reasoning-box">'
                            f'📄 <strong>{step["tool_response"]}</strong><br>'
                            f'<em>{step["snippet"][:150]}...</em>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ─── Helper: Process a query with streaming ──────────────────────────────────

def _process_query(user_query: str):
    """Process a query: retrieve context, stream LLM response, store result."""
    try:
        # Initialize agent if needed
        if st.session_state.agent is None:
            st.session_state.agent = create_agent()

        # Prepare context (local, instant)
        prompt, reasoning_trace = prepare_query(
            user_query, chat_history=st.session_state.messages
        )

        # Stream the LLM response
        with st.chat_message("assistant", avatar="🤖"):
            full_response = st.write_stream(
                query_agent_stream(st.session_state.agent, prompt)
            )

            # Show sources after streaming completes
            if reasoning_trace and show_reasoning:
                with st.expander("📚 Sources Used", expanded=False):
                    for step in reasoning_trace:
                        if "tool" in step:
                            st.markdown(
                                f'<div class="reasoning-box">'
                                f'🔧 <strong>Retrieval Query:</strong> {step["input"].get("query", "")}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        elif "tool_response" in step:
                            st.markdown(
                                f'<div class="reasoning-box">'
                                f'📄 <strong>{step["tool_response"]}</strong><br>'
                                f'<em>{step["snippet"][:150]}...</em>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

        # Store message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "reasoning": reasoning_trace,
        })

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            friendly_msg = (
                "⏳ **Rate limit reached.** Please wait a moment and try again. "
                "Consider adding a GROQ_API_KEY for faster, more reliable responses."
            )
            st.warning(friendly_msg)
        else:
            friendly_msg = f"❌ Error: {error_str}"
            st.error(friendly_msg)
        st.session_state.messages.append(
            {"role": "assistant", "content": friendly_msg, "reasoning": []}
        )


# ─── Process pending query from sample buttons ───────────────────────────────

pending = st.session_state.pending_query
if pending:
    st.session_state.pending_query = None
    _process_query(pending)

# ─── Chat input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about past projects, tech stacks, proposals..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    _process_query(user_input)
