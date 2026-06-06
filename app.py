"""
Streamlit Dashboard — Internal RFP Analyst Chatbot.
Professional chatbot UI with streaming, citations, and agent reasoning traces.
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
from agent import create_agent, query_agent

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

    .sample-btn {
        background: rgba(108, 99, 255, 0.1);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        color: #c0c0c0;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
        width: 100%;
        text-align: left;
    }
    .sample-btn:hover {
        background: rgba(108, 99, 255, 0.2);
        border-color: #6C63FF;
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
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{int(time.time())}"
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

    # Ingestion
    st.markdown("### 📥 Document Ingestion")

    if st.button("🚀 Ingest Documents", use_container_width=True, type="primary"):
        with st.spinner("Processing documents..."):
            try:
                progress_bar = st.progress(0, text="Loading PDFs...")
                progress_bar.progress(20, text="Loading PDFs...")
                ingest_documents()
                progress_bar.progress(60, text="Embedding chunks...")
                time.sleep(0.5)
                progress_bar.progress(90, text="Storing vectors...")
                time.sleep(0.3)
                progress_bar.progress(100, text="Complete!")
                st.success("✅ Documents ingested successfully!")
                # Reset agent to pick up new data
                st.session_state.agent = None
                time.sleep(1)
                st.rerun()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    if "limit: 0" in error_str or "limit:0" in error_str:
                        st.error(
                            "🚫 **Daily API quota exhausted.** The Gemini free tier daily limit has been reached. "
                            "Please wait a few minutes for the quota to reset, then try again."
                        )
                    else:
                        st.warning(
                            "⏳ **Rate limit reached.** Please wait a moment and try again."
                        )
                else:
                    st.error(f"❌ Error: {error_str}")

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
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.3, 0.1)
    num_sources = st.slider("Sources to Retrieve", 1, 8, 4)
    show_reasoning = st.toggle("Show Agent Reasoning", value=True)

    # Reset
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"session_{int(time.time())}"
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
        if msg["role"] == "assistant" and "reasoning" in msg and show_reasoning:
            with st.expander("🧠 Agent Reasoning Trace", expanded=False):
                for step in msg["reasoning"]:
                    if "tool" in step:
                        st.markdown(
                            f'<div class="reasoning-box">'
                            f'🔧 <strong>Tool:</strong> {step["tool"]}<br>'
                            f'📥 <strong>Input:</strong> {step["input"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    elif "tool_response" in step:
                        st.markdown(
                            f'<div class="reasoning-box">'
                            f'📤 <strong>Response from:</strong> {step["tool_response"]}<br>'
                            f'<em>{step["snippet"][:150]}...</em>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

# ─── Process pending query from sample buttons ───────────────────────────────

pending = st.session_state.pending_query
if pending:
    st.session_state.pending_query = None  # clear immediately to avoid re-processing
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analyzing knowledge base..."):
            try:
                if st.session_state.agent is None:
                    st.session_state.agent = create_agent()

                result = query_agent(
                    st.session_state.agent,
                    pending,
                    thread_id=st.session_state.thread_id,
                )

                answer = result["answer"]
                reasoning = result["reasoning_trace"]

                st.markdown(answer)

                if reasoning and show_reasoning:
                    with st.expander("🧠 Agent Reasoning Trace", expanded=False):
                        for step in reasoning:
                            if "tool" in step:
                                st.markdown(
                                    f'<div class="reasoning-box">'
                                    f'🔧 <strong>Tool:</strong> {step["tool"]}<br>'
                                    f'📥 <strong>Input:</strong> {step["input"]}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            elif "tool_response" in step:
                                st.markdown(
                                    f'<div class="reasoning-box">'
                                    f'📤 <strong>Response from:</strong> {step["tool_response"]}<br>'
                                    f'<em>{step["snippet"][:150]}...</em>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "reasoning": reasoning,
                    }
                )

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    if "limit: 0" in error_str or "limit:0" in error_str:
                        friendly_msg = (
                            "🚫 **Daily API quota exhausted.** The Gemini free tier daily limit has been reached. "
                            "Please wait a few minutes for the quota to reset, then try again."
                        )
                        st.error(friendly_msg)
                    else:
                        friendly_msg = (
                            "⏳ **Rate limit reached.** Please wait a moment and try again."
                        )
                        st.warning(friendly_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": friendly_msg, "reasoning": []}
                    )
                else:
                    error_msg = f"❌ Error: {error_str}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "reasoning": []}
                    )

# ─── Chat input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about past projects, tech stacks, proposals..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analyzing knowledge base..."):
            try:
                # Initialize agent if needed
                if st.session_state.agent is None:
                    st.session_state.agent = create_agent()

                result = query_agent(
                    st.session_state.agent,
                    user_input,
                    thread_id=st.session_state.thread_id,
                )

                answer = result["answer"]
                reasoning = result["reasoning_trace"]

                st.markdown(answer)

                # Show reasoning
                if reasoning and show_reasoning:
                    with st.expander("🧠 Agent Reasoning Trace", expanded=False):
                        for step in reasoning:
                            if "tool" in step:
                                st.markdown(
                                    f'<div class="reasoning-box">'
                                    f'🔧 <strong>Tool:</strong> {step["tool"]}<br>'
                                    f'📥 <strong>Input:</strong> {step["input"]}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            elif "tool_response" in step:
                                st.markdown(
                                    f'<div class="reasoning-box">'
                                    f'📤 <strong>Response from:</strong> {step["tool_response"]}<br>'
                                    f'<em>{step["snippet"][:150]}...</em>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                # Store message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "reasoning": reasoning,
                    }
                )

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    if "limit: 0" in error_str or "limit:0" in error_str:
                        friendly_msg = (
                            "🚫 **Daily API quota exhausted.** The Gemini free tier daily limit has been reached. "
                            "Please wait a few minutes for the quota to reset, then try again."
                        )
                        st.error(friendly_msg)
                    else:
                        friendly_msg = (
                            "⏳ **Rate limit reached.** Please wait a moment and try again."
                        )
                        st.warning(friendly_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": friendly_msg, "reasoning": []}
                    )
                else:
                    error_msg = f"❌ Error: {error_str}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "reasoning": []}
                    )
