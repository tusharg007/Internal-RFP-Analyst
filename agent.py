"""
RAG Query Engine — Simple, fast retrieval-augmented generation.
Single LLM call per query (not multi-step ReAct). Uses local embeddings
for retrieval and Gemini for generation only.
"""

import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    AGENT_SYSTEM_PROMPT,
    RETRIEVAL_K,
)
from rag_engine import similarity_search, get_vectorstore_stats, get_retriever


def get_llm():
    """Initialize Google Gemini LLM."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
    )


def create_agent():
    """Create the RAG query engine (LLM instance).
    
    Returns the LLM — embeddings/retrieval are handled locally
    so only the LLM needs initialization.
    """
    return get_llm()


def query_agent(agent, user_query: str, thread_id: str = "default", chat_history: list = None):
    """Single-call RAG query: retrieve context → one LLM call → answer.
    
    This replaces the multi-step ReAct agent with a simple, fast pipeline:
    1. Retrieve relevant chunks (local embeddings, instant)
    2. Get project list from metadata (no API call)
    3. Single LLM call with all context
    
    Typical response time: 3-8 seconds.
    """
    llm = agent  # agent IS the LLM instance

    # ── Step 1: Retrieve relevant documents (local, instant) ──
    try:
        results = similarity_search(user_query, k=RETRIEVAL_K)
    except Exception:
        results = []

    context_parts = []
    sources_used = []
    for doc, score in results:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source: {source}, Page {int(page) + 1}] (relevance: {score:.2f})\n"
            f"{doc.page_content}"
        )
        sources_used.append({"source": source, "page": page, "score": f"{score:.2f}"})

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

    # ── Step 2: Get project list from metadata (no API call) ──
    stats = get_vectorstore_stats()
    project_list = "\n".join(
        f"  - {name}" for name in stats.get("document_names", [])
    ) or "  No documents ingested yet."

    # ── Step 3: Build conversation history context ──
    history_text = ""
    if chat_history:
        recent = [m for m in chat_history[-6:] if m.get("role") in ("user", "assistant")]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:300]  # truncate to save tokens
            history_text += f"{role}: {content}\n"

    # ── Step 4: Single LLM call ──
    prompt = f"""{AGENT_SYSTEM_PROMPT}

── Available Documents in Knowledge Base ──
{project_list}
Total: {stats.get('total_documents', 0)} documents, {stats.get('total_chunks', 0)} indexed chunks

── Retrieved Context (most relevant passages) ──
{context}

{f"── Recent Conversation ──{chr(10)}{history_text}" if history_text else ""}
── Current Question ──
{user_query}

Provide a thorough, well-structured answer grounded in the retrieved context above. 
Cite sources using [Source: document name, Page N] format."""

    response = llm.invoke([HumanMessage(content=prompt)])

    # ── Build reasoning trace (shows what was retrieved) ──
    reasoning_trace = []
    if sources_used:
        reasoning_trace.append({
            "tool": "search_knowledge_base",
            "input": {"query": user_query},
        })
        for s in sources_used[:3]:  # show top 3 sources
            reasoning_trace.append({
                "tool_response": f"{s['source']} (Page {s['page']})",
                "snippet": next(
                    (p[:200] for p in context_parts if s["source"] in p), "..."
                ),
            })

    return {
        "answer": response.content,
        "reasoning_trace": reasoning_trace,
        "all_messages": [],
    }


if __name__ == "__main__":
    print("Creating RAG engine...")
    llm = create_agent()

    test_queries = [
        "What tech stack did we use for the banking audit?",
        "Which projects used Azure?",
        "List all projects with their timelines",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("=" * 60)
        result = query_agent(llm, q)
        print(f"A: {result['answer'][:500]}")
