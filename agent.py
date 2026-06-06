"""
RAG Query Engine — Fast retrieval-augmented generation with streaming.
Uses Groq (primary, fastest) or Gemini (fallback) for LLM generation.
Local embeddings for retrieval — zero API overhead for search.
"""

from langchain_core.messages import HumanMessage

from config import (
    GROQ_API_KEY,
    GOOGLE_API_KEY,
    GROQ_MODEL,
    GEMINI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    AGENT_SYSTEM_PROMPT,
    RETRIEVAL_K,
)
from rag_engine import similarity_search, get_vectorstore_stats


# ─── LLM Initialization ──────────────────────────────────────────────────────

def _get_provider_name():
    """Return which LLM provider is active."""
    if GROQ_API_KEY:
        return f"Groq ({GROQ_MODEL})"
    elif GOOGLE_API_KEY:
        return f"Gemini ({GEMINI_MODEL})"
    return "None"


def get_llm():
    """Get LLM with automatic provider selection. Groq preferred (faster)."""
    if GROQ_API_KEY:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    elif GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )
    else:
        raise ValueError(
            "No API key found. Set GROQ_API_KEY (recommended, https://console.groq.com) "
            "or GOOGLE_API_KEY (https://aistudio.google.com/apikey)"
        )


def create_agent():
    """Create the LLM instance."""
    return get_llm()


# ─── Context Retrieval ────────────────────────────────────────────────────────

def _retrieve_context(user_query: str):
    """Retrieve relevant documents and format context. Local, instant."""
    try:
        results = similarity_search(user_query, k=RETRIEVAL_K)
    except Exception:
        return "", [], []

    context_parts = []
    sources_used = []
    for doc, score in results:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source: {source}, Page {int(page) + 1}]\n{doc.page_content}"
        )
        sources_used.append({"source": source, "page": page, "score": f"{score:.2f}"})

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."
    return context, context_parts, sources_used


def _build_prompt(user_query, context, chat_history=None):
    """Build the full prompt with context, history, and question."""
    stats = get_vectorstore_stats()
    project_list = "\n".join(
        f"  - {name}" for name in stats.get("document_names", [])
    ) or "  No documents ingested yet."

    history_text = ""
    if chat_history:
        recent = [m for m in chat_history[-6:] if m.get("role") in ("user", "assistant")]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content'][:300]}\n"

    return f"""{AGENT_SYSTEM_PROMPT}

── Available Documents ──
{project_list}
Total: {stats.get('total_documents', 0)} documents, {stats.get('total_chunks', 0)} chunks

── Retrieved Context ──
{context}

{f"── Recent Conversation ──{chr(10)}{history_text}" if history_text else ""}
── Question ──
{user_query}

Answer thoroughly with source citations."""


# ─── Query Functions ──────────────────────────────────────────────────────────

def prepare_query(user_query: str, chat_history: list = None):
    """Prepare retrieval context and prompt. Returns (sources, prompt)."""
    context, context_parts, sources_used = _retrieve_context(user_query)
    prompt = _build_prompt(user_query, context, chat_history)

    reasoning_trace = []
    if sources_used:
        reasoning_trace.append({
            "tool": "search_knowledge_base",
            "input": {"query": user_query},
        })
        for s in sources_used[:3]:
            reasoning_trace.append({
                "tool_response": f"{s['source']} (Page {s['page']})",
                "snippet": next(
                    (p[:200] for p in context_parts if s["source"] in p), "..."
                ),
            })

    return prompt, reasoning_trace


def query_agent_stream(llm, prompt):
    """Stream LLM response chunks. Yields text as it's generated."""
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content


def query_agent(llm, user_query: str, thread_id: str = "default", chat_history: list = None):
    """Non-streaming query (backward compatible). Single LLM call."""
    prompt, reasoning_trace = prepare_query(user_query, chat_history)
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "answer": response.content,
        "reasoning_trace": reasoning_trace,
        "all_messages": [],
    }


if __name__ == "__main__":
    print(f"Active provider: {_get_provider_name()}")
    llm = create_agent()
    result = query_agent(llm, "List all projects with their timelines")
    print(result["answer"][:500])
