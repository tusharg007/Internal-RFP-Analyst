"""
LangGraph ReAct Agent — AI-powered RFP Analyst with multiple tools.
Uses Google Gemini as LLM, ChromaDB retriever, and conversation memory.
Includes rate-limit retry logic for Gemini free-tier quotas.
"""

import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# ─── Rate-Limit Constants ─────────────────────────────────────────────────────
LLM_MAX_RETRIES = 2          # keep low to avoid long hangs
LLM_INITIAL_BACKOFF = 5      # seconds
LLM_MAX_BACKOFF = 20         # never wait more than 20s per retry

from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    AGENT_SYSTEM_PROMPT,
    RETRIEVAL_K,
)
from rag_engine import similarity_search, get_vectorstore_stats, load_vectorstore


# ─── Tool Definitions ────────────────────────────────────────────────────────


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for information about past projects,
    proposals, RFP responses, and case studies. Use this tool when the user
    asks about specific project details, tech stacks, teams, budgets, or outcomes.
    Returns relevant passages with source citations."""
    try:
        results = similarity_search(query, k=RETRIEVAL_K)
        if not results:
            return "No relevant documents found for this query."

        formatted = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(
                f"--- Result {i} (Relevance: {score:.2f}) ---\n"
                f"[Source: {source}, Page {int(page) + 1}]\n"
                f"{doc.page_content}\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@tool
def get_document_summary(document_name: str) -> str:
    """Get a summary of a specific document from the knowledge base.
    Use this when the user asks about a particular project or document.
    Provide a partial or full document name to search for."""
    try:
        vectorstore = load_vectorstore()
        all_data = vectorstore._collection.get()

        matching_chunks = []
        for i, meta in enumerate(all_data["metadatas"]):
            source = meta.get("source_file", "")
            if document_name.lower() in source.lower():
                matching_chunks.append(
                    {"content": all_data["documents"][i], "metadata": meta}
                )

        if not matching_chunks:
            return f"No document found matching '{document_name}'. Use list_all_projects to see available documents."

        # Return first few chunks as summary
        summary_parts = [f"Document: {matching_chunks[0]['metadata'].get('source_file', 'Unknown')}\n"]
        for chunk in matching_chunks[:5]:
            summary_parts.append(chunk["content"])

        return "\n\n".join(summary_parts)
    except Exception as e:
        return f"Error retrieving document summary: {str(e)}"


@tool
def list_all_projects() -> str:
    """List all projects and documents available in the knowledge base.
    Use this when the user asks what projects exist, wants an overview,
    or needs to know what information is available."""
    try:
        stats = get_vectorstore_stats()
        if stats["status"] == "not_initialized":
            return "Knowledge base is not initialized. Please ingest documents first."

        lines = [
            f"Knowledge Base Overview:",
            f"Total Documents: {stats['total_documents']}",
            f"Total Indexed Chunks: {stats['total_chunks']}",
            f"\nAvailable Documents:",
        ]
        for i, name in enumerate(stats["document_names"], 1):
            # Clean up filename for display
            display_name = name.replace("_", " ").replace(".pdf", "")
            # Remove leading number prefix
            if display_name[:3].strip().isdigit():
                display_name = display_name[3:].strip()
            lines.append(f"  {i}. {display_name} ({name})")

        return "\n".join(lines)
    except Exception as e:
        return f"Error listing projects: {str(e)}"


@tool
def compare_projects(project_keywords: str) -> str:
    """Compare multiple projects by searching for each and presenting differences.
    Provide comma-separated keywords for the projects to compare.
    Example: 'banking, healthcare' or 'Azure, AWS'.
    Use this when the user wants to compare tech stacks, budgets, or approaches."""
    try:
        keywords = [k.strip() for k in project_keywords.split(",")]
        comparison_data = {}

        for keyword in keywords:
            results = similarity_search(keyword, k=3)
            if results:
                sources = set()
                content_parts = []
                for doc, score in results:
                    source = doc.metadata.get("source_file", "Unknown")
                    sources.add(source)
                    content_parts.append(doc.page_content)
                comparison_data[keyword] = {
                    "sources": list(sources),
                    "content": "\n".join(content_parts[:2]),
                }

        if not comparison_data:
            return "No matching projects found for the given keywords."

        output = ["Project Comparison Results:\n"]
        for keyword, data in comparison_data.items():
            output.append(f"=== Projects matching '{keyword}' ===")
            output.append(f"Found in: {', '.join(data['sources'])}")
            output.append(f"Key Details:\n{data['content']}\n")

        return "\n".join(output)
    except Exception as e:
        return f"Error comparing projects: {str(e)}"


# ─── Agent Setup ──────────────────────────────────────────────────────────────

TOOLS = [search_knowledge_base, get_document_summary, list_all_projects, compare_projects]


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
    """Create a LangGraph ReAct agent with tools and memory."""
    llm = get_llm()
    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        checkpointer=memory,
        prompt=AGENT_SYSTEM_PROMPT,
    )
    return agent


def query_agent(agent, user_query: str, thread_id: str = "default"):
    """Send a query to the agent and get a response with reasoning trace.
    
    Includes automatic retry with exponential backoff for Gemini rate limits.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Retry loop for rate-limit errors
    backoff = LLM_INITIAL_BACKOFF
    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_query)]},
                config=config,
            )
            break  # success
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            is_rate_limit = (
                "429" in error_str
                or "resource_exhausted" in error_str
                or "quota" in error_str
            )
            # If quota is fully exhausted (limit: 0), fail immediately
            if "limit: 0" in error_str or "limit:0" in error_str:
                raise
            if is_rate_limit and attempt < LLM_MAX_RETRIES:
                wait = min(backoff, LLM_MAX_BACKOFF)
                logger.warning(
                    f"LLM rate limit hit (attempt {attempt}/{LLM_MAX_RETRIES}). "
                    f"Retrying in {wait}s..."
                )
                print(f"⏳ LLM rate limit — retrying in {wait}s ({attempt}/{LLM_MAX_RETRIES})...")
                time.sleep(wait)
                backoff *= 2
            else:
                raise
    else:
        raise last_error  # all retries exhausted

    # Extract the final AI message
    messages = response["messages"]
    final_message = messages[-1]

    # Collect tool calls for reasoning trace
    reasoning_trace = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                reasoning_trace.append(
                    {"tool": tc["name"], "input": tc["args"]}
                )
        if msg.type == "tool":
            reasoning_trace.append(
                {"tool_response": msg.name, "snippet": msg.content[:200] + "..."}
            )

    return {
        "answer": final_message.content,
        "reasoning_trace": reasoning_trace,
        "all_messages": messages,
    }


if __name__ == "__main__":
    print("Creating agent...")
    agent = create_agent()

    test_queries = [
        "What tech stack did we use for the banking audit?",
        "Which projects used Azure?",
        "Compare banking and healthcare projects",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("=" * 60)
        result = query_agent(agent, q)
        print(f"A: {result['answer'][:500]}")
        print(f"\nTools used: {[r.get('tool', r.get('tool_response', '?')) for r in result['reasoning_trace']]}")
