"""Runtime helpers for simple and agentic RAG modes."""

from __future__ import annotations

from config import AGENT_MODE
from rag_engine import get_vectorstore_stats
from rfp_analyst.agent.graph import (
    prepare_query_payload as graph_prepare_query_payload,
    run_agent_graph,
    run_query as graph_run_query,
    stream_query_response as graph_stream_query_response,
)
from rfp_analyst.agent.prompts import build_simple_prompt
from rfp_analyst.agent.state import AgentState
from rfp_analyst.tools.search_kb import search_knowledge_base


KNOWLEDGE_BASE_NOT_READY_MESSAGE = (
    "Knowledge base is not ready. Generate or upload PDFs and click Ingest Documents."
)


def get_agent_mode() -> str:
    return AGENT_MODE if AGENT_MODE in {"simple", "agentic"} else "agentic"


def _stream_text(text: str, chunk_size: int = 120):
    for index in range(0, len(text), chunk_size):
        yield text[index:index + chunk_size]


def prepare_simple_query(user_query: str, chat_history: list | None = None) -> dict:
    stats = get_vectorstore_stats()
    if stats.get("status") != "ready":
        return {
            "mode": "simple",
            "prompt": "",
            "documents": [],
            "reasoning_trace": [{"tool": "knowledge_base_status", "input": {"status": "not_initialized"}}],
            "prebuilt_answer": KNOWLEDGE_BASE_NOT_READY_MESSAGE,
        }

    result = search_knowledge_base(user_query)
    if not result["documents"]:
        return {
            "mode": "simple",
            "prompt": "",
            "documents": [],
            "reasoning_trace": [{"tool": "search_knowledge_base", "input": {"query": user_query, "mode": "simple"}}],
            "prebuilt_answer": "I couldn't find relevant documents for this request. Please ingest more documents or refine the question.",
        }

    prompt = build_simple_prompt(user_query, result["context"], stats, chat_history)
    reasoning_trace = [{"tool": "search_knowledge_base", "input": {"query": user_query, "mode": "simple"}}]
    for source in result["sources"][:3]:
        reasoning_trace.append(
            {
                "tool_response": f"{source['source']} (Page {source['page'] + 1})",
                "snippet": source["snippet"],
            }
        )
    return {
        "mode": "simple",
        "prompt": prompt,
        "documents": result["documents"],
        "reasoning_trace": reasoning_trace,
        "prebuilt_answer": None,
    }


def prepare_agentic_query(user_query: str, chat_history: list | None = None) -> dict:
    state = run_agent_graph(AgentState(query=user_query, chat_history=chat_history or []))
    return {
        "mode": "agentic",
        "prompt": state.prompt,
        "documents": state.retrieved_documents,
        "reasoning_trace": state.tool_trace,
        "prebuilt_answer": state.final_answer or None,
    }


def prepare_query_payload(user_query: str, chat_history: list | None = None) -> dict:
    """Deprecated compatibility wrapper for the canonical graph runtime."""
    return graph_prepare_query_payload(user_query, chat_history)


def stream_query_response(llm, payload):
    """Deprecated compatibility wrapper for the canonical graph runtime."""
    if isinstance(payload, str):
        payload = {"prompt": payload, "response_mode": "llm", "retrieved_documents": []}
    yield from graph_stream_query_response(llm, payload)


def run_query(llm, user_query: str, chat_history: list | None = None) -> dict:
    """Deprecated compatibility wrapper for the canonical graph runtime."""
    return graph_run_query(llm, user_query, chat_history=chat_history)
