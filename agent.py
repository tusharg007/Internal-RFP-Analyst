"""RAG query engine with provider selection and streaming generation."""

import os

from config import GEMINI_MODEL, GROQ_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, get_api_keys
from rfp_analyst.agent.graph import (
    KB_NOT_READY_MESSAGE as KB_NOT_READY_MESSAGE,
    NO_SCOPE_DOCUMENTS_MESSAGE as NO_SCOPE_DOCUMENTS_MESSAGE,
    prepare_query_payload,
    run_query,
    stream_query_response,
)
from rfp_analyst.exceptions import LLMProviderNotConfiguredError
from rfp_analyst.ui.explainability import query_explanation


LLM_CONFIG_WARNING = (
    "No LLM provider is configured. Add GROQ_API_KEY or GOOGLE_API_KEY in your .env "
    "file locally, or in Streamlit secrets when deployed."
)
LLM_AUTH_ERROR_MESSAGE = (
    "LLM authentication failed. Your API key is invalid or expired. Update "
    "GROQ_API_KEY or GOOGLE_API_KEY in .env locally or Streamlit Secrets on deployment."
)
LLM_TOKEN_BUDGET_ERROR_MESSAGE = (
    "The analysis exceeded the current model's token budget. The request was "
    "compacted automatically, but the remaining evidence is still too large. "
    "Reduce the document scope or retry with a higher-capacity provider."
)
LLM_AUTH_ERROR_MARKERS = (
    "401",
    "invalid_api_key",
    "invalid api key",
    "authentication",
    "unauthorized",
)
LLM_TOKEN_BUDGET_ERROR_MARKERS = (
    "413",
    "request too large",
    "tokens per minute",
    "rate_limit_exceeded",
    "context length",
    "maximum context",
)


def is_debug_mode_enabled() -> bool:
    """Return whether safe diagnostic categories may be shown (never raw errors)."""
    return os.getenv("RFP_ANALYST_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def is_llm_auth_error(error: Exception) -> bool:
    """Detect provider authentication failures without exposing secret-bearing details."""
    error_text = str(error).lower()
    return any(marker in error_text for marker in LLM_AUTH_ERROR_MARKERS)


def is_llm_token_budget_error(error: Exception) -> bool:
    """Detect token-limit and request-size failures without exposing raw provider payloads."""
    error_text = str(error).lower()
    return any(marker in error_text for marker in LLM_TOKEN_BUDGET_ERROR_MARKERS)


def format_llm_error(error: Exception, debug: bool = False) -> str | None:
    """Return a safe user-facing LLM error message when the error is recognized."""
    if not is_llm_auth_error(error):
        if not is_llm_token_budget_error(error):
            return None
        if debug:
            return f"{LLM_TOKEN_BUDGET_ERROR_MESSAGE}\n\nError type: {type(error).__name__}"
        return LLM_TOKEN_BUDGET_ERROR_MESSAGE
    if debug:
        return f"{LLM_AUTH_ERROR_MESSAGE}\n\nError type: {type(error).__name__}"

    return LLM_AUTH_ERROR_MESSAGE


def sanitize_answer_text(answer: str) -> str:
    """Remove unsupported placeholder citations before showing an answer."""
    return str(answer or "").replace(
        "[Source: None]",
        "The retrieved evidence does not support this claim.",
    )


def _get_provider_name():
    """Return which LLM provider is active."""
    groq_api_key, google_api_key = get_api_keys()
    if groq_api_key:
        return f"Groq ({GROQ_MODEL})"
    if google_api_key:
        return f"Gemini ({GEMINI_MODEL})"
    return "Not configured"


def is_llm_provider_configured() -> bool:
    """Return whether any supported LLM provider is configured."""
    groq_api_key, google_api_key = get_api_keys()
    return bool(groq_api_key or google_api_key)


def get_llm():
    """Get the active LLM instance."""
    groq_api_key, google_api_key = get_api_keys()
    if groq_api_key:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_MODEL,
            api_key=groq_api_key,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    if google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=google_api_key,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )
    raise LLMProviderNotConfiguredError(LLM_CONFIG_WARNING)


def create_agent():
    """Create the LLM instance."""
    return get_llm()


def _payload_to_reasoning_trace(payload: dict) -> list[dict]:
    traces = []
    if "requested_retrieval_mode" in payload:
        traces.append({"tool": "execution_explanation", "explanation": query_explanation(payload)})
    source_label = {
        "private_kb": "Private KB",
        "kb": "Private KB",
        "web_search": "Web Search",
        "web": "Web Search",
        "direct": "Direct",
    }.get(payload.get("source_used"))
    if source_label:
        traces.append(
            {
                "tool": "source_used",
                "input_summary": "Answer source",
                "output_summary": source_label,
                "source_used": source_label,
            }
        )
    for step in payload.get("traces", []):
        tool_name = step.get("tool")
        if tool_name == "search_knowledge_base":
            traces.append(
                {
                    "tool": tool_name,
                    "input": step.get("input", {"query": payload.get("user_query", "")}),
                    "input_summary": step.get("input_summary", ""),
                    "output_summary": step.get("output_summary", ""),
                }
            )
            grouped_sources = {}
            for document in step.get("documents", []):
                dedupe_key = (
                    document.get("source"),
                    document.get("page"),
                )
                if dedupe_key in grouped_sources:
                    grouped_sources[dedupe_key]["match_count"] += 1
                    grouped_sources[dedupe_key]["chunk_ids"].append(document.get("chunk_id", ""))
                    if float(document.get("score", 0) or 0) > float(grouped_sources[dedupe_key]["score"] or 0):
                        grouped_sources[dedupe_key]["score"] = document.get("score", "0.00")
                    continue
                grouped_sources[dedupe_key] = {
                    "source": document.get("source", "Unknown"),
                    "page": int(document.get("page", 1) or 1),
                    "score": document.get("score", "0.00"),
                    "chunk_id": document.get("chunk_id", ""),
                    "chunk_ids": [document.get("chunk_id", "")],
                    "document_origin": document.get("document_origin", "sample"),
                    "match_count": 1,
                    "score_type": document.get("score_type", "vector_relevance"),
                }
            for source_card in grouped_sources.values():
                match_suffix = (
                    f" · {source_card['match_count']} matching chunks"
                    if source_card["match_count"] > 1
                    else ""
                )
                traces.append(
                    {
                        "tool_response": f"{source_card['source']} (Page {source_card['page']})",
                        "snippet": (
                            "graph-backed original evidence"
                            if source_card["score_type"] == "graph_witness"
                            else f"retrieval score {source_card['score']}"
                        )
                        + match_suffix,
                        "source": source_card["source"],
                        "page": source_card["page"],
                        "score": source_card["score"],
                        "chunk_id": source_card["chunk_id"],
                        "chunk_ids": source_card["chunk_ids"],
                        "document_origin": source_card["document_origin"],
                        "match_count": source_card["match_count"],
                    }
                )
        elif tool_name and tool_name not in {"health_check"}:
            traces.append(
                {
                    "tool": tool_name,
                    "input": {"query": payload.get("user_query", "")},
                    "input_summary": step.get("input_summary", ""),
                    "output_summary": step.get("output_summary", step.get("verification_status", "")),
                }
            )
    return traces or payload.get("traces", [])


def prepare_query(
    user_query: str,
    chat_history: list | None = None,
    retrieval_scope: str = "all",
    vectorstore_stats: dict | None = None,
    llm=None,
    *,
    retrieval_mode=None,
    retrieval_provider=None,
):
    """Prepare the graph payload and convert traces for the existing UI."""
    payload = prepare_query_payload(
        user_query=user_query,
        chat_history=chat_history,
        retrieval_scope=retrieval_scope,
        vectorstore_stats=vectorstore_stats,
        llm=llm,
        retrieval_mode=retrieval_mode,
        retrieval_provider=retrieval_provider,
    )
    reasoning_trace = _payload_to_reasoning_trace(payload)
    payload["_ui_reasoning_trace"] = reasoning_trace
    return payload, reasoning_trace


def query_agent_stream(llm, prompt_or_payload):
    """Stream query output using the graph payload or a legacy prompt string."""
    try:
        if isinstance(prompt_or_payload, dict):
            yield from stream_query_response(llm, prompt_or_payload)
            return

        payload = {
            "prompt": str(prompt_or_payload),
            "response_mode": "llm",
        }
        yield from stream_query_response(llm, payload)
    except Exception as exc:
        safe_message = format_llm_error(exc, debug=is_debug_mode_enabled())
        if safe_message:
            yield safe_message
            return
        raise


def query_agent(
    llm,
    user_query: str,
    thread_id: str = "default",
    chat_history: list | None = None,
    retrieval_scope: str = "all",
    *,
    retrieval_mode=None,
    retrieval_provider=None,
):
    """Non-streaming query retained for backward compatibility."""
    try:
        result = run_query(
            llm,
            user_query,
            thread_id=thread_id,
            chat_history=chat_history,
            retrieval_scope=retrieval_scope,
            retrieval_mode=retrieval_mode,
            retrieval_provider=retrieval_provider,
        )
    except Exception as exc:
        safe_message = format_llm_error(exc, debug=is_debug_mode_enabled())
        if safe_message:
            return {
                "answer": safe_message,
                "reasoning_trace": [],
                "all_messages": [],
                "payload": {},
            }
        raise
    result["answer"] = sanitize_answer_text(result.get("answer", ""))
    result["reasoning_trace"] = _payload_to_reasoning_trace(result.get("payload", {}))
    return result


if __name__ == "__main__":
    print(f"Active provider: {_get_provider_name()}")
    llm = create_agent()
    result = query_agent(llm, "List all projects with their timelines")
    print(result["answer"][:500])
