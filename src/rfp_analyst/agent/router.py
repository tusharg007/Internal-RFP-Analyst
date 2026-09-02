"""LLM-backed routing for the agentic RAG workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from config import (
    GEMINI_MODEL,
    GRADING_TEMPERATURE,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    get_api_keys,
)
from rfp_analyst.agent.prompts import ROUTER_PROMPT
from rfp_analyst.agent.schemas_decisions import RouteDecision

if TYPE_CHECKING:
    from rfp_analyst.agent.graph import QueryState


_DIRECT_CONVERSATION_PREFIXES = (
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "good morning",
    "good afternoon",
    "good evening",
)


def _create_router_llm():
    """Create a low-temperature provider instance for routing, if configured."""
    groq_api_key, google_api_key = get_api_keys()
    if groq_api_key:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_MODEL,
            api_key=groq_api_key,
            temperature=GRADING_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    if google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=google_api_key,
            temperature=GRADING_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )
    return None


def _fallback_route(query: str) -> Literal["kb", "direct"]:
    """Provide an offline-safe route for the deterministic graph backend."""
    normalized = (query or "").strip().lower()
    if any(
        normalized == prefix or normalized.startswith(f"{prefix} ") or normalized.startswith(f"{prefix}!")
        for prefix in _DIRECT_CONVERSATION_PREFIXES
    ):
        return "direct"
    return "kb"


def _refine_kb_intent(query: str) -> str:
    """Apply the legacy RFP intent rules after the LLM has selected the KB route."""
    lowered = (query or "").lower().strip()
    analysis_signals = (
        "extract requirements",
        "find gaps",
        "find case studies",
        "compare fit",
        "verify recommendations",
        "proposal outline",
    )
    if any(signal in lowered for signal in analysis_signals):
        return "rfp_analysis"
    if any(token in lowered for token in ("proposal", "draft", "write", "respond to rfp", "rfp response")):
        return "proposal"
    if any(token in lowered for token in ("compare", "difference", "versus", " vs ", "contrast")):
        return "compare"
    return "search"


def route_question(state: "QueryState") -> dict:
    """Route a query with structured LLM output, then refine KB-specific intent."""
    query = state.get("user_query", "")
    allow_llm_routing = state.get("allow_llm_routing", True)
    llm = state.get("router_llm") if allow_llm_routing else None
    if llm is None and allow_llm_routing:
        llm = _create_router_llm()

    route = _fallback_route(query)
    if llm is not None:
        try:
            router_llm = llm.with_structured_output(RouteDecision, method="json_mode")
            decision = router_llm.invoke(ROUTER_PROMPT.format(question=query))
            route = decision.route
        except Exception:
            route = _fallback_route(query)

    intent = _refine_kb_intent(query) if route == "kb" else "direct"
    return {
        "intent": intent,
        "source_used": "kb" if route == "kb" else "direct",
        "current_query": query,
    }


def route_after_router(state: dict) -> Literal["retrieve_kb", "direct_answer"]:
    """Select the Phase 5 graph branch after structured routing."""
    if state.get("source_used") == "direct" or state.get("intent") == "direct":
        return "direct_answer"
    return "retrieve_kb"
