"""LLM-backed grading for private-KB and web-search evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import (
    GEMINI_MODEL,
    GRADING_TEMPERATURE,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    get_api_keys,
)
from rfp_analyst.agent.prompts import KB_GRADER_PROMPT, WEB_GRADER_PROMPT
from rfp_analyst.agent.schemas_decisions import EvidenceGrade

if TYPE_CHECKING:
    from rfp_analyst.agent.graph import QueryState


def _create_grader_llm():
    """Create a deterministic provider instance for evidence-grading decisions."""
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


def _format_kb_evidence(documents: list[dict]) -> str:
    return "\n\n".join(
        f"Source: {document.get('source', 'Unknown')}\n{document.get('content', '')}"
        for document in documents
    )


def _grade_with_llm(prompt: str, fallback_grade: str, state: "QueryState") -> str:
    allow_llm_grading = state.get("allow_llm_grading", True)
    llm = state.get("grader_llm") if allow_llm_grading else None
    if llm is None and allow_llm_grading:
        llm = _create_grader_llm()
    if llm is None:
        return fallback_grade

    try:
        grader_llm = llm.with_structured_output(EvidenceGrade, method="json_mode")
        decision = grader_llm.invoke(prompt)
        return decision.grade
    except Exception:
        return fallback_grade


def grade_kb_evidence(state: "QueryState") -> dict:
    """Grade whether pre-filtered private-KB evidence can answer the question."""
    documents = state.get("retrieved_documents", [])
    context = _format_kb_evidence(documents)
    grade = _grade_with_llm(
        KB_GRADER_PROMPT.format(question=state.get("user_query", ""), context=context),
        fallback_grade="good" if documents else "weak",
        state=state,
    )
    return {"kb_grade": grade}


def grade_web_evidence(state: "QueryState") -> dict:
    """Grade whether web-search evidence can answer the question."""
    web_results = str(state.get("web_results", "") or "")
    grade = _grade_with_llm(
        WEB_GRADER_PROMPT.format(
            question=state.get("user_query", ""),
            web_results=web_results,
        ),
        fallback_grade="good" if web_results.strip() else "weak",
        state=state,
    )
    return {"web_grade": grade}
