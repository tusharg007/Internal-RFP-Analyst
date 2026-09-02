"""LLM-based query rewriting for bounded retrieval retries."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from config import (
    GEMINI_MODEL,
    GENERATION_TEMPERATURE,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    get_api_keys,
)
from rfp_analyst.agent.prompts import QUERY_REWRITER_PROMPT
from rfp_analyst.agent.schemas_decisions import QueryRewrite

if TYPE_CHECKING:
    from rfp_analyst.agent.graph import QueryState


def _create_rewriter_llm():
    """Create the configured provider used for query rewriting."""
    groq_api_key, google_api_key = get_api_keys()
    if groq_api_key:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_MODEL,
            api_key=groq_api_key,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    if google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=google_api_key,
            temperature=GENERATION_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )
    return None


def rewrite_query(state: "QueryState") -> dict:
    """Rewrite the original question for another bounded retrieval attempt."""
    original_query = state.get("user_query", "")
    rewritten_query = original_query
    allow_llm_rewriting = state.get("allow_llm_rewriting", True)
    llm = state.get("rewriter_llm") if allow_llm_rewriting else None
    if llm is None and allow_llm_rewriting:
        llm = _create_rewriter_llm()

    if llm is not None:
        try:
            rewriter_llm = llm.with_structured_output(QueryRewrite, method="json_mode")
            rewritten = rewriter_llm.invoke(
                QUERY_REWRITER_PROMPT.format(question=original_query)
            )
            rewritten_query = rewritten.rewritten_query.strip() or original_query
        except Exception:
            rewritten_query = original_query

    normalized_original = " ".join(original_query.lower().split())
    if re.search(r"\b(resume|curriculum vitae|cv)\b", normalized_original):
        rewritten_query = (
            f"{rewritten_query}. Search the uploaded resume for its explicitly stated "
            "Technical Skills, technologies, frameworks, programming languages, databases, and tools."
        )
    if (
        re.search(r"\b(all|every|list|inventory)\b", normalized_original)
        and re.search(r"\b(projects?|case studies)\b", normalized_original)
    ):
        rewritten_query = f"{rewritten_query}. Preserve complete coverage of all project documents."

    return {
        "current_query": rewritten_query,
        "retry_count": int(state.get("retry_count", 0) or 0) + 1,
    }
