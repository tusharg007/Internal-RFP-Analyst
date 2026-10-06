"""Tavily web-search fallback for the agentic RAG workflow."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from config import TAVILY_API_KEY, TAVILY_MAX_RESULTS

if TYPE_CHECKING:
    from rfp_analyst.agent.graph import QueryState


LOGGER = logging.getLogger(__name__)


def _format_result_item(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)

    lines = []
    if item.get("title"):
        lines.append(f"Title: {item['title']}")
    if item.get("url"):
        lines.append(f"URL: {item['url']}")
    content = item.get("content") or item.get("raw_content") or item.get("snippet")
    if content:
        lines.append(f"Evidence: {content}")
    return "\n".join(lines) or str(item)


def _format_web_results(result: Any) -> str:
    """Normalize the dict and list response formats emitted by Tavily."""
    if isinstance(result, dict):
        lines = []
        answer = result.get("answer")
        if answer:
            lines.append(f"Tavily answer: {answer}")
        entries = result.get("results", [])
        if isinstance(entries, list):
            lines.extend(_format_result_item(item) for item in entries)
        elif entries:
            lines.append(_format_result_item(entries))
        # Tavily can return an error payload as a normal dictionary instead of
        # raising. Error metadata is not evidence and must not be graded as a
        # successful nonempty search result.
        return "\n\n".join(line for line in lines if line).strip()

    if isinstance(result, list):
        return "\n\n".join(_format_result_item(item) for item in result if item).strip()

    return str(result or "")


def search_web(state: "QueryState") -> dict:
    """Search Tavily and normalize the response for evidence grading and generation."""
    if not state.get("allow_web_search", True):
        return {"web_results": "", "source_used": "web"}
    if not TAVILY_API_KEY:
        LOGGER.warning("TAVILY_API_KEY is not configured; skipping web-search fallback.")
        return {"web_results": "", "source_used": "web"}

    # TavilySearch reads credentials from the environment; this also supports Streamlit secrets.
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    try:
        from langchain_tavily import TavilySearch

        web_search = TavilySearch(
            max_results=TAVILY_MAX_RESULTS,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
        result = web_search.invoke({"query": state.get("current_query") or state.get("user_query", "")})
    except Exception:
        LOGGER.warning("Tavily web-search fallback failed; details redacted")
        return {"web_results": "", "source_used": "web"}

    return {"web_results": _format_web_results(result), "source_used": "web"}
