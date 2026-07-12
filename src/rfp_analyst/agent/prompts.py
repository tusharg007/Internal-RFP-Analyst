"""Prompts and deterministic intent classification."""

from __future__ import annotations

import re

from config import AGENT_SYSTEM_PROMPT


def classify_query_intent(query: str) -> str:
    lower = query.lower()
    if any(keyword in lower for keyword in ("compare", "versus", "vs", "difference")):
        return "compare_projects"
    if any(keyword in lower for keyword in ("proposal", "outline", "respond to rfp", "write proposal")):
        return "proposal_writer"
    if any(keyword in lower for keyword in ("rfp", "requirement", "gap", "case study", "fit")):
        return "rfp_gap_analysis"
    return "search"


def is_ambiguous_query(query: str) -> bool:
    lower = query.lower().strip()
    pronouns = {"it", "that", "this", "they", "them", "those", "one", "ones", "other"}
    tokens = re.findall(r"[a-zA-Z]+", lower)
    if len(tokens) <= 3:
        return True
    if any(token in pronouns for token in tokens) and not any(
        keyword in lower for keyword in ("banking", "healthcare", "insurance", "project", "proposal", "rfp")
    ):
        return True
    return False


def _render_history(chat_history: list | None) -> str:
    if not chat_history:
        return ""
    rendered = []
    for item in chat_history[-6:]:
        role = item.get("role", "user").title()
        rendered.append(f"{role}: {item.get('content', '')[:300]}")
    return "\n".join(rendered)


def build_no_documents_message(query: str) -> str:
    return (
        "I couldn't find any indexed project documents to answer this yet. "
        "Please ingest documents first, then try your question again. "
        f"Your question was: {query}"
    )


def build_ambiguous_question_message(query: str) -> str:
    return (
        "Your question is a bit ambiguous. Please clarify which project, proposal, or document set you mean "
        f"before I answer: {query}"
    )


def build_simple_prompt(user_query: str, context: str, stats: dict, chat_history: list | None = None) -> str:
    history = _render_history(chat_history)
    project_list = "\n".join(f"  - {name}" for name in stats.get("document_names", [])) or "  No documents ingested yet."
    history_block = f"\nRecent Conversation:\n{history}\n" if history else ""
    return f"""{AGENT_SYSTEM_PROMPT}

Available Documents:
{project_list}
Total: {stats.get('total_documents', 0)} documents, {stats.get('total_chunks', 0)} chunks

Retrieved Context:
{context}
{history_block}
Question:
{user_query}

Answer thoroughly with source citations."""


def build_agentic_prompt(state) -> str:
    history = _render_history(state.chat_history)
    tool_summaries = []
    for name, payload in state.tool_outputs.items():
        if isinstance(payload, dict):
            if "comparison_markdown" in payload:
                tool_summaries.append(f"Tool {name}:\n{payload['comparison_markdown']}")
            elif "outline" in payload:
                tool_summaries.append(f"Tool {name}:\n{payload['outline']}")
            elif "summary" in payload:
                tool_summaries.append(f"Tool {name}: {payload['summary']}")

    evidence = "\n\n".join(
        f"[Source: {source['source']}, Page {source['page'] + 1}]\n{source['snippet']}"
        for source in state.sources[:8]
    ) or "No evidence retrieved."

    history_block = f"\nRecent Conversation:\n{history}\n" if history else ""
    tool_block = "\n\n".join(tool_summaries) if tool_summaries else "No specialized tool outputs."
    return f"""{AGENT_SYSTEM_PROMPT}

Intent: {state.intent}
Available Documents: {state.stats.get('total_documents', 0)}

Tool Outputs:
{tool_block}

Evidence:
{evidence}
{history_block}
Question:
{state.query}

Write a grounded final answer with explicit [Source: <document>, Page <number>] citations for every major claim."""
