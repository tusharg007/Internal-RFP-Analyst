"""LangGraph-backed query workflow with deterministic fallback."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Callable, TypedDict

from config import (
    AGENT_SYSTEM_PROMPT,
    MAX_CASE_STUDIES,
    MAX_CHUNKS_PER_CASE_STUDY,
    MAX_CONTEXT_CHARS_PER_CHUNK,
    MAX_HISTORY_MESSAGES,
    MAX_PROMPT_TOKENS,
    MAX_TARGET_CHUNKS,
    MIN_RELEVANCE_SCORE,
    RETRIEVAL_K,
    RFP_ANALYSIS_MAX_OUTPUT_TOKENS,
)
from rag_engine import (
    KB_NOT_READY_MESSAGE,
    NO_SCOPE_DOCUMENTS_MESSAGE,
    get_vectorstore_stats,
    similarity_search,
)
from rfp_analyst.tools.compare_projects import compare_projects
from rfp_analyst.tools.proposal_writer import generate_proposal_outline
from rfp_analyst.tools.rfp_gap_analyzer import (
    extract_rfp_requirements,
    find_relevant_case_studies,
)
from rfp_analyst.tools.source_verifier import verify_answer_grounding

CLARIFICATION_MESSAGE = (
    "Could you clarify which project, document, comparison, or proposal you want me to work on?"
)
FOLLOWUP_CLARIFICATION_MESSAGE = (
    "Which previously discussed document or project should I use for this follow-up?"
)
UNSUPPORTED_CLAIM_MESSAGE = "The retrieved evidence does not support this claim."
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I could not find sufficiently relevant evidence in the selected document scope."
)
MISSING_UPLOAD_CONTEXT_MESSAGE = (
    "No indexed uploaded documents were found. Upload and ingest the target RFP before running cross-corpus analysis."
)
NO_RELEVANT_UPLOAD_TARGET_MESSAGE = (
    "Uploaded documents are indexed, but no sufficiently relevant target evidence matched this analysis request. "
    "Select a specific uploaded document or make the target requirements more explicit."
)
VAGUE_REFERENCE_PATTERN = re.compile(
    r"\b(here|this|that|it|those|these|above|the above project|the project|same one)\b",
    re.IGNORECASE,
)
PLURAL_REFERENCE_PATTERN = re.compile(r"\b(those|these|projects|documents|the above)\b", re.IGNORECASE)
RFP_ANALYSIS_ORCHESTRATION_PHRASES = (
    "find case studies",
    "compare fit",
    "generate proposal",
    "verify recommendations",
    "proposal outline",
    "return technical requirements",
    "return requirements",
    "three case studies",
    "case studies",
)
RFP_TARGET_FOCUS_AREAS = (
    "requirements",
    "business problem",
    "architecture",
    "technologies",
    "security/compliance",
    "timeline",
    "risks",
    "measurable outcomes",
)
RFP_TARGET_MAX_CHUNKS = 4
RFP_TARGET_FALLBACK_MARGIN = 0.15

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:
    END = "__end__"
    START = "__start__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


class QueryState(TypedDict, total=False):
    user_query: str
    chat_history: list[dict[str, Any]]
    vectorstore_stats: dict[str, Any]
    retrieval_k: int
    retrieval_scope: str
    retrieval_fn: Callable[[str, int, str], list]
    traces: list[dict[str, Any]]
    kb_ready: bool
    intent: str
    planned_tools: list[str]
    retrieved_documents: list[dict[str, Any]]
    retrieval_context: str
    specialized_notes: str
    tool_outputs: dict[str, Any]
    prompt: str
    answer: str
    response_mode: str
    resolved_query: str
    resolved_entities: list[dict[str, Any]]
    grounded: bool
    graph_backend: str
    prompt_budget: dict[str, Any]


class DeterministicCompiledGraph:
    """Fallback runner that mirrors the LangGraph node flow."""

    def __init__(self):
        self.node_order = [
            "health_check",
            "classify_intent",
            "plan_tools",
            "execute_retrieval",
            "execute_specialized_tool",
            "synthesize_prompt",
            "evidence_availability_check",
            "final_response",
        ]

    def invoke(self, state: QueryState) -> QueryState:
        current = dict(state)
        current["graph_backend"] = "deterministic-fallback"
        for node_name in self.node_order:
            node_fn = NODE_FUNCTIONS[node_name]
            updates = node_fn(current)
            if updates:
                current.update(updates)
        return current


@lru_cache(maxsize=1)
def compile_query_graph():
    """Compile the query graph, using a deterministic fallback when needed."""
    if not LANGGRAPH_AVAILABLE:
        return DeterministicCompiledGraph()

    workflow = StateGraph(QueryState)
    workflow.add_node("health_check", health_check)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("plan_tools", plan_tools)
    workflow.add_node("execute_retrieval", execute_retrieval)
    workflow.add_node("execute_specialized_tool", execute_specialized_tool)
    workflow.add_node("synthesize_prompt", synthesize_prompt)
    workflow.add_node("evidence_availability_check", evidence_availability_check)
    workflow.add_node("final_response", final_response)

    workflow.add_edge(START, "health_check")
    workflow.add_edge("health_check", "classify_intent")
    workflow.add_edge("classify_intent", "plan_tools")
    workflow.add_edge("plan_tools", "execute_retrieval")
    workflow.add_edge("execute_retrieval", "execute_specialized_tool")
    workflow.add_edge("execute_specialized_tool", "synthesize_prompt")
    workflow.add_edge("synthesize_prompt", "evidence_availability_check")
    workflow.add_edge("evidence_availability_check", "final_response")
    workflow.add_edge("final_response", END)

    return workflow.compile()


def _append_trace(state: QueryState, step: str, details: dict[str, Any]) -> list[dict[str, Any]]:
    trace = list(state.get("traces", []))
    trace.append({"step": step, **details})
    return trace


def _is_ambiguous_query(query: str) -> bool:
    lowered = query.lower().strip()
    if len(lowered) < 12:
        return True
    ambiguous_phrases = {
        "help me",
        "tell me more",
        "what about that",
        "can you help",
        "do it",
        "continue",
    }
    return lowered in ambiguous_phrases


def _is_vague_followup(query: str) -> bool:
    return bool(VAGUE_REFERENCE_PATTERN.search(query or ""))


def _dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for entity in entities:
        source = entity.get("source")
        if not source:
            continue
        key = (source, entity.get("document_origin", "sample"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return unique


def _extract_recent_entities(
    chat_history: list[dict[str, Any]] | None,
    scope: str,
) -> list[dict[str, Any]]:
    """Extract recent retrieved source entities from persisted UI traces."""
    if not chat_history:
        return []

    entities: list[dict[str, Any]] = []
    for message in reversed(chat_history[-8:]):
        reasoning = message.get("reasoning") or []
        for step in reversed(reasoning):
            source = step.get("source")
            if not source and step.get("tool_response"):
                source = str(step["tool_response"]).split(" (Page ", 1)[0]
            if not source:
                continue
            origin = step.get("document_origin", "sample")
            if scope in {"sample", "upload"} and origin != scope:
                continue
            entities.append(
                {
                    "source": source,
                    "document_origin": origin,
                    "page": step.get("page"),
                }
            )
        if entities:
            break
    return _dedupe_entities(entities)


def _resolve_conversational_query(
    query: str,
    chat_history: list[dict[str, Any]] | None,
    scope: str,
) -> tuple[str, list[dict[str, Any]], str]:
    """Resolve vague follow-up references using recent retrieved entities."""
    if not _is_vague_followup(query):
        return query, [], "not_needed"

    entities = _extract_recent_entities(chat_history, scope)
    if not entities:
        return query, [], "ambiguous"

    if len(entities) > 1 and not PLURAL_REFERENCE_PATTERN.search(query):
        return query, entities, "ambiguous"

    entity_names = ", ".join(entity["source"] for entity in entities[:4])
    resolved_query = f"{query}\n\nResolved follow-up target document(s): {entity_names}"
    return resolved_query, entities, "resolved"


def _build_rfp_target_query(query: str) -> str:
    """Focus upload retrieval on target requirements, not orchestration verbs."""
    cleaned = (query or "").replace("\n", " ")
    for phrase in RFP_ANALYSIS_ORCHESTRATION_PHRASES:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    focus = ", ".join(RFP_TARGET_FOCUS_AREAS)
    if cleaned:
        return f"{cleaned}. Focus on: {focus}."
    return f"Focus on: {focus}."


def _dedupe_documents_by_source_page(documents: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in documents:
        key = (item.get("source", "Unknown"), int(item.get("page", 0) or 0))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def _classify_query(query: str) -> str:
    lowered = query.lower().strip()
    if any(
        phrase in lowered
        for phrase in (
            "what sources did you use",
            "which documents were used",
            "show the citations from your last response",
            "sources for the previous answer",
        )
    ):
        return "previous_sources"
    if _is_ambiguous_query(lowered):
        return "ambiguous"
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


def _previous_answer_sources(chat_history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return the prior assistant answer's displayed sources without retrieval."""
    for message in reversed(chat_history or []):
        if message.get("role") != "assistant":
            continue
        sources = []
        seen = set()
        for step in message.get("reasoning") or []:
            source = step.get("source")
            page = step.get("page")
            if not source or page is None:
                continue
            key = (source, int(page), step.get("document_origin", "sample"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "source_file": source,
                    "page": int(page),
                    "document_origin": step.get("document_origin", "sample"),
                }
            )
        return sources
    return []


def _format_previous_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "The previous answer did not contain any source citations."
    return "\n".join(
        f"- {item['source_file']}, Page {item['page']}, origin: {item['document_origin']}"
        for item in sources
    )


def _format_sources(results: list[dict[str, Any]]) -> str:
    parts = []
    for item in results:
        page_number = item["page"] + 1 if isinstance(item["page"], int) else item["page"]
        parts.append(
            f"[Source: {item['source']}, Page {page_number}]\n{item['content']}"
        )
    return "\n\n---\n\n".join(parts) if parts else "No relevant documents found."


def _build_history_text(chat_history: list[dict[str, Any]] | None) -> str:
    if not chat_history:
        return ""
    recent = [
        message
        for message in chat_history[-6:]
        if message.get("role") in {"user", "assistant"}
    ]
    lines = []
    for message in recent:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message.get('content', '')[:300]}")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text or "") + 3) // 4)


def _truncate_text(text: str, limit: int = MAX_CONTEXT_CHARS_PER_CHUNK) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(limit - 3, 0)].rstrip() + "..."


def _format_compact_citation(source: str, page: int, origin: str) -> str:
    return f"{source} (Page {int(page) + 1}, origin={origin})"


def _compact_document_entry(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": document.get("source", "Unknown"),
        "page": int(document.get("page", 0) or 0),
        "score": float(document.get("score", 0.0) or 0.0),
        "chunk_id": document.get("chunk_id", ""),
        "document_origin": document.get("document_origin", "sample"),
        "content": _truncate_text(document.get("content", "")),
    }


def _group_case_study_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for document in documents:
        if document.get("document_origin") != "sample":
            continue
        source = document.get("source", "Unknown")
        group = grouped.setdefault(
            source,
            {
                "source": source,
                "score": float(document.get("score", 0.0) or 0.0),
                "documents": [],
            },
        )
        group["score"] = max(group["score"], float(document.get("score", 0.0) or 0.0))
        group["documents"].append(document)

    normalized = []
    for source, group in grouped.items():
        docs = sorted(
            group["documents"],
            key=lambda item: (-float(item.get("score", 0.0) or 0.0), item.get("page", 0), item.get("chunk_id", "")),
        )
        docs = _dedupe_documents_by_source_page(docs, MAX_CHUNKS_PER_CASE_STUDY)
        normalized.append({"source": source, "score": group["score"], "documents": docs})
    normalized.sort(key=lambda item: (-item["score"], item["source"]))
    return normalized[:MAX_CASE_STUDIES]


def _dedupe_documents_by_content_signature(documents: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in documents:
        signature = re.sub(r"\d+", "", item.get("content", "").lower())[:120]
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def compact_tool_outputs_for_prompt(tool_outputs: dict[str, Any]) -> dict[str, Any]:
    requirements = []
    for item in (tool_outputs.get("extract_rfp_requirements", {}) or {}).get("requirements", [])[:8]:
        requirements.append(
            {
                "id": item.get("id", ""),
                "text": _truncate_text(item.get("text", ""), 220),
                "source_file": item.get("source_file", ""),
                "page": item.get("page"),
                "document_origin": item.get("document_origin", ""),
            }
        )

    gaps = []
    for item in (tool_outputs.get("extract_rfp_requirements", {}) or {}).get("gaps", [])[:10]:
        gaps.append(
            {
                "requirement_id": item.get("requirement_id", ""),
                "detail": _truncate_text(item.get("detail", ""), 220),
                "status": item.get("status", "absent_or_ambiguous"),
            }
        )

    case_matches = []
    for match in (tool_outputs.get("find_relevant_case_studies", {}) or {}).get("matches", [])[:MAX_CASE_STUDIES]:
        pages = [int(page) + 1 for page in match.get("pages", [])[:2]]
        case_matches.append(
            {
                "source": match.get("source", "Unknown"),
                "fit_reason": _truncate_text(" ; ".join(match.get("snippets", [])[:2]), 220),
                "fit_score": match.get("fit_score", len(match.get("matched_requirements", []))),
                "matched_requirements": match.get("matched_requirements", []),
                "missing_coverage": match.get("missing_coverage", []),
                "citations": [f"Page {page}" for page in pages],
            }
        )

    comparison_rows = []
    for row in (tool_outputs.get("compare_projects", {}) or {}).get("rows", [])[:3]:
        comparison_rows.append(
            {
                "source": row.get("source", "Unknown"),
                "timeline": _truncate_text(row.get("timeline", ""), 120),
                "budget": _truncate_text(row.get("budget", ""), 80),
                "tech_stack": _truncate_text(row.get("tech_stack", ""), 140),
                "outcomes": _truncate_text(row.get("outcomes", ""), 140),
            }
        )

    proposal = tool_outputs.get("proposal_writer", {}) or {}
    proposal_headings = []
    for line in str(proposal.get("outline", "") or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            proposal_headings.append(stripped[3:])
        elif stripped.startswith("- "):
            proposal_headings.append(_truncate_text(stripped[2:], 160))
        if len(proposal_headings) >= 8:
            break
    if not proposal_headings and proposal.get("outline"):
        proposal_headings.append(_truncate_text(proposal.get("outline", ""), 220))

    return {
        "requirements": requirements,
        "gaps": gaps,
        "selected_case_studies": case_matches,
        "comparison": comparison_rows,
        "proposal": proposal_headings,
    }


def _render_compact_tool_outputs(tool_outputs: dict[str, Any], verbose: bool = True) -> str:
    compact = compact_tool_outputs_for_prompt(tool_outputs)
    lines = []

    requirements = compact.get("requirements", [])
    if requirements:
        lines.append("-- Requirement Summary --")
        for item in requirements[:6]:
            citation = ""
            if item.get("source_file") and item.get("page") is not None:
                citation = f" [{item['source_file']}, Page {int(item['page']) + 1}]"
            lines.append(f"- {item.get('id', '')}: {item.get('text', '')}{citation}")

    gaps = compact.get("gaps", [])
    if gaps:
        lines.append("-- Inferred Requirement Gaps --")
        for item in gaps[:8]:
            lines.append(f"- {item.get('requirement_id', '')}: {item.get('detail', '')}")

    case_studies = compact.get("selected_case_studies", [])
    if case_studies:
        lines.append("-- Selected Case Studies --")
        case_limit = MAX_CASE_STUDIES if verbose else min(MAX_CASE_STUDIES, 2)
        for item in case_studies[:case_limit]:
            detail = f": {item.get('fit_reason', '')}" if verbose and item.get("fit_reason") else ""
            citations = ", ".join(item.get("citations", [])[:2])
            fit_score = item.get("fit_score")
            score_text = f" fit={fit_score}" if fit_score is not None else ""
            matched = f"; matched={', '.join(item.get('matched_requirements', []))}" if item.get("matched_requirements") else ""
            missing = f"; missing={', '.join(item.get('missing_coverage', []))}" if item.get("missing_coverage") else ""
            citation_text = f" [{citations}]" if citations else ""
            lines.append(f"- {item.get('source', 'Unknown')}{score_text}{matched}{missing}{citation_text}{detail}")

    comparison_rows = compact.get("comparison", [])
    if comparison_rows:
        lines.append("-- Comparison Summary --")
        for row in comparison_rows[:3]:
            if verbose:
                lines.append(
                    f"- {row['source']}: timeline={row['timeline']}; budget={row['budget']}; "
                    f"stack={row['tech_stack']}; outcomes={row['outcomes']}"
                )
            else:
                lines.append(f"- {row['source']}: stack={row['tech_stack']}; outcomes={row['outcomes']}")

    proposal = compact.get("proposal", [])
    if proposal:
        lines.append("-- Proposal Outline --")
        proposal_limit = 8 if verbose else 4
        for item in proposal[:proposal_limit]:
            lines.append(f"- {item}")

    return "\n".join(lines).strip()


def _build_compact_prompt_sections(
    state: QueryState,
    history_messages: list[dict[str, Any]],
    target_documents: list[dict[str, Any]],
    sample_case_groups: list[dict[str, Any]],
    include_verbose_tools: bool,
) -> dict[str, str]:
    stats = state.get("vectorstore_stats", {})
    scope_label = state.get("retrieval_scope", "all")
    project_list = "\n".join(f"  - {name}" for name in stats.get("document_names", [])[:12]) or "  No documents ingested yet."
    tool_block = _render_compact_tool_outputs(state.get("tool_outputs", {}), verbose=include_verbose_tools)

    target_lines = []
    for document in target_documents:
        target_lines.append(
            f"- {_format_compact_citation(document['source'], document['page'], document['document_origin'])}: "
            f"{document['content']}"
        )
    target_block = "\n".join(target_lines) if target_lines else "- No uploaded target evidence selected."

    sample_lines = []
    for group in sample_case_groups:
        sample_lines.append(f"- {group['source']}")
        for document in group.get("documents", []):
            sample_lines.append(
                f"  * Page {int(document['page']) + 1}: {document['content']}"
            )
    sample_block = "\n".join(sample_lines) if sample_lines else "- No sample case-study evidence selected."

    history_lines = []
    for message in history_messages:
        role = "User" if message.get("role") == "user" else "Assistant"
        history_lines.append(f"{role}: {_truncate_text(message.get('content', ''), 240)}")
    history_block = "\n".join(history_lines)

    sections = [
        AGENT_SYSTEM_PROMPT.strip(),
        "\n-- Workflow Intent --",
        state.get("intent", "search"),
        "\n-- Planned Tools --",
        ", ".join(state.get("planned_tools", [])) or "none",
        "\n-- Document Scope --",
        scope_label,
        "\n-- Available Documents --",
        f"{project_list}\nTotal: {stats.get('total_documents', 0)} documents, {stats.get('total_chunks', 0)} chunks",
        "\n-- Uploaded Target Evidence --",
        target_block,
        "\n-- Sample Case Study Evidence --",
        sample_block,
    ]
    if tool_block:
        sections.extend(["\n-- Compact Tool Outputs --", tool_block])
    if history_block:
        sections.extend(["\n-- Recent Conversation --", history_block])
    sections.extend(
        [
            "\n-- Question --",
            state.get("user_query", ""),
            "\nAnswer thoroughly with source citations. If the evidence is incomplete, say so explicitly.",
            'Never write "[Source: None]". If a claim is unsupported, write: The retrieved evidence does not support this claim.',
        ]
    )
    return {
        "prompt": "\n".join(section for section in sections if section),
        "tool_block": tool_block,
    }


def health_check(state: QueryState) -> QueryState:
    stats = dict(state.get("vectorstore_stats") or get_vectorstore_stats())
    kb_ready = stats.get("status") == "ready" and int(stats.get("total_chunks", 0) or 0) > 0
    return {
        "vectorstore_stats": stats,
        "kb_ready": kb_ready,
        "traces": _append_trace(
            state,
            "health_check",
            {
                "tool": "health_check",
                "kb_ready": kb_ready,
                "document_count": int(stats.get("total_documents", 0) or 0),
                "chunk_count": int(stats.get("total_chunks", 0) or 0),
            },
        ),
    }


def classify_intent(state: QueryState) -> QueryState:
    retrieval_scope = state.get("retrieval_scope", "all")
    resolved_query, resolved_entities, resolution_status = _resolve_conversational_query(
        state.get("user_query", ""),
        state.get("chat_history"),
        retrieval_scope,
    )
    intent = _classify_query(resolved_query if resolution_status == "resolved" else state.get("user_query", ""))
    if intent == "previous_sources":
        sources = _previous_answer_sources(state.get("chat_history"))
        return {
            "intent": intent,
            "response_mode": "direct",
            "answer": _format_previous_sources(sources),
            "tool_outputs": {"previous_sources": sources},
            "resolved_query": resolved_query,
            "resolved_entities": [],
            "traces": _append_trace(
                state,
                "classify_intent",
                {
                    "tool": "previous_sources",
                    "intent": intent,
                    "input_summary": state.get("user_query", "")[:120],
                    "output_summary": f"Returned {len(sources)} source citation(s) from the previous answer.",
                },
            ),
        }
    if resolution_status == "ambiguous":
        intent = "ambiguous"
    response_mode = "clarification" if intent == "ambiguous" else "llm"
    answer = FOLLOWUP_CLARIFICATION_MESSAGE if resolution_status == "ambiguous" else ""
    answer = answer or (CLARIFICATION_MESSAGE if intent == "ambiguous" else "")
    return {
        "intent": intent,
        "response_mode": response_mode,
        "answer": answer,
        "resolved_query": resolved_query,
        "resolved_entities": resolved_entities,
        "traces": _append_trace(
            state,
            "classify_intent",
            {
                "tool": "intent_classifier",
                "intent": intent,
                "input_summary": state.get("user_query", "")[:120],
                "resolution_status": resolution_status,
                "resolved_entities": [entity["source"] for entity in resolved_entities],
            },
        ),
    }


def plan_tools(state: QueryState) -> QueryState:
    intent = state.get("intent", "search")
    planned_tools: list[str] = []
    if intent == "search":
        planned_tools = ["search_knowledge_base"]
    elif intent == "compare":
        planned_tools = ["search_knowledge_base", "compare_projects"]
    elif intent == "proposal":
        planned_tools = [
            "search_knowledge_base",
            "extract_rfp_requirements",
            "find_relevant_case_studies",
            "proposal_writer",
        ]
    elif intent == "rfp_analysis":
        planned_tools = [
            "search_knowledge_base",
            "extract_rfp_requirements",
            "find_relevant_case_studies",
            "compare_projects",
            "proposal_writer",
        ]

    return {
        "planned_tools": planned_tools,
        "traces": _append_trace(
            state,
            "plan_tools",
            {
                "tool": "tool_planner",
                "planned_tools": planned_tools,
                "output_summary": f"Selected {len(planned_tools)} tool(s): {', '.join(planned_tools) or 'none'}",
            },
        ),
    }


def execute_retrieval(state: QueryState) -> QueryState:
    planned_tools = state.get("planned_tools", [])
    is_rfp_analysis = state.get("intent") == "rfp_analysis"
    retrieval_scope = "upload" if is_rfp_analysis else state.get("retrieval_scope", "all")
    if "search_knowledge_base" not in planned_tools:
        return {
            "retrieved_documents": [],
            "retrieval_context": "",
        }

    if not state.get("kb_ready"):
        traces = _append_trace(
            state,
            "execute_retrieval",
            {
                "tool": "search_knowledge_base",
                "status": "skipped",
                "reason": KB_NOT_READY_MESSAGE,
                "scope": retrieval_scope,
            },
        )
        return {
            "retrieved_documents": [],
            "retrieval_context": "",
            "response_mode": "fallback",
            "answer": KB_NOT_READY_MESSAGE,
            "traces": traces,
        }

    retrieval_fn = state.get("retrieval_fn") or similarity_search
    retrieval_k = int(state.get("retrieval_k", RETRIEVAL_K))
    base_query = state.get("resolved_query") or state.get("user_query", "")
    retrieval_query = _build_rfp_target_query(base_query) if is_rfp_analysis else base_query
    try:
        raw_results = retrieval_fn(retrieval_query, retrieval_k, retrieval_scope)
    except TypeError:
        raw_results = retrieval_fn(retrieval_query, retrieval_k)

    documents = []
    for doc, score in raw_results:
        metadata = getattr(doc, "metadata", {}) or {}
        page = metadata.get("page", 0)
        try:
            page = int(page)
        except Exception:
            page = 0
        documents.append(
            {
                "source": metadata.get("source_file", "Unknown"),
                "page": page,
                "score": float(score),
                "content": getattr(doc, "page_content", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "document_origin": metadata.get("document_origin", "sample"),
            }
        )

    if is_rfp_analysis:
        documents = [item for item in documents if item["document_origin"] == "upload"]

    scope_chunk_counts = state.get("vectorstore_stats", {}).get("scope_chunk_counts", {})
    indexed_upload_chunk_count = int(scope_chunk_counts.get("upload", 0) or 0)
    indexed_upload_document_count = int(state.get("vectorstore_stats", {}).get("indexed_upload_document_count", 0) or 0)
    indexed_upload_files = list(state.get("vectorstore_stats", {}).get("indexed_upload_files", []) or [])
    relevant_documents = [item for item in documents if item["score"] >= MIN_RELEVANCE_SCORE]
    below_threshold_count = len(documents) - len(relevant_documents)
    selected_documents = relevant_documents
    target_fallback_used = False

    if is_rfp_analysis:
        selected_documents = _dedupe_documents_by_source_page(selected_documents, RFP_TARGET_MAX_CHUNKS)
        if not selected_documents and indexed_upload_chunk_count > 0:
            near_threshold = [
                item for item in documents if item["score"] >= max(MIN_RELEVANCE_SCORE - RFP_TARGET_FALLBACK_MARGIN, 0.0)
            ]
            if near_threshold:
                target_fallback_used = True
                selected_documents = _dedupe_documents_by_source_page(near_threshold, RFP_TARGET_MAX_CHUNKS)

    documents = selected_documents

    traces = _append_trace(
        state,
        "execute_retrieval",
        {
            "tool": "search_knowledge_base",
            "status": "completed",
            "scope": retrieval_scope,
            "input": {
                "query": retrieval_query,
                "original_query": state.get("user_query", ""),
                "k": retrieval_k,
                "scope": retrieval_scope,
            },
            "input_summary": f"Search scope={retrieval_scope}; k={retrieval_k}",
            "output_summary": (
                f"Retrieved {len(documents)} relevant chunk(s); "
                f"filtered {below_threshold_count} below threshold {MIN_RELEVANCE_SCORE:.2f}"
            ),
            "documents": [
                {
                    "source": item["source"],
                    "page": item["page"] + 1,
                    "score": f"{item['score']:.2f}",
                    "chunk_id": item["chunk_id"],
                    "document_origin": item["document_origin"],
                }
                for item in documents[:6]
            ],
        },
    )

    if is_rfp_analysis:
        traces = _append_trace(
            {"traces": traces},
            "execute_retrieval",
            {
                "tool": "target_context_retrieval",
                "status": "completed",
                "indexed_upload_chunk_count": indexed_upload_chunk_count,
                "indexed_upload_document_count": indexed_upload_document_count,
                "indexed_upload_files": indexed_upload_files,
                "raw_upload_hits": len(raw_results),
                "qualifying_upload_hits": len(relevant_documents),
                "relevance_threshold": MIN_RELEVANCE_SCORE,
                "target_fallback_used": target_fallback_used,
                "selected_upload_source_files": sorted({item["source"] for item in documents}),
                "input_summary": "Upload-only target retrieval for cross-corpus analysis.",
                "output_summary": (
                    f"Indexed upload chunks={indexed_upload_chunk_count}; "
                    f"selected {len(documents)} upload chunk(s)"
                    + (" using bounded fallback." if target_fallback_used else ".")
                ),
            },
        )

    if is_rfp_analysis and indexed_upload_chunk_count == 0:
        return {
            "retrieved_documents": [],
            "retrieval_context": "",
            "response_mode": "fallback",
            "answer": MISSING_UPLOAD_CONTEXT_MESSAGE,
            "traces": traces,
        }
    if is_rfp_analysis and not documents:
        return {
            "retrieved_documents": [],
            "retrieval_context": "",
            "response_mode": "fallback",
            "answer": NO_RELEVANT_UPLOAD_TARGET_MESSAGE,
            "traces": traces,
        }
    if not documents and retrieval_scope in {"sample", "upload"} and int(scope_chunk_counts.get(retrieval_scope, 0) or 0) == 0:
        return {
            "retrieved_documents": [],
            "retrieval_context": "",
            "response_mode": "fallback",
            "answer": NO_SCOPE_DOCUMENTS_MESSAGE,
            "traces": traces,
        }

    if not documents:
        return {
            "retrieved_documents": [],
            "retrieval_context": "",
            "response_mode": "fallback",
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "traces": traces,
        }

    return {
        "retrieved_documents": documents,
        "retrieval_context": _format_sources(documents),
        "traces": traces,
    }


def _scoped_search(state: QueryState, scope: str) -> Callable[[str, int], list]:
    retrieval_fn = state.get("retrieval_fn") or similarity_search

    def search(query: str, k: int = RETRIEVAL_K):
        try:
            return retrieval_fn(query, k, scope)
        except TypeError:
            return retrieval_fn(query, k)

    return search


def _documents_to_text(documents: list[dict[str, Any]]) -> str:
    return "\n\n".join(item.get("content", "") for item in documents if item.get("content"))


def _normalize_tool_documents(documents: list[object]) -> list[dict[str, Any]]:
    normalized = []
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        normalized.append(
            {
                "source": metadata.get("source_file", "Unknown"),
                "page": int(metadata.get("page", 0) or 0),
                "score": 1.0,
                "content": getattr(document, "page_content", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "document_origin": metadata.get("document_origin", "sample"),
            }
        )
    return normalized


def _attach_requirement_source_metadata(requirements: list[dict[str, Any]], documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_documents = [document for document in documents if document.get("document_origin") == "upload"]
    if not target_documents:
        return requirements
    fallback = sorted(
        target_documents,
        key=lambda item: (-float(item.get("score", 0.0) or 0.0), item.get("source", ""), item.get("page", 0)),
    )[0]
    enriched = []
    for requirement in requirements:
        item = dict(requirement)
        item.setdefault("source_file", fallback.get("source", "Unknown"))
        item.setdefault("page", int(fallback.get("page", 0) or 0))
        item.setdefault("document_origin", fallback.get("document_origin", "upload"))
        enriched.append(item)
    return enriched


def execute_specialized_tool(state: QueryState) -> QueryState:
    intent = state.get("intent", "search")
    if state.get("response_mode") in {"fallback", "clarification", "direct"}:
        return {"specialized_notes": "", "tool_outputs": state.get("tool_outputs", {})}

    query = state.get("resolved_query") or state.get("user_query", "")
    outputs = dict(state.get("tool_outputs", {}))
    traces = list(state.get("traces", []))
    notes = []

    if intent == "compare":
        result = compare_projects(
            query,
            search_fn=_scoped_search(state, state.get("retrieval_scope", "all")),
            k=int(state.get("retrieval_k", RETRIEVAL_K)),
        )
        outputs["compare_projects"] = result
        summary = f"Compared {len(result.get('rows', []))} projects across 4 dimensions"
        notes.append(result.get("comparison_markdown", ""))
        traces.append(
            {
                "step": "execute_specialized_tool",
                "tool": "compare_projects",
                "status": "completed",
                "input_summary": f"Comparison scope={state.get('retrieval_scope', 'all')}",
                "output_summary": summary,
            }
        )

    if intent in {"proposal", "rfp_analysis"}:
        requirements_result = extract_rfp_requirements(
            _documents_to_text(state.get("retrieved_documents", []))
        )
        requirements = _attach_requirement_source_metadata(
            requirements_result.get("requirements", []),
            state.get("retrieved_documents", []),
        )
        requirements_result = {**requirements_result, "requirements": requirements}
        outputs["extract_rfp_requirements"] = requirements_result
        traces.append(
            {
                "step": "execute_specialized_tool",
                "tool": "extract_rfp_requirements",
                "status": "completed",
                "input_summary": "Uploaded target evidence" if intent == "rfp_analysis" else "Retrieved target evidence",
                "output_summary": requirements_result.get("summary", "Extracted requirements."),
            }
        )

        case_scope = "sample" if intent == "rfp_analysis" else state.get("retrieval_scope", "all")
        case_studies = find_relevant_case_studies(
            requirements,
            search_fn=_scoped_search(state, case_scope),
            k=int(state.get("retrieval_k", RETRIEVAL_K)),
        )
        outputs["find_relevant_case_studies"] = case_studies
        selected_names = [item.get("source", "Unknown") for item in case_studies.get("matches", [])[:5]]
        traces.append(
            {
                "step": "execute_specialized_tool",
                "tool": "find_relevant_case_studies",
                "status": "completed",
                "input_summary": f"Case-study scope={case_scope}",
                "output_summary": f"Selected {len(selected_names)} case study match(es): {', '.join(selected_names) or 'none'}",
            }
        )

        if intent == "rfp_analysis":
            comparison = compare_projects(
                "Compare fit for: " + " ".join(item.get("text", "") for item in requirements),
                search_fn=_scoped_search(state, "sample"),
                k=int(state.get("retrieval_k", RETRIEVAL_K)),
            )
            outputs["compare_projects"] = comparison
            traces.append(
                {
                    "step": "execute_specialized_tool",
                    "tool": "compare_projects",
                    "status": "completed",
                    "input_summary": "Compare sample case-study fit against uploaded requirements",
                    "output_summary": f"Compared {len(comparison.get('rows', []))} projects across 4 dimensions",
                }
            )

        proposal = generate_proposal_outline(query, case_studies, requirements)
        outputs["proposal_writer"] = proposal
        notes.extend(
            [
                requirements_result.get("summary", ""),
                f"Selected case studies: {', '.join(selected_names) or 'none'}",
                proposal.get("outline", ""),
            ]
        )
        traces.append(
            {
                "step": "execute_specialized_tool",
                "tool": "proposal_writer",
                "status": "completed",
                "input_summary": f"{len(requirements)} requirements and {len(selected_names)} case study match(es)",
                "output_summary": "Generated a structured proposal outline from executed tool outputs.",
            }
        )

        supporting = _normalize_tool_documents(case_studies.get("documents", []))
        existing_ids = {item.get("chunk_id") for item in state.get("retrieved_documents", [])}
        supporting = [item for item in supporting if not item.get("chunk_id") or item.get("chunk_id") not in existing_ids]
        return {
            "specialized_notes": "\n\n".join(item for item in notes if item),
            "tool_outputs": outputs,
            "retrieved_documents": [*state.get("retrieved_documents", []), *supporting],
            "retrieval_context": _format_sources([*state.get("retrieved_documents", []), *supporting]),
            "traces": traces,
        }

    return {
        "specialized_notes": "\n\n".join(item for item in notes if item),
        "tool_outputs": outputs,
        "traces": traces,
    }


def synthesize_prompt(state: QueryState) -> QueryState:
    response_mode = state.get("response_mode", "llm")
    if response_mode in {"clarification", "fallback", "direct"}:
        return {
            "prompt": "",
            "traces": _append_trace(
                state,
                "synthesize_prompt",
                {
                    "tool": "prompt_synthesizer",
                    "status": "skipped",
                    "reason": response_mode,
                    "output_summary": f"Skipped because response_mode={response_mode}",
                },
            ),
        }

    intent = state.get("intent", "search")
    retrieved_documents = [_compact_document_entry(item) for item in state.get("retrieved_documents", [])]
    target_documents = [
        item for item in retrieved_documents if item.get("document_origin") == "upload"
    ]
    target_documents = sorted(
        target_documents,
        key=lambda item: (-float(item.get("score", 0.0) or 0.0), item.get("source", ""), item.get("page", 0)),
    )
    target_documents = _dedupe_documents_by_source_page(target_documents, MAX_TARGET_CHUNKS)
    if intent == "rfp_analysis":
        target_documents = _dedupe_documents_by_content_signature(target_documents, MAX_TARGET_CHUNKS)

    sample_groups = _group_case_study_documents(retrieved_documents)
    history_messages = [
        message
        for message in (state.get("chat_history") or [])
        if message.get("role") in {"user", "assistant"}
    ][-MAX_HISTORY_MESSAGES:]

    verbose_tools = True
    chunks_dropped = 0
    sample_case_groups = sample_groups

    sections = _build_compact_prompt_sections(
        state,
        history_messages=history_messages,
        target_documents=target_documents,
        sample_case_groups=sample_case_groups,
        include_verbose_tools=verbose_tools,
    )
    prompt = sections["prompt"]
    estimated_input_tokens = _estimate_tokens(prompt)
    projected_total_tokens = estimated_input_tokens + RFP_ANALYSIS_MAX_OUTPUT_TOKENS

    while estimated_input_tokens > MAX_PROMPT_TOKENS:
        dropped = False

        for group in reversed(sample_case_groups):
            if len(group.get("documents", [])) > 1:
                group["documents"].pop()
                chunks_dropped += 1
                dropped = True
                break

        if not dropped and len(sample_case_groups) > 1:
            sample_case_groups.pop()
            chunks_dropped += 1
            dropped = True

        if not dropped and len(history_messages) > 1:
            history_messages.pop(0)
            dropped = True

        if not dropped and verbose_tools:
            verbose_tools = False
            dropped = True

        if not dropped and len(sample_case_groups) > 0:
            sample_case_groups.pop()
            chunks_dropped += 1
            dropped = True

        if not dropped:
            break

        sections = _build_compact_prompt_sections(
            state,
            history_messages=history_messages,
            target_documents=target_documents,
            sample_case_groups=sample_case_groups,
            include_verbose_tools=verbose_tools,
        )
        prompt = sections["prompt"]
        estimated_input_tokens = _estimate_tokens(prompt)
        projected_total_tokens = estimated_input_tokens + RFP_ANALYSIS_MAX_OUTPUT_TOKENS

    prompt_budget = {
        "estimated_input_tokens": estimated_input_tokens,
        "reserved_output_tokens": RFP_ANALYSIS_MAX_OUTPUT_TOKENS,
        "projected_total_tokens": projected_total_tokens,
        "budget_limit": MAX_PROMPT_TOKENS,
        "target_chunks_included": len(target_documents),
        "sample_chunks_included": sum(len(group.get("documents", [])) for group in sample_case_groups),
        "chunks_dropped": chunks_dropped,
        "history_messages_included": len(history_messages),
        "status": "within_budget" if estimated_input_tokens <= MAX_PROMPT_TOKENS else "over_budget",
    }
    return {
        "prompt": prompt,
        "prompt_budget": prompt_budget,
        "traces": _append_trace(
            {"traces": _append_trace(
                state,
                "synthesize_prompt",
                {
                    "tool": "prompt_synthesizer",
                    "status": "completed",
                    "intent": intent,
                    "scope": state.get("retrieval_scope", "all"),
                    "output_summary": "Built grounded answer prompt from compact retrieved evidence and conversation context.",
                },
            )},
            "synthesize_prompt",
            {"tool": "prompt_budget", **prompt_budget},
        ),
    }


def evidence_availability_check(state: QueryState) -> QueryState:
    response_mode = state.get("response_mode", "llm")
    documents = state.get("retrieved_documents", [])
    grounded = bool(documents) if response_mode == "llm" else response_mode == "clarification"
    answer = state.get("answer", "")
    if response_mode == "llm" and not documents:
        grounded = False
        answer = "I could not find grounded evidence for that request in the current knowledge base."
        response_mode = "fallback"
    if "[Source: None]" in answer:
        answer = answer.replace("[Source: None]", UNSUPPORTED_CLAIM_MESSAGE)

    return {
        "grounded": grounded,
        "response_mode": response_mode,
        "answer": answer,
        "traces": _append_trace(
            state,
            "verify_grounding",
            {
                "tool": "evidence_availability_check",
                "grounded": grounded,
                "response_mode": response_mode,
                "verification_status": "available" if grounded else "unavailable",
                "output_summary": "Checked whether sufficient evidence is available before answer generation.",
            },
        ),
    }


def final_response(state: QueryState) -> QueryState:
    response_mode = state.get("response_mode", "llm")
    traces = _append_trace(
        state,
        "final_response",
        {
            "tool": "final_response",
            "response_mode": response_mode,
            "graph_backend": state.get(
                "graph_backend",
                "langgraph" if LANGGRAPH_AVAILABLE else "deterministic-fallback",
            ),
            "output_summary": f"Prepared {response_mode} response.",
        },
    )
    return {
        "traces": traces,
    }


NODE_FUNCTIONS = {
    "health_check": health_check,
    "classify_intent": classify_intent,
    "plan_tools": plan_tools,
    "execute_retrieval": execute_retrieval,
    "execute_specialized_tool": execute_specialized_tool,
    "synthesize_prompt": synthesize_prompt,
    "evidence_availability_check": evidence_availability_check,
    "final_response": final_response,
}


def prepare_query_payload(
    user_query: str,
    chat_history: list[dict[str, Any]] | None = None,
    vectorstore_stats: dict[str, Any] | None = None,
    retrieval_fn: Callable[[str, int, str], list] | None = None,
    retrieval_k: int = RETRIEVAL_K,
    retrieval_scope: str = "all",
) -> dict[str, Any]:
    """Run the orchestration graph and return the full query payload."""
    graph = compile_query_graph()
    initial_state: QueryState = {
        "user_query": user_query,
        "chat_history": chat_history or [],
        "vectorstore_stats": vectorstore_stats or {},
        "retrieval_fn": retrieval_fn or similarity_search,
        "retrieval_k": retrieval_k,
        "retrieval_scope": retrieval_scope,
        "traces": [],
        "response_mode": "llm",
        "resolved_query": user_query,
        "resolved_entities": [],
        "tool_outputs": {},
        "graph_backend": "langgraph" if LANGGRAPH_AVAILABLE else "deterministic-fallback",
    }
    final_state = graph.invoke(initial_state)
    return dict(final_state)


def run_agent_graph(state, search_fn=None, stats_fn=None):
    """Backward-compatible AgentState wrapper around the current query graph."""
    from rfp_analyst.agent.prompts import build_no_documents_message

    stats = stats_fn() if stats_fn else None
    payload = prepare_query_payload(
        user_query=state.query,
        chat_history=state.chat_history,
        vectorstore_stats=stats,
        retrieval_fn=search_fn,
    )

    intent_map = {
        "search": "search",
        "compare": "compare_projects",
        "proposal": "proposal_writer",
        "ambiguous": "ambiguous",
    }
    state.intent = intent_map.get(payload.get("intent", "search"), payload.get("intent", "search"))
    state.stats = payload.get("vectorstore_stats", {})
    state.prompt = payload.get("prompt", "")
    state.final_answer = payload.get("answer", "")
    if payload.get("response_mode") == "fallback" and "Please ingest documents first" not in state.final_answer:
        state.final_answer = build_no_documents_message(state.query)

    state.tool_trace = payload.get("traces", [])
    state.retrieved_documents = payload.get("retrieved_documents", [])
    state.sources = [
        {
            "source": document.get("source", "Unknown"),
            "page": document.get("page", 0),
            "snippet": document.get("content", ""),
        }
        for document in state.retrieved_documents
    ]
    state.tool_outputs = {
        "planned_tools": payload.get("planned_tools", []),
        "retrieval_context": payload.get("retrieval_context", ""),
    }
    return state


def _human_message(content: str):
    from langchain_core.messages import HumanMessage

    return HumanMessage(content=content)


_BAD_CITATION_PATTERN = re.compile(
    r"\[Source:\s*(None|respective documents?|the documents?|various documents?)\s*(?:,\s*Page\s*\d+)?\]",
    re.IGNORECASE,
)
_GENERIC_SOURCE_PATTERN = re.compile(r"\[Source:\s*[^,\]]+\]", re.IGNORECASE)


def _sanitize_answer_text(answer: str) -> str:
    sanitized = str(answer or "").replace("[Source: None]", UNSUPPORTED_CLAIM_MESSAGE)
    sanitized = _BAD_CITATION_PATTERN.sub(UNSUPPORTED_CLAIM_MESSAGE, sanitized)
    sanitized = _GENERIC_SOURCE_PATTERN.sub(UNSUPPORTED_CLAIM_MESSAGE, sanitized)
    return sanitized


def _has_vague_or_invalid_citations(answer: str) -> bool:
    if _BAD_CITATION_PATTERN.search(answer or ""):
        return True
    for citation in re.findall(r"\[Source:[^\]]+\]", answer or "", flags=re.IGNORECASE):
        if not re.search(r"\.pdf\s*,\s*Page\s+\d+", citation):
            return True
    return False


def _repair_unsupported_answer(answer: str, unsupported_claims: list[str]) -> str:
    unsupported = [claim.strip() for claim in unsupported_claims if claim.strip()]
    repaired_lines = []
    changed = False
    for raw_line in _sanitize_answer_text(answer).splitlines():
        line = raw_line.rstrip()
        if not line:
            repaired_lines.append(line)
            continue
        if _has_vague_or_invalid_citations(line):
            repaired_lines.append(UNSUPPORTED_CLAIM_MESSAGE)
            changed = True
            continue
        if any(claim in line for claim in unsupported):
            if line.lstrip().startswith("-"):
                repaired_lines.append("- The retrieved evidence does not support this claim.")
            else:
                repaired_lines.append(UNSUPPORTED_CLAIM_MESSAGE)
            changed = True
            continue
        repaired_lines.append(line)
    if not changed and unsupported:
        repaired_lines.append(UNSUPPORTED_CLAIM_MESSAGE)
    return "\n".join(repaired_lines).strip()


def _verify_generated_answer(payload: dict[str, Any], answer: str) -> tuple[str, dict[str, Any]]:
    """Verify a completed generated answer and append the canonical visible trace."""
    sanitized_answer = _sanitize_answer_text(answer)
    verification = verify_answer_grounding(sanitized_answer, payload.get("retrieved_documents", []))
    trace = {
        "step": "verify_grounding",
        "tool": "grounding_verifier",
        "status": "completed",
        "verification_status": "grounded" if verification["is_grounded"] else "unsupported",
        "input_summary": "Complete generated answer and retrieved source metadata",
        "output_summary": (
            f"Verified {len(verification['checked_claims'])} claim(s); "
            f"identified {len(verification['unsupported_claims'])} unsupported claim(s)."
        ),
        "unsupported_claims": verification["unsupported_claims"][:3],
    }
    payload.setdefault("traces", []).append(trace)

    final_answer = sanitized_answer
    final_verification = verification
    repair_needed = (not verification["is_grounded"]) or _has_vague_or_invalid_citations(sanitized_answer)
    if repair_needed:
        repaired_answer = _repair_unsupported_answer(sanitized_answer, verification["unsupported_claims"])
        payload.setdefault("traces", []).append(
            {
                "step": "answer_repair",
                "tool": "answer_repair",
                "status": "completed",
                "input_summary": "Unsupported or vague-cited claims from first grounding pass",
                "output_summary": "Removed or qualified unsupported claims once.",
                "claims_repaired": len(verification["unsupported_claims"]),
            }
        )
        final_answer = repaired_answer
        final_verification = verify_answer_grounding(final_answer, payload.get("retrieved_documents", []))

    payload.setdefault("traces", []).append(
        {
            "step": "verify_grounding",
            "tool": "final_grounding_verifier",
            "status": "completed",
            "verification_status": "grounded" if final_verification["is_grounded"] else "unsupported",
            "input_summary": "Final answer after optional bounded repair",
            "output_summary": (
                f"Verified {len(final_verification['checked_claims'])} final claim(s); "
                f"identified {len(final_verification['unsupported_claims'])} unsupported claim(s)."
            ),
            "unsupported_claims": final_verification["unsupported_claims"][:3],
        }
    )

    ui_trace = payload.get("_ui_reasoning_trace")
    if isinstance(ui_trace, list):
        ui_trace.append(
            {
                "tool": "grounding_verifier",
                "input_summary": trace["input_summary"],
                "output_summary": trace["output_summary"],
            }
        )
        if repair_needed:
            ui_trace.append(
                {
                    "tool": "answer_repair",
                    "input_summary": "Unsupported or vague-cited claims",
                    "output_summary": "Removed or qualified unsupported claims once.",
                }
            )
        ui_trace.append(
            {
                "tool": "final_grounding_verifier",
                "input_summary": "Final repaired answer",
                "output_summary": "Completed final grounding check.",
            }
        )
    return final_answer, final_verification


def stream_query_response(llm, payload: dict[str, Any]):
    """Stream either a direct fallback response or an LLM completion."""
    if payload.get("response_mode") in {"clarification", "fallback", "direct"}:
        yield _sanitize_answer_text(payload.get("answer", ""))
        return

    prompt = payload.get("prompt", "")
    complete_answer = ""
    for chunk in llm.stream([_human_message(prompt)]):
        if chunk.content:
            sanitized = _sanitize_answer_text(chunk.content)
            complete_answer += sanitized
            yield sanitized
    repaired_answer, verification = _verify_generated_answer(payload, complete_answer)
    if repaired_answer != complete_answer:
        yield "\n\nGrounding repair:\n" + repaired_answer
    elif not verification["is_grounded"]:
        yield "\n\nGrounding check: the retrieved evidence does not support every generated claim."


def run_query(
    llm,
    user_query: str,
    thread_id: str = "default",
    chat_history: list[dict[str, Any]] | None = None,
    vectorstore_stats: dict[str, Any] | None = None,
    retrieval_fn: Callable[[str, int, str], list] | None = None,
    retrieval_scope: str = "all",
) -> dict[str, Any]:
    """Execute the graph and optionally call the LLM when grounding is ready."""
    del thread_id
    payload = prepare_query_payload(
        user_query=user_query,
        chat_history=chat_history,
        vectorstore_stats=vectorstore_stats,
        retrieval_fn=retrieval_fn,
        retrieval_scope=retrieval_scope,
    )

    if payload.get("response_mode") in {"clarification", "fallback", "direct"}:
        answer = _sanitize_answer_text(payload.get("answer", ""))
    else:
        response = llm.invoke([_human_message(payload.get("prompt", ""))])
        answer, verification = _verify_generated_answer(payload, response.content)
        if not verification["is_grounded"]:
            answer += "\n\nGrounding check: some claims are not supported by the retrieved evidence."

    return {
        "answer": answer,
        "reasoning_trace": payload.get("traces", []),
        "all_messages": [],
        "payload": payload,
    }
