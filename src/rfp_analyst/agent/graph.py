"""LangGraph-backed query workflow with deterministic fallback."""

from __future__ import annotations

import html
import re
from functools import lru_cache
from typing import Any, Callable, Literal, TypedDict

from config import (
    AGENT_SYSTEM_PROMPT,
    MAX_CASE_STUDIES,
    MAX_CHUNKS_PER_CASE_STUDY,
    MAX_CONTEXT_CHARS_PER_CHUNK,
    MAX_HISTORY_MESSAGES,
    MAX_PROMPT_TOKENS,
    MAX_TARGET_CHUNKS,
    MAX_QUERY_RETRIES,
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
from rfp_analyst.agent.grader import grade_kb_evidence, grade_web_evidence
from rfp_analyst.agent.query_rewriter import rewrite_query
from rfp_analyst.agent.router import route_after_router, route_question
from rfp_analyst.agent.prompts import (
    DIRECT_ANSWER_PROMPT,
    KB_GENERATION_PROMPT,
    WEB_GENERATION_PROMPT,
)
from rfp_analyst.tools.compare_projects import compare_projects
from rfp_analyst.tools.project_catalog import build_project_timeline_catalog
from rfp_analyst.tools.proposal_writer import generate_proposal_outline
from rfp_analyst.tools.rfp_gap_analyzer import (
    extract_rfp_requirements,
    find_relevant_case_studies,
)
from rfp_analyst.tools.source_verifier import verify_answer_grounding
from rfp_analyst.tools.web_search import search_web

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
ADAPTIVE_RETRIEVAL_FALLBACK_MARGIN = 0.15
PERSONAL_DOCUMENT_SCORE_BOOST = 0.25
PERSONAL_DOCUMENT_K = 20
PROJECT_CATALOG_CHUNKS_PER_DOCUMENT = 3

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
    source_used: str
    current_query: str
    router_llm: Any
    allow_llm_routing: bool
    kb_grade: str
    web_grade: str
    web_results: str
    retry_count: int
    grader_llm: Any
    allow_llm_grading: bool
    rewriter_llm: Any
    allow_llm_rewriting: bool
    llm: Any


class DeterministicCompiledGraph:
    """Fallback runner that mirrors the LangGraph node flow."""

    def __init__(self):
        self.node_order = [
            "health_check",
            "route_question",
            "retrieve_kb",
            "grade_kb_evidence",
            "execute_tools",
            "synthesize_prompt",
            "generate_from_kb",
        ]

    def invoke(self, state: QueryState) -> QueryState:
        current = dict(state)
        current["graph_backend"] = "deterministic-fallback"
        current["allow_llm_routing"] = False
        current["allow_llm_grading"] = False
        current["allow_llm_rewriting"] = False
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
    workflow.add_node("route_question", route_question_node)
    workflow.add_node("retrieve_kb", retrieve_kb)
    workflow.add_node("grade_kb_evidence", grade_kb_evidence_node)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("synthesize_prompt", synthesize_prompt)
    workflow.add_node("generate_from_kb", generate_from_kb)
    workflow.add_node("search_web", search_web_node)
    workflow.add_node("grade_web_evidence", grade_web_evidence_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate_from_web", generate_from_web)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("answer_insufficient", answer_insufficient)

    workflow.add_edge(START, "health_check")
    workflow.add_edge("health_check", "route_question")
    workflow.add_conditional_edges(
        "route_question",
        route_after_router,
        {"retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"},
    )
    workflow.add_edge("retrieve_kb", "grade_kb_evidence")
    workflow.add_conditional_edges(
        "grade_kb_evidence",
        decide_after_kb_grade,
        {"execute_tools": "execute_tools", "search_web": "search_web"},
    )
    workflow.add_edge("execute_tools", "synthesize_prompt")
    workflow.add_edge("synthesize_prompt", "generate_from_kb")
    workflow.add_edge("search_web", "grade_web_evidence")
    workflow.add_conditional_edges(
        "grade_web_evidence",
        decide_after_web_grade,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "answer_insufficient": "answer_insufficient",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve_kb")
    workflow.add_edge("generate_from_kb", END)
    workflow.add_edge("generate_from_web", END)
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("answer_insufficient", END)

    return workflow.compile()


def _append_trace(state: QueryState, step: str, details: dict[str, Any]) -> list[dict[str, Any]]:
    trace = list(state.get("traces", []))
    trace.append({"step": step, **details})
    return trace


def decide_after_kb_grade(state: QueryState) -> Literal["execute_tools", "search_web"]:
    """Select tool execution or web fallback after grading KB evidence."""
    return "execute_tools" if state.get("kb_grade") == "good" else "search_web"


def decide_after_web_grade(
    state: QueryState,
) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    """Generate from good web evidence or retry only within the configured bound."""
    if state.get("web_grade") == "good":
        return "generate_from_web"
    if int(state.get("retry_count", 0) or 0) < MAX_QUERY_RETRIES:
        return "rewrite_query"
    return "answer_insufficient"


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


def _is_project_catalog_query(query: str) -> bool:
    normalized = " ".join(str(query or "").lower().split())
    broad_request = bool(re.search(r"\b(list|show|all|every|inventory|catalog)\b", normalized))
    project_request = bool(re.search(r"\b(projects?|case studies)\b", normalized))
    repeated_field = bool(
        re.search(r"\b(timelines?|milestones?|durations?|budgets?|tech(?:nology)? stacks?)\b", normalized)
    )
    return broad_request and project_request and repeated_field


def _is_personal_document_query(query: str) -> bool:
    normalized = " ".join(str(query or "").lower().split())
    return bool(
        re.search(r"\b(resume|curriculum vitae|cv|professional profile)\b", normalized)
        and re.search(r"\b(my|mine|uploaded|document|profile|skills?|stack|experience)\b", normalized)
    )


def _scope_stats(state: QueryState, scope: str) -> tuple[int, int]:
    stats = state.get("vectorstore_stats", {}) or {}
    chunk_count = int((stats.get("scope_chunk_counts", {}) or {}).get(scope, 0) or 0)
    if scope == "upload":
        document_count = int(stats.get("indexed_upload_document_count", 0) or 0)
    elif scope == "sample":
        document_count = int(stats.get("indexed_sample_document_count", 0) or 0)
        if not document_count:
            document_count = len(stats.get("indexed_sample_files", []) or [])
    else:
        document_count = int(stats.get("total_documents", 0) or 0)
    return chunk_count, document_count


def _adaptive_retrieval_plan(
    state: QueryState,
    base_query: str,
    requested_scope: str,
    requested_k: int,
) -> tuple[str, str, int, str]:
    """Choose a query-aware scope and candidate budget without changing UI scope."""

    if _is_project_catalog_query(base_query):
        scope = "sample" if requested_scope == "all" else requested_scope
        chunk_count, document_count = _scope_stats(state, scope)
        desired_k = max(requested_k, max(document_count, 1) * PROJECT_CATALOG_CHUNKS_PER_DOCUMENT)
        if chunk_count:
            desired_k = min(desired_k, chunk_count)
        query = f"{base_query}. Timeline & Milestones. Total Duration."
        return query, scope, desired_k, "project_catalog"

    if _is_personal_document_query(base_query):
        scope = "upload" if requested_scope == "all" else requested_scope
        chunk_count, _ = _scope_stats(state, scope)
        desired_k = max(requested_k, PERSONAL_DOCUMENT_K)
        if chunk_count:
            desired_k = min(desired_k, chunk_count)
        query = (
            f"{base_query}. Uploaded resume Technical Skills technologies frameworks "
            "programming languages tools backend agents RAG."
        )
        return query, scope, desired_k, "personal_document"

    return base_query, requested_scope, requested_k, "standard"


def _personal_filename_match(source: str) -> bool:
    normalized = re.sub(r"[_-]+", " ", str(source)).lower()
    return any(marker in normalized for marker in ("resume", "curriculum vitae", " cv ", "profile"))


def _select_catalog_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep timeline-bearing chunks while preserving broad source coverage."""

    candidates = [
        item
        for item in documents
        if item["score"] >= max(MIN_RELEVANCE_SCORE - ADAPTIVE_RETRIEVAL_FALLBACK_MARGIN, 0.0)
    ]
    candidates.sort(
        key=lambda item: (
            not bool(re.search(r"Timeline\s*&\s*Milestones|Total\s+Duration", item.get("content", ""), re.I)),
            -float(item.get("score", 0.0) or 0.0),
            item.get("source", ""),
            item.get("page", 0),
        )
    )
    selected: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}
    for item in candidates:
        source = str(item.get("source", "Unknown"))
        if per_source.get(source, 0) >= 2:
            continue
        selected.append(item)
        per_source[source] = per_source.get(source, 0) + 1
    return selected


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
    user_query = state.get("user_query", "")
    resolved_query, resolved_entities, resolution_status = _resolve_conversational_query(
        user_query,
        state.get("chat_history"),
        retrieval_scope,
    )
    if any(
        phrase in user_query.lower()
        for phrase in (
            "what sources did you use",
            "which documents were used",
            "show the citations from your last response",
            "sources for the previous answer",
        )
    ):
        sources = _previous_answer_sources(state.get("chat_history"))
        return {
            "intent": "previous_sources",
            "response_mode": "direct",
            "answer": _format_previous_sources(sources),
            "tool_outputs": {"previous_sources": sources},
            "resolved_query": resolved_query,
            "resolved_entities": [],
            "source_used": "direct",
            "current_query": user_query,
            "traces": _append_trace(
                state,
                "classify_intent",
                {
                    "tool": "previous_sources",
                    "intent": "previous_sources",
                    "input_summary": user_query[:120],
                    "output_summary": f"Returned {len(sources)} source citation(s) from the previous answer.",
                },
            ),
        }
    if resolution_status == "ambiguous" or _is_ambiguous_query(user_query):
        intent = "ambiguous"
        routing = {"source_used": "direct", "current_query": user_query}
    else:
        routing = route_question(state)
        intent = routing["intent"]

    response_mode = "clarification" if intent == "ambiguous" else "llm"
    if intent == "direct":
        response_mode = "direct"
    answer = FOLLOWUP_CLARIFICATION_MESSAGE if resolution_status == "ambiguous" else ""
    answer = answer or (CLARIFICATION_MESSAGE if intent == "ambiguous" else "")
    if intent == "direct":
        answer = "Hello! How can I help with the Internal RFP knowledge base?"
    return {
        "intent": intent,
        "response_mode": response_mode,
        "answer": answer,
        "resolved_query": resolved_query,
        "resolved_entities": resolved_entities,
        "source_used": routing["source_used"],
        "current_query": routing["current_query"],
        "traces": _append_trace(
            state,
            "classify_intent",
            {
                "tool": "intent_classifier",
                "intent": intent,
                "route": routing["source_used"],
                "input_summary": user_query[:120],
                "resolution_status": resolution_status,
                "resolved_entities": [entity["source"] for entity in resolved_entities],
            },
        ),
    }


def route_question_node(state: QueryState) -> QueryState:
    """Preserve conversational safeguards around the structured router node."""
    result = classify_intent(state)
    if result.get("intent") not in {"direct", "ambiguous", "previous_sources"}:
        result["current_query"] = result.get("resolved_query") or result.get("current_query")
    return result


def plan_tools(state: QueryState) -> QueryState:
    intent = state.get("intent", "search")
    planned_tools: list[str] = []
    if intent == "search":
        planned_tools = ["search_knowledge_base"]
        query = state.get("resolved_query") or state.get("current_query") or state.get("user_query", "")
        if _is_project_catalog_query(query):
            planned_tools.append("project_catalog")
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


def retrieve_kb(state: QueryState) -> QueryState:
    """Plan KB tools, then execute the existing relevance-filtered retrieval."""
    planned_state = dict(state)
    planned_state.update(plan_tools(state))
    result = execute_retrieval(planned_state)
    result["planned_tools"] = planned_state["planned_tools"]
    return result


def grade_kb_evidence_node(state: QueryState) -> QueryState:
    """Grade KB evidence and retain the legacy visible availability trace."""
    result = grade_kb_evidence(state)
    grade = result.get("kb_grade", "weak")
    result["traces"] = _append_trace(
        state,
        "grade_kb_evidence",
        {
            "tool": "evidence_availability_check",
            "status": "completed",
            "grade": grade,
            "input_summary": "Pre-filtered Private KB evidence",
            "output_summary": f"Private KB evidence graded {grade}.",
        },
    )
    return result


def grade_web_evidence_node(state: QueryState) -> QueryState:
    """Grade web evidence while preserving an inspectable trace."""
    result = grade_web_evidence(state)
    grade = result.get("web_grade", "weak")
    result["traces"] = _append_trace(
        state,
        "grade_web_evidence",
        {
            "tool": "web_evidence_grader",
            "status": "completed",
            "grade": grade,
            "input_summary": "Web Search evidence",
            "output_summary": f"Web Search evidence graded {grade}.",
        },
    )
    return result


def search_web_node(state: QueryState) -> QueryState:
    """Run web fallback and expose only a concise, user-visible trace."""
    result = search_web(state)
    has_results = bool(str(result.get("web_results", "")).strip())
    result["traces"] = _append_trace(
        state,
        "search_web",
        {
            "tool": "web_search",
            "status": "completed" if has_results else "unavailable",
            "input_summary": state.get("current_query") or state.get("user_query", ""),
            "output_summary": (
                "Retrieved Web Search evidence."
                if has_results
                else "No Web Search evidence was available."
            ),
        },
    )
    return result


def rewrite_query_node(state: QueryState) -> QueryState:
    """Rewrite a query and record the bounded retry without hidden reasoning."""
    result = rewrite_query(state)
    result["traces"] = _append_trace(
        state,
        "rewrite_query",
        {
            "tool": "query_rewriter",
            "status": "completed",
            "retry_count": result.get("retry_count", 0),
            "input_summary": state.get("current_query") or state.get("user_query", ""),
            "output_summary": f"Prepared bounded retrieval retry {result.get('retry_count', 0)}.",
        },
    )
    return result


def execute_retrieval(state: QueryState) -> QueryState:
    planned_tools = state.get("planned_tools", [])
    is_rfp_analysis = state.get("intent") == "rfp_analysis"
    requested_scope = state.get("retrieval_scope", "all")
    retrieval_scope = "upload" if is_rfp_analysis else requested_scope
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
    base_query = state.get("current_query") or state.get("resolved_query") or state.get("user_query", "")
    if is_rfp_analysis:
        retrieval_query = _build_rfp_target_query(base_query)
        retrieval_mode = "rfp_analysis"
    else:
        retrieval_query, retrieval_scope, retrieval_k, retrieval_mode = _adaptive_retrieval_plan(
            state,
            base_query,
            requested_scope,
            retrieval_k,
        )
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
        raw_score = float(score)
        source = metadata.get("source_file", "Unknown")
        adjusted_score = raw_score
        if retrieval_mode == "personal_document" and _personal_filename_match(source):
            adjusted_score = min(1.0, raw_score + PERSONAL_DOCUMENT_SCORE_BOOST)
        documents.append(
            {
                "source": source,
                "page": page,
                "score": adjusted_score,
                "raw_score": raw_score,
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

    if retrieval_mode == "project_catalog":
        selected_documents = _select_catalog_documents(documents)
    elif retrieval_mode == "personal_document" and not selected_documents:
        near_threshold = [
            item
            for item in documents
            if _personal_filename_match(item["source"])
            and item["score"] >= max(MIN_RELEVANCE_SCORE - ADAPTIVE_RETRIEVAL_FALLBACK_MARGIN, 0.0)
        ]
        selected_documents = _dedupe_documents_by_source_page(near_threshold, PERSONAL_DOCUMENT_K)

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
            "requested_scope": requested_scope,
            "retrieval_mode": retrieval_mode,
            "input": {
                "query": retrieval_query,
                "original_query": state.get("user_query", ""),
                "k": retrieval_k,
                "scope": retrieval_scope,
            },
            "input_summary": (
                f"Search scope={retrieval_scope}; k={retrieval_k}"
                + (f"; adaptive mode={retrieval_mode}" if retrieval_mode != "standard" else "")
            ),
            "output_summary": (
                f"Retrieved {len(documents)} relevant chunk(s); "
                f"filtered {below_threshold_count} below threshold {MIN_RELEVANCE_SCORE:.2f}"
            ),
            "documents": [
                {
                    "source": item["source"],
                    "page": item["page"] + 1,
                    "score": f"{item['score']:.2f}",
                    "raw_score": f"{item.get('raw_score', item['score']):.2f}",
                    "chunk_id": item["chunk_id"],
                    "document_origin": item["document_origin"],
                }
                for item in documents[:20]
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
        "response_mode": "llm",
        "answer": "",
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

    if "project_catalog" in state.get("planned_tools", []):
        result = build_project_timeline_catalog(state.get("retrieved_documents", []))
        outputs["project_catalog"] = result
        if result.get("answer_markdown"):
            notes.append(result["answer_markdown"])
        traces.append(
            {
                "step": "execute_specialized_tool",
                "tool": "project_catalog",
                "status": "completed",
                "input_summary": "Retrieved timeline-bearing project evidence",
                "output_summary": (
                    f"Extracted grounded timelines for {result.get('project_count', 0)} project document(s)."
                ),
            }
        )

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


def execute_tools(state: QueryState) -> QueryState:
    """Run the specialized tools selected by the existing intent planner."""
    return execute_specialized_tool(state)


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


def _invoke_generation(state: QueryState, prompt: str, source_used: str, verify_kb: bool) -> QueryState:
    """Invoke the supplied LLM inside the graph and preserve visible grounding traces."""
    llm = state.get("llm")
    if llm is None:
        return {
            "answer": state.get("answer", ""),
            "response_mode": state.get("response_mode", "llm"),
            "source_used": source_used,
            "traces": _append_trace(
                state,
                "final_response",
                {
                    "tool": "final_response",
                    "response_mode": "llm",
                    "output_summary": "Prepared generation prompt; no LLM instance was supplied.",
                },
            ),
        }

    response = llm.invoke([_human_message(prompt)])
    answer = str(getattr(response, "content", response) or "")
    traces = _append_trace(
        state,
        "final_response",
        {
            "tool": "final_response",
            "response_mode": "llm",
            "source_used": source_used,
            "output_summary": "Generated an answer inside the graph.",
        },
    )
    if not verify_kb:
        return {
            "answer": _sanitize_answer_text(answer),
            "response_mode": "llm",
            "source_used": source_used,
            "traces": traces,
        }

    verification_payload = {
        "retrieved_documents": state.get("retrieved_documents", []),
        "traces": traces,
    }
    answer, verification = _verify_generated_answer(verification_payload, answer)
    return {
        "answer": answer,
        "response_mode": "llm",
        "source_used": source_used,
        "grounded": verification["is_grounded"],
        "traces": verification_payload["traces"],
    }


def generate_from_kb(state: QueryState) -> QueryState:
    """Generate and ground a private-KB answer within the graph."""
    catalog = (state.get("tool_outputs", {}) or {}).get("project_catalog", {}) or {}
    catalog_answer = str(catalog.get("answer_markdown", "") or "").strip()
    if catalog_answer:
        traces = _append_trace(
            state,
            "final_response",
            {
                "tool": "final_response",
                "response_mode": "deterministic_grounded",
                "source_used": "private_kb",
                "output_summary": "Rendered the project inventory directly from extracted KB fields.",
            },
        )
        verification_payload = {
            "retrieved_documents": state.get("retrieved_documents", []),
            "traces": traces,
        }
        answer, verification = _verify_generated_answer(verification_payload, catalog_answer)
        return {
            "answer": answer,
            "response_mode": "llm",
            "source_used": "private_kb",
            "grounded": verification["is_grounded"],
            "traces": verification_payload["traces"],
        }

    prompt = KB_GENERATION_PROMPT.format(
        question=state.get("user_query", ""),
        context=state.get("retrieval_context", ""),
    )
    if state.get("specialized_notes"):
        prompt += "\n\nSpecialized analysis:\n" + state["specialized_notes"]
    return _invoke_generation(state, prompt, "private_kb", verify_kb=True)


def generate_from_web(state: QueryState) -> QueryState:
    """Generate an answer from web evidence after successful web grading."""
    prompt = WEB_GENERATION_PROMPT.format(
        question=state.get("user_query", ""),
        web_context=state.get("web_results", ""),
    )
    return _invoke_generation(state, prompt, "web_search", verify_kb=False)


def direct_answer(state: QueryState) -> QueryState:
    """Answer a conversational message without retrieval."""
    if state.get("intent") == "previous_sources" or state.get("response_mode") == "clarification":
        return {
            "source_used": "direct",
            "traces": _append_trace(
                state,
                "final_response",
                {"tool": "final_response", "response_mode": "direct", "output_summary": "Returned prior source citations."},
            ),
        }
    llm = state.get("llm")
    answer = "Hello! How can I help with the Internal RFP knowledge base?"
    if llm is not None:
        response = llm.invoke(
            [_human_message(DIRECT_ANSWER_PROMPT.format(question=state.get("user_query", "")))]
        )
        answer = str(getattr(response, "content", response) or answer)
    return {
        "answer": _sanitize_answer_text(answer),
        "response_mode": "direct",
        "source_used": "direct",
        "traces": _append_trace(
            state,
            "final_response",
            {"tool": "final_response", "response_mode": "direct", "output_summary": "Generated a direct response without retrieval."},
        ),
    }


def answer_insufficient(state: QueryState) -> QueryState:
    """Return a graceful response after bounded KB and web evidence attempts."""
    answer = state.get("answer") or (
        f"{INSUFFICIENT_EVIDENCE_MESSAGE} I could not find enough reliable evidence "
        "in the private knowledge base or web search results to answer confidently."
    )
    return {
        "answer": answer,
        "response_mode": "fallback",
        "source_used": "insufficient_evidence",
        "traces": _append_trace(
            state,
            "final_response",
            {"tool": "final_response", "response_mode": "fallback", "output_summary": "Returned insufficient-evidence response after bounded retries."},
        ),
    }


NODE_FUNCTIONS = {
    "health_check": health_check,
    "route_question": route_question_node,
    "retrieve_kb": retrieve_kb,
    "grade_kb_evidence": grade_kb_evidence_node,
    "execute_tools": execute_tools,
    "synthesize_prompt": synthesize_prompt,
    "generate_from_kb": generate_from_kb,
}


def prepare_query_payload(
    user_query: str,
    chat_history: list[dict[str, Any]] | None = None,
    vectorstore_stats: dict[str, Any] | None = None,
    retrieval_fn: Callable[[str, int, str], list] | None = None,
    retrieval_k: int = RETRIEVAL_K,
    retrieval_scope: str = "all",
    llm: Any | None = None,
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
        "llm": llm,
        "current_query": user_query,
        "retry_count": 0,
        "web_results": "",
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
    sanitized = html.unescape(str(answer or ""))
    sanitized = sanitized.replace("\t", " ").replace("[Source: None]", UNSUPPORTED_CLAIM_MESSAGE)
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
            if line.strip().startswith("|") and line.strip().endswith("|"):
                changed = True
                continue
            repaired_lines.append(UNSUPPORTED_CLAIM_MESSAGE)
            changed = True
            continue
        if any(claim in line for claim in unsupported):
            if line.strip().startswith("|") and line.strip().endswith("|"):
                changed = True
                continue
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
    """Stream graph-produced answers, retaining legacy prompt streaming support."""
    if "answer" in payload:
        yield _sanitize_answer_text(payload.get("answer", ""))
        return
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
        llm=llm,
        chat_history=chat_history,
        vectorstore_stats=vectorstore_stats,
        retrieval_fn=retrieval_fn,
        retrieval_scope=retrieval_scope,
    )

    answer = _sanitize_answer_text(payload.get("answer", ""))

    return {
        "answer": answer,
        "reasoning_trace": payload.get("traces", []),
        "all_messages": [],
        "payload": payload,
    }
