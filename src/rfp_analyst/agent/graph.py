"""Deterministic agent graph for orchestration."""

from __future__ import annotations

from rag_engine import get_vectorstore_stats
from rfp_analyst.agent.prompts import (
    build_agentic_prompt,
    build_ambiguous_question_message,
    build_no_documents_message,
    classify_query_intent,
    is_ambiguous_query,
)
from rfp_analyst.agent.state import AgentState
from rfp_analyst.tools.compare_projects import compare_projects
from rfp_analyst.tools.proposal_writer import generate_proposal_outline
from rfp_analyst.tools.rfp_gap_analyzer import extract_rfp_requirements, find_relevant_case_studies
from rfp_analyst.tools.search_kb import search_knowledge_base


def _add_source_trace(state: AgentState, sources: list[dict]) -> None:
    for source in sources[:5]:
        state.tool_trace.append(
            {
                "tool_response": f"{source['source']} (Page {source['page'] + 1})",
                "snippet": source["snippet"],
            }
        )


def run_agent_graph(state: AgentState, search_fn=None, stats_fn=None) -> AgentState:
    """Run the agent workflow from intent classification to synthesis prompt creation."""
    state.stats = stats_fn() if stats_fn else get_vectorstore_stats()
    if state.stats.get("status") != "ready" or state.stats.get("total_documents", 0) == 0:
        state.tool_trace.append({"tool": "knowledge_base_status", "input": {"status": "not_initialized"}})
        state.final_answer = build_no_documents_message(state.query)
        return state

    if is_ambiguous_query(state.query):
        state.tool_trace.append({"tool": "ambiguity_check", "input": {"status": "ambiguous"}})
        state.final_answer = build_ambiguous_question_message(state.query)
        return state

    state.intent = classify_query_intent(state.query)
    state.tool_trace.append({"tool": "classify_intent", "input": {"query": state.query, "intent": state.intent}})

    if state.intent == "compare_projects":
        comparison = compare_projects(state.query, search_fn=search_fn)
        state.tool_outputs["compare_projects"] = comparison
        state.retrieved_documents = comparison["documents"]
        state.sources = comparison["sources"]
        state.tool_trace.append({"tool": "compare_projects", "input": {"query": state.query}})
        _add_source_trace(state, state.sources)
    elif state.intent in {"rfp_gap_analysis", "proposal_writer"}:
        requirements = extract_rfp_requirements(state.query)
        state.tool_outputs["extract_rfp_requirements"] = requirements
        state.tool_trace.append({"tool": "extract_rfp_requirements", "input": {"count": len(requirements['requirements'])}})

        case_studies = find_relevant_case_studies(requirements["requirements"], search_fn=search_fn)
        state.tool_outputs["find_relevant_case_studies"] = case_studies
        state.retrieved_documents = case_studies["documents"]
        state.sources = [
            {"source": match["source"], "page": match["pages"][0] if match["pages"] else 0, "snippet": match["snippets"][0] if match["snippets"] else ""}
            for match in case_studies["matches"]
        ]
        state.tool_trace.append({"tool": "find_relevant_case_studies", "input": {"matches": len(case_studies['matches'])}})
        _add_source_trace(state, state.sources)

        if state.intent == "proposal_writer":
            outline = generate_proposal_outline(state.query, case_studies, requirements["requirements"])
            state.tool_outputs["generate_proposal_outline"] = outline
            state.tool_trace.append({"tool": "generate_proposal_outline", "input": {"sections": 5}})
    else:
        search_result = search_knowledge_base(state.query, search_fn=search_fn)
        state.tool_outputs["search_knowledge_base"] = search_result
        state.retrieved_documents = search_result["documents"]
        state.sources = search_result["sources"]
        state.tool_trace.append({"tool": "search_knowledge_base", "input": {"query": state.query}})
        _add_source_trace(state, state.sources)

    if not state.retrieved_documents and not state.sources:
        state.final_answer = "I couldn't find relevant documents for this request. Please ingest more documents or refine the question."
        return state

    state.prompt = build_agentic_prompt(state)
    state.tool_trace.append({"tool": "verify_answer_grounding", "input": {"status": "planned"}})
    return state
