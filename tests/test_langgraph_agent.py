from types import SimpleNamespace

import importlib
import re
import sys
from types import ModuleType

from config import MAX_PROMPT_TOKENS, MAX_QUERY_RETRIES, RFP_ANALYSIS_MAX_OUTPUT_TOKENS
import agent as ui_agent

from rfp_analyst.agent.graph import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    MISSING_UPLOAD_CONTEXT_MESSAGE,
    NO_RELEVANT_UPLOAD_TARGET_MESSAGE,
    compile_query_graph,
    prepare_query_payload,
    run_query,
    decide_after_kb_grade,
    decide_after_web_grade,
)
from rfp_analyst.agent.grader import grade_kb_evidence, grade_web_evidence
from rfp_analyst.agent.query_rewriter import rewrite_query
from rfp_analyst.agent.router import route_after_router, route_question
from rfp_analyst.agent.schemas_decisions import EvidenceGrade, QueryRewrite, RouteDecision
from rfp_analyst.agent.prompts import (
    KB_GENERATION_PROMPT,
    KB_GRADER_PROMPT,
    QUERY_REWRITER_PROMPT,
    ROUTER_PROMPT,
    WEB_GENERATION_PROMPT,
    WEB_GRADER_PROMPT,
)
from rfp_analyst.tools import web_search
from rfp_analyst.tools.source_verifier import verify_answer_grounding

graph_module = importlib.import_module("rfp_analyst.agent.graph")


def test_specialized_prompts_have_required_contracts():
    assert "Private KB" in KB_GENERATION_PROMPT
    assert "[Source: <doc>, Page <page>]" in KB_GENERATION_PROMPT
    assert "Markdown" in KB_GENERATION_PROMPT
    assert "Web Search" in WEB_GENERATION_PROMPT
    assert "[Title](URL)" in WEB_GENERATION_PROMPT
    assert "RFP analysis" in ROUTER_PROMPT
    assert '{"grade": "good"}' in KB_GRADER_PROMPT.format(question="q", context="c")
    assert '{"grade": "weak"}' in WEB_GRADER_PROMPT.format(question="q", web_results="w")
    assert "Do not answer" in QUERY_REWRITER_PROMPT.format(question="q")
    assert "context" in KB_GENERATION_PROMPT.format(question="q", context="c").lower()
    assert "web search context" in WEB_GENERATION_PROMPT.format(question="q", web_context="w").lower()


class FakeDoc:
    def __init__(self, content: str, source: str, page: int = 0, origin: str = "sample"):
        self.page_content = content
        self.metadata = {
            "source_file": source,
            "page": page,
            "document_origin": origin,
            "chunk_id": f"{origin}-{source}-{page}",
        }


def fake_retrieval(query: str, k: int):
    return [
        (
            FakeDoc(
                content=f"Evidence for: {query}",
                source="banking_case_study.pdf",
                page=0,
            ),
            0.91,
        )
    ][:k]


def ready_stats():
    return {
        "status": "ready",
        "total_documents": 3,
        "total_chunks": 9,
        "document_names": [
            "banking_case_study.pdf",
            "healthcare_migration.pdf",
            "insurance_automation.pdf",
        ],
    }


def not_ready_stats():
    return {
        "status": "not_initialized",
        "total_documents": 0,
        "total_chunks": 0,
        "document_names": [],
    }


def test_graph_compiles():
    graph = compile_query_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")


class FakeStructuredRouter:
    def __init__(self, decision):
        self.decision = decision
        self.prompt = ""

    def invoke(self, prompt):
        self.prompt = prompt
        return self.decision


class FakeRouterLLM:
    def __init__(self, decision):
        self.schema = None
        self.method = None
        self.structured_router = FakeStructuredRouter(decision)

    def with_structured_output(self, schema, method):
        self.schema = schema
        self.method = method
        return self.structured_router


def test_router_uses_structured_output_and_refines_kb_intent():
    llm = FakeRouterLLM(RouteDecision(route="kb"))

    result = route_question(
        {"user_query": "Compare the healthcare and insurance projects", "router_llm": llm}
    )

    assert llm.schema is RouteDecision
    assert llm.method == "json_mode"
    assert "Compare the healthcare" in llm.structured_router.prompt
    assert result == {
        "intent": "compare",
        "source_used": "kb",
        "current_query": "Compare the healthcare and insurance projects",
    }


def test_router_returns_direct_branch_for_simple_conversation():
    result = route_question(
        {"user_query": "Thanks!", "router_llm": FakeRouterLLM(RouteDecision(route="direct"))}
    )

    assert result["intent"] == "direct"
    assert result["source_used"] == "direct"
    assert route_after_router(result) == "direct_answer"


def test_router_falls_back_without_an_llm_provider():
    result = route_question(
        {"user_query": "Write a proposal for the banking RFP", "allow_llm_routing": False}
    )

    assert result["intent"] == "proposal"
    assert result["source_used"] == "kb"
    assert route_after_router(result) == "retrieve_kb"


def test_kb_grader_uses_structured_output():
    llm = FakeRouterLLM(EvidenceGrade(grade="good"))

    result = grade_kb_evidence(
        {
            "user_query": "What cloud platform did the banking project use?",
            "retrieved_documents": [
                {"source": "banking_case_study.pdf", "content": "The project used Azure services."}
            ],
            "grader_llm": llm,
        }
    )

    assert result == {"kb_grade": "good"}
    assert llm.schema is EvidenceGrade
    assert llm.method == "json_mode"
    assert "banking_case_study.pdf" in llm.structured_router.prompt


def test_web_grader_uses_structured_output():
    llm = FakeRouterLLM(EvidenceGrade(grade="weak"))

    result = grade_web_evidence(
        {
            "user_query": "What is the current cloud market outlook?",
            "web_results": "A short, incomplete result.",
            "grader_llm": llm,
        }
    )

    assert result == {"web_grade": "weak"}
    assert llm.schema is EvidenceGrade
    assert llm.method == "json_mode"
    assert "incomplete result" in llm.structured_router.prompt


def test_grader_fallbacks_and_conditional_decisions():
    assert grade_kb_evidence({"retrieved_documents": [], "allow_llm_grading": False}) == {"kb_grade": "weak"}
    assert grade_web_evidence({"web_results": "", "allow_llm_grading": False}) == {"web_grade": "weak"}
    assert decide_after_kb_grade({"kb_grade": "good"}) == "execute_tools"
    assert decide_after_kb_grade({"kb_grade": "weak"}) == "search_web"
    assert decide_after_web_grade({"web_grade": "good", "retry_count": 0}) == "generate_from_web"
    assert decide_after_web_grade({"web_grade": "weak", "retry_count": 0}) == "rewrite_query"
    assert decide_after_web_grade({"web_grade": "weak", "retry_count": 1}) == "answer_insufficient"


def test_web_search_degrades_gracefully_without_tavily_key(monkeypatch, caplog):
    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "")

    result = web_search.search_web({"user_query": "What is the cloud market outlook?"})

    assert result == {"web_results": "", "source_used": "web"}
    assert "TAVILY_API_KEY is not configured" in caplog.text


def test_web_search_formats_dict_and_list_tavily_responses(monkeypatch):
    created = []

    class FakeTavilySearch:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(kwargs)

        def invoke(self, _payload):
            return {
                "answer": "Tavily summary",
                "results": [{"title": "Reference", "url": "https://example.test", "content": "Evidence"}],
            }

    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setitem(sys.modules, "langchain_tavily", ModuleType("langchain_tavily"))
    sys.modules["langchain_tavily"].TavilySearch = FakeTavilySearch

    result = web_search.search_web({"current_query": "test query"})

    assert created == [
        {
            "max_results": 5,
            "topic": "general",
            "include_answer": True,
            "include_raw_content": False,
        }
    ]
    assert "Tavily summary" in result["web_results"]
    assert "https://example.test" in result["web_results"]
    assert "Evidence" in web_search._format_web_results([{"title": "List", "content": "List evidence"}])
    assert web_search._format_web_results({"error": "network unavailable"}) == ""


def test_query_rewriter_uses_structured_output_and_tracks_retries():
    llm = FakeRouterLLM(QueryRewrite(rewritten_query="banking RFP Azure implementation details"))

    result = rewrite_query(
        {
            "user_query": "What did the banking RFP need?",
            "retry_count": 0,
            "rewriter_llm": llm,
        }
    )

    assert result == {
        "current_query": "banking RFP Azure implementation details",
        "retry_count": 1,
    }
    assert llm.schema is QueryRewrite
    assert llm.method == "json_mode"
    assert "Preserve the original intent" in llm.structured_router.prompt


class GraphGenerationLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, _messages):
        return SimpleNamespace(content=self.content)


def test_cyclic_graph_direct_route_generates_inside_graph():
    payload = prepare_query_payload(
        "Hello there, Internal RFP Analyst",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
        llm=GraphGenerationLLM("Hello! How can I help?"),
    )

    assert payload["intent"] == "direct"
    assert payload["source_used"] == "direct"
    assert payload["answer"] == "Hello! How can I help?"
    assert all(step.get("tool") != "search_knowledge_base" for step in payload["traces"])


def test_cyclic_graph_kb_route_generates_inside_graph():
    payload = prepare_query_payload(
        "What technology was used in the banking audit?",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
        llm=GraphGenerationLLM(
            "Evidence for the banking audit technology. [Source: banking_case_study.pdf, Page 1]"
        ),
    )

    assert payload["source_used"] == "private_kb"
    assert payload["answer"].startswith("Evidence for the banking audit technology")
    tools = [step.get("tool") for step in payload["traces"]]
    assert "search_knowledge_base" in tools
    assert "grounding_verifier" in tools


def test_project_catalog_query_retrieves_broad_sample_coverage_and_preserves_table():
    calls = {}

    def retrieve_catalog(query, k, scope):
        calls.update({"query": query, "k": k, "scope": scope})
        return [
            (
                FakeDoc(
                    (
                        "Timeline & Milestones\n"
                        f"- Phase 1: Discovery for project {index}\n"
                        f"- Phase 2: Delivery for project {index}\n"
                        f"Total Duration: {10 + index} weeks"
                    ),
                    f"0{index}_Project_{index}.pdf",
                    page=1,
                    origin="sample",
                ),
                0.42,
            )
            for index in range(1, 4)
        ]

    payload = prepare_query_payload(
        "List all projects with their timelines",
        vectorstore_stats={
            **ready_stats(),
            "indexed_sample_document_count": 3,
            "scope_chunk_counts": {"sample": 30, "upload": 0, "all": 30},
        },
        retrieval_fn=retrieve_catalog,
        retrieval_scope="all",
    )

    assert calls["scope"] == "sample"
    assert calls["k"] == 9
    assert "Timeline & Milestones" in calls["query"]
    assert payload["planned_tools"] == ["search_knowledge_base", "project_catalog"]
    assert payload["answer"].count("[Source:") == 3
    assert "The retrieved evidence does not support this claim." not in payload["answer"]
    assert "| Project | Timeline & Milestones | Total Duration | Source |" in payload["answer"]


def test_resume_query_targets_uploads_and_uses_filename_aware_ranking():
    calls = {}

    def retrieve_resume(query, k, scope):
        calls.update({"query": query, "k": k, "scope": scope})
        return [
            (
                FakeDoc(
                    "Technical Skills\nPython, LangGraph, FastAPI, Qdrant, PostgreSQL, Redis, Docker",
                    "Tushar_Ghosh_Atlys_AI_Intern_Resume.pdf",
                    page=0,
                    origin="upload",
                ),
                0.29,
            )
        ]

    payload = prepare_query_payload(
        "what is my tech stack in resume",
        vectorstore_stats={
            **ready_stats(),
            "indexed_upload_document_count": 3,
            "scope_chunk_counts": {"sample": 9, "upload": 12, "all": 21},
        },
        retrieval_fn=retrieve_resume,
        retrieval_scope="all",
    )

    assert calls["scope"] == "upload"
    assert calls["k"] == 12
    assert "Technical Skills" in calls["query"]
    assert payload["retrieved_documents"][0]["source"].endswith("Resume.pdf")
    assert payload["retrieved_documents"][0]["raw_score"] == 0.29
    assert payload["retrieved_documents"][0]["score"] == 0.54
    assert payload["kb_grade"] == "good"
    assert all(step.get("tool") != "web_search" for step in payload["traces"])


def test_grounding_verifier_accepts_markdown_escaped_filename_citations():
    result = verify_answer_grounding(
        "Python and FastAPI are listed. [Source: My\\_Resume.pdf, Page 1]",
        [
            {
                "source": "My_Resume.pdf",
                "page": 0,
                "content": "Technical Skills include Python and FastAPI.",
            }
        ],
    )

    assert result["is_grounded"] is True


def test_grounding_verifier_uses_filename_year_as_document_identity():
    result = verify_answer_grounding(
        "Banking Digital Audit 2024 lasted 16 weeks. "
        "[Source: 01_Banking_Digital_Audit_2024.pdf, Page 2]",
        [
            {
                "source": "01_Banking_Digital_Audit_2024.pdf",
                "page": 1,
                "content": "Timeline and milestones. Total Duration: 16 weeks.",
            }
        ],
    )

    assert result["is_grounded"] is True


def test_grounding_repair_drops_unsupported_table_row_without_corrupting_table():
    answer = (
        "| Project | Timeline | Source |\n"
        "| --- | --- | --- |\n"
        "| Supported | 12 weeks | [Source: supported.pdf, Page 1] |\n"
        "| Unsupported | 99 weeks | [Source: missing.pdf, Page 1] |"
    )

    repaired = graph_module._repair_unsupported_answer(answer, ["| Unsupported | 99 weeks |"])

    assert "| Project | Timeline | Source |" in repaired
    assert "| --- | --- | --- |" in repaired
    assert "| Supported |" in repaired
    assert "| Unsupported |" not in repaired
    assert "The retrieved evidence does not support this claim." not in repaired


def test_cyclic_graph_uses_web_fallback_after_weak_kb_evidence(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "search_web",
        lambda _state: {"web_results": "Web evidence with a useful URL", "source_used": "web"},
    )
    compile_query_graph.cache_clear()

    payload = prepare_query_payload(
        "What is the current cloud market outlook?",
        vectorstore_stats=ready_stats(),
        retrieval_fn=lambda *_args: [],
        llm=GraphGenerationLLM("Web-backed answer"),
    )

    assert payload["source_used"] == "web_search"
    assert payload["answer"] == "Web-backed answer"
    compile_query_graph.cache_clear()


def test_cyclic_graph_bounds_query_rewriting_after_weak_web_evidence(monkeypatch):
    retrieval_calls = []

    def empty_retrieval(query, *_args):
        retrieval_calls.append(query)
        return []

    monkeypatch.setattr(
        graph_module,
        "search_web",
        lambda _state: {"web_results": "", "source_used": "web"},
    )
    compile_query_graph.cache_clear()

    payload = prepare_query_payload(
        "Find the zxqv-991 nonexistent consulting initiative",
        vectorstore_stats=ready_stats(),
        retrieval_fn=empty_retrieval,
    )

    assert payload["retry_count"] == MAX_QUERY_RETRIES
    assert len(retrieval_calls) == MAX_QUERY_RETRIES + 1
    assert payload["source_used"] == "insufficient_evidence"
    tools = [step.get("tool") for step in payload["traces"]]
    assert tools.count("query_rewriter") == MAX_QUERY_RETRIES
    assert tools.count("web_search") == MAX_QUERY_RETRIES + 1
    compile_query_graph.cache_clear()


def test_ui_reasoning_trace_displays_source_type_labels():
    assert ui_agent._payload_to_reasoning_trace({"source_used": "private_kb"})[0]["source_used"] == "Private KB"
    assert ui_agent._payload_to_reasoning_trace({"source_used": "web_search"})[0]["source_used"] == "Web Search"
    assert ui_agent._payload_to_reasoning_trace({"source_used": "direct"})[0]["source_used"] == "Direct"


def test_search_intent_routes_to_search_knowledge_base():
    payload = prepare_query_payload(
        user_query="What tech stack did we use for the banking audit?",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
    )

    assert payload["intent"] == "search"
    assert payload["planned_tools"] == ["search_knowledge_base"]
    assert any(step.get("tool") == "search_knowledge_base" for step in payload["traces"])


def test_compare_intent_routes_to_compare_projects():
    payload = prepare_query_payload(
        user_query="Compare the healthcare and insurance projects",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
    )

    assert payload["intent"] == "compare"
    assert "compare_projects" in payload["planned_tools"]
    assert any(step.get("tool") == "compare_projects" for step in payload["traces"])


def test_proposal_intent_routes_to_proposal_writer():
    payload = prepare_query_payload(
        user_query="Write a proposal response for a banking modernization RFP",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
    )

    assert payload["intent"] == "proposal"
    assert "proposal_writer" in payload["planned_tools"]
    assert any(step.get("tool") == "proposal_writer" for step in payload["traces"])


def test_no_kb_returns_graceful_fallback():
    payload = prepare_query_payload(
        user_query="What projects used Azure?",
        vectorstore_stats=not_ready_stats(),
        retrieval_fn=fake_retrieval,
    )

    assert payload["response_mode"] == "fallback"
    assert "Knowledge base is not ready" in payload["answer"]
    assert any(
        step.get("tool") == "search_knowledge_base" and step.get("status") == "skipped"
        for step in payload["traces"]
    )


def test_ambiguous_query_returns_clarification():
    payload = prepare_query_payload(
        user_query="help me",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
    )

    assert payload["intent"] == "ambiguous"
    assert payload["response_mode"] == "clarification"
    assert "clarify" in payload["answer"].lower()


def test_vague_followup_uses_prior_entities():
    captured = {}
    history = [
        {
            "role": "assistant",
            "content": "The uploaded client profile uses React.",
            "reasoning": [
                {
                    "tool_response": "client_profile.pdf (Page 1)",
                    "source": "client_profile.pdf",
                    "page": 1,
                    "chunk_id": "client-profile-1",
                    "document_origin": "upload",
                }
            ],
        }
    ]

    def retrieval(query: str, k: int, scope: str):
        captured["query"] = query
        captured["scope"] = scope
        return fake_retrieval(query, k)

    payload = prepare_query_payload(
        user_query="what is tech stack here?",
        chat_history=history,
        vectorstore_stats=ready_stats(),
        retrieval_fn=retrieval,
        retrieval_scope="upload",
    )

    assert payload["response_mode"] == "llm"
    assert "client_profile.pdf" in captured["query"]
    assert captured["scope"] == "upload"
    assert payload["resolved_entities"][0]["source"] == "client_profile.pdf"


def test_ambiguous_followup_asks_for_clarification():
    payload = prepare_query_payload(
        user_query="what is tech stack here?",
        chat_history=[],
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
        retrieval_scope="all",
    )

    assert payload["response_mode"] == "clarification"
    assert "which previously discussed" in payload["answer"].lower()


def test_scope_is_retained_across_followups():
    captured = {}
    history = [
        {
            "role": "assistant",
            "content": "Resume answer",
            "reasoning": [
                {
                    "tool_response": "client_profile.pdf (Page 1)",
                    "source": "client_profile.pdf",
                    "page": 1,
                    "chunk_id": "client-profile-1",
                    "document_origin": "upload",
                }
            ],
        }
    ]

    def retrieval(query: str, k: int, scope: str):
        captured["scope"] = scope
        return fake_retrieval(query, k)

    prepare_query_payload(
        user_query="what about this?",
        chat_history=history,
        vectorstore_stats=ready_stats(),
        retrieval_fn=retrieval,
        retrieval_scope="upload",
    )

    assert captured["scope"] == "upload"


def test_complex_rfp_query_invokes_multiple_tools_with_summaries():
    payload = prepare_query_payload(
        user_query="Write a proposal response for a banking modernization RFP",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
    )

    tools = [step.get("tool") for step in payload["traces"]]
    assert "intent_classifier" in tools
    assert "search_knowledge_base" in tools
    assert "proposal_writer" in tools
    assert "evidence_availability_check" in tools
    assert all("thought" not in str(step).lower() for step in payload["traces"])
    assert any(step.get("output_summary") for step in payload["traces"])


def test_compare_tool_is_actually_called(monkeypatch):
    called = {}

    def fake_compare(query, search_fn=None, k=6):
        called["query"] = query
        called["search_fn"] = search_fn
        return {"rows": [{"source": "a.pdf"}, {"source": "b.pdf"}], "comparison_markdown": "comparison"}

    monkeypatch.setattr(graph_module, "compare_projects", fake_compare)
    payload = prepare_query_payload(
        "Compare project A versus project B",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
        retrieval_scope="sample",
    )

    assert called["query"]
    assert callable(called["search_fn"])
    assert len(payload["tool_outputs"]["compare_projects"]["rows"]) == 2
    assert any("Compared 2 projects across 4 dimensions" in step.get("output_summary", "") for step in payload["traces"])


def test_proposal_tools_are_actually_called(monkeypatch):
    calls = []
    monkeypatch.setattr(graph_module, "extract_rfp_requirements", lambda text: calls.append("extract") or {"requirements": [{"id": "REQ-01", "text": "Must use Azure"}], "summary": "Extracted 1 requirement(s)."})
    monkeypatch.setattr(graph_module, "find_relevant_case_studies", lambda requirements, search_fn=None, k=6: calls.append("cases") or {"matches": [{"source": "banking.pdf", "pages": [0]}], "documents": []})
    monkeypatch.setattr(graph_module, "generate_proposal_outline", lambda query, cases, requirements=None: calls.append("outline") or {"outline": "Outline naming banking.pdf"})

    payload = prepare_query_payload(
        "Write a proposal response for an Azure RFP",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
    )

    assert calls == ["extract", "cases", "outline"]
    assert "Outline naming banking.pdf" in payload["prompt"]


def test_cross_corpus_rfp_analysis_uses_upload_target_and_sample_cases(monkeypatch):
    calls = []

    def scoped_retrieval(query, k, scope):
        calls.append({"query": query, "scope": scope})
        source = "target_rfp.pdf" if scope == "upload" else "banking_case.pdf"
        return [(FakeDoc("Must support Azure dashboards.", source, origin=scope), 0.92)]

    payload = prepare_query_payload(
        "Extract requirements, find case studies, compare fit, and create proposal outline",
        vectorstore_stats={
            **ready_stats(),
            "indexed_upload_document_count": 1,
            "indexed_upload_files": ["target_rfp.pdf"],
            "scope_chunk_counts": {"upload": 2, "sample": 7},
        },
        retrieval_fn=scoped_retrieval,
        retrieval_scope="all",
    )

    assert payload["intent"] == "rfp_analysis"
    assert calls[0]["scope"] == "upload"
    assert "sample" in [call["scope"] for call in calls[1:]]
    assert "compare fit" not in calls[0]["query"].lower()
    assert "case studies" not in calls[0]["query"].lower()
    assert "requirements" in calls[0]["query"].lower()
    assert "architecture" in calls[0]["query"].lower()
    assert all(item["document_origin"] == "upload" for item in payload["retrieved_documents"] if item["source"] == "target_rfp.pdf")
    assert "banking_case.pdf" in payload["prompt"]
    target_trace = next(step for step in payload["traces"] if step.get("tool") == "target_context_retrieval")
    assert target_trace["indexed_upload_chunk_count"] == 2
    assert target_trace["target_fallback_used"] is False
    assert target_trace["selected_upload_source_files"] == ["target_rfp.pdf"]


def test_rfp_analysis_without_upload_context_returns_clean_fallback():
    def no_upload(query, k, scope):
        assert scope == "upload"
        return []

    payload = prepare_query_payload(
        "Extract requirements and find case studies for my uploaded RFP",
        vectorstore_stats={
            **ready_stats(),
            "indexed_upload_document_count": 0,
            "indexed_upload_files": [],
            "scope_chunk_counts": {"upload": 0, "sample": 9},
        },
        retrieval_fn=no_upload,
    )

    assert payload["response_mode"] == "fallback"
    assert payload["answer"] == MISSING_UPLOAD_CONTEXT_MESSAGE


def test_rfp_analysis_indexed_uploads_without_qualifying_hits_returns_target_message():
    def low_scores(query, k, scope):
        assert scope == "upload"
        return [
            (FakeDoc("Weakly related content", "target_rfp.pdf", page=0, origin="upload"), 0.12),
            (FakeDoc("Another weakly related note", "target_rfp.pdf", page=1, origin="upload"), 0.08),
        ]

    payload = prepare_query_payload(
        "Use uploaded documents as target requirements and numbered PDFs as internal case studies. Return technical requirements, gaps, three case studies and a proposal outline.",
        vectorstore_stats={
            **ready_stats(),
            "indexed_upload_document_count": 1,
            "indexed_upload_files": ["target_rfp.pdf"],
            "scope_chunk_counts": {"upload": 8, "sample": 9, "all": 17},
        },
        retrieval_fn=low_scores,
        retrieval_scope="all",
    )

    assert payload["response_mode"] == "fallback"
    assert payload["answer"] == NO_RELEVANT_UPLOAD_TARGET_MESSAGE
    assert payload["answer"] != MISSING_UPLOAD_CONTEXT_MESSAGE


def test_rfp_analysis_uses_bounded_upload_fallback_when_hits_are_slightly_below_threshold():
    def near_threshold(query, k, scope):
        assert scope == "upload"
        return [
            (FakeDoc("Requirement A", "target_rfp.pdf", page=0, origin="upload"), 0.47),
            (FakeDoc("Requirement A duplicate chunk", "target_rfp.pdf", page=0, origin="upload"), 0.46),
            (FakeDoc("Requirement B", "target_rfp.pdf", page=1, origin="upload"), 0.45),
            (FakeDoc("Requirement C", "target_rfp_2.pdf", page=0, origin="upload"), 0.44),
            (FakeDoc("Requirement D", "target_rfp_3.pdf", page=0, origin="upload"), 0.43),
        ]

    payload = prepare_query_payload(
        "Use uploaded documents as target requirements and numbered PDFs as internal case studies. Return technical requirements, gaps, three case studies and a proposal outline.",
        vectorstore_stats={
            **ready_stats(),
            "indexed_upload_document_count": 3,
            "indexed_upload_files": ["target_rfp.pdf", "target_rfp_2.pdf", "target_rfp_3.pdf"],
            "scope_chunk_counts": {"upload": 10, "sample": 9, "all": 19},
        },
        retrieval_fn=near_threshold,
        retrieval_scope="all",
    )

    assert payload["response_mode"] == "llm"
    upload_docs = [item for item in payload["retrieved_documents"] if item["document_origin"] == "upload"]
    assert 2 <= len(upload_docs) <= 4
    assert len({(item["source"], item["page"]) for item in upload_docs}) == len(upload_docs)
    target_trace = next(step for step in payload["traces"] if step.get("tool") == "target_context_retrieval")
    assert target_trace["target_fallback_used"] is True
    assert target_trace["qualifying_upload_hits"] == 0


def test_rfp_analysis_target_evidence_stays_upload_only():
    def scoped_retrieval(query, k, scope):
        if scope == "upload":
            return [
                (FakeDoc("Uploaded requirement", "target_rfp.pdf", page=0, origin="upload"), 0.91),
                (FakeDoc("Mislabeled sample should be dropped", "wrong.pdf", page=0, origin="sample"), 0.93),
            ]
        return [(FakeDoc("Sample case study", "case_study.pdf", page=0, origin="sample"), 0.95)]

    payload = prepare_query_payload(
        "Use uploaded documents as target requirements and numbered PDFs as internal case studies. Return technical requirements, gaps, three case studies and a proposal outline.",
        vectorstore_stats={
            **ready_stats(),
            "indexed_upload_document_count": 1,
            "indexed_upload_files": ["target_rfp.pdf"],
            "scope_chunk_counts": {"upload": 4, "sample": 7, "all": 11},
        },
        retrieval_fn=scoped_retrieval,
        retrieval_scope="all",
    )

    upload_docs = [item for item in payload["retrieved_documents"] if item["source"] == "target_rfp.pdf"]
    assert upload_docs
    assert all(item["document_origin"] == "upload" for item in upload_docs)
    assert all(item["source"] != "wrong.pdf" for item in payload["retrieved_documents"])


def test_rfp_analysis_case_study_evidence_stays_sample_only():
    def scoped_retrieval(query, k, scope):
        source = "target_rfp.pdf" if scope == "upload" else "sample_case.pdf"
        origin = "upload" if scope == "upload" else "sample"
        return [(FakeDoc("Evidence", source, page=0, origin=origin), 0.93)]

    payload = prepare_query_payload(
        "Use uploaded documents as target requirements and numbered PDFs as internal case studies. Return technical requirements, gaps, three case studies and a proposal outline.",
        vectorstore_stats={
            **ready_stats(),
            "indexed_upload_document_count": 1,
            "indexed_upload_files": ["target_rfp.pdf"],
            "scope_chunk_counts": {"upload": 2, "sample": 6, "all": 8},
        },
        retrieval_fn=scoped_retrieval,
        retrieval_scope="all",
    )

    case_docs = [item for item in payload["retrieved_documents"] if item["source"] == "sample_case.pdf"]
    assert case_docs
    assert all(item["document_origin"] == "sample" for item in case_docs)


def test_rfp_analysis_target_prompt_budget_is_bounded():
    def many_upload_hits(query, k, scope):
        assert scope == "upload"
        return [
            (FakeDoc(f"Requirement {index}", f"target_{index}.pdf", page=0, origin="upload"), 0.9 - (index * 0.01))
            for index in range(6)
        ]

    payload = prepare_query_payload(
        "Use uploaded documents as target requirements and numbered PDFs as internal case studies. Return technical requirements, gaps, three case studies and a proposal outline.",
        vectorstore_stats={
            **ready_stats(),
            "indexed_upload_document_count": 6,
            "indexed_upload_files": [f"target_{index}.pdf" for index in range(6)],
            "scope_chunk_counts": {"upload": 12, "sample": 9, "all": 21},
        },
        retrieval_fn=many_upload_hits,
        retrieval_scope="all",
    )

    upload_docs = [item for item in payload["retrieved_documents"] if item["document_origin"] == "upload"]
    assert len(upload_docs) <= 4


def test_non_rfp_queries_do_not_get_upload_target_fallback():
    def near_threshold(query, k, scope):
        return [(FakeDoc("Weakly related upload note", "target_rfp.pdf", page=0, origin="upload"), 0.47)]

    payload = prepare_query_payload(
        "What is the CEO phone number?",
        vectorstore_stats={**ready_stats(), "scope_chunk_counts": {"upload": 3, "sample": 9, "all": 12}},
        retrieval_fn=near_threshold,
        retrieval_scope="upload",
    )

    assert payload["response_mode"] == "fallback"
    assert payload["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert all(step.get("tool") != "target_context_retrieval" for step in payload["traces"])


def test_previous_sources_does_not_retrieve():
    def forbidden_retrieval(*args):
        raise AssertionError("retrieval must not run")

    history = [{"role": "assistant", "content": "answer", "reasoning": [{"source": "client_profile.pdf", "page": 2, "document_origin": "upload"}]}]
    payload = prepare_query_payload(
        "What sources did you use?",
        chat_history=history,
        vectorstore_stats=ready_stats(),
        retrieval_fn=forbidden_retrieval,
    )

    assert payload["intent"] == "previous_sources"
    assert payload["tool_outputs"]["previous_sources"] == [{"source_file": "client_profile.pdf", "page": 2, "document_origin": "upload"}]
    assert "client_profile.pdf" in payload["answer"]


def test_low_relevance_results_return_insufficient_evidence():
    def unrelated(query, k, scope="all"):
        return [(FakeDoc("Unrelated project text", "unrelated.pdf"), 0.1)]

    for question in ("What is the CEO phone number?", "Describe the aerospace blockchain architecture"):
        payload = prepare_query_payload(question, vectorstore_stats=ready_stats(), retrieval_fn=unrelated)
        assert payload["response_mode"] == "fallback"
        assert "sufficiently relevant evidence" in payload["answer"]
        assert "unrelated.pdf" not in payload["answer"]


def test_generated_answer_is_verified_after_llm_generation():
    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(content="Azure is used. [Source: banking_case_study.pdf, Page 1]")

    result = run_query(
        FakeLLM(),
        "What technology was used?",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
    )
    tools = [step.get("tool") for step in result["payload"]["traces"]]
    assert tools.index("grounding_verifier") > tools.index("final_response")


def test_rfp_analysis_prompt_is_compact_and_within_budget(monkeypatch):
    phrase = "Azure dashboards with HIPAA controls and phased rollout milestones."

    def scoped_retrieval(query, k, scope):
        if scope == "upload":
            return [
                (FakeDoc(f"{phrase} Requirement chunk {index}", f"target_{index}.pdf", page=index, origin="upload"), 0.95 - (index * 0.02))
                for index in range(6)
            ]
        return [(FakeDoc("Sample retrieval evidence", "sample.pdf", page=0, origin="sample"), 0.9)]

    monkeypatch.setattr(
        graph_module,
        "extract_rfp_requirements",
        lambda _text: {
            "requirements": [
                {
                    "id": "REQ-01",
                    "text": phrase,
                    "source_file": "target_0.pdf",
                    "page": 0,
                    "document_origin": "upload",
                }
            ],
            "summary": "Extracted 1 requirement.",
        },
    )

    def fake_cases(requirements, search_fn=None, k=6):
        assert search_fn is not None
        docs = []
        matches = []
        for case_index in range(4):
            source = f"case_{case_index}.pdf"
            matches.append(
                {
                    "source": source,
                    "pages": [0, 1, 2],
                    "matched_requirements": ["REQ-01"],
                    "snippets": [f"Fit reason {case_index}"] * 3,
                }
            )
            for page in range(3):
                docs.append(FakeDoc(f"Case {case_index} page {page} tech stack evidence", source, page=page, origin="sample"))
        return {"matches": matches, "documents": docs}

    monkeypatch.setattr(graph_module, "find_relevant_case_studies", fake_cases)
    monkeypatch.setattr(
        graph_module,
        "compare_projects",
        lambda *args, **kwargs: {
            "rows": [
                {"source": "case_0.pdf", "timeline": "12 weeks", "budget": "$100k", "tech_stack": "Azure", "outcomes": "Dashboard launch"},
                {"source": "case_1.pdf", "timeline": "16 weeks", "budget": "$120k", "tech_stack": "Python", "outcomes": "Workflow automation"},
                {"source": "case_2.pdf", "timeline": "20 weeks", "budget": "$140k", "tech_stack": "Snowflake", "outcomes": "Reporting uplift"},
                {"source": "case_3.pdf", "timeline": "24 weeks", "budget": "$160k", "tech_stack": "Kubernetes", "outcomes": "Platform rebuild"},
            ]
        },
    )
    monkeypatch.setattr(
        graph_module,
        "generate_proposal_outline",
        lambda query, case_studies, requirements=None: {
            "outline": "\n".join(
                [
                    "## Executive Summary",
                    "- Map Azure migration to client requirements.",
                    "## Delivery Plan",
                    "- Run phased rollout.",
                    "## Risks",
                    "- Track HIPAA controls.",
                ]
            )
        },
    )

    payload = prepare_query_payload(
        "Use uploaded documents as target requirements and numbered PDFs as internal case studies. Return technical requirements, gaps, three case studies and a proposal outline.",
        vectorstore_stats={
            **ready_stats(),
            "indexed_upload_document_count": 6,
            "indexed_upload_files": [f"target_{index}.pdf" for index in range(6)],
            "scope_chunk_counts": {"upload": 12, "sample": 12, "all": 24},
        },
        retrieval_fn=scoped_retrieval,
        retrieval_scope="all",
    )

    prompt = payload["prompt"]
    budget = payload["prompt_budget"]

    assert "Document(" not in prompt
    assert "page_content" not in prompt
    assert prompt.count(phrase) <= 2
    assert budget["estimated_input_tokens"] <= MAX_PROMPT_TOKENS
    assert budget["projected_total_tokens"] == budget["estimated_input_tokens"] + RFP_ANALYSIS_MAX_OUTPUT_TOKENS
    assert budget["target_chunks_included"] <= 4
    assert budget["status"] == "within_budget"
    assert any(step.get("tool") == "prompt_budget" for step in payload["traces"])

    upload_lines = [line for line in prompt.splitlines() if "origin=upload" in line]
    assert 1 <= len(upload_lines) <= 4
    assert "-- Uploaded Target Evidence --" in prompt
    assert "-- Sample Case Study Evidence --" in prompt

    case_names = sorted(set(re.findall(r"case_[0-9]\\.pdf", prompt)))
    assert len(case_names) <= 3
    for name in case_names:
        assert prompt.count(name) <= 3
    assert "REQ-01" in prompt
    assert "target_0.pdf, Page 1" in prompt


def test_prompt_budget_drops_lowest_scoring_evidence_first(monkeypatch):
    monkeypatch.setattr(graph_module, "MAX_PROMPT_TOKENS", 250)

    def scoped_retrieval(query, k, scope):
        if scope == "upload":
            return [(FakeDoc("Target evidence", "target.pdf", page=0, origin="upload"), 0.95)]
        return [(FakeDoc("Sample fallback", "sample.pdf", page=0, origin="sample"), 0.9)]

    monkeypatch.setattr(
        graph_module,
        "extract_rfp_requirements",
        lambda _text: {"requirements": [{"id": "REQ-01", "text": "Must support Azure"}], "summary": "Extracted."},
    )
    monkeypatch.setattr(
        graph_module,
        "find_relevant_case_studies",
        lambda requirements, search_fn=None, k=6: {
            "matches": [
                {"source": "alpha.pdf", "pages": [0], "matched_requirements": ["REQ-01"], "snippets": ["strong fit"]},
                {"source": "beta.pdf", "pages": [0], "matched_requirements": ["REQ-01"], "snippets": ["medium fit"]},
                {"source": "gamma.pdf", "pages": [0], "matched_requirements": ["REQ-01"], "snippets": ["lower fit"]},
                {"source": "delta.pdf", "pages": [0], "matched_requirements": ["REQ-01"], "snippets": ["lowest fit"]},
            ],
            "documents": [
                FakeDoc("alpha evidence" * 40, "alpha.pdf", page=0, origin="sample"),
                FakeDoc("beta evidence" * 40, "beta.pdf", page=0, origin="sample"),
                FakeDoc("gamma evidence" * 40, "gamma.pdf", page=0, origin="sample"),
                FakeDoc("delta evidence" * 40, "delta.pdf", page=0, origin="sample"),
            ],
        },
    )
    monkeypatch.setattr(
        graph_module,
        "compare_projects",
        lambda *args, **kwargs: {"rows": [{"source": "alpha.pdf", "timeline": "12 weeks", "budget": "$100k", "tech_stack": "Azure", "outcomes": "Launch"}]},
    )
    monkeypatch.setattr(graph_module, "generate_proposal_outline", lambda *args, **kwargs: {"outline": "## Summary\n- concise"})

    payload = prepare_query_payload(
        "Use uploaded documents as target requirements and numbered PDFs as internal case studies. Return technical requirements, gaps, three case studies and a proposal outline.",
        vectorstore_stats={
            **ready_stats(),
            "indexed_upload_document_count": 1,
            "indexed_upload_files": ["target.pdf"],
            "scope_chunk_counts": {"upload": 2, "sample": 8, "all": 10},
        },
        retrieval_fn=scoped_retrieval,
        retrieval_scope="all",
    )

    assert payload["prompt_budget"]["chunks_dropped"] >= 1
    assert "alpha.pdf" in payload["prompt"]
    assert "delta.pdf" not in payload["prompt"]


def test_complete_documents_remain_available_for_post_generation_verification(monkeypatch):
    captured = {}

    def fake_verify(answer, documents):
        captured["documents"] = documents
        return {"is_grounded": True, "checked_claims": ["ok"], "unsupported_claims": []}

    monkeypatch.setattr(graph_module, "verify_answer_grounding", fake_verify)

    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(content="Grounded answer. [Source: banking_case_study.pdf, Page 1]")

    result = run_query(
        FakeLLM(),
        "What technology was used?",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
    )

    assert result["answer"].startswith("Grounded answer")
    assert captured["documents"]
    assert captured["documents"][0]["content"].startswith("Evidence for:")


def test_unsupported_claims_trigger_one_repair_pass(monkeypatch):
    calls = {"count": 0}

    def fake_verify(answer, documents):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "is_grounded": False,
                "checked_claims": ["Unsupported Kubernetes claim"],
                "unsupported_claims": ["Unsupported Kubernetes claim"],
            }
        return {"is_grounded": True, "checked_claims": [], "unsupported_claims": []}

    monkeypatch.setattr(graph_module, "verify_answer_grounding", fake_verify)

    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(
                content=(
                    "Supported Azure claim [Source: banking_case_study.pdf, Page 1].\n"
                    "Unsupported Kubernetes claim [Source: respective documents]."
                )
            )

    result = run_query(
        FakeLLM(),
        "What technology was used?",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
    )

    tools = [step.get("tool") for step in result["payload"]["traces"]]
    assert calls["count"] == 2
    assert tools.count("answer_repair") == 1
    assert "final_grounding_verifier" in tools
    assert "[Source: respective documents]" not in result["answer"]
    assert "Unsupported Kubernetes claim" not in result["answer"]


def test_vague_citations_are_removed_by_repair(monkeypatch):
    def fake_verify(answer, documents):
        return {"is_grounded": True, "checked_claims": [], "unsupported_claims": []}

    monkeypatch.setattr(graph_module, "verify_answer_grounding", fake_verify)

    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(content="Azure is relevant [Source: respective documents].")

    result = run_query(
        FakeLLM(),
        "What technology was used?",
        vectorstore_stats=ready_stats(),
        retrieval_fn=fake_retrieval,
    )

    assert "[Source: respective documents]" not in result["answer"]
    assert "[Source: None]" not in result["answer"]
    assert "The retrieved evidence does not support this claim." in result["answer"]
