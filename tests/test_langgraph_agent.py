from types import SimpleNamespace

import importlib
import re

from config import MAX_PROMPT_TOKENS, RFP_ANALYSIS_MAX_OUTPUT_TOKENS

from rfp_analyst.agent.graph import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    MISSING_UPLOAD_CONTEXT_MESSAGE,
    NO_RELEVANT_UPLOAD_TARGET_MESSAGE,
    compile_query_graph,
    prepare_query_payload,
    run_query,
)

graph_module = importlib.import_module("rfp_analyst.agent.graph")


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

