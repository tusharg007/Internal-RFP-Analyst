"""Generic regressions for transport retries, tool stages and typed comparisons."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rfp_analyst.agent import graph, grader
from rfp_analyst.agent.schemas_decisions import EvidenceGrade
from rfp_analyst.graph.reader import GraphReadFailure
from rfp_analyst.retrieval.decisions import plan_retrieval, matches_project_reference
from rfp_analyst.tools.compare_projects import _extract_field, compare_projects
from tests.test_hybrid_retrieval import (
    environment as environment,
    no_providers as no_providers,
    provider,
    ready,
)
from tests.test_langgraph_agent import FakeRouterLLM


def test_timeout_retry_preserves_supported_plan_and_never_uses_vectors(environment, monkeypatch):
    service = provider(environment, policy="graph_only", strict=True)
    original = service.reader.fetch
    calls = []

    def flaky(*args, **kwargs):
        calls.append(args[3])
        if len(calls) == 1:
            raise GraphReadFailure("timeout")
        return original(*args, **kwargs)

    monkeypatch.setattr(service.reader, "fetch", flaky)
    monkeypatch.setattr(
        graph, "rewrite_query", Mock(side_effect=AssertionError("Semantic rewrite"))
    )
    vector = Mock(side_effect=AssertionError("Strict graph-only vector fallback"))
    q = "What is the total timeline for Healthcare Migration?"
    p = graph.prepare_query_payload(
        q,
        retrieval_provider=service,
        retrieval_fn=vector,
        vectorstore_stats=ready(environment),
        allow_llm_routing=False,
        allow_llm_grading=False,
        allow_web_search=False,
    )
    assert p["retry_count"] == 1 and p["current_query"] == q
    assert len(calls) == 2 and calls[0] == calls[1]
    assert p["retrieval_mode"] == "graph_only" and p["graph_paths"]
    assert not p["graph_fallback_reason"]
    assert service.events[0].fallback_reason == "timeout"
    assert service.events[-1].fallback_reason == ""
    assert any(t.get("retry_kind") == "transport_same_query" for t in p["traces"])
    vector.assert_not_called()


def test_final_head_check_deadline_is_timeout_not_snapshot_mutation(environment, monkeypatch):
    from rfp_analyst.graph import reader as reader_module

    driver = environment[1]
    monkeypatch.setattr(
        reader_module,
        "monotonic",
        lambda: 10.0 if driver.calls and driver.calls[-1][0] == reader_module.CHECK_HEAD else 0.0,
    )
    result = provider(environment, policy="graph_only", strict=True).retrieve(
        "timeline for Healthcare Migration", k=6, scope="sample", vector_supplier=Mock()
    )
    assert result.fallback_reason == "timeout" and not result.documents
    assert driver.head == environment[0].version


@pytest.mark.parametrize("reason", ["disabled", "not_configured", "invalid_configuration"])
def test_transport_reset_does_not_bypass_permanent_unavailability(environment, reason):
    p = provider(environment, policy="graph_only", strict=True)
    p._unavailable_reason = reason
    p.reset_transient_failure()
    assert p._unavailable_reason == reason


def test_exhausted_retry_never_resets_graph_circuit(environment, monkeypatch):
    p = provider(environment, policy="graph_only", strict=True)
    p._unavailable_reason = "timeout"
    monkeypatch.setattr(
        graph, "rewrite_query", lambda state: {"current_query": "unchanged", "retry_count": 2}
    )
    graph.rewrite_query_node(
        {
            "user_query": "timeline for Healthcare Migration",
            "retrieval_provider": p,
            "graph_fallback_reason": "timeout",
            "retry_count": graph.MAX_QUERY_RETRIES,
        }
    )
    assert p._unavailable_reason == "timeout"


def test_named_field_comparison_has_two_sources_and_four_original_witnesses(environment):
    query = "Compare timelines and budget ranges of Healthcare Migration and Insurance Automation projects."
    decision = plan_retrieval(query)
    assert decision.mode == "hybrid"
    assert decision.plan.query_type == "project_fields"
    assert decision.plan.fields == ("timeline", "budget")
    service = provider(environment, policy="graph_only", strict=True)
    vector = Mock(side_effect=AssertionError("Vector fallback"))
    result = service.retrieve(query, k=6, scope="sample", vector_supplier=vector)
    assert len(result.paths) == 2
    assert {d["source"] for d in result.documents} == {"healthcare.pdf", "insurance.pdf"}
    assert {(d["source"], d["page"]) for d in result.documents} >= {
        (source, page) for source in ("healthcare.pdf", "insurance.pdf") for page in (1, 2)
    }
    assert all(len(path["assertion_ids"]) == 2 for path in result.paths)
    vector.assert_not_called()


@pytest.mark.parametrize(
    "reference,name",
    [
        ("the hospital migration project", "Hospital Data Migration to Secure Cloud"),
        ("claims automation projects", "Claims Processing Automation"),
    ],
)
def test_shortened_ordered_project_names(reference, name):
    assert matches_project_reference(reference, name)
    assert not matches_project_reference("unrelated migration", name)


def test_duplicate_shortened_names_remain_ambiguous(environment):
    from rfp_analyst.retrieval.hybrid import _resolve_projects

    subject = SimpleNamespace(
        entity=SimpleNamespace(name="Hospital Data Migration"),
        identity=SimpleNamespace(source_file="hospital.pdf"),
    )
    with pytest.raises(GraphReadFailure, match="ambiguous_project_reference"):
        _resolve_projects(("hospital migration",), [subject, subject])


def test_timeline_section_does_not_end_at_a_phase_line():
    text = (
        "Timeline & Milestones\nPhase 1: Setup (Weeks 1-3)\n"
        "Phase 2: Delivery (Weeks 4-16)\nTotal Duration: 16 weeks\n"
        "Budget Range\nEstimated project cost: $100,000 - $200,000 USD"
    )
    result = _extract_field(text, "Timeline & Milestones")
    assert "Phase 2" in result and "Total Duration: 16 weeks" in result
    assert "$100,000" not in result


def test_pre_tool_grading_is_distinct_from_final_answer_grading():
    llm = FakeRouterLLM(EvidenceGrade(grade="good"))
    state = {
        "user_query": "Find matching projects",
        "grader_llm": llm,
        "retrieved_documents": [{"source": "target.pdf", "content": "Must use Azure."}],
    }
    grader.grade_kb_evidence(state | {"kb_grading_stage": "tool_input"})
    assert "NEXT DETERMINISTIC TOOL STEP" in llm.structured_router.prompt
    grader.grade_kb_evidence(state | {"kb_grading_stage": "answer"})
    assert "sufficient" in llm.structured_router.prompt
    assert "NEXT DETERMINISTIC TOOL STEP" not in llm.structured_router.prompt


def test_inline_requirement_matching_does_not_use_upload_target(environment):
    service = provider(environment, policy="graph_only", strict=True)
    vector = Mock(side_effect=AssertionError("Inline graph matching used vectors"))
    payload = graph.prepare_query_payload(
        "Find case studies for an RFP requiring HIPAA and achieved fraud reduction.",
        retrieval_scope="sample",
        retrieval_provider=service,
        retrieval_fn=vector,
        vectorstore_stats=ready(environment),
        allow_llm_routing=False,
        allow_llm_grading=False,
        allow_web_search=False,
    )
    assert payload["intent"] == "rfp_analysis"
    assert payload["tool_outputs"]["find_relevant_case_studies"]["matches"]
    assert all(d["document_origin"] == "sample" for d in payload["retrieved_documents"])
    assert any(t.get("input_summary") == "User-provided requirements" for t in payload["traces"])
    assert not any(t.get("tool") == "target_context_retrieval" for t in payload["traces"])
    assert any("Reduced fraud" in d["content"] for d in payload["retrieved_documents"])
    vector.assert_not_called()


def test_proposal_qualification_is_not_a_delivered_outcome_filter():
    p = plan_retrieval(
        "Find case studies for uploaded RFP requiring Azure and HIPAA. Distinguish proposals from delivered evidence."
    )
    assert not p.plan.delivered_keyword
    assert (
        plan_retrieval(
            "Which projects delivered fraud reduction for banking?"
        ).plan.delivered_keyword
        == "fraud reduction"
    )


def test_hybrid_unsupported_plan_returns_exact_vector_supplier_evidence(environment):
    service = provider(environment, policy="hybrid", strict=True)
    documents = [{"source": "project.pdf", "content": "Team size: 9", "chunk_id": "original"}]
    vectors = Mock(return_value=documents)
    result = service.retrieve(
        "How many professionals are on the team?", k=6, scope="sample", vector_supplier=vectors
    )
    assert result.documents is documents
    assert result.fallback_reason == "unsupported_plan" and result.effective_mode == "vector_only"
    vectors.assert_called_once()
    assert not environment[1].calls


def test_compare_tool_finishes_before_final_grading_and_retains_all_fields(
    environment, monkeypatch
):
    service = provider(environment, policy="graph_only", strict=True)
    observed = []

    def grade(state):
        observed.append(
            (
                state.get("kb_grading_stage"),
                bool(state.get("tool_outputs", {}).get("compare_projects")),
            )
        )
        return {"kb_grade": "good"}

    monkeypatch.setattr(graph, "grade_kb_evidence", grade)
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content="Healthcare Migration has a total duration of 16 weeks. [Source: healthcare.pdf, Page 2]"
    )
    p = graph.prepare_query_payload(
        "Compare timelines and budget ranges of Healthcare Migration and Insurance Automation projects.",
        retrieval_provider=service,
        vectorstore_stats=ready(environment),
        llm=llm,
        allow_llm_routing=False,
        allow_web_search=False,
    )
    assert observed == [("tool_input", False), ("answer", True)]
    assert p["tool_outputs"]["compare_projects"]["rows"]
    assert p["grounded"] and p["generation_kind"] == "llm_kb"
    assert len(p["graph_paths"]) == 2
    assert all(
        "16 weeks" in row["timeline"] for row in p["tool_outputs"]["compare_projects"]["rows"]
    )


def test_comparison_supplement_retains_only_requested_sources():
    from langchain_core.documents import Document

    def doc(source, page, text):
        return Document(page_content=text, metadata={"source_file": source, "page": page})

    def search(query, k):
        if query.startswith("budget"):
            return [
                (
                    doc(
                        "Hospital_Data_Migration.pdf",
                        2,
                        "Budget Range\nEstimated cost: $100,000 USD",
                    ),
                    0.9,
                ),
                (doc("Other.pdf", 2, "Budget Range\nEstimated cost: $999,999 USD"), 0.9),
            ]
        return [
            (
                doc(
                    "Hospital_Data_Migration.pdf",
                    1,
                    "Timeline & Milestones\nTotal Duration: 12 weeks",
                ),
                0.9,
            ),
            (
                doc(
                    "Claims_Processing_Automation.pdf",
                    1,
                    "Timeline & Milestones\nTotal Duration: 15 weeks",
                ),
                0.9,
            ),
        ]

    result = compare_projects(
        "Compare timelines and budget ranges of hospital migration and claims automation projects.",
        search_fn=search,
    )
    assert any(d.metadata["page"] == 2 for d in result["documents"])
    assert all(d.metadata["source_file"] != "Other.pdf" for d in result["documents"])
    assert "Page 3" in result["comparison_evidence"]


@pytest.fixture
def public_pdf_environment(tmp_path, monkeypatch):
    """Real bundled PDF structure with an injected driver, never a live benchmark."""
    import document_generator
    from evals.run_kb_evals import _ensure_eval_upload
    from rfp_analyst.graph.ingestion import IndexedChunk, build_snapshot
    from rfp_analyst.graph.reader import Neo4jGraphReader
    from rfp_analyst.ingestion.chunking import chunk_loaded_sources
    from rfp_analyst.ingestion.loaders import load_pdf_sources
    from rfp_analyst.retrieval.hybrid import CorpusSnapshot
    from tests.test_graph_store import settings
    from tests.test_hybrid_retrieval import ReadDriver

    samples, uploads = tmp_path / "samples", tmp_path / "uploads"
    monkeypatch.setattr(document_generator, "DATA_DIR", samples)
    document_generator.generate_all_documents()
    _ensure_eval_upload(uploads)
    loaded = [*load_pdf_sources(samples), *load_pdf_sources(uploads, document_origin="upload")]
    inputs = [IndexedChunk.from_document(d) for d in chunk_loaded_sources(loaded)]
    snapshot = build_snapshot("public-pdf-unit-test", inputs)
    captured = CorpusSnapshot(snapshot.inputs)
    driver = ReadDriver(snapshot)
    reader = Neo4jGraphReader(settings(role="reader"), driver=driver)
    yield snapshot, driver, reader, captured, Mock(side_effect=AssertionError("Vector search"))
    reader.close()


def test_public_pdf_comparison_preserves_each_budget_and_timeline(public_pdf_environment):
    service = provider(public_pdf_environment, policy="graph_only", strict=True)
    result = compare_projects(
        "Compare the timelines and budget ranges of the healthcare migration and insurance claims automation projects.",
        search_fn=graph._scoped_search({"retrieval_provider": service}, "sample"),
    )
    assert len(result["rows"]) == 2
    healthcare = next(r for r in result["rows"] if "Healthcare" in r["source"])
    insurance = next(r for r in result["rows"] if "Insurance" in r["source"])
    assert "Total Duration: 18 weeks" in healthcare["timeline"]
    assert "$1,200,000 - $1,500,000 USD" in healthcare["budget"]
    assert "Total Duration: 16 weeks" in insurance["timeline"]
    assert "$650,000 - $800,000 USD" in insurance["budget"]
    assert len(result["comparison_evidence"].splitlines()) == 4
    assert all(e.paths and not e.fallback_reason for e in service.events)
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content=(
            "Healthcare migration proposes 18 weeks. [Source: 02_Healthcare_Data_Migration_to_Azure_Cloud.pdf, Page 2]\n"
            "Healthcare estimated project cost: $1,200,000 - $1,500,000 USD. [Source: 02_Healthcare_Data_Migration_to_Azure_Cloud.pdf, Page 3]\n"
            "Insurance automation specifies 16 weeks. [Source: 04_Insurance_Claims_Processing_Automation.pdf, Page 2]\n"
            "Insurance estimated project cost: $650,000 - $800,000 USD. [Source: 04_Insurance_Claims_Processing_Automation.pdf, Page 3]"
        )
    )
    p = graph.prepare_query_payload(
        result["query"],
        retrieval_provider=service,
        retrieval_fn=public_pdf_environment[4],
        vectorstore_stats=ready(public_pdf_environment),
        llm=llm,
        allow_llm_routing=False,
        allow_llm_grading=False,
        allow_web_search=False,
    )
    assert p["generation_kind"] == "llm_kb" and p["grounded"]
    assert "$1,200,000" in p["answer"] and "$650,000" in p["answer"]
    assert p["answer"].count("[Source:") == 4


def test_exact_public_timeline_has_original_chunk_provenance(public_pdf_environment):
    service = provider(public_pdf_environment, policy="graph_only", strict=True)
    result = service.retrieve(
        "What is the total timeline for Banking Sector Digital Audit 2024?",
        k=6,
        scope="sample",
        vector_supplier=public_pdf_environment[4],
    )
    assert result.effective_mode == "graph_only" and not result.fallback_reason
    assert len(result.paths) == 1
    assert any("Total Duration: 16 weeks" in d["content"] for d in result.documents)
    original = {i.chunk_id: i.text for i in public_pdf_environment[3].inputs}
    assert all(original[d["chunk_id"]] == d["content"] for d in result.documents)
    public_pdf_environment[4].assert_not_called()


@pytest.mark.parametrize("mode", ["vector_only", "graph_only"])
def test_good_tool_input_cannot_bypass_weak_final_answer_grade(environment, monkeypatch, mode):
    llm = Mock(side_effect=AssertionError("Generation with insufficient final evidence"))
    llm.invoke.side_effect = AssertionError("Generation with insufficient final evidence")
    monkeypatch.setattr(
        graph,
        "grade_kb_evidence",
        lambda state: {
            "kb_grade": "good" if state.get("kb_grading_stage") == "tool_input" else "weak"
        },
    )
    p = graph.prepare_query_payload(
        "Compare timelines and budget ranges of Healthcare Migration and Insurance Automation projects.",
        retrieval_provider=provider(environment, policy=mode, strict=True),
        retrieval_fn=environment[4],
        vectorstore_stats=ready(environment),
        llm=llm,
        allow_llm_routing=False,
        allow_web_search=False,
    )
    assert p["tool_outputs"]["compare_projects"]["rows"]
    assert p["generation_kind"] == "insufficient"
    llm.invoke.assert_not_called()


def test_uploaded_pdf_tool_chain_keeps_proposal_modality_and_provenance(public_pdf_environment):
    service = provider(public_pdf_environment, policy="graph_only", strict=True)
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content=(
            "Healthcare Data Migration to Azure Cloud is a proposal, not delivered evidence. "
            "[Source: 02_Healthcare_Data_Migration_to_Azure_Cloud.pdf, Page 1]"
        )
    )
    p = graph.prepare_query_payload(
        "Find case studies for the uploaded RFP requiring Azure migration and HIPAA. "
        "Distinguish proposals from delivered evidence.",
        retrieval_provider=service,
        retrieval_fn=public_pdf_environment[4],
        vectorstore_stats=ready(public_pdf_environment),
        llm=llm,
        allow_llm_routing=False,
        allow_llm_grading=False,
        allow_web_search=False,
    )
    assert p["generation_kind"] == "llm_kb" and p["grounded"]
    assert {d["document_origin"] for d in p["generation_evidence"]} == {"sample", "upload"}
    assert p["tool_outputs"]["extract_rfp_requirements"]["requirements"]
    assert any(
        "Healthcare" in r["source"]
        for r in p["tool_outputs"]["find_relevant_case_studies"]["matches"]
    )
    assert "proposal" in p["answer"].lower() and "not delivered" in p["answer"].lower()
    assert service.events[0].paths and not service.events[0].fallback_reason
    assert any(
        e.paths and e.decision.plan.query_type == "project_constraints" for e in service.events
    )
    # Non-domain target prose (e.g. the document heading) may safely decline a
    # secondary strict lookup. Do not fabricate a relationship for that prose.
    assert {e.fallback_reason for e in service.events} <= {"", "unsupported_plan"}
    public_pdf_environment[4].assert_not_called()


def test_derived_requirement_fit_is_not_a_named_project_field_comparison():
    plan = plan_retrieval(
        "Compare fit for: Find case studies for an RFP requiring HIPAA and achieved fraud reduction."
    ).plan
    assert plan.query_type == "project_constraints" and not plan.project_refs


def test_inline_clinical_pdf_matching_keeps_achieved_original_outcomes(public_pdf_environment):
    service = provider(public_pdf_environment, policy="graph_only", strict=True)
    source = "07_Pharma_Clinical_Trial_Data_Platform.pdf"
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content=(
            f"Clinical data reconciliation time was reduced by 75%. [Source: {source}, Page 3]\n"
            f"85% of routine regulatory submissions were automated. [Source: {source}, Page 3]\n"
            f"FDA 21 CFR Part 11 compliance certification was achieved. [Source: {source}, Page 3]"
        )
    )
    p = graph.prepare_query_payload(
        "Find case studies for an RFP requiring FDA 21 CFR Part 11, automated "
        "regulatory submissions, and achieved clinical data reconciliation improvements.",
        retrieval_scope="sample",
        retrieval_provider=service,
        retrieval_fn=public_pdf_environment[4],
        vectorstore_stats=ready(public_pdf_environment),
        llm=llm,
        allow_llm_routing=False,
        allow_llm_grading=False,
        allow_web_search=False,
    )
    assert p["generation_kind"] == "llm_kb" and p["grounded"]
    assert all(d["document_origin"] == "sample" for d in p["generation_evidence"])
    assert any(d["source"] == source and d["page"] == 2 for d in p["generation_evidence"])
    assert "75%" in p["answer"] and "85%" in p["answer"]
    assert any(
        r["source"] == source for r in p["tool_outputs"]["find_relevant_case_studies"]["matches"]
    )
    assert all(not e.fallback_reason for e in service.events)
    public_pdf_environment[4].assert_not_called()
