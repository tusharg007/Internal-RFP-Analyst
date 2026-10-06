"""Production audit regressions. No external services or model calls."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rfp_analyst.agent import graph, grader, router
from rfp_analyst.graph.extraction import ChunkExtraction, ExtractedFact, validate_extraction
from rfp_analyst.graph.ingestion import ExtractionRejected, build_snapshot
from rfp_analyst.tools import web_search
from rfp_analyst.tools.source_verifier import verify_answer_grounding
from tests.test_graph_ingestion import project


@pytest.mark.parametrize("claim", [
    "Banking Digital Audit uses blockchain.",
    "Azure",
    "# Banking Digital Audit",
])
def test_invalid_page_is_rejected_even_in_short_claims_and_headings(claim):
    result = verify_answer_grounding(
        claim + " [Source: Banking_Digital_Audit.pdf, Page 999]",
        [{"source": "Banking_Digital_Audit.pdf", "page": 0, "content": "Banking digital audit uses Python."}],
    )
    assert not result["is_grounded"]
    assert result["invalid_citations"]


def test_mixed_valid_and_invalid_citations_cannot_mask_bad_identity():
    result = verify_answer_grounding(
        "This project uses Microsoft Azure. [Source: valid.pdf, Page 1] [Source: made_up.pdf, Page 1]",
        [{"source": "valid.pdf", "page": 0, "content": "This project uses Microsoft Azure."}],
    )
    assert not result["is_grounded"]


@pytest.mark.parametrize("web", [False, True])
@pytest.mark.parametrize("factory_error", [False, True])
def test_grader_errors_fail_weak_without_logging_provider_payload(monkeypatch, caplog, web, factory_error):
    error = RuntimeError("SENSITIVE credential and document contents")
    llm = Mock()
    llm.with_structured_output.side_effect = error
    if factory_error:
        monkeypatch.setattr(grader, "_create_grader_llm", Mock(side_effect=error))
    state = {"retrieved_documents": [{"content": "nonempty"}], "web_results": "nonempty"}
    if not factory_error:
        state["grader_llm"] = llm
    result = grader.grade_web_evidence(state) if web else grader.grade_kb_evidence(state)
    assert result == {"web_grade" if web else "kb_grade": "weak"}
    assert "SENSITIVE" not in caplog.text


@pytest.mark.parametrize("quote", [
    "- Cloud: We do not use Azure.",
    "- Cloud: Azure is excluded.",
    "- Cloud: If approved, use Azure.",
    "- Cloud: We might use Azure.",
    "- Cloud: AWS instead of Azure.",
])
def test_relationship_span_rejects_negative_or_conditional_mentions(quote):
    fact = ExtractedFact(
        predicate="uses_technology", object_name="Microsoft Azure",
        quote=quote, start=0, end=len(quote), confidence=1.0,
    )
    with pytest.raises(ValueError, match="requires review"):
        validate_extraction(ChunkExtraction(facts=(fact,)), quote)
    with pytest.raises(ExtractionRejected):
        build_snapshot("audit-test", project(tech=quote))


@pytest.mark.parametrize(("question", "expected"), [
    ("What security controls does the healthcare proposal describe?", "search"),
    ("Compare proposals for banking and healthcare", "compare"),
    ("Find relevant case studies matching the uploaded RFP", "rfp_analysis"),
    ("Draft a proposal for Azure migration", "proposal"),
    ("Please write an RFP response", "proposal"),
])
def test_action_oriented_intent_refinement(question, expected):
    assert router._refine_kb_intent(question) == expected


def test_actual_vector_prompt_budget_and_verifier_use_only_submitted_chunks(monkeypatch):
    monkeypatch.setattr(graph, "MAX_PROMPT_TOKENS", 500)
    calls, verified = [], []

    class LLM:
        def invoke(self, messages):
            calls.append(messages[0].content)
            return SimpleNamespace(content="Answer supported by Azure.")

    def verify(payload, answer):
        verified.extend(payload["retrieved_documents"])
        return answer, {"is_grounded": True}

    monkeypatch.setattr(graph, "_verify_generated_answer", verify)
    docs = [
        {"source": "one.pdf", "page": 0, "content": "Azure " * 90, "chunk_id": "one"},
        {"source": "two.pdf", "page": 0, "content": "UNUSED " * 1000, "chunk_id": "two"},
    ]
    result = graph.generate_from_kb({
        "user_query": "What technology was used?", "llm": LLM(),
        "retrieved_documents": docs, "retrieval_context": graph._format_sources(docs),
        "specialized_notes": "DERIVED " * 1000, "allow_llm_grading": False,
    })
    assert len(calls) == 1
    assert graph._estimate_tokens(calls[0]) <= 500
    assert result["prompt"] == calls[0]
    assert result["generation_contexts"] == [graph._format_sources(docs[:1])]
    assert verified == docs[:1]
    assert result["prompt_budget"]["derived_notes_dropped"]
    assert "UNUSED" not in calls[0] and "DERIVED" not in calls[0]


def test_actual_web_prompt_budget_captures_and_grades_clipped_context(monkeypatch):
    monkeypatch.setattr(graph, "MAX_PROMPT_TOKENS", 500)
    calls, graded = [], []
    monkeypatch.setattr(graph, "grade_web_evidence", lambda s: graded.append(s["web_results"]) or {"web_grade": "good"})
    llm = Mock()
    llm.invoke.side_effect = lambda messages: calls.append(messages[0].content) or SimpleNamespace(content="Web response")
    result = graph.generate_from_web({
        "llm": llm, "user_query": "External question?", "web_results": "WEB " * 2000,
    })
    assert graph._estimate_tokens(calls[0]) <= 500
    assert result["generation_contexts"] == graded
    assert len(graded[0]) < 8000


def test_oversized_question_does_not_invoke_generation(monkeypatch):
    monkeypatch.setattr(graph, "MAX_PROMPT_TOKENS", 10)
    llm = Mock()
    result = graph.generate_from_web({"llm": llm, "user_query": "query " * 100, "web_results": "evidence"})
    llm.invoke.assert_not_called()
    assert result["generation_kind"] == "insufficient"


def test_tavily_errors_do_not_emit_secrets(monkeypatch, caplog):
    import langchain_tavily

    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(langchain_tavily, "TavilySearch", Mock(side_effect=RuntimeError("SENSITIVE test key")))
    assert web_search.search_web({"user_query": "external"})["web_results"] == ""
    assert "SENSITIVE" not in caplog.text


def test_overlapping_judges_keep_redacted_logging_until_last_close():
    from evals.ragas_judge import (
        JudgeSettings, RagasJudge, _acquire_redacted_logging,
    )

    logger = logging.getLogger("instructor")
    original = logger.level
    settings = JudgeSettings("groq", "test", "synthetic", embedding_model="local")
    first, second = RagasJudge(settings), RagasJudge(settings)
    try:
        for judge in (first, second):
            _acquire_redacted_logging()
            judge._logging_acquired = True
        asyncio.run(first.close())
        assert logger.level == logging.CRITICAL
        asyncio.run(first.close())  # Idempotent close cannot decrement twice.
        assert logger.level == logging.CRITICAL
        asyncio.run(second.close())
        assert logger.level == original
    finally:
        asyncio.run(first.close())
        asyncio.run(second.close())


def test_retrieval_only_evaluation_never_calls_live_controls(monkeypatch):
    from evals.run_kb_evals import _run_case
    from rfp_analyst.agent import query_rewriter
    from langchain_core.documents import Document

    forbidden = Mock(side_effect=AssertionError("Live provider invoked"))
    monkeypatch.setattr(router, "_create_router_llm", forbidden)
    monkeypatch.setattr(grader, "_create_grader_llm", forbidden)
    monkeypatch.setattr(query_rewriter, "_create_rewriter_llm", forbidden)
    result = _run_case(
        {"question": "What technologies support healthcare analytics?", "expected_sources": ["case.pdf"]},
        vectorstore_stats={"status": "ready", "total_chunks": 1, "total_documents": 1},
        retrieval_fn=lambda *args: [(Document(
            page_content="Healthcare analytics uses Microsoft Azure and Python.",
            metadata={"source_file": "case.pdf", "page": 0, "document_origin": "sample"},
        ), 0.9)],
    )
    assert result["passed"]
    forbidden.assert_not_called()


def test_owned_graph_driver_is_closed_on_currentness_failure():
    from rfp_analyst.graph.reader import CHECK_HEAD, GraphReadFailure, Neo4jGraphReader
    from tests.test_graph_store import FakeDriver, settings

    driver = FakeDriver()
    driver.fail_statement = CHECK_HEAD
    reader = Neo4jGraphReader(settings(), driver=driver, owns_driver=True)
    with pytest.raises(GraphReadFailure, match="unavailable"):
        reader.current_version("audit-test")
    assert driver.close_count == 1
    reader.close()
    assert driver.close_count == 1


def test_oversized_grading_prompt_does_not_call_provider(monkeypatch):
    monkeypatch.setattr(grader, "MAX_PROMPT_TOKENS", 10)
    factory = Mock(side_effect=AssertionError("Provider must not be called"))
    monkeypatch.setattr(grader, "_create_grader_llm", factory)
    assert grader.grade_web_evidence({"web_results": "long evidence " * 100}) == {"web_grade": "weak"}
    factory.assert_not_called()


def test_real_kb_answer_metrics_use_one_execution(monkeypatch):
    from evals import run_kb_evals

    query = Mock(return_value={
        "answer": "Azure is used.",
        "payload": {
            "response_mode": "llm",
            "retrieved_documents": [{"source": "actual.pdf", "document_origin": "sample"}],
            "traces": [{"tool": "search_knowledge_base"}],
        },
    })
    monkeypatch.setattr(run_kb_evals, "run_query", query)
    monkeypatch.setattr(run_kb_evals, "prepare_query_payload", Mock(side_effect=AssertionError("Second run")))
    result = run_kb_evals._run_case(
        {"question": "Which technology is used?", "expected_sources": ["actual.pdf"]},
        vectorstore_stats={}, retrieval_fn=Mock(), llm=Mock(),
    )
    assert result["passed"] and result["mode"] == "llm_answer"
    assert result["retrieved_sources"] == ["actual.pdf"]
    query.assert_called_once()


def test_failed_final_grounding_withholds_answer_after_one_repair(monkeypatch):
    counter = []

    def reject(answer, docs):
        counter.append(answer)
        return {"is_grounded": False, "checked_claims": [answer], "unsupported_claims": [answer]}

    monkeypatch.setattr(graph, "verify_answer_grounding", reject)
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="Unsupported assertion with an invented citation.")
    doc = {"source": "case.pdf", "page": 0, "content": "Microsoft Azure is used.", "chunk_id": "case"}
    result = graph.generate_from_kb({
        "llm": llm, "user_query": "What technologies were used?",
        "retrieved_documents": [doc], "retrieval_context": graph._format_sources([doc]),
    })
    assert len(counter) == 2
    assert result["answer"] == graph.INSUFFICIENT_EVIDENCE_MESSAGE
    assert result["response_mode"] == "fallback"
    assert result["source_used"] == "insufficient_evidence"
    assert result["generation_kind"] == "insufficient"
    assert result["generation_contexts"]  # Preserve the rejected attempt for audit.
    assert [t["tool"] for t in result["traces"]].count("answer_repair") == 1
    assert result["traces"][-1]["tool"] == "grounding_gate"
