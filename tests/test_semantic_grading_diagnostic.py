"""Boundary observation/concept witnesses only; no external calls or corpus reads."""

from types import SimpleNamespace

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from evals.diagnose_semantic_grading import (
    SOURCE, GradingObserver, ObservedPipeline, concept_check, diagnostic_summary, sha,
)
from evals.ragas_pipeline import FrozenPipeline
from rfp_analyst.agent.grader import _format_kb_evidence
from rfp_analyst.agent.prompts import KB_GRADER_PROMPT
from rfp_analyst.agent.schemas_decisions import EvidenceGrade


def document(text):
    return {"source": SOURCE, "page": 1, "chunk_id": "public-test-chunk", "content": text,
            "score": 0.62, "raw_score": 0.62, "document_origin": "sample"}


FULL = ("Evaluate data quality, lineage, and governance. "
        "Benchmark digital maturity against industry peers. "
        "Deliver a prioritized transformation roadmap with ROI projections.")


def test_verbatim_source_equivalents_cover_five_reference_concepts():
    result = concept_check([document(FULL)])
    assert result["sufficiency"] == "sufficient"
    assert result["missing_expected_concepts"] == []
    assert all(result["witnesses"].values())


def test_executive_summary_does_not_invent_priority_roi_or_peers():
    result = concept_check([document("Digital transformation maturity, data governance, and a modernization roadmap.")])
    assert result["missing_expected_concepts"] == [
        "peer benchmarking", "prioritized modernization roadmap", "ROI projections",
    ]


def test_wrong_project_cannot_supply_banking_concepts():
    row = document(FULL)
    row["source"] = "other-public-case.pdf"
    assert len(concept_check([row])["missing_expected_concepts"]) == 5


def test_before_grading_capture_exact_input_prompt_raw_output_and_parsed_grade(tmp_path):
    observer = GradingObserver(tmp_path)
    state = {"user_query": "original", "current_query": "rewritten", "retry_count": 1,
             "retrieved_documents": [document(FULL)]}
    observer.on_chain_start(None, state, run_id="node", metadata={"langgraph_node": "grade_kb_evidence"})
    context = _format_kb_evidence(state["retrieved_documents"])
    prompt = KB_GRADER_PROMPT.format(question="original", context=context)
    observer.on_chat_model_start({"id": ["ChatGoogleGenerativeAI"]}, [[HumanMessage(prompt)]],
                                run_id="model", metadata={"langgraph_node": "grade_kb_evidence",
                                "ls_provider": "google_genai", "ls_model_name": "test-model"})
    response = SimpleNamespace(generations=[[SimpleNamespace(
        text='{"grade":"weak"}', message=AIMessage('{"grade":"weak"}'), generation_info=None,
    )]])
    observer.on_llm_end(response, run_id="model")
    observer.on_chain_end(EvidenceGrade(grade="weak"), run_id="node")
    observer.on_chain_start(None, {}, run_id="update", metadata={"langgraph_node": "grade_kb_evidence"})
    observer.on_chain_end({"kb_grade": "weak"}, run_id="update")
    entry = observer.attempts[0]
    assert entry["grader_evidence"] == context
    assert entry["evidence_hash"] == sha(context)
    assert entry["prompt_hash"] == sha(prompt)
    assert entry["prompt_matches_captured_evidence"] is True
    assert entry["raw_grade"] == entry["parsed_grade"] == entry["application_grade"] == "weak"
    assert entry["rationale"] is None  # Never invent a reason or add it to the schema.
    assert entry["retrieved_chunks"][0]["score"] == 0.62
    assert entry["retrieved_chunks"][0]["page_number"] == 2
    assert diagnostic_summary(observer)["attempts"][0]["classification"] == "PROVIDER/GRADER COMPATIBILITY"


def test_missing_concepts_are_retrieval_evidence_failure(tmp_path):
    observer = GradingObserver(tmp_path)
    state = {"user_query": "query", "current_query": "query", "retrieved_documents": [document("Data governance.")]}
    observer.on_chain_start(None, state, run_id="node", metadata={"langgraph_node": "grade_kb_evidence"})
    observer.on_chain_end({"kb_grade": "weak"}, run_id="node")
    assert diagnostic_summary(observer)["attempts"][0]["classification"] == "RETRIEVAL EVIDENCE FAILURE"


def test_rewrite_evidence_comparison_checks_exact_content_not_only_ids(tmp_path):
    observer = GradingObserver(tmp_path)
    for retry, content in enumerate([FULL, "Data governance."]):
        state = {"user_query": "query", "current_query": f"query-{retry}", "retry_count": retry,
                 "retrieved_documents": [document(content)]}
        observer.on_chain_start(None, state, run_id=f"node-{retry}", metadata={"langgraph_node": "grade_kb_evidence"})
    result = diagnostic_summary(observer)
    assert result["exact_grader_evidence_identical"] is False
    assert result["chunk_sets_identical"] is True
    assert result["rewrite_changed_evidence"] is True


def test_duplicate_sequence_callback_does_not_create_fake_grading_attempt(tmp_path):
    observer = GradingObserver(tmp_path)
    state = {"user_query": "query", "current_query": "query", "retrieved_documents": [document(FULL)]}
    observer.on_chain_start(None, state, run_id="sequence", metadata={"langgraph_node": "grade_kb_evidence"})
    observer.on_chain_start(None, state, run_id="callable", metadata={"langgraph_node": "grade_kb_evidence"})
    assert len(observer.attempts) == 1


def test_vector_observer_returns_original_results_unmodified(monkeypatch, tmp_path):
    results = [(SimpleNamespace(metadata={"source_file": SOURCE, "page": 1, "chunk_id": "public-test"},
                                page_content=FULL), 0.49)]
    monkeypatch.setattr(FrozenPipeline, "vector_search", lambda *a: results)
    pipeline = object.__new__(ObservedPipeline)
    pipeline.observer = GradingObserver(tmp_path)
    assert pipeline.vector_search("query", 6, "sample") is results
    assert pipeline.observer.retrievals[0]["chunks"][0]["score"] == 0.49


def test_transient_windows_file_lock_is_retried(monkeypatch, tmp_path):
    from evals import diagnose_semantic_grading as diagnostic

    original = diagnostic.write_report
    calls = []

    def locked_once(path, report):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError("simulated sharing violation")
        original(path, report)

    monkeypatch.setattr(diagnostic, "write_report", locked_once)
    monkeypatch.setattr(diagnostic.time, "sleep", lambda _: None)
    observer = GradingObserver(tmp_path)
    observer.checkpoint(strict=True)
    assert len(calls) == 2
    assert observer.writer_errors == []


def test_recorder_io_failure_cannot_change_application_grade(monkeypatch, tmp_path):
    from evals import diagnose_semantic_grading as diagnostic

    def locked(*args):
        raise PermissionError("simulated sharing violation")

    monkeypatch.setattr(diagnostic, "write_report", locked)
    monkeypatch.setattr(diagnostic.time, "sleep", lambda _: None)
    observer = GradingObserver(tmp_path)
    state = {"user_query": "query", "current_query": "query", "retrieved_documents": [document(FULL)]}
    observer.on_chain_start(None, state, run_id="node", metadata={"langgraph_node": "grade_kb_evidence"})
    observer.on_chain_end({"kb_grade": "weak"}, run_id="node")
    assert observer.attempts[0]["application_grade"] == "weak"
    assert observer.writer_errors
    with pytest.raises(PermissionError):
        observer.checkpoint(strict=True)  # Final persistence must not report false success.


def test_concurrent_checkpoints_are_serialized(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    from evals import diagnose_semantic_grading as diagnostic

    active = 0
    maximum = 0
    count_lock = threading.Lock()

    def writer(*args):
        nonlocal active, maximum
        with count_lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.005)
        with count_lock:
            active -= 1

    monkeypatch.setattr(diagnostic, "write_report", writer)
    observer = GradingObserver(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: observer.checkpoint(strict=True), range(12)))
    assert maximum == 1
