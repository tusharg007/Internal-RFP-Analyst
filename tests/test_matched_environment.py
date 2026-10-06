"""Evaluation isolation and fail-fast tests; no network, index mutation, or judge."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from evals import matched_environment as matched
from evals.provider_guard import ProviderAbort, ProviderGuard, stop_on_provider_failure
from rfp_analyst.graph.ingestion import IndexedChunk, build_snapshot, digest


class RateLimitError(Exception):
    status_code = 429


class FailingChat(BaseChatModel):
    @property
    def _llm_type(self):
        return "unit-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RateLimitError("THIS_SECRET_MESSAGE_MUST_NOT_BE_EXPORTED")


def test_provider_guard_escapes_production_exception_fallback_and_resets():
    llm = FailingChat()

    def application(_):
        try:
            return llm.invoke("public test input")
        except Exception:
            return "normal degraded response"

    node = RunnableLambda(application)
    with stop_on_provider_failure(), pytest.raises(ProviderAbort) as exc:
        node.invoke({}, config={"metadata": {
            "langgraph_node": "grade_kb_evidence", "ls_provider": "groq",
            "ls_model_name": "public-test-model",
        }})
    assert exc.value.failure == {
        "stage": "grade_kb_evidence", "provider": "failing", "model": "public-test-model",
        "error_type": "RateLimitError", "http_status": 429, "category": "rate_limit",
        "retrieval_quality_failure": False,
    }
    assert "SECRET" not in json.dumps(exc.value.failure)
    assert node.invoke({}) == "normal degraded response"


def test_guard_context_propagates_to_evaluation_worker_thread():
    async def run():
        with stop_on_provider_failure():
            await asyncio.to_thread(
                FailingChat().invoke, "public", config={"metadata": {"langgraph_node": "route_question"}}
            )

    with pytest.raises(ProviderAbort) as exc:
        asyncio.run(run())
    assert exc.value.failure["stage"] == "route_question"


@pytest.mark.parametrize("status", [429, 503])
def test_provider_error_never_scored_as_retrieval_failure(status):
    guard = ProviderGuard()
    error = Exception("private diagnostic")
    error.status_code = status
    with pytest.raises(ProviderAbort) as exc:
        guard.on_llm_error(error, run_id="test")
    assert exc.value.failure["http_status"] == status
    assert exc.value.failure["retrieval_quality_failure"] is False


def snapshot_and_raw():
    inputs = [IndexedChunk(
        chunk_id="public-chunk-1", source_file="eval_target_rfp.pdf", document_origin="upload",
        file_hash="a" * 64, page_index=0, chunk_index=0, text="Public synthetic test fixture",
    )]
    snapshot = build_snapshot(matched.CORPUS_ID, inputs)
    indexed = digest([i.model_dump() for i in inputs])
    raw = {"nodes": [
        {"labels": ["RFPCorpusHead"], "properties": {
            "corpus_id": snapshot.corpus_id, "active_version": snapshot.version,
        }},
        {"labels": ["RFPSnapshot"], "properties": {
            "corpus_id": snapshot.corpus_id, "corpus_version": snapshot.version, "indexed_digest": indexed,
        }},
        *[{"labels": ["RFPDocument"], "properties": d.model_dump()} for d in snapshot.documents],
        *[{"labels": ["RFPChunk"], "properties": c.model_dump()} for c in snapshot.chunks],
    ], "edges": [{
        "type": "IN_DOCUMENT", "properties": {}, "start": snapshot.chunks[0].model_dump(),
        "end": snapshot.documents[0].model_dump(),
    }]}
    return snapshot, raw, indexed


def test_exact_manifest_parity():
    snapshot, raw, indexed = snapshot_and_raw()
    result = matched.compare_manifest(raw, snapshot, indexed)
    assert result["parity"] is True
    assert result["expected_chunk_sha256"] == result["actual_chunk_sha256"]


@pytest.mark.parametrize("field,value", [
    ("chunk_id", "different-id"), ("page_index", 9), ("text_hash", "b" * 64),
    ("document_origin", "sample"), ("file_hash", "c" * 64), ("span_end", 2),
])
def test_equal_counts_do_not_establish_parity(field, value):
    snapshot, raw, indexed = snapshot_and_raw()
    raw["nodes"][-1]["properties"][field] = value
    assert matched.compare_manifest(raw, snapshot, indexed)["parity"] is False


def test_duplicate_missing_edges_and_cross_namespace_are_rejected():
    snapshot, raw, indexed = snapshot_and_raw()
    duplicate = copy.deepcopy(raw)
    duplicate["nodes"].append(duplicate["nodes"][-1])
    assert not matched.compare_manifest(duplicate, snapshot, indexed)["parity"]
    missing = copy.deepcopy(raw)
    missing["edges"] = []
    assert not matched.compare_manifest(missing, snapshot, indexed)["parity"]
    raw["edges"][0]["end"]["corpus_id"] = matched.PROTECTED
    assert "cross_namespace_relationship" in matched.compare_manifest(raw, snapshot, indexed)["failures"]


def test_protected_fingerprint_observes_properties_and_edges_without_exporting_them():
    _, raw, _ = snapshot_and_raw()
    before = matched.protected_fingerprint(raw)
    raw["nodes"][-1]["properties"]["source_file"] = "private-filename.pdf"
    after = matched.protected_fingerprint(raw)
    assert before != after
    assert "private-filename" not in json.dumps(after)


def test_live_namespace_rejected_before_any_connection(tmp_path):
    with pytest.raises(ValueError, match="isolated"):
        matched.run(SimpleNamespace(corpus_id=matched.PROTECTED, output_dir=tmp_path))


def test_capture_preserves_partial_report_and_stops_after_provider_abort(monkeypatch, tmp_path):
    calls = []

    async def evaluate(pipeline, cases, mode, **kwargs):
        calls.append((cases[0]["id"], mode))
        if len(calls) == 2:
            raise ProviderAbort({"stage": "grade_kb_evidence", "provider": "groq",
                                 "model": "test-model", "http_status": 429})
        return {"cases": [{"id": cases[0]["id"], "status": "completed",
                           "pipeline_latency_seconds": 0, "scores": {}}], "failures": []}

    monkeypatch.setattr(matched, "evaluate_mode", evaluate)
    monkeypatch.setattr(matched, "assert_frozen", lambda *a: None)
    monkeypatch.setattr(matched, "benchmark_checks", lambda *a: {})
    monkeypatch.setattr(matched, "aggregate_cases", lambda *a: {})
    pipeline = SimpleNamespace(
        start_generation=lambda: None, generation_metadata={}, corpus_hash="public",
        corpus_id=matched.CORPUS_ID, index=tmp_path, snapshot=None, load_current=lambda: None,
    )
    output = tmp_path / "capture.json"
    result = asyncio.run(matched.capture(pipeline, [{"id": "test", "question": "public",
                                                   "query_class": "relationship"}], {}, output))
    saved = json.loads(output.read_text())
    assert result["status"] == saved["status"] == "stopped_provider_failure"
    assert saved["completed_executions"] == 1
    assert saved["aborted_execution"]["http_status"] == 429
    assert len(calls) == 2  # Third mode must not execute.
    assert not saved["capture_valid"]


@pytest.mark.parametrize("count,parity,failures,valid", [
    (48, True, [], True), (47, True, [], False), (48, False, [], False),
    (48, True, [{"http_status": 429}], False),
])
def test_48_successful_executions_and_parity_required(count, parity, failures, valid):
    reports = {"test": {"cases": [{"status": "completed"} for _ in range(count)]}}
    assert matched.capture_complete(reports, 48, failures, parity) is valid
