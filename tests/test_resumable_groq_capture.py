import asyncio
import json

import pytest

from evals.groq_quota_safety import MODEL, GroqTPMPacer, QuotaSafetyStop, assert_tpd_reset_confirmed
from evals.resumable_groq_capture import (
    MODE_ORDER,
    _load_prior,
    execute_batch,
    validate_and_merge,
)
from evals.groq_matched_capture import GroqCallAudit


def metadata(tag="same"):
    return {"experiment_hash": tag, "git_commit": "abc", "provider": "groq", "model": MODEL}


def test_tpm_pacer_waits_for_reservations_and_records_actual_usage(tmp_path):
    now = [1000.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    pacer = GroqTPMPacer(tmp_path / "ledger.db", tpm_limit=80, tpd_limit=500,
                         clock=lambda: now[0], sleeper=sleep, estimator=lambda _messages: 30)
    first = pacer.before_call(["private prompt"], max_output_tokens=20)
    second = pacer.before_call(["other prompt"], max_output_tokens=20)
    assert first["reserved_tokens"] == 50
    assert second["pacing_wait_seconds"] > 0
    assert waits
    pacer.finish_call(first["request_id"], actual_tokens=32)
    assert "private prompt" not in (tmp_path / "ledger.db").read_bytes().decode("latin1")


def test_tpm_pacer_fails_closed_for_oversized_call(tmp_path):
    pacer = GroqTPMPacer(tmp_path / "ledger.db", tpm_limit=50, estimator=lambda _messages: 31)
    with pytest.raises(QuotaSafetyStop, match="TPM"):
        pacer.before_call(["large"], max_output_tokens=20)


def test_daily_budget_stops_without_waiting(tmp_path):
    pacer = GroqTPMPacer(tmp_path / "ledger.db", tpm_limit=100, tpd_limit=60,
                         estimator=lambda _messages: 30, sleeper=lambda _: pytest.fail("must not sleep"))
    pacer.before_call(["one"], max_output_tokens=20)
    with pytest.raises(QuotaSafetyStop, match="TPD"):
        pacer.before_call(["two"], max_output_tokens=20)


def test_tpd_reset_confirmation_required():
    with pytest.raises(QuotaSafetyStop, match="operator confirmation"):
        assert_tpd_reset_confirmed(confirmed=False)
    assert_tpd_reset_confirmed(confirmed=True) is None


def test_batch_runs_each_case_in_vector_graph_hybrid_order_and_checkpoints(tmp_path):
    cases = [{"id": "a"}, {"id": "b"}]
    calls = []

    async def execute(case, mode):
        calls.append((case["id"], mode))
        return {"status": "completed", "answer": "captured"}

    batch = asyncio.run(execute_batch(cases, metadata=metadata(), completed={}, execute=execute,
        output=tmp_path, batch_size=2, calls_path=tmp_path / "calls.json"))
    assert batch["status"] == "batch_completed"
    assert calls == [(case["id"], mode) for case in cases for mode in MODE_ORDER]
    assert len(batch["executions"]) == 6
    saved = json.loads(next(tmp_path.glob("batch-*.json")).read_text())
    assert [row["execution_sequence"] for row in saved["executions"]] == list(range(1, 7))


def test_quota_stop_preserves_success_and_does_not_record_unexecuted_as_failures(tmp_path):
    from evals.provider_guard import ProviderAbort

    cases = [{"id": "a"}, {"id": "b"}]
    calls = []

    async def execute(case, mode):
        calls.append((case["id"], mode))
        if case["id"] == "a" and mode == "graph_only":
            raise ProviderAbort({"stage": "grade_web_evidence", "provider": "groq", "model": "model",
                                 "http_status": 429, "rate_limit": {"dimension": "TPD"}})
        return {"status": "completed", "answer": "ok"}

    batch = asyncio.run(execute_batch(cases, metadata=metadata(), completed={}, execute=execute,
        output=tmp_path, batch_size=2, calls_path=tmp_path / "calls.json"))
    assert batch["status"] == "incomplete_due_to_provider_quota"
    assert [(row["case_id"], row["retrieval_mode"]) for row in batch["executions"]] == [("a", "vector_only")]
    assert batch["provider_failure"]["quota_dimension"] == "TPD"
    assert {tuple((row["case_id"], row["retrieval_mode"])) for row in batch["unexecuted"]} == {
        ("a", "graph_only"), ("a", "hybrid"), ("b", "vector_only"),
        ("b", "graph_only"), ("b", "hybrid"),
    }


def test_resume_reuses_only_completed_entries_with_exact_metadata(tmp_path):
    cases = [{"id": "a"}]
    calls = []

    async def execute(case, mode):
        calls.append(mode)
        return {"status": "completed", "answer": "ok"}

    first = asyncio.run(execute_batch(cases, metadata=metadata(), completed={}, execute=execute,
        output=tmp_path, batch_size=1, calls_path=tmp_path / "calls.json"))
    completed = _load_prior(tmp_path, metadata())
    assert len(completed) == 3
    assert not _load_prior(tmp_path, metadata("different"))
    assert calls == list(MODE_ORDER)
    assert first["status"] == "batch_completed"
    second_calls = []

    async def execute_again(case, mode):
        second_calls.append(mode)
        return {"status": "completed", "answer": "ok"}

    asyncio.run(execute_batch(cases, metadata=metadata("different"), completed={}, execute=execute_again,
        output=tmp_path, batch_size=1, calls_path=tmp_path / "calls.json"))
    assert second_calls == list(MODE_ORDER)


def test_merger_rejects_metadata_mismatch_and_incomplete_batches(tmp_path):
    cases = [{"id": "a"}]
    one = {"status": "batch_completed", "experiment_metadata": metadata(), "executions": []}
    two = {"status": "batch_completed", "experiment_metadata": metadata("different"), "executions": []}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(one))
    b.write_text(json.dumps(two))
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_and_merge([a, b], cases)
    two["experiment_metadata"] = metadata()
    two["status"] = "incomplete_due_to_provider_quota"
    b.write_text(json.dumps(two))
    with pytest.raises(ValueError, match="coverage incomplete"):
        validate_and_merge([a, b], cases)


def test_merger_requires_exact_coverage_and_execution_order(tmp_path):
    cases = [{"id": "a"}]
    rows = [{"case_id": "a", "retrieval_mode": mode, "status": "completed",
             "experiment_hash": "same", "provider": "groq", "model": MODEL, "execution_sequence": i}
            for i, mode in enumerate(reversed(MODE_ORDER), 1)]
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps({"status": "batch_completed", "experiment_metadata": metadata(),
                                      "executions": rows}))
    with pytest.raises(ValueError, match="execution order mismatch"):
        validate_and_merge([batch_path], cases)


def test_merger_accepts_complete_same_metadata_case_major_capture(tmp_path):
    cases = [{"id": "a"}, {"id": "b"}]
    rows = []
    sequence = 0
    for case in cases:
        for mode in MODE_ORDER:
            sequence += 1
            rows.append({"case_id": case["id"], "retrieval_mode": mode, "status": "completed",
                         "provider": "groq", "model": MODEL, "experiment_hash": "same",
                         "execution_sequence": sequence})
    path = tmp_path / "batch-0001.json"
    path.write_text(json.dumps({"status": "batch_completed", "experiment_metadata": metadata(),
                                "executions": rows}))
    merged = validate_and_merge([path], cases)
    assert merged["status"] == "valid_complete_capture"
    assert merged["completed_executions"] == 6
    assert merged["judge"] == "not_run:capture_only"


def test_partial_quota_batch_can_resume_and_merge_only_after_full_coverage(tmp_path):
    from evals.provider_guard import ProviderAbort

    cases = [{"id": "a"}]

    async def first_attempt(_case, mode):
        if mode == "graph_only":
            raise ProviderAbort({"stage": "grade_kb_evidence", "http_status": 429,
                                 "rate_limit": {"dimension": "TPD"}})
        return {"status": "completed", "answer": "vector result"}

    directory = tmp_path / "batches"
    first = asyncio.run(execute_batch(cases, metadata=metadata(), completed={}, execute=first_attempt,
        output=directory, batch_size=1, calls_path=tmp_path / "calls.json"))
    assert first["status"] == "incomplete_due_to_provider_quota"
    prior = _load_prior(directory, metadata())

    async def resumed(_case, mode):
        return {"status": "completed", "answer": mode}

    second = asyncio.run(execute_batch(cases, metadata=metadata(), completed=prior, execute=resumed,
        output=directory, batch_size=1, calls_path=tmp_path / "calls.json"))
    assert [row["retrieval_mode"] for row in second["executions"]] == ["graph_only", "hybrid"]
    merged = validate_and_merge(sorted(directory.glob("batch-*.json")), cases)
    assert merged["completed_executions"] == 3
    assert merged["interrupted_batches"][0]["status"] == "incomplete_due_to_provider_quota"


def test_audit_appends_prior_provider_telemetry_across_resumes(tmp_path):
    path = tmp_path / "calls.json"
    prior_event = {"stage": "grade_kb_evidence", "provider": "groq", "model": "model",
                   "status": "completed", "latency_seconds": 1.2}
    path.write_text(json.dumps({"events": [prior_event], "node_outputs": [],
                                "diagnostic_writer_errors": []}))
    audit = GroqCallAudit(path)
    audit.phase = "capture"
    audit.checkpoint(strict=True)
    saved = json.loads(path.read_text())
    assert saved["events"] == [prior_event]

