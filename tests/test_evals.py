from pathlib import Path

from evals.run_evals import run_offline_smoke_eval
from rfp_analyst.evals import format_latency, load_eval_snapshot


def test_eval_latency_formatting_from_seconds():
    assert format_latency(0.23, "seconds") == "230 ms"
    assert format_latency(1.5, "seconds") == "1.50 s"


def test_eval_snapshot_missing_file(tmp_path: Path):
    snapshot = load_eval_snapshot(tmp_path / "results.json")

    assert snapshot["status"] == "missing"
    assert snapshot["message"] == "No evaluation run found"


def test_eval_snapshot_missing_real_kb_message(tmp_path: Path):
    snapshot = load_eval_snapshot(
        tmp_path / "real_kb_results.json",
        missing_message="No real KB evaluation run found",
    )

    assert snapshot["status"] == "missing"
    assert snapshot["message"] == "No real KB evaluation run found"


def test_offline_smoke_eval_is_labeled(tmp_path: Path):
    payload = run_offline_smoke_eval(tmp_path / "offline_smoke_results.json")

    assert payload["evaluation_name"] == "Offline Smoke Evaluation"
    assert payload["evaluation_type"] == "offline_smoke_eval"
    assert "mock" in payload["notes"].lower()
