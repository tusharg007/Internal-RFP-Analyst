"""Static compatibility probing only; every provider call is mocked."""

import json
from types import SimpleNamespace

import pytest

from evals.static_grader_probe import (
    MODELS, digest, error_metadata, invoke_google, probe_models, safe_continuation, saved_input,
)
from rfp_analyst.agent.prompts import KB_GRADER_PROMPT


def listing():
    return {f"models/{m}": {"name": f"models/{m}", "supported_actions": ["generateContent"]} for m in MODELS}


def test_saved_input_reconstructs_identical_prompt_and_preserves_file(tmp_path):
    evidence = "Source: public.pdf\nExact frozen evidence."
    prompt = KB_GRADER_PROMPT.format(question="original", context=evidence)
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"attempts": [{"original_query": "original", "grader_evidence": evidence,
                       "evidence_hash": digest(evidence), "prompt_hash": digest(prompt),
                       "prompt_matches_captured_evidence": True}]}), encoding="utf-8")
    original = trace.read_bytes()
    assert saved_input(trace)[:2] == (prompt, evidence)
    assert trace.read_bytes() == original


def test_changed_prompt_is_rejected_before_calls(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"attempts": [{"original_query": "changed", "grader_evidence": "evidence",
                       "evidence_hash": digest("evidence"), "prompt_hash": digest("different prompt"),
                       "prompt_matches_captured_evidence": True}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        saved_input(trace)


def test_each_available_model_receives_identical_input_exactly_once():
    calls = []

    def invoke(model, text):
        calls.append((model, text))
        grade = "weak" if model == MODELS[-1] else "good"
        return {"raw_response": {"content": json.dumps({"grade": grade})},
                "parsed_grade": grade, "provider_status": "success"}

    rows = probe_models(listing(), "exact prompt", "exact evidence", invoke, lambda _: None)
    assert calls == [(m, "exact prompt") for m in MODELS]
    assert len({r["prompt_hash"] for r in rows}) == len({r["evidence_hash"] for r in rows}) == 1
    assert all(r["generation_requests"] == 1 for r in rows)
    assert [r["classification"] for r in rows] == ["compatible"] * 3 + ["incompatible-for-this-grader"]


def test_unlisted_and_non_generation_models_are_never_called():
    available = listing()
    del available[f"models/{MODELS[0]}"]
    available[f"models/{MODELS[1]}"]["supported_actions"] = ["countTokens"]
    called = []

    def invoke(model, _):
        called.append(model)
        return {"parsed_grade": "weak", "provider_status": "success"}

    rows = probe_models(available, "prompt", "evidence", invoke, lambda _: None)
    assert called == list(MODELS[2:])
    assert [r["provider_status"] for r in rows[:2]] == ["unavailable", "unavailable"]


def test_strong_is_not_compatible_with_existing_contract():
    rows = probe_models(listing(), "p", "e", lambda *_: {
        "parsed_grade": None, "raw_response": {"content": '{"grade":"strong"}'},
        "provider_status": "invalid_contract"}, lambda _: None)
    assert all(r["classification"] == "unavailable/provider-error" for r in rows)


class QuotaError(Exception):
    code = 429
    status = "RESOURCE_EXHAUSTED"

    def __init__(self, details=None):
        super().__init__("must not be persisted: fake secret")
        self.details = details or {}


def test_unknown_quota_stops_remaining_models_without_retry():
    calls = []

    def invoke(model, _):
        calls.append(model)
        raise QuotaError()

    rows = probe_models(listing(), "p", "e", invoke, lambda _: None)
    assert calls == [MODELS[0]]
    assert rows[0]["http_status"] == 429
    assert all(r["provider_status"] == "not_called_quota_safety" for r in rows[1:])
    assert "fake secret" not in json.dumps(rows)


def test_explicit_model_quota_honors_backoff_then_calls_other_models_once():
    calls, waits = [], []
    details = {"error": {"details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7.5s"},
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel", "quotaMetric": "requests",
             "quotaDimensions": {"model": MODELS[0], "project": "private-project"}}]},
    ]}}

    def invoke(model, _):
        calls.append(model)
        if model == MODELS[0]:
            raise QuotaError(details)
        return {"parsed_grade": "good", "provider_status": "success"}

    rows = probe_models(listing(), "p", "e", invoke, lambda _: None, wait=waits.append)
    assert calls == list(MODELS)
    assert sum(waits) == 8.5
    assert max(waits) <= 5
    assert "private-project" not in json.dumps(rows)


@pytest.mark.parametrize("delay", [None, -1, 61, float("inf")])
def test_unsafe_backoff_is_not_ignored(delay):
    assert safe_continuation({"retry_delay_seconds": delay, "quota_violations": [
        {"quota_id": "PerModel", "model": MODELS[0]}]}, MODELS[0]) is None


def test_global_quota_cannot_be_bypassed_by_switching_model():
    assert safe_continuation({"retry_delay_seconds": 5, "quota_violations": [
        {"quota_id": "PerProject", "model": MODELS[0]}]}, MODELS[0]) is None


def test_wrapped_error_retains_status_not_exception_text():
    outer = RuntimeError("secret text")
    outer.__cause__ = QuotaError()
    assert error_metadata(outer)["http_status"] == 429
    assert "secret text" not in json.dumps(error_metadata(outer))


def test_google_invocation_preserves_contract_and_closes_client(monkeypatch):
    from config import GRADING_TEMPERATURE, LLM_MAX_TOKENS
    from langchain_core.messages import AIMessage
    import langchain_google_genai
    from rfp_analyst.agent.schemas_decisions import EvidenceGrade

    recorded = {}

    class FakeModel:
        def __init__(self, **kwargs):
            recorded["parameters"] = kwargs
            self.client = SimpleNamespace(close=lambda: recorded.update(closed=True))

        def with_structured_output(self, schema, **kwargs):
            assert schema is EvidenceGrade
            recorded["structured"] = kwargs
            return self

        def invoke(self, text):
            recorded["prompt"] = text
            return {"raw": AIMessage('{"grade":"good"}'), "parsed": EvidenceGrade(grade="good"),
                    "parsing_error": None}

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeModel)
    result = invoke_google(MODELS[0], "exact prompt", "test-key")
    assert recorded["parameters"]["temperature"] == GRADING_TEMPERATURE
    assert recorded["parameters"]["max_output_tokens"] == LLM_MAX_TOKENS
    assert recorded["parameters"]["max_retries"] == 0
    assert recorded["structured"] == {"method": "json_mode", "include_raw": True}
    assert recorded["prompt"] == "exact prompt"
    assert recorded["closed"] is True
    assert result["parsed_grade"] == "good"
    assert result["http_status"] is None  # Never invent an observed transport status.
