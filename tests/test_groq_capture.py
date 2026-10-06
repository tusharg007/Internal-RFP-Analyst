"""Groq evaluation pinning, audit and smoke gates; no external calls."""

import json
import os
from types import SimpleNamespace

import pytest

from evals.groq_matched_capture import MODEL, GroqCallAudit, groq_environment, smoke_checks
from evals.provider_guard import ProviderAbort


def test_groq_pinning_restores_google_and_corpus_settings(monkeypatch):
    import config

    monkeypatch.setattr(config, "_get_streamlit_secret", lambda _: "")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-placeholder")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-placeholder")
    monkeypatch.setenv("RFP_GRAPH_CORPUS_ID", "internal-rfp")
    gemini_model = config.GEMINI_MODEL
    with groq_environment():
        assert config.get_api_keys() == ("test-groq-placeholder", "")
        assert config.get_retrieval_settings()["corpus_id"] == "rfp-eval-frozen-641e9d4dad22"
    assert os.environ["GOOGLE_API_KEY"] == "test-google-placeholder"
    assert os.environ["RFP_GRAPH_CORPUS_ID"] == "internal-rfp"
    assert config.GEMINI_MODEL == gemini_model


def test_groq_pinning_restores_absent_settings(monkeypatch):
    import config

    monkeypatch.setattr(config, "_get_streamlit_secret", lambda _: "")
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("RFP_GRAPH_CORPUS_ID", raising=False)
    with groq_environment():
        pass
    assert "GOOGLE_API_KEY" not in os.environ
    assert "RFP_GRAPH_CORPUS_ID" not in os.environ


def test_google_secret_precedence_refuses_evaluation_pin(monkeypatch):
    import config

    monkeypatch.setattr(config, "_get_streamlit_secret", lambda name: "hidden" if name == "GOOGLE_API_KEY" else "")
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    monkeypatch.setenv("GOOGLE_API_KEY", "original-placeholder")
    with pytest.raises(ValueError, match="Groq-only"):
        with groq_environment():
            pytest.fail("Must not permit a selectable Gemini fallback")
    assert os.environ["GOOGLE_API_KEY"] == "original-placeholder"


def test_missing_groq_key_refuses_pin(monkeypatch):
    import config

    monkeypatch.setattr(config, "get_api_keys", lambda: ("", ""))
    with pytest.raises(ValueError, match="Groq-only"):
        with groq_environment():
            pass


def test_audit_records_actual_grade_provider_and_model_without_prompt(tmp_path):
    audit = GroqCallAudit(tmp_path / "calls.json")
    audit.on_chat_model_start({"id": ["ChatGroq"]}, ["SECRET_PROMPT"], run_id="call",
                             metadata={"ls_provider": "groq", "ls_model_name": MODEL,
                                       "langgraph_node": "grade_kb_evidence"})
    response = SimpleNamespace(generations=[[SimpleNamespace(text='{"grade":"good"}',
                message=SimpleNamespace(response_metadata={"model_name": MODEL}))]])
    audit.on_llm_end(response, run_id="call")
    report = json.loads(audit.path.read_text())
    assert report["events"][0]["status"] == "completed"
    assert report["events"][0]["raw_grade"] == "good"
    assert report["events"][0]["response_model"] == MODEL
    assert report["gemini_calls"] == report["judge_calls"] == 0
    assert report["groq_calls"] == 1
    assert "SECRET_PROMPT" not in audit.path.read_text()


@pytest.mark.parametrize("classname,model", [("ChatGoogleGenerativeAI", "gemini-3.8-flash"),
                                             ("ChatGroq", "wrong-model")])
def test_audit_blocks_non_pinned_call_before_transport(tmp_path, classname, model):
    audit = GroqCallAudit(tmp_path / "calls.json")
    with pytest.raises(ProviderAbort) as exc:
        audit.on_chat_model_start({"id": [classname]}, [], run_id="test",
                                 metadata={"ls_model_name": model, "langgraph_node": "route_question"})
    assert exc.value.failure["category"] == "unexpected_provider_or_model"
    assert audit.events[0]["status"] == "blocked_before_transport"


def test_429_preserves_failed_stage_without_exception_body(tmp_path):
    audit = GroqCallAudit(tmp_path / "calls.json")
    audit.on_chat_model_start({"id": ["ChatGroq"]}, [], run_id="call",
                             metadata={"ls_model_name": MODEL, "langgraph_node": "generate_from_kb"})
    error = Exception("sensitive provider body")
    error.status_code = 429
    audit.on_llm_error(error, run_id="call")
    event = json.loads(audit.path.read_text())["events"][0]
    assert event["status"] == "error"
    assert event["http_status"] == 429
    assert event["stage"] == "generate_from_kb"
    assert not audit.pending
    assert "sensitive" not in audit.path.read_text()


def success():
    payload = {"kb_grade": "good", "generation_kind": "llm_kb", "answer": "Grounded answer.",
               "grounded": True, "retrieval_mode": "vector_only",
               "traces": [{"tool": "grounding_verifier"}, {"tool": "final_grounding_verifier"}]}
    audit = SimpleNamespace(events=[{"phase": "smoke", "stage": stage, "status": "completed",
                                   "class": "ChatGroq", "model": MODEL}
                                  for stage in ("route_question", "grade_kb_evidence", "generate_from_kb")])
    return payload, audit


def test_success_requires_all_end_to_end_smoke_boundaries():
    assert all(smoke_checks(*success()).values())


@pytest.mark.parametrize("field,value", [("kb_grade", "weak"), ("grounded", False),
    ("generation_kind", "insufficient"), ("retrieval_mode", "hybrid"), ("answer", "")])
def test_failed_smoke_cannot_open_full_capture_gate(field, value):
    payload, audit = success()
    payload[field] = value
    assert not all(smoke_checks(payload, audit).values())


def test_missing_route_or_generation_call_cannot_pass_smoke():
    payload, audit = success()
    audit.events = [audit.events[1]]
    checks = smoke_checks(payload, audit)
    assert checks["route_question_completed"] is False
    assert checks["generation_succeeded"] is False


def test_recorded_grounding_true_without_verifier_trace_cannot_pass():
    payload, audit = success()
    payload["traces"] = []
    assert smoke_checks(payload, audit)["grounding_executed"] is False
