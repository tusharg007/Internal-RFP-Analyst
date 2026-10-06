"""Operational provider selection/audit tests; no provider or Neo4j calls."""

import json
import os

import pytest

from evals.gemini_matched_capture import GoogleCallAudit, google_environment
from evals.provider_guard import ProviderAbort, ProviderGuard, provider_status_code


@pytest.mark.parametrize("code", [400, 403, 404, 429, 503])
def test_google_wrapped_status_preserved_without_exception_text(code):
    sdk_error = Exception("secret SDK body")
    sdk_error.code = code
    wrapper = Exception("secret classified Google error")
    wrapper.__cause__ = sdk_error
    assert provider_status_code(wrapper) == code
    guard = ProviderGuard()
    with pytest.raises(ProviderAbort) as exc:
        guard.on_llm_error(wrapper, run_id="test")
    assert exc.value.failure["http_status"] == code
    assert "secret" not in json.dumps(exc.value.failure)
    assert exc.value.failure["retrieval_quality_failure"] is False


def test_status_chain_is_bounded_and_cycle_safe():
    error = Exception("private")
    error.__cause__ = error
    assert provider_status_code(error) is None


def test_google_selection_uses_existing_resolver_and_restores_environment(monkeypatch):
    import config

    monkeypatch.setattr(config, "_get_streamlit_secret", lambda name: "")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-public-placeholder-not-a-real-key")
    monkeypatch.setenv("GROQ_API_KEY", "preexisting-test-placeholder")
    monkeypatch.setenv("RFP_GRAPH_CORPUS_ID", "internal-rfp")
    with google_environment():
        groq, google = config.get_api_keys()
        assert groq == ""
        assert google == "test-public-placeholder-not-a-real-key"
        assert config.get_retrieval_settings()["corpus_id"] == "rfp-eval-frozen-641e9d4dad22"
    assert os.environ["GROQ_API_KEY"] == "preexisting-test-placeholder"
    assert os.environ["RFP_GRAPH_CORPUS_ID"] == "internal-rfp"


def test_streamlit_groq_precedence_stops_selection_without_changing_secret(monkeypatch):
    import config

    monkeypatch.setattr(config, "_get_streamlit_secret", lambda name: "hidden" if name == "GROQ_API_KEY" else "")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-placeholder")
    with pytest.raises(ValueError, match="Google-only"):
        with google_environment():
            pytest.fail("Must not permit a Groq-selecting process")


def test_audit_records_successful_google_call_without_prompts(tmp_path):
    path = tmp_path / "calls.json"
    audit = GoogleCallAudit(path)
    audit.on_chat_model_start({"id": ["ChatGoogleGenerativeAI"]}, ["SECRET_PROMPT"], run_id="test",
                             metadata={"ls_provider": "google_genai", "ls_model_name": "test-model",
                                       "langgraph_node": "generate_from_kb"})
    audit.on_llm_end("SECRET_ANSWER", run_id="test")
    report = json.loads(path.read_text())
    assert report["events"][0]["status"] == "completed"
    assert report["events"][0]["model"] == "test-model"
    assert report["groq_calls"] == report["judge_calls"] == 0
    assert "SECRET" not in path.read_text()


def test_audit_blocks_groq_before_transport(tmp_path):
    audit = GoogleCallAudit(tmp_path / "calls.json")
    with pytest.raises(ProviderAbort) as exc:
        audit.on_chat_model_start({"id": ["ChatGroq"]}, ["private"], run_id="test",
                                 metadata={"ls_provider": "groq", "langgraph_node": "route_question"})
    assert exc.value.failure["category"] == "unexpected_provider"
    assert audit.events[0]["status"] == "blocked_before_transport"
