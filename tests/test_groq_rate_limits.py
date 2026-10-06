"""Quota diagnosis and error preservation without live calls, sleeps or retries."""

import json

import httpx
import pytest
from groq import RateLimitError

from evals.groq_rate_limits import HEADERS, extract_groq_limit, header_value, inspect_saved, quota_excerpt
from evals.provider_guard import ProviderAbort, ProviderGuard


def quota_error(dimension="tokens per minute", abbreviation="TPM"):
    headers = {
        "retry-after": "7", "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "995", "x-ratelimit-reset-requests": "1h2m3s",
        "x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "0",
        "x-ratelimit-reset-tokens": "7.66s", "authorization": "NEVER_EXPORT_SECRET",
    }
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://example.invalid"))
    body = {"error": {"code": "rate_limit_exceeded", "type": "tokens",
                      "message": f"Rate limit reached for model openai/gpt-oss-120b in organization org_PRIVATE "
                                 f"on {dimension} ({abbreviation}): Limit 8000, Used 7500, Requested 1000. "
                                 "Please try again in 7.66s. NEVER_EXPORT_SECRET private-rfp.pdf"}}
    return RateLimitError("SDK_SECRET_MESSAGE", response=response, body=body)


@pytest.mark.parametrize("dimension,abbreviation", [
    ("requests per minute", "RPM"), ("tokens per minute", "TPM"),
    ("requests per day", "RPD"), ("tokens per day", "TPD"),
    ("input tokens per minute", "ITPM"), ("output tokens per minute", "OTPM"),
])
def test_dimension_requires_explicit_provider_quota_clause(dimension, abbreviation):
    record = extract_groq_limit(quota_error(dimension, abbreviation))
    assert record["dimension"] == abbreviation
    assert record["dimension_evidence"] == "explicit provider quota clause"


def test_extracts_requested_headers_and_preserves_missing_vs_known():
    record = extract_groq_limit(quota_error())
    assert set(record["headers"]) == set(HEADERS)
    assert record["headers"]["retry-after"] == "7"
    assert record["headers"]["x-ratelimit-reset-tokens"] == "7.66s"
    assert record["http_status"] == 429
    assert record["header_windows"] == {"request_headers": "RPD", "token_headers": "TPM"}
    text = json.dumps(record)
    for secret in ("SDK_SECRET_MESSAGE", "org_PRIVATE", "NEVER_EXPORT_SECRET", "private-rfp.pdf", "authorization"):
        assert secret not in text


def test_headers_alone_cannot_identify_failed_dimension():
    error = quota_error()
    error.body = None
    record = extract_groq_limit(error)
    assert record["headers"]["x-ratelimit-remaining-tokens"] == "0"
    assert record["dimension"] == "unknown"


def test_fast_request_reset_does_not_mean_rpm():
    error = quota_error("requests per day", "RPD")
    error.response.headers["x-ratelimit-reset-requests"] = "0.5s"
    record = extract_groq_limit(error)
    assert record["dimension"] == "RPD"
    assert record["header_windows"]["request_headers"] == "RPD"


def test_wrapped_and_cyclic_sdk_errors_are_bounded():
    outer = Exception("private")
    outer.__cause__ = quota_error()
    assert extract_groq_limit(outer)["dimension"] == "TPM"
    outer.__cause__ = outer
    assert extract_groq_limit(outer)["dimension"] == "unknown"


@pytest.mark.parametrize("body", [None, "private response body", [], {"error": "private"},
                                  {"error": {"message": "private prompt", "type": "tokens"}}])
def test_malformed_or_non_quota_body_never_becomes_dimension(body):
    safe, dimension, _ = quota_excerpt(body)
    assert dimension == "unknown"
    assert "private" not in json.dumps(safe)


def test_conflicting_or_mismatched_dimensions_remain_unknown():
    body = quota_error().body
    body["error"]["message"] += " on requests per day (RPD): Limit 1000, Used 1000, Requested 1"
    assert quota_excerpt(body)[1] == "unknown"
    body["error"]["message"] = "Rate limit reached on requests per day (RPM): Limit 1, Used 1, Requested 1"
    assert quota_excerpt(body)[1] == "unknown"


@pytest.mark.parametrize("name,value,expected", [
    ("retry-after", "7.5", "7.5"),
    ("retry-after", "Wed, 21 Oct 2015 07:28:00 GMT", "Wed, 21 Oct 2015 07:28:00 GMT"),
    ("retry-after", "gsk_SECRET", None),
    ("x-ratelimit-reset-requests", "2m59.56s", "2m59.56s"),
    ("x-ratelimit-limit-requests", "1000", "1000"),
    ("x-ratelimit-limit-tokens", "private text", None),
])
def test_header_formats_do_not_export_arbitrary_text(name, value, expected):
    assert header_value(name, value) == expected


def test_guard_persists_metadata_before_abort_even_if_audit_hook_is_skipped():
    guard = ProviderGuard()
    guard.on_chat_model_start({"id": ["ChatGroq"]}, [], run_id="call", metadata={
        "langgraph_node": "grade_web_evidence", "ls_provider": "groq", "ls_model_name": "openai/gpt-oss-120b"})
    with pytest.raises(ProviderAbort) as exc:
        guard.on_llm_error(quota_error(), run_id="call")
    assert exc.value.failure["rate_limit"]["dimension"] == "TPM"
    assert exc.value.failure["retrieval_quality_failure"] is False


def test_historical_report_is_unknown_and_stays_unchanged(tmp_path):
    directory = tmp_path
    failure = {"provider": "groq", "http_status": 429, "error_type": "RateLimitError"}
    (directory / "run_summary.json").write_text(json.dumps({
        "provider_failures": [failure], "manifest_parity": {"parity": True}, "corpus_integrity": {},
    }), encoding="utf-8")
    (directory / "application_calls.json").write_text(json.dumps({
        "events": [{"status": "error", "http_status": 429}],
    }), encoding="utf-8")
    originals = {p: p.read_bytes() for p in directory.glob("*.json")}
    report = inspect_saved(directory)
    assert report["dimension"] == "unknown"
    assert report["headers"] == dict.fromkeys(HEADERS)
    assert report["response_body"] is None
    assert report["pacing_added"] is False
    assert report["new_provider_calls"] == report["wait_seconds"] == 0
    assert all(p.read_bytes() == content for p, content in originals.items())


def test_unknown_error_without_response_is_safe():
    error = Exception("private body")
    error.status_code = 429
    result = extract_groq_limit(error)
    assert result["http_status"] == 429
    assert result["response_body"] is None
    assert result["headers"] == dict.fromkeys(HEADERS)


def test_audit_copies_quota_details_into_provider_trace(tmp_path):
    from evals.groq_matched_capture import MODEL, GroqCallAudit

    audit = GroqCallAudit(tmp_path / "audit.json")
    audit.on_chat_model_start({"id": ["ChatGroq"]}, [], run_id="call", metadata={
        "ls_model_name": MODEL, "langgraph_node": "grade_web_evidence"})
    audit.on_llm_error(quota_error(), run_id="call")
    saved = json.loads(audit.path.read_text())
    assert saved["events"][0]["rate_limit"]["headers"]["retry-after"] == "7"
    assert saved["events"][0]["rate_limit"]["dimension"] == "TPM"
