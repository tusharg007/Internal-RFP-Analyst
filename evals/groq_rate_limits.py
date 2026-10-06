"""Offline quota diagnosis and safe error telemetry; no pacing or provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from email.utils import parsedate_to_datetime
from pathlib import Path

HEADERS = (
    "retry-after", "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests", "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
)
DOCUMENTATION = "https://console.groq.com/docs/rate-limits"
DIMENSIONS = {
    "requests per minute": "RPM", "tokens per minute": "TPM",
    "requests per day": "RPD", "tokens per day": "TPD",
    "input tokens per minute": "ITPM", "output tokens per minute": "OTPM",
}


def header_value(name, value):
    """Retain only known numeric/duration/date formats, never arbitrary header text."""
    if value is None:
        return None
    value = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return value
    if "reset" in name and re.fullmatch(r"(?:\d+(?:\.\d+)?(?:ms|s|m|h|d))+", value):
        return value
    if name == "retry-after":
        try:
            if parsedate_to_datetime(value).tzinfo is not None:
                return value
        except (ValueError, TypeError, OverflowError):
            pass
    return None


def quota_excerpt(body):
    """Only retain a matched provider quota clause, not org IDs, prompts or secrets."""
    if not isinstance(body, dict) or not isinstance(body.get("error"), dict):
        return None, "unknown", "quota message unavailable"
    error = body["error"]
    message = error.get("message")
    safe = {k: v for k in ("type", "code") if isinstance((v := error.get(k)), str)
            and re.fullmatch(r"[a-z_]{1,80}", v)}
    if not isinstance(message, str) or not re.search(r"rate limit (?:reached|exceeded)", message, re.I):
        return {"error": safe} if safe else None, "unknown", "explicit quota dimension unavailable"
    pattern = (
        r"\bon\s+(input tokens per minute|output tokens per minute|requests per minute|"
        r"tokens per minute|requests per day|tokens per day)\s*\((RPM|TPM|RPD|TPD|ITPM|OTPM)\)"
        r"\s*:\s*Limit\s+([\d.]+),\s*Used\s+([\d.]+),\s*Requested\s+([\d.]+)"
    )
    matches = list(re.finditer(pattern, message, re.I))
    dimensions = {DIMENSIONS[m.group(1).lower()] for m in matches
                  if DIMENSIONS[m.group(1).lower()] == m.group(2).upper()}
    clauses = [m.group() for m in matches]
    if clauses:
        safe["message"] = " | ".join(clauses)
    dimension = next(iter(dimensions)) if len(dimensions) == 1 else "unknown"
    return {"error": safe}, dimension, "explicit provider quota clause" if dimension != "unknown" else "absent or conflicting quota dimensions"


def extract_groq_limit(error):
    """Bounded SDK cause inspection; never stringify exceptions or serialize requests."""
    result = {"http_status": None, "headers": dict.fromkeys(HEADERS), "response_body": None,
              "response_body_capture": "quota excerpt only; no request/org/credential data",
              "dimension": "unknown", "dimension_evidence": "no provider quota clause captured"}
    seen = set()
    for _ in range(8):
        if error is None or id(error) in seen:
            break
        seen.add(id(error))
        response = getattr(error, "response", None)
        status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            result["http_status"] = status
        headers = getattr(response, "headers", {}) or {}
        for name in HEADERS:
            value = header_value(name, headers.get(name))
            if value is not None:
                result["headers"][name] = value
        body, dimension, basis = quota_excerpt(getattr(error, "body", None))
        if body is not None:
            result.update(response_body=body, dimension=dimension, dimension_evidence=basis)
        error = error.__cause__ or error.__context__
    # Header windows are documented, but a 429 plus headers alone does not
    # establish which limit triggered it. Never turn RPD headers into RPM.
    result["header_windows"] = {"request_headers": "RPD", "token_headers": "TPM"}
    return result


def inspect_saved(directory):
    """Read historical artifacts only. Missing fields remain unknown, not guessed."""
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    calls = json.loads((directory / "application_calls.json").read_text(encoding="utf-8"))
    failures = summary["provider_failures"]
    failed = next(f for f in failures if f.get("provider") == "groq" and f.get("http_status") == 429)
    associated = [e for e in calls["events"] if e.get("status") == "error" and e.get("http_status") == 429]
    telemetry = failed.get("rate_limit") or next((e["rate_limit"] for e in associated if e.get("rate_limit")), None)
    result = {
        "source_directory": str(directory), "failure": failed,
        "associated_provider_errors": associated,
        "http_status": failed["http_status"], "groq_sdk_error_type": failed["error_type"],
        "response_body": telemetry.get("response_body") if telemetry else None,
        "headers": telemetry["headers"] if telemetry else dict.fromkeys(HEADERS),
        "dimension": telemetry["dimension"] if telemetry else "unknown",
        "missing_data_reason": None if telemetry else "Historical recorder omitted SDK response body and HTTP headers.",
        "header_windows": {"request_headers": "RPD", "token_headers": "TPM"},
        "provider_documentation": DOCUMENTATION,
        "historical_manifest_parity": summary["manifest_parity"]["parity"],
        "historical_corpus_integrity": summary["corpus_integrity"],
        "new_provider_calls": 0, "new_benchmark_started": False,
        "pacing_added": False, "wait_seconds": 0,
        "two_case_smoke": "not_run: exact quota dimension not established",
        "retrieval_quality_failure": False,
    }
    return result


if __name__ == "__main__":
    from evals.ragas_reports import write_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a new output path; preserve existing reports")
    originals = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in args.source_dir.glob("*.json")}
    report = inspect_saved(args.source_dir.resolve())
    report["historical_reports_unchanged"] = originals == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.source_dir.glob("*.json")}
    write_report(args.output, report)
    print(json.dumps({k: report[k] for k in ("http_status", "groq_sdk_error_type", "dimension", "headers",
                     "missing_data_reason", "historical_reports_unchanged", "new_provider_calls", "pacing_added")}, indent=2))
