"""One static grader request per listed Google model; no application execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

MODELS = ("gemini-3.1-pro-preview", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.8-flash")
ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "evals/results/semantic-grader-diagnostic-20261006-rerun/grading_trace.json"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def saved_input(path):
    from rfp_analyst.agent.prompts import KB_GRADER_PROMPT

    original = path.read_bytes()
    entry = json.loads(original)["attempts"][0]
    evidence = entry["grader_evidence"]
    prompt = KB_GRADER_PROMPT.format(question=entry["original_query"], context=evidence)
    if digest(prompt) != entry["prompt_hash"] or digest(evidence) != entry["evidence_hash"]:
        raise ValueError("Saved prompt/evidence integrity mismatch; no calls allowed")
    if not entry["prompt_matches_captured_evidence"]:
        raise ValueError("Original diagnostic did not establish exact prompt identity")
    return prompt, evidence, hashlib.sha256(original).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def error_metadata(error):
    """Whitelist numeric status and quota/backoff fields, never exception messages."""
    from evals.provider_guard import provider_status_code

    result = {"error_type": type(error).__name__, "http_status": provider_status_code(error),
              "provider_code": None, "retry_delay_seconds": None, "quota_violations": []}
    seen = set()
    for _ in range(8):
        if error is None or id(error) in seen:
            break
        seen.add(id(error))
        status = getattr(error, "status", None)
        if isinstance(status, str) and re.fullmatch(r"[A-Z_]+", status):
            result["provider_code"] = status
        body = getattr(error, "details", None)
        if isinstance(body, dict):
            body = body.get("error", body)
            for detail in body.get("details", []):
                if not isinstance(detail, dict):
                    continue
                if str(detail.get("@type", "")).endswith("RetryInfo"):
                    delay = detail.get("retryDelay", "")
                    if isinstance(delay, str) and re.fullmatch(r"\d+(?:\.\d+)?s", delay):
                        result["retry_delay_seconds"] = float(delay[:-1])
                if str(detail.get("@type", "")).endswith("QuotaFailure"):
                    for violation in detail.get("violations", []):
                        result["quota_violations"].append({
                            "quota_id": violation.get("quotaId"),
                            "quota_metric": violation.get("quotaMetric"),
                            "model": violation.get("quotaDimensions", {}).get("model"),
                        })
        error = error.__cause__ or error.__context__
    return result


def safe_continuation(error, failed_model):
    """Only explicit model-isolated quota AND bounded server backoff permit continuation."""
    delay = error.get("retry_delay_seconds")
    violations = error.get("quota_violations", [])
    isolated = bool(violations) and all(
        v.get("model") in {failed_model, f"models/{failed_model}"}
        and "permodel" in str(v.get("quota_id", "")).lower()
        for v in violations
    )
    if isolated and isinstance(delay, (float, int)) and math.isfinite(delay) and 0 <= delay <= 60:
        return max(1.0, delay + 1.0)
    return None


def probe_models(listed, prompt, evidence, invoke, checkpoint, wait=time.sleep):
    results = []
    blocked = None
    for model in MODELS:
        name = f"models/{model}"
        row = {"model_id": name, "requested_model": model, "provider": "google_genai",
               "prompt_hash": digest(prompt), "evidence_hash": digest(evidence),
               "raw_response": None, "parsed_grade": None, "http_status": None,
               "provider_status": "not_called", "latency_seconds": None, "generation_requests": 0,
               "classification": "unavailable/provider-error"}
        results.append(row)
        metadata = listed.get(name)
        row["availability"] = metadata
        if metadata is None or "generateContent" not in metadata["supported_actions"]:
            row["provider_status"] = "unavailable"
        elif blocked:
            row.update({"provider_status": "not_called_quota_safety", "blocked_by_model": blocked})
        else:
            row.update({"provider_status": "started", "generation_requests": 1})
            checkpoint(results)
            started = time.perf_counter()
            try:
                output = invoke(model, prompt)
                row.update(output)
                row["classification"] = (
                    "compatible" if row["parsed_grade"] == "good" else
                    "incompatible-for-this-grader" if row["parsed_grade"] == "weak" else
                    "unavailable/provider-error"
                )
            except Exception as exc:
                error = error_metadata(exc)
                row.update({"provider_status": "provider_error", **error})
                row["latency_seconds"] = round(time.perf_counter() - started, 6)
                if error["http_status"] == 429:
                    delay = safe_continuation(error, model)
                    row["continuation_backoff_seconds"] = delay
                    if delay is None:
                        blocked = model
                    else:
                        checkpoint(results)
                        print(f"{model}: HTTP 429; waiting {delay:.1f}s before a DIFFERENT model", flush=True)
                        remaining = delay
                        while remaining > 0:
                            interval = min(5.0, remaining)
                            wait(interval)
                            remaining -= interval
            finally:
                if row["latency_seconds"] is None:
                    row["latency_seconds"] = round(time.perf_counter() - started, 6)
            print(f"{model}: {row['provider_status']} / {row['classification']}", flush=True)
        checkpoint(results)
    return results


def invoke_google(model, prompt, key):
    from config import GRADING_TEMPERATURE, LLM_MAX_TOKENS
    from langchain_google_genai import ChatGoogleGenerativeAI
    from rfp_analyst.agent.schemas_decisions import EvidenceGrade

    llm = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=GRADING_TEMPERATURE,
                                max_output_tokens=LLM_MAX_TOKENS, max_retries=0)
    try:
        # include_raw only changes the local parser wrapper, not the HTTP model input.
        output = llm.with_structured_output(EvidenceGrade, method="json_mode", include_raw=True).invoke(prompt)
        raw = output["raw"]
        parsed = output["parsed"]
        return {"raw_response": {"content": raw.content, "response_metadata": raw.response_metadata,
                                 "usage_metadata": raw.usage_metadata},
                "parsed_grade": parsed.grade if parsed is not None else None,
                "parsing_error_type": type(output["parsing_error"]).__name__ if output["parsing_error"] else None,
                "provider_status": "success" if parsed is not None else "invalid_contract",
                "http_status": None,
                "http_status_note": "Successful SDK response; raw HTTP status is not exposed by this adapter."}
    finally:
        if llm.client is not None:
            llm.client.close()


def protected_hashes():
    paths = list((ROOT / "src/rfp_analyst").rglob("*.py"))
    paths += [ROOT / p for p in (
        "config.py", "agent.py", "rag_engine.py", "evals/retrieval_questions.yaml",
        "evals/retrieval_benchmark.lock.json", "evals/golden_questions.yaml",
    )]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run(output):
    from config import GEMINI_MODEL, GRADING_TEMPERATURE, LLM_MAX_TOKENS, get_api_keys
    from google import genai
    from google.genai import types

    if output.exists():
        raise ValueError("Preserve existing reports; use a fresh output directory")
    prompt, evidence, trace_hash = saved_input(TRACE)
    protected = protected_hashes()
    _, key = get_api_keys()
    if not key:
        raise ValueError("Configured Google key is missing")
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "availability_check",
              "source_trace": str(TRACE), "source_trace_sha256": trace_hash,
              "prompt_hash": digest(prompt), "evidence_hash": digest(evidence),
              "prompt": prompt, "evidence": evidence, "configured_gemini_model": GEMINI_MODEL,
              "contract": {"method": "json_mode", "schema": "EvidenceGrade", "allowed_grades": ["good", "weak"],
                           "temperature": GRADING_TEMPERATURE, "max_output_tokens": LLM_MAX_TOKENS,
                           "automatic_retries": 0},
              "models": [], "availability": {}, "benchmark_started": False,
              "judge": "not_run", "interpretation": "Single-input grader compatibility, not answer quality."}
    def checkpoint(rows):
        write_json(output, {**report, "models": rows})
    checkpoint([])
    try:
        with genai.Client(api_key=key, http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1))) as client:
            listing = {m.name: {"name": m.name, "version": m.version,
                               "supported_actions": list(m.supported_actions or [])}
                       for m in client.models.list()}
        report["availability"] = {name: listing.get(name) for name in (f"models/{m}" for m in MODELS)}
        report["availability_checked_at"] = datetime.now(timezone.utc).isoformat()
        report["status"] = "probing"
        checkpoint([])
        report["models"] = probe_models(listing, prompt, evidence,
                                       lambda model, text: invoke_google(model, text, key), checkpoint)
        report["status"] = "completed" if all(r["generation_requests"] for r in report["models"]) else "completed_with_unavailable_or_blocked_models"
    except Exception as exc:
        report.update({"status": "setup_error", "setup_error": error_metadata(exc)})
    finally:
        report["source_trace_unchanged"] = hashlib.sha256(TRACE.read_bytes()).hexdigest() == trace_hash
        report["protected_files_unchanged"] = protected_hashes() == protected
        report["identical_hashes_across_models"] = all(
            r["prompt_hash"] == digest(prompt) and r["evidence_hash"] == digest(evidence)
            for r in report["models"])
        write_json(output, report)
    print(json.dumps({"status": report["status"], "models": [{k: r.get(k) for k in (
        "model_id", "provider_status", "parsed_grade", "http_status", "classification", "latency_seconds"
    )} for r in report["models"]]}, indent=2))
    return 0 if report["status"].startswith("completed") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.output.resolve()))
