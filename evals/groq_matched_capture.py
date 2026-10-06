"""Groq-only evaluation smoke, then the unchanged matched capture (no judge).

Provider pinning is process-local. Existing application factories and fallback
implementation are untouched; callbacks block a non-Groq call before transport.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

MODEL = "openai/gpt-oss-120b"


@contextmanager
def groq_environment():
    overrides = {"GOOGLE_API_KEY": " ", "RFP_GRAPH_CORPUS_ID": "rfp-eval-frozen-641e9d4dad22"}
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        import config

        groq, google = config.get_api_keys()
        if not groq or google or config.GROQ_MODEL != MODEL:
            raise ValueError("Groq-only GPT-OSS evaluation selection could not be established")
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class GroqCallAudit(BaseCallbackHandler):
    raise_error = True

    def __init__(self, path, *, pacer=None):
        self.path = path
        # Resumable capture appends telemetry across process invocations. The
        # legacy runner uses fresh output directories, so this is inert there.
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            previous = {}
        self.events = list(previous.get("events", []))
        self.pending = {}
        self.phase = "smoke"
        self.chain_stages = {}
        self.node_outputs = list(previous.get("node_outputs", []))
        self.writer_errors = list(previous.get("diagnostic_writer_errors", []))
        self.pacer = pacer
        self.lock = threading.RLock()

    def checkpoint(self, *, strict=False):
        from evals.ragas_reports import write_report

        with self.lock:
            for attempt in range(4):
                try:
                    write_report(self.path, {
                        "events": self.events, "node_outputs": self.node_outputs,
                        "gemini_calls": 0, "judge_calls": 0,
                        "groq_calls": sum(e.get("class") == "ChatGroq" and e.get("status") not in {
                                              "blocked_before_transport", "blocked_by_local_quota"}
                                          for e in self.events),
                        "diagnostic_writer_errors": self.writer_errors,
                    })
                    return
                except OSError as exc:
                    if isinstance(exc, PermissionError) and attempt < 3:
                        time.sleep(0.05 * (attempt + 1))
                        continue
                    self.writer_errors.append({"error_type": type(exc).__name__})
                    if strict:
                        raise
                    return

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        from evals.provider_guard import ProviderAbort

        meta = metadata or {}
        params = kwargs.get("invocation_params", {})
        identity = (serialized or {}).get("id", [])
        event = {"phase": self.phase, "stage": meta.get("langgraph_node", "unknown"),
                 "provider": meta.get("ls_provider", "unknown"),
                 "class": identity[-1] if identity else "unknown",
                 "model": meta.get("ls_model_name") or params.get("model_name") or params.get("model"),
                 "status": "started", "http_status": None, "started_monotonic": time.perf_counter()}
        with self.lock:
            self.events.append(event)
            self.pending[run_id] = event
            if event["class"] != "ChatGroq" or event["model"] != MODEL:
                event["status"] = "blocked_before_transport"
                self.checkpoint()
                raise ProviderAbort({k: event[k] for k in ("stage", "provider", "model", "http_status")} | {
                    "category": "unexpected_provider_or_model", "retrieval_quality_failure": False})
            if self.pacer is not None:
                from evals.groq_quota_safety import QuotaSafetyStop

                try:
                    reservation = self.pacer.before_call(
                        messages,
                        max_output_tokens=params.get("max_tokens") or params.get("max_completion_tokens") or 2048,
                    )
                except QuotaSafetyStop as exc:
                    event.update({"status": "blocked_by_local_quota", "quota_dimension": exc.dimension,
                                  "quota_detail": exc.detail, "http_status": None})
                    self.pending.pop(run_id, None)
                    self.checkpoint()
                    raise
                event.update({k: v for k, v in reservation.items() if k != "request_id"})
                event["pacer_request_id"] = reservation["request_id"]
            self.checkpoint()

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self.lock:
            event = self.pending.pop(run_id, None)
            if event:
                latency = time.perf_counter() - event.pop("started_monotonic")
                pacing = float(event.get("pacing_wait_seconds", 0.0))
                event.update({"status": "completed", "provider_status": "success",
                              "latency_seconds": round(latency, 6),
                              "provider_latency_seconds_excluding_pacing": round(max(0.0, latency - pacing), 6)})
                generation = response.generations[0][0]
                raw = getattr(generation, "message", None)
                event["response_model"] = getattr(raw, "response_metadata", {}).get("model_name")
                usage = getattr(response, "llm_output", None) or {}
                usage = usage.get("token_usage") or usage.get("usage") or {}
                message_usage = getattr(raw, "usage_metadata", None) or {}
                actual_tokens = (message_usage.get("total_tokens") if message_usage else None) or usage.get("total_tokens")
                if actual_tokens is None:
                    actual_tokens = (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)
                if actual_tokens:
                    event["provider_reported_tokens"] = int(actual_tokens)
                if self.pacer is not None and event.get("pacer_request_id"):
                    self.pacer.finish_call(event["pacer_request_id"], actual_tokens=actual_tokens or None)
                if event["stage"] in {"grade_kb_evidence", "grade_web_evidence"}:
                    try:
                        decision = json.loads(generation.text)
                        event["raw_grade"] = decision.get("grade") if isinstance(decision, dict) else None
                    except (ValueError, TypeError):
                        event["raw_grade"] = None
            self.checkpoint()

    def mark_failure(self, failure):
        with self.lock:
            for run_id, event in list(self.pending.items()):
                if event["stage"] == failure["stage"]:
                    event.update({"status": "error", "provider_status": "provider_error",
                                  "http_status": failure.get("http_status"),
                                  "error_type": failure.get("error_type"),
                                  "latency_seconds": round(time.perf_counter() - event.pop("started_monotonic"), 6)})
                    if failure.get("rate_limit"):
                        event["rate_limit"] = failure["rate_limit"]
                    if self.pacer is not None and event.get("pacer_request_id"):
                        self.pacer.finish_call(event["pacer_request_id"], outcome="provider_error")
                    self.pending.pop(run_id)
            self.checkpoint()

    def on_llm_error(self, error, *, run_id, **kwargs):
        from evals.provider_guard import provider_status_code

        event = self.pending.get(run_id)
        if event:
            failure = {"stage": event["stage"], "http_status": provider_status_code(error),
                       "error_type": type(error).__name__}
            if failure["http_status"] == 429:
                from evals.groq_rate_limits import extract_groq_limit

                failure["rate_limit"] = extract_groq_limit(error)
            self.mark_failure(failure)

    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kwargs):
        with self.lock:
            self.chain_stages[run_id] = (metadata or {}).get("langgraph_node")

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        with self.lock:
            stage = self.chain_stages.pop(run_id, None)
            fields = {}
            if isinstance(outputs, dict):
                fields = {k: outputs[k] for k in ("intent", "source_used", "kb_grade", "web_grade") if k in outputs}
            elif type(outputs).__name__ == "EvidenceGrade":
                fields = {"parsed_grade": outputs.grade}
            if stage in {"route_question", "grade_kb_evidence", "grade_web_evidence"} and fields:
                self.node_outputs.append({"phase": self.phase, "stage": stage, **fields})
                self.checkpoint()


_audit = ContextVar("rfp_groq_matched_application_audit", default=None)
register_configure_hook(_audit, inheritable=True)


def smoke_checks(payload, audit):
    events = [e for e in audit.events if e["phase"] == "smoke"]
    tools = {t.get("tool") for t in payload.get("traces", [])}
    return {
        "groq_gpt_oss_only": bool(events) and all(e["class"] == "ChatGroq" and e["model"] == MODEL
                                                  and e["status"] == "completed" for e in events),
        "route_question_completed": any(e["stage"] == "route_question" and e["status"] == "completed" for e in events),
        "grader_succeeded": payload.get("kb_grade") == "good" and any(
            e["stage"] == "grade_kb_evidence" and e["status"] == "completed" for e in events),
        "generation_succeeded": payload.get("generation_kind") == "llm_kb" and bool(payload.get("answer", "").strip())
        and any(e["stage"] == "generate_from_kb" and e["status"] == "completed" for e in events),
        "grounding_executed": {"grounding_verifier", "final_grounding_verifier"} <= tools,
        "final_answer_grounded": payload.get("grounded") is True,
        "vector_only_effective": payload.get("retrieval_mode") == "vector_only" and not payload.get("graph_fallback_reason"),
    }


def summary(output):
    from evals.matched_environment import MODE_ORDER
    from evals.ragas_reports import write_report

    def read(name):
        return json.loads((output / name).read_text(encoding="utf-8"))

    capture = read("capture.json") if (output / "capture.json").exists() else None
    smoke = read("smoke.json")
    calls = read("application_calls.json")
    environment = read("environment_after_run.json")
    reports = {r["retrieval_mode"]: r for r in capture["reports"]} if capture else {}
    result = {
        "status": capture["status"] if capture else "benchmark_not_started_smoke_failed",
        "smoke": smoke, "provider_selection": read("provider_selection.json"),
        "full_benchmark": {"started": capture is not None, "expected_executions": 48,
                           "completed_executions": capture["completed_executions"] if capture else 0,
                           "capture_valid": capture["capture_valid"] if capture else False},
        "provider_failures": smoke["provider_failures"] + (capture["provider_failures"] if capture else []),
        "manifest": environment["manifest"], "manifest_parity": environment["comparison"],
        "corpus_integrity": read("corpus_fingerprints.json"),
        "protected_files": read("protected_files.json"),
        "gemini_calls": calls["gemini_calls"], "judge_calls": calls["judge_calls"],
        "groq_calls": calls["groq_calls"],
        "per_mode": {mode: {"completed_executions": sum(r["status"] == "completed" for r in reports[mode]["cases"])
                            if mode in reports else 0,
                            "aggregate": reports[mode].get("aggregate") if mode in reports else None,
                            "case_results": [{k: r.get(k) for k in ("id", "status", "benchmark", "deterministic")}
                                             for r in reports[mode]["cases"]] if mode in reports else []}
                     for mode in MODE_ORDER},
        "interpretation": "No paired quality comparison until 48/48 valid matched executions without provider/pipeline failures.",
    }
    write_report(output / "run_summary.json", result)
    return result


def run(output):
    from evals import matched_environment as matched
    from evals.gemini_matched_capture import graph_fingerprints
    from evals.provider_guard import ProviderAbort, stop_on_provider_failure
    from evals.ragas_adapter import adapt_execution
    from evals.ragas_reports import write_report
    from evals.static_grader_probe import protected_hashes

    output = output.resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory; retain all earlier reports")
    protected_before = protected_hashes()
    with groq_environment():
        from agent import get_llm
        from rfp_analyst.agent.grader import _create_grader_llm
        from rfp_analyst.agent.query_rewriter import _create_rewriter_llm
        from rfp_analyst.agent.router import _create_router_llm

        selection = {role: {"class": type(llm).__name__, "model": getattr(llm, "model_name", None)}
                     for role, llm in (("generation", get_llm()), ("routing", _create_router_llm()),
                                       ("grading", _create_grader_llm()), ("rewriting", _create_rewriter_llm()))}
        if any(v != {"class": "ChatGroq", "model": MODEL} for v in selection.values()):
            raise ValueError("An existing application factory did not select Groq GPT-OSS")
        write_report(output / "provider_selection.json", {"factories": selection, "groq_resolved": True,
                     "google_resolved": False, "judge": "not_run:capture_only",
                     "pinning": "process-local; production Gemini fallback unchanged"})
        args = SimpleNamespace(corpus_id=matched.CORPUS_ID, index=matched.INDEX, collection="rfp_kb_v2",
                               output_dir=output, publish=False, capture=False)
        if matched.run(args) != 0:
            raise ValueError("Corpus preflight failed")
        write_report(output / "environment_before_smoke.json", json.loads((output / "environment.json").read_text(encoding="utf-8")))
        before = graph_fingerprints()
        pipeline = matched.FrozenPipeline(matched.INDEX, collection="rfp_kb_v2")
        cases, _ = matched.validate_frozen(pipeline)
        case = next(c for c in cases if c["id"] == "semantic_modernization")
        pipeline.start_generation()
        smoke = {"case_id": case["id"], "requested_retrieval_mode": "vector_only",
                 "generation": pipeline.generation_metadata, "status": "running",
                 "judge": "not_run:capture_only", "provider_failures": []}
        write_report(output / "smoke.json", smoke)
        audit = GroqCallAudit(output / "application_calls.json")
        token = _audit.set(audit)
        started = time.perf_counter()
        exit_code = 1
        try:
            try:
                with stop_on_provider_failure():
                    payload = pipeline.execute(case, "vector_only")
                adapted = adapt_execution(case, payload, "vector_only")
                smoke.update({"sample": adapted.sample.model_dump(), "deterministic": adapted.deterministic,
                              "provenance": adapted.provenance, "generation_kind": payload.get("generation_kind"),
                              "kb_grade": payload.get("kb_grade"), "grounded": payload.get("grounded"),
                              "traces": payload.get("traces", []), "checks": smoke_checks(payload, audit)})
                smoke["status"] = "passed" if all(smoke["checks"].values()) else "failed_smoke_gate"
            except ProviderAbort as exc:
                smoke.update({"status": "stopped_provider_failure", "provider_failures": [
                    {"case_id": case["id"], "requested_retrieval_mode": "vector_only", **exc.failure}]})
                audit.mark_failure(exc.failure)
            except Exception as exc:
                smoke.update({"status": "stopped_pipeline_failure", "error_type": type(exc).__name__})
            finally:
                smoke["pipeline_latency_seconds"] = round(time.perf_counter() - started, 6)
                write_report(output / "smoke.json", smoke)
            print(f"Groq semantic_modernization smoke: {smoke['status']}", flush=True)
            if smoke["status"] == "passed":
                # Verify smoke parity before permitting ANY of the 48 executions.
                matched.run(args)
                write_report(output / "environment_after_smoke.json", json.loads((output / "environment.json").read_text(encoding="utf-8")))
                if graph_fingerprints() != before:
                    raise ValueError("Corpus fingerprints changed during smoke")
                audit.phase = "benchmark"
                args.capture = True
                exit_code = matched.run(args)
                captured = json.loads((output / "capture.json").read_text(encoding="utf-8"))
                for failure in captured["provider_failures"]:
                    audit.mark_failure(failure)
        finally:
            _audit.reset(token)
            audit.checkpoint(strict=True)
            pipeline.load_current()
            args.capture = False
            matched.run(args)
            write_report(output / "environment_after_run.json", json.loads((output / "environment.json").read_text(encoding="utf-8")))
            after = graph_fingerprints()
            write_report(output / "corpus_fingerprints.json", {
                "before": before, "after": after, "internal_rfp_unchanged": before["internal_rfp"] == after["internal_rfp"],
                "evaluation_corpus_unchanged": before["evaluation_corpus"] == after["evaluation_corpus"],
                "frozen_chroma_unchanged": pipeline.load_current().fingerprint == pipeline.corpus_hash})
            protected_after = protected_hashes()
            write_report(output / "protected_files.json", {"before": protected_before, "after": protected_after,
                         "unchanged": protected_before == protected_after})
        result = summary(output)
        print(json.dumps({k: result[k] for k in ("status", "full_benchmark", "provider_failures", "gemini_calls", "judge_calls")}, indent=2))
        return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        raise SystemExit(run(args.output_dir))
    except Exception as exc:
        print(f"Groq matched capture setup/verification stopped ({type(exc).__name__}); details redacted.")
        raise SystemExit(2) from None
