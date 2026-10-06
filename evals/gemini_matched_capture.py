"""Operational Gemini-only smoke gate followed by the EXISTING matched capture.

No new application provider abstraction or changes to pipeline policy. Only
public frozen inputs, existing provider factories and read-only corpus checks.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook


class GoogleCallAudit(BaseCallbackHandler):
    """Observe only safe provider/model/node/status metadata, never input text."""

    def __init__(self, path):
        self.path = path
        self.events = []
        self.pending = {}
        self.phase = "smoke"

    def checkpoint(self):
        from evals.ragas_reports import write_report

        write_report(self.path, {"events": self.events, "groq_calls": 0,
                                "judge_calls": 0, "scope": "application_callbacks_only"})

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        from evals.provider_guard import ProviderAbort

        meta = metadata or {}
        params = kwargs.get("invocation_params", {})
        identity = (serialized or {}).get("id", [])
        classname = identity[-1] if identity else "unknown"
        event = {
            "phase": self.phase, "stage": meta.get("langgraph_node", "unknown"),
            "provider": meta.get("ls_provider", "unknown"), "class": classname,
            "model": meta.get("ls_model_name") or params.get("model") or "unknown",
            "status": "started",
        }
        self.events.append(event)
        self.pending[run_id] = event
        # Abort BEFORE transport if configuration unexpectedly selects another provider.
        if classname != "ChatGoogleGenerativeAI":
            event["status"] = "blocked_before_transport"
            self.checkpoint()
            raise ProviderAbort({**event, "http_status": None,
                                 "category": "unexpected_provider", "retrieval_quality_failure": False})
        self.checkpoint()

    def on_llm_end(self, response, *, run_id, **kwargs):
        if run_id in self.pending:
            self.pending.pop(run_id)["status"] = "completed"
        self.checkpoint()

    def on_llm_error(self, error, *, run_id, **kwargs):
        from evals.provider_guard import provider_status_code

        if run_id in self.pending:
            self.pending.pop(run_id).update({"status": "error", "http_status": provider_status_code(error)})
        self.checkpoint()


_audit = ContextVar("rfp_google_application_call_audit", default=None)
register_configure_hook(_audit, inheritable=True)


@contextmanager
def google_environment():
    """Process-local credential masking through the existing resolver.

An empty string falls through to .env. A whitespace value is truthy at lookup,
then stripped to empty by _resolve_key. Fail if Streamlit secret precedence
defeats masking; never edit secrets, .env, or the production configuration.
"""
    overrides = {"GROQ_API_KEY": " ", "RFP_GRAPH_CORPUS_ID": "rfp-eval-frozen-641e9d4dad22"}
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        import config

        groq, google = config.get_api_keys()
        if groq or not google:
            raise ValueError("Google-only provider selection could not be established")
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def graph_fingerprints():
    from evals.matched_environment import CORPUS_ID, PROTECTED, GraphInspection, digest_records, protected_fingerprint

    with GraphInspection() as graph:
        public = graph.namespace(CORPUS_ID)
        return {
            "internal_rfp": protected_fingerprint(graph.namespace(PROTECTED)),
            "evaluation_corpus": {
                "corpus_id": CORPUS_ID, "nodes": len(public["nodes"]),
                "relationships": len(public["edges"]),
                "node_sha256": digest_records(public["nodes"]),
                "relationship_sha256": digest_records(public["edges"]),
            },
        }


def summarize_saved_run(output):
    """Summarize actual saved artifacts only; never execute or re-score a case."""
    from evals.matched_environment import LOCK, MODE_ORDER
    from evals.ragas_reports import write_report

    def read(name):
        return json.loads((output / name).read_text(encoding="utf-8"))

    smoke = read("smoke.json")
    environment = read("environment.json")
    integrity = read("corpus_fingerprints.json")
    calls = read("application_calls.json")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    capture = read("capture.json") if (output / "capture.json").exists() else None
    reports = {r["retrieval_mode"]: r for r in capture["reports"]} if capture else {}
    per_mode = {}
    for mode in MODE_ORDER:
        report = reports.get(mode)
        rows = {r["id"]: r for r in report["cases"]} if report else {}
        per_mode[mode] = {
            "planned_executions": len(lock["signature"]["case_ids"]),
            "completed_executions": sum(r["status"] == "completed" for r in rows.values()),
            "deterministic_aggregates": report.get("aggregate", {}).get("deterministic") if report else None,
            "case_results": {case_id: {
                "status": rows[case_id]["status"] if case_id in rows else "not_run",
                "benchmark": rows[case_id].get("benchmark") if case_id in rows else None,
            } for case_id in lock["signature"]["case_ids"]},
        }
    result = {
        "status": capture["status"] if capture else "benchmark_not_started_smoke_failed",
        "provider_selection": read("provider_selection.json"),
        "manifest": environment["manifest"], "manifest_comparison": environment["comparison"],
        "corpus_integrity": integrity, "smoke": smoke,
        "full_benchmark": {
            "expected_executions": 48, "started": capture is not None,
            "completed_executions": capture["completed_executions"] if capture else 0,
            "capture_valid": capture["capture_valid"] if capture else False,
        },
        "per_mode": per_mode,
        "provider_failures": smoke["provider_failures"] + (capture["provider_failures"] if capture else []),
        "groq_calls": calls["groq_calls"], "judge_calls": calls["judge_calls"],
        "interpretation": "No quality conclusion without 48 valid completed matched executions; errors are not quality scores.",
    }
    write_report(output / "run_summary.json", result)
    return result


def run(output):
    from evals import matched_environment as matched
    from evals.provider_guard import ProviderAbort, stop_on_provider_failure
    from evals.ragas_adapter import adapt_execution
    from evals.ragas_reports import write_report

    output = output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory; preserve all prior reports")
    with google_environment():
        from agent import get_llm
        from rfp_analyst.agent.grader import _create_grader_llm
        from rfp_analyst.agent.query_rewriter import _create_rewriter_llm
        from rfp_analyst.agent.router import _create_router_llm

        selection = {
            role: {"class": type(llm).__name__, "model": getattr(llm, "model", None)}
            for role, llm in (
                ("generation", get_llm()), ("routing", _create_router_llm()),
                ("grading", _create_grader_llm()), ("rewriting", _create_rewriter_llm()),
            )
        }
        if any(value["class"] != "ChatGoogleGenerativeAI" for value in selection.values()):
            raise ValueError("An existing factory did not select Google")
        write_report(output / "provider_selection.json", {"factories": selection,
                     "groq_resolved": False, "google_resolved": True, "judge": "not_run:capture_only"})
        args = SimpleNamespace(corpus_id=matched.CORPUS_ID, index=matched.INDEX,
                               collection="rfp_kb_v2", output_dir=output,
                               publish=False, capture=False)
        if matched.run(args) != 0:
            raise ValueError("Corpus preflight failed")
        write_report(output / "environment_before_smoke.json", json.loads(
            (output / "environment.json").read_text(encoding="utf-8")))
        before = graph_fingerprints()
        write_report(output / "corpus_fingerprints.json", {"before": before})
        pipeline = matched.FrozenPipeline(matched.INDEX, collection="rfp_kb_v2")
        cases, _ = matched.validate_frozen(pipeline)
        case = cases[0]  # The first frozen case, not selected after inspecting scores.
        pipeline.start_generation()
        smoke = {
            "case_id": case["id"], "requested_retrieval_mode": "vector_only",
            "generation": pipeline.generation_metadata, "status": "running",
            "judge": "not_run:capture_only", "provider_failures": [],
        }
        write_report(output / "smoke.json", smoke)
        audit = GoogleCallAudit(output / "application_calls.json")
        token = _audit.set(audit)
        started = time.perf_counter()
        exit_code = 1
        try:
            try:
                with stop_on_provider_failure():
                    payload = pipeline.execute(case, "vector_only")
                adapted = adapt_execution(case, payload, "vector_only")
                smoke.update({"sample": adapted.sample.model_dump(),
                              "deterministic": adapted.deterministic,
                              "generation_kind": adapted.generation_kind,
                              "kb_grade": payload.get("kb_grade"),
                              "traces": [{k: t.get(k) for k in (
                                  "tool", "step", "status", "grade", "verification_status"
                              ) if k in t} for t in payload.get("traces", [])]})
                tools = {t.get("tool") for t in payload.get("traces", [])}
                smoke["checks"] = {
                    "google_only": bool(audit.events) and all(
                        e["class"] == "ChatGoogleGenerativeAI" for e in audit.events),
                    "grader_succeeded": payload.get("kb_grade") == "good",
                    "generation_succeeded": adapted.generation_kind == "llm_kb" and any(
                        e["stage"] == "generate_from_kb" and e["status"] == "completed"
                        for e in audit.events),
                    "grounding_executed": {"grounding_verifier", "final_grounding_verifier"} <= tools,
                }
                smoke["status"] = "passed" if all(smoke["checks"].values()) else "failed_smoke_gate"
            except ProviderAbort as exc:
                smoke["status"] = "stopped_provider_failure"
                smoke["provider_failures"] = [{"case_id": case["id"],
                    "requested_retrieval_mode": "vector_only", **exc.failure}]
                # The fail-fast callback may abort before the audit error hook executes.
                for event in audit.pending.values():
                    event.update({"status": "error", "http_status": exc.failure.get("http_status")})
                audit.pending.clear()
                audit.checkpoint()
            except Exception as exc:
                smoke.update({"status": "stopped_pipeline_failure", "error_type": type(exc).__name__})
            finally:
                smoke["pipeline_latency_seconds"] = round(time.perf_counter() - started, 6)
                write_report(output / "smoke.json", smoke)
            # No further application calls unless the ONE smoke execution passed.
            if smoke["status"] == "passed":
                audit.phase = "benchmark"
                args.capture = True
                exit_code = matched.run(args)
        finally:
            _audit.reset(token)
            pipeline.load_current()
            # Recheck exact manifest parity, not just graph fingerprints.
            args.capture = False
            matched.run(args)
            write_report(output / "environment_after_run.json", json.loads(
                (output / "environment.json").read_text(encoding="utf-8")))
            after = graph_fingerprints()
            write_report(output / "corpus_fingerprints.json", {
                "before": before, "after": after,
                "internal_rfp_unchanged": before["internal_rfp"] == after["internal_rfp"],
                "evaluation_corpus_unchanged": before["evaluation_corpus"] == after["evaluation_corpus"],
                "frozen_chroma_unchanged": pipeline.load_current().fingerprint == pipeline.corpus_hash,
            })
        summarize_saved_run(output)
        print(json.dumps({"smoke_status": smoke["status"], "provider_failures": smoke["provider_failures"],
                          "full_capture_started": smoke["status"] == "passed", "output": str(output)}, indent=2))
        return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args.output_dir)
    except Exception as exc:
        print(f"Gemini capture setup/verification failed ({type(exc).__name__}); details redacted.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
