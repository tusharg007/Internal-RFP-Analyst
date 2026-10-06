"""Resumable, quota-safe Groq capture for the frozen public 48-run experiment.

This is deterministic capture only: no RAGAS judge is initialized. Every case
runs vector_only, graph_only, hybrid in that order. Provider calls are opt-in
and are blocked until the operator explicitly confirms the known TPD reset.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from evals.groq_quota_safety import (
    MODEL, GroqTPMPacer, QuotaSafetyStop, assert_tpd_reset_confirmed,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evals/results/resumable-groq"
MODE_ORDER = ("vector_only", "graph_only", "hybrid")
SMOKE_CASE_IDS = ("semantic_modernization", "semantic_healthcare_security")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode())


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _evaluation_code_hash() -> str:
    files = (
        Path(__file__), Path(__file__).with_name("groq_quota_safety.py"),
        Path(__file__).with_name("groq_matched_capture.py"), Path(__file__).with_name("run_ragas.py"),
    )
    return _canonical_hash({str(path.name): _sha(path.read_bytes()) for path in files})


def experiment_metadata(pipeline, environment: dict) -> dict:
    """Capture secret-free, experiment-defining inputs. Exact match gates reuse."""
    import config
    from evals.matched_environment import LOCK, QUESTIONS

    from rfp_analyst.agent import prompts

    safe_config = {
        name: getattr(config, name)
        for name in (
            "GROQ_MODEL", "GENERATION_TEMPERATURE", "GRADING_TEMPERATURE", "RETRIEVAL_K",
            "MIN_RELEVANCE_SCORE", "MAX_QUERY_RETRIES", "MAX_PROMPT_TOKENS",
            "MAX_CONTEXT_CHARS_PER_CHUNK", "EMBEDDING_MODEL", "COLLECTION_NAME",
            "LLM_MAX_TOKENS", "RFP_ANALYSIS_MAX_OUTPUT_TOKENS", "MAX_HISTORY_MESSAGES",
            "MAX_TARGET_CHUNKS", "MAX_CASE_STUDIES", "MAX_CHUNKS_PER_CASE_STUDY",
        ) if hasattr(config, name)
    }
    retrieval_config = config.get_retrieval_settings()
    graph_config = config.get_neo4j_settings("reader")
    safe_config["configured_retrieval_policy"] = retrieval_config["policy"]
    safe_config["configured_graph_corpus_id"] = retrieval_config["corpus_id"]
    safe_config["neo4j"] = {key: graph_config[key] for key in (
        "enabled", "database", "connection_timeout_seconds", "acquisition_timeout_seconds",
        "transaction_timeout_seconds", "max_connection_pool_size",
    )}
    manifest = environment["manifest"]
    prompts_hash = _sha(Path(prompts.__file__).read_bytes())
    generation_settings = dict(pipeline.generation_metadata)
    body = {
        "metadata_schema": "resumable-groq-capture-v1",
        "git_commit": _git_commit(),
        "evaluation_code_hash": _evaluation_code_hash(),
        "provider": "groq",
        "model": MODEL,
        "config_hash": _canonical_hash({"safe_values": safe_config,
                                          "config_source_sha256": _sha(Path(config.__file__).read_bytes())}),
        "prompt_version": prompts_hash,
        "prompt_hash": prompts_hash,
        "question_set_hash": _sha(QUESTIONS.read_bytes()),
        "freeze_lock_hash": _sha(LOCK.read_bytes()),
        "frozen_corpus_hash": pipeline.corpus_hash,
        "chroma_manifest_hash": _canonical_hash({
            "indexed_digest": manifest["indexed_digest"],
            "documents": manifest["documents"], "chunks": manifest["chunks"],
        }),
        "chroma_document_count": manifest["document_count"],
        "chroma_chunk_count": manifest["chunk_count"],
        "neo4j_corpus_id": manifest["graph_corpus_id"],
        "neo4j_corpus_version": manifest["graph_version"],
        "generation_settings": generation_settings,
        "judge": "not_run:capture_only",
    }
    body["experiment_hash"] = _canonical_hash(body)
    return body


def _load_prior(batch_dir: Path, metadata: dict):
    executions = {}
    for path in sorted(batch_dir.glob("batch-*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        if batch.get("experiment_metadata") != metadata:
            continue  # Never reuse an execution from a different experiment.
        for row in batch.get("executions", []):
            key = (row["case_id"], row["retrieval_mode"])
            if row.get("status") == "completed":
                if key in executions:
                    raise ValueError(f"duplicate completed execution in resume history: {key}")
                executions[key] = row
    return executions


async def execute_batch(cases, *, metadata, completed, execute, output: Path, batch_size: int,
                        calls_path: Path, parity_check=None):
    """Pure coordinator used by CLI and offline tests; persists each success."""
    from evals.provider_guard import ProviderAbort

    output.mkdir(parents=True, exist_ok=True)
    index = 1 + max((int(p.stem.split("-")[-1]) for p in output.glob("batch-*.json")), default=0)
    batch_path = output / f"batch-{index:04d}.json"
    batch = {"status": "running", "experiment_metadata": metadata, "executions": [],
             "provider_calls_file": str(calls_path), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "completed_before_batch": len(completed), "provider_failure": None,
             "pacing": {"mechanism": "rolling token reservations", "tpm_limit": 8000,
                        "wait_seconds": 0.0, "wait_is_excluded_from_pipeline_latency": True}}
    sequence = max((int(row.get("execution_sequence", 0)) for row in completed.values()), default=0) + 1
    touched_cases = set()
    _atomic_json(batch_path, batch)

    def pipeline_stop(case_id, mode, exc):
        batch["status"] = "incomplete_due_to_pipeline_failure"
        batch["pipeline_failure"] = {"case_id": case_id, "retrieval_mode": mode,
                                     "error_type": type(exc).__name__,
                                     "retrieval_quality_failure": False}
        batch["unexecuted"] = [
            {"case_id": c["id"], "retrieval_mode": m}
            for c in cases for m in MODE_ORDER
            if (c["id"], m) not in completed and not any(
                x["case_id"] == c["id"] and x["retrieval_mode"] == m for x in batch["executions"])
        ]
        _atomic_json(batch_path, batch)
        return batch

    for case in cases:
        case_id = case["id"]
        if all((case_id, mode) in completed for mode in MODE_ORDER):
            continue
        if case_id not in touched_cases and len(touched_cases) >= batch_size:
            break
        touched_cases.add(case_id)
        for mode in MODE_ORDER:
            if (case_id, mode) in completed:
                continue
            if parity_check:
                try:
                    parity_check()
                except Exception as exc:
                    return pipeline_stop(case_id, mode, exc)
            try:
                row = await execute(case, mode)
            except (QuotaSafetyStop, ProviderAbort) as exc:
                if isinstance(exc, QuotaSafetyStop):
                    dimension, failure = exc.dimension, {"category": "local_quota_guard", "detail": exc.detail}
                elif isinstance(exc, ProviderAbort):
                    failure = exc.failure
                    dimension = ((failure.get("rate_limit") or {}).get("dimension") or "unknown")
                quota_error = isinstance(exc, QuotaSafetyStop) or (
                    isinstance(exc, ProviderAbort) and failure.get("http_status") == 429
                )
                batch["status"] = ("incomplete_due_to_provider_quota" if quota_error
                                    else "incomplete_due_to_provider_error")
                batch["provider_failure"] = {"case_id": case_id, "retrieval_mode": mode,
                                              "provider": "groq", "model": MODEL,
                                              "quota_dimension": dimension, **failure}
                batch["unexecuted"] = [
                    {"case_id": c["id"], "retrieval_mode": m}
                    for c in cases for m in MODE_ORDER
                    if (c["id"], m) not in completed and not any(
                        x["case_id"] == c["id"] and x["retrieval_mode"] == m for x in batch["executions"])
                ]
                _atomic_json(batch_path, batch)
                return batch
            except Exception as exc:
                return pipeline_stop(case_id, mode, exc)
            if row.get("status") != "completed":
                batch["status"] = "incomplete_due_to_pipeline_failure"
                batch["pipeline_failure"] = {"case_id": case_id, "retrieval_mode": mode,
                                              "error_type": row.get("error_type", "unknown"),
                                              "retrieval_quality_failure": False}
                batch["unexecuted"] = [
                    {"case_id": c["id"], "retrieval_mode": m}
                    for c in cases for m in MODE_ORDER
                    if (c["id"], m) not in completed and not any(
                        x["case_id"] == c["id"] and x["retrieval_mode"] == m for x in batch["executions"])
                ]
                _atomic_json(batch_path, batch)
                return batch
            row = {**row, "case_id": case_id, "retrieval_mode": mode, "status": "completed",
                   "provider": "groq", "model": MODEL, "experiment_hash": metadata["experiment_hash"],
                   "execution_sequence": sequence}
            sequence += 1
            batch["executions"].append(row)
            completed[(case_id, mode)] = row
            batch["pacing"]["wait_seconds"] = round(sum(
                float(item.get("provider_pacing_wait_seconds", 0.0)) for item in batch["executions"]
            ), 3)
            _atomic_json(batch_path, batch)  # Each completed run survives interruption.
        if parity_check:
            try:
                parity_check()
            except Exception as exc:
                return pipeline_stop(case_id, "post_case_parity", exc)
    batch["status"] = "batch_completed"
    _atomic_json(batch_path, batch)
    return batch


def validate_and_merge(batch_paths: list[Path], cases: list[dict]) -> dict:
    """Strictly validate a complete matched experiment before any aggregation."""
    batches = [json.loads(p.read_text(encoding="utf-8")) for p in batch_paths]
    if not batches:
        raise ValueError("no batches supplied")
    metadata = batches[0].get("experiment_metadata")
    if not metadata:
        raise ValueError("batch missing experiment metadata")
    for batch in batches:
        if batch.get("experiment_metadata") != metadata:
            raise ValueError("experiment-defining metadata mismatch; refusing aggregation")
        if batch.get("status") not in {
            "batch_completed", "running", "incomplete_due_to_provider_quota",
            "incomplete_due_to_provider_error", "incomplete_due_to_pipeline_failure",
        }:
            raise ValueError(f"unknown batch status: {batch.get('status')}")
    rows = sorted((row for batch in batches for row in batch.get("executions", [])),
                  key=lambda row: int(row.get("execution_sequence", 0)))
    keys = [(row.get("case_id"), row.get("retrieval_mode")) for row in rows]
    expected = {(case["id"], mode) for case in cases for mode in MODE_ORDER}
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate case/mode execution; refusing aggregation")
    if set(keys) != expected:
        raise ValueError(f"execution coverage incomplete: expected {len(expected)}, found {len(keys)}")
    expected_order = [(case["id"], mode) for case in cases for mode in MODE_ORDER]
    if keys != expected_order:
        raise ValueError("execution order mismatch; each case must run vector_only, graph_only, hybrid")
    if any(row.get("status") != "completed"
           or row.get("experiment_hash") != metadata["experiment_hash"]
           or row.get("provider") != metadata.get("provider")
           or row.get("model") != metadata.get("model") for row in rows):
        raise ValueError("execution status/metadata hash mismatch")
    return {"status": "valid_complete_capture", "expected_executions": len(expected),
            "completed_executions": len(rows), "experiment_metadata": metadata,
            "executions": rows,
            "interrupted_batches": [
                {"status": batch["status"], "provider_failure": batch.get("provider_failure"),
                 "pipeline_failure": batch.get("pipeline_failure"),
                 "unexecuted_count_at_stop": len(batch.get("unexecuted", []))}
                for batch in batches if batch["status"] != "batch_completed"
            ],
            "judge": "not_run:capture_only"}


def expand_batch_paths(arguments: list[Path]) -> list[Path]:
    paths = []
    for item in arguments:
        item = Path(item)
        if item.is_dir():
            paths.extend(sorted(item.glob("batch-*.json")))
        elif any(char in str(item) for char in "*?["):
            paths.extend(sorted(item.parent.glob(item.name)))
        else:
            paths.append(item)
    return paths


def _existing_quota_gate(output: Path, confirmed: bool):
    assert_tpd_reset_confirmed(confirmed=confirmed)
    # Persist the operator acknowledgement with the run; no key or API telemetry is stored here.
    _atomic_json(output / "quota_reset_acknowledgement.json", {
        "confirmed_by_operator": True,
        "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "known_previous_block": {"dimension": "TPD", "limit": 200000, "used": 199554,
                                 "last_request_tokens": 716, "source": "user-provided provider CSV"},
    })


async def _run_live(args):
    # This function is deliberately not called by tests. Provider use requires two
    # explicit operator actions: TPD-reset confirmation and smoke command first.
    from evals import matched_environment as matched
    from evals.groq_matched_capture import GroqCallAudit, _audit, groq_environment
    from evals.provider_guard import ProviderAbort, stop_on_provider_failure
    from evals.run_ragas import evaluate_mode

    output = args.output.resolve()
    _existing_quota_gate(output, args.confirm_tpd_reset)
    environment_dir = output / "preflight"
    environment_dir.mkdir(parents=True, exist_ok=True)
    with groq_environment():
        provider_args = argparse.Namespace(corpus_id=matched.CORPUS_ID, index=matched.INDEX,
            collection="rfp_kb_v2", output_dir=environment_dir, publish=False, capture=False)
        if matched.run(provider_args) != 0:
            raise ValueError("frozen corpus parity preflight failed")
        env = json.loads((environment_dir / "environment.json").read_text(encoding="utf-8"))
        pipeline = matched.FrozenPipeline(matched.INDEX, collection="rfp_kb_v2")
        cases, _lock = matched.validate_frozen(pipeline)
        import agent
        from rfp_analyst.agent.grader import _create_grader_llm
        from rfp_analyst.agent.query_rewriter import _create_rewriter_llm
        from rfp_analyst.agent.router import _create_router_llm

        factories = {"generation": agent.get_llm(), "routing": _create_router_llm(),
                     "grading": _create_grader_llm(), "rewriting": _create_rewriter_llm()}
        provider_selection = {role: {"class": type(llm).__name__,
                                     "model": getattr(llm, "model_name", None) or getattr(llm, "model", None)}
                              for role, llm in factories.items()}
        if any(item != {"class": "ChatGroq", "model": MODEL} for item in provider_selection.values()):
            raise ValueError("existing application factory did not select pinned Groq model")
        pipeline.start_generation()
        metadata = experiment_metadata(pipeline, env)
        run_dir = output / metadata["experiment_hash"]
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_dir / "experiment_metadata.json", metadata)
        _atomic_json(run_dir / "provider_selection.json", {"factories": provider_selection,
            "judge": "not_run:capture_only", "google_provider_calls": 0})
        calls_path = run_dir / "application_calls.json"
        pacer = GroqTPMPacer(output / "groq-tpm-ledger.sqlite3")
        audit = GroqCallAudit(calls_path, pacer=pacer)
        token = _audit.set(audit)
        try:
            if args.command == "smoke":
                selected = [next(c for c in cases if c["id"] == case_id) for case_id in SMOKE_CASE_IDS]
                smoke_path = run_dir / "smoke.json"
                if smoke_path.exists():
                    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
                    if smoke.get("experiment_metadata") != metadata:
                        raise ValueError("smoke metadata mismatch; provider executions cannot be reused")
                    if smoke.get("status") == "passed":
                        print("Matching two-case smoke already passed; no provider calls repeated.")
                        return 0
                else:
                    smoke = {"status": "running", "experiment_metadata": metadata,
                             "required_cases": list(SMOKE_CASE_IDS), "completed": [], "failures": [],
                             "executions": [], "judge": "not_run:capture_only"}
                smoke["status"] = "running"
                _atomic_json(smoke_path, smoke)
                for case in selected:
                    if case["id"] in smoke["completed"]:
                        continue
                    event_start = len(audit.events)
                    try:
                        with stop_on_provider_failure():
                            result = await evaluate_mode(pipeline, [case], "vector_only", judge=None)
                    except (ProviderAbort, QuotaSafetyStop) as exc:
                        if isinstance(exc, QuotaSafetyStop):
                            failure = {"stage": "local_quota_pacer", "provider": "groq", "model": MODEL,
                                       "category": "local_quota_guard", "quota_dimension": exc.dimension,
                                       "detail": exc.detail, "http_status": None}
                        else:
                            failure = exc.failure
                        is_quota = isinstance(exc, QuotaSafetyStop) or failure.get("http_status") == 429
                        smoke["status"] = ("incomplete_due_to_provider_quota" if is_quota
                                           else "incomplete_due_to_provider_error")
                        smoke["failures"].append({"case_id": case["id"], **failure})
                        if isinstance(exc, ProviderAbort):
                            audit.mark_failure(exc.failure)
                        _atomic_json(smoke_path, smoke)
                        return 2
                    row = result["cases"][0]
                    row["smoke"] = True
                    row["provider_call_event_range"] = [event_start, len(audit.events)]
                    smoke["completed"].append(case["id"])
                    smoke.setdefault("executions", []).append(row)
                    _atomic_json(smoke_path, smoke)
                if matched.run(provider_args) != 0:
                    smoke["status"] = "failed_smoke_gate"
                    smoke["manifest_parity"] = False
                    _atomic_json(smoke_path, smoke)
                    return 2
                after_smoke = json.loads((environment_dir / "environment.json").read_text(encoding="utf-8"))
                smoke["manifest_parity"] = (
                    after_smoke.get("comparison", {}).get("parity") is True
                    and after_smoke.get("manifest", {}).get("indexed_digest") == env["manifest"]["indexed_digest"]
                    and after_smoke.get("manifest", {}).get("graph_version") == env["manifest"]["graph_version"]
                )
                stage_pairs = {(e["stage"], e["status"]) for e in audit.events}
                smoke["checks"] = {
                    "all_required_stages_ran": all(any(stage == expected and status == "completed" for stage, status in stage_pairs)
                        for expected in ("route_question", "grade_kb_evidence", "generate_from_kb"))
                        and any(stage in {"grounding_verifier", "final_grounding_verifier"} and status == "completed"
                                for stage, status in stage_pairs),
                    "all_calls_groq_model": all(e["class"] == "ChatGroq" and e["model"] == MODEL and e["status"] == "completed" for e in audit.events),
                    "both_cases_completed": set(smoke["completed"]) == set(SMOKE_CASE_IDS),
                    "both_cases_generated_answers": all(bool(row.get("sample", {}).get("response", "").strip())
                                                          for row in smoke.get("executions", [])),
                    "all_answers_grounded": all(row.get("deterministic", {}).get("grounding_passed") is True
                                                 for row in smoke.get("executions", [])),
                    "manifest_parity_before_after": smoke["manifest_parity"],
                }
                smoke["status"] = "passed" if all(smoke["checks"].values()) else "failed_smoke_gate"
                _atomic_json(smoke_path, smoke)
                print(f"Two-case Groq smoke: {smoke['status']}")
                return 0 if smoke["status"] == "passed" else 2

            smoke = json.loads((run_dir / "smoke.json").read_text(encoding="utf-8"))
            if smoke.get("status") != "passed" or smoke.get("experiment_metadata") != metadata:
                raise ValueError("matching two-case smoke must pass under identical metadata before batches")
            existing = _load_prior(run_dir / "batches", metadata)

            async def execute(case, mode):
                audit.phase = "capture"
                call_start = len(audit.events)
                with stop_on_provider_failure():
                    result = await evaluate_mode(pipeline, [case], mode, judge=None)
                row = result["cases"][0]
                call_events = audit.events[call_start:]
                pacing_wait = sum(float(event.get("pacing_wait_seconds", 0.0)) for event in call_events)
                row["provider_pacing_wait_seconds"] = round(pacing_wait, 3)
                if row.get("pipeline_latency_seconds") is not None:
                    row["pipeline_latency_seconds"] = round(max(
                        0.0, row["pipeline_latency_seconds"] - pacing_wait), 6)
                row["deterministic_failures"] = result["failures"]
                row["query_class"] = case.get("query_class")
                row["benchmark"] = matched.benchmark_checks(case, row, pipeline.snapshot)
                if matched.backend_failure(row):
                    row.update({"status": "error", "error_type": "RetrievalBackendFailure"})
                return row

            def parity_check():
                pipeline.load_current()
                if matched.run(provider_args) != 0:
                    raise ValueError("corpus parity failed during capture")

            result = await execute_batch(cases, metadata=metadata, completed=existing, execute=execute,
                output=run_dir / "batches", batch_size=args.batch_size,
                calls_path=calls_path, parity_check=parity_check)
            print(json.dumps({"status": result["status"], "completed_this_batch": len(result["executions"]),
                              "failure": result.get("provider_failure")}, indent=2))
            return 0 if result["status"] == "batch_completed" else 2
        finally:
            _audit.reset(token)
            audit.checkpoint(strict=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("smoke", "run-batch"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        p.add_argument("--confirm-tpd-reset", action="store_true",
                       help="operator attests Groq daily quota reset; mandatory because TPD is currently exhausted")
        if name == "run-batch":
            p.add_argument("--batch-size", type=int, default=2, help="maximum cases touched in this batch")
    merge = sub.add_parser("merge")
    merge.add_argument("--batches", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    return result


def main():
    args = parser().parse_args()
    if args.command in {"smoke", "run-batch"}:
        if args.command == "run-batch" and args.batch_size < 1:
            raise SystemExit("--batch-size must be positive")
        if not args.confirm_tpd_reset:
            print("Blocked before provider use: confirm that Groq TPD quota has reset.")
            return 2
        return asyncio.run(_run_live(args))
    from evals.matched_environment import validate_frozen
    from evals.ragas_pipeline import FrozenPipeline
    from evals.matched_environment import INDEX

    pipeline = FrozenPipeline(INDEX, collection="rfp_kb_v2")
    cases, _ = validate_frozen(pipeline)
    merged = validate_and_merge(expand_batch_paths(args.batches), cases)
    _atomic_json(args.output, merged)
    print(f"Validated {merged['completed_executions']}/48 execution records: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
