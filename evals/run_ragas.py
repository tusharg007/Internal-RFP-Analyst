"""Opt-in semantic evaluation of actual Internal RFP Analyst executions."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml

from evals.ragas_adapter import METRICS, MODES, adapt_execution
from evals.ragas_reports import aggregate_cases, new_report, setup_failure_report, write_report

ANNOTATIONS = Path(__file__).with_name("ragas_annotations.yaml")


def load_cases(questions: Path, annotations: Path | None = ANNOTATIONS) -> list[dict]:
    from evals.run_kb_evals import load_golden_cases

    cases = load_golden_cases(questions)
    if not cases or any(not c.get("id") or not c.get("question") for c in cases):
        raise ValueError("Golden cases need unique IDs and questions")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate golden case ID")
    if annotations:
        extras = yaml.safe_load(annotations.read_text(encoding="utf-8")) or {}
        if set(extras) - {case["id"] for case in cases}:
            raise ValueError("Annotations contain unknown case IDs")
        for case in cases:
            extra = extras.get(case["id"], {})
            if set(extra) - {"reference", "reference_source", "expected_intent"}:
                raise ValueError("Annotations may not change frozen golden expectations")
            case.update(extra)
    return cases


async def evaluate_mode(pipeline, cases, mode, *, judge=None, output: Path | None = None):
    judge_metadata = (
        judge.settings.public_metadata()
        if judge
        else {"provider": None, "model": None, "status": "not_run"}
    )
    report = new_report(
        mode,
        cases=cases,
        corpus_hash=pipeline.corpus_hash,
        judge=judge_metadata,
        generation=pipeline.generation_metadata,
    )
    if judge and getattr(judge, "provider_failure", None):
        report.update({"status": "incomplete_due_to_judge_provider_error",
                       "unexecuted_case_ids": [c["id"] for c in cases],
                       "judge_http_status": judge.provider_failure})
        report["aggregate"] = aggregate_cases([])
        if output:
            write_report(output, report)
        return report
    for case in cases:
        start = time.perf_counter()
        row = {
            "id": case["id"],
            "status": "completed",
            "requested_retrieval_mode": mode,
            "question": case["question"],
            "category": case.get("category"),
            "judge_latency_seconds": 0,
        }
        try:
            payload = await asyncio.to_thread(pipeline.execute, case, mode)
            row["pipeline_latency_seconds"] = round(time.perf_counter() - start, 6)
            if case.get("query_class"):
                # Actual post-filter/fusion records from this execution, not a second search.
                row["retrieval_evidence"] = [
                    {
                        key: doc.get(key)
                        for key in (
                            "source",
                            "page",
                            "chunk_id",
                            "evidence_id",
                            "document_origin",
                            "content",
                        )
                    }
                    for doc in payload.get("retrieved_documents", [])
                ]
            execution = adapt_execution(case, payload, mode)
            row.update(
                {
                    "sample": execution.sample.model_dump(),
                    "generation_kind": execution.generation_kind,
                    "deterministic": execution.deterministic,
                    "provenance": execution.provenance,
                }
            )
            for name, value in execution.deterministic.items():
                if value is False:
                    report["failures"].append(
                        {"case_id": case["id"], "stage": "deterministic", "check": name}
                    )
            score_start = time.perf_counter()
            row["scores"] = (
                await judge.score(execution)
                if judge
                else {
                    name: {"status": "not_run", "score": None, "reason": "capture_only"}
                    for name in METRICS
                }
            )
            row["judge_latency_seconds"] = (
                round(time.perf_counter() - score_start, 6) if judge else 0
            )
            for name, result in row["scores"].items():
                if result["status"] == "error":
                    report["failures"].append(
                        {
                            "case_id": case["id"],
                            "stage": "judge",
                            "metric": name,
                            "error_type": result["error_type"],
                            **({"http_status": result["http_status"]}
                               if result.get("http_status") else {}),
                        }
                    )
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "pipeline_latency_seconds": round(time.perf_counter() - start, 6),
                    "scores": {
                        name: {"status": "error", "score": None, "error_type": type(exc).__name__}
                        for name in METRICS
                    },
                }
            )
            report["failures"].append(
                {
                    "case_id": case["id"],
                    "stage": "pipeline_or_adapter",
                    "error_type": type(exc).__name__,
                }
            )
        report["cases"].append(row)
        report["aggregate"] = aggregate_cases(report["cases"])
        provider_error = next((v.get("http_status") for v in row.get("scores", {}).values()
                               if v.get("status") == "error" and v.get("http_status")), None)
        if provider_error:
            report.update({"status": "incomplete_due_to_judge_provider_error",
                           "judge_http_status": provider_error,
                           "unexecuted_case_ids": [c["id"] for c in cases[len(report["cases"]):]]})
        if output:
            write_report(output, report)  # Checkpoint completed cases before any costly next call.
        if provider_error:
            break
    return report


def parser(*, comparison=False):
    result = argparse.ArgumentParser(description=__doc__)
    if not comparison:
        result.add_argument("--mode", choices=MODES, required=True)
    result.add_argument(
        "--questions", type=Path, default=Path(__file__).with_name("golden_questions.yaml")
    )
    result.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    result.add_argument("--no-annotations", action="store_true")
    result.add_argument(
        "--index", type=Path, help="Existing frozen Chroma index; never rebuilt by evaluation"
    )
    result.add_argument("--collection", help="Existing Chroma collection name")
    result.add_argument(
        "--prepare-samples",
        type=Path,
        help="Explicitly create a NEW persistent public-sample workspace",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path("evals/ragas_results") / ("comparison.json" if comparison else "results.json"),
    )
    result.add_argument(
        "--capture-only",
        action="store_true",
        help="Run real generation but no semantic judge calls",
    )
    result.add_argument(
        "--allow-judge",
        action="store_true",
        help="Consent to send captured evidence/answers to configured paid judge",
    )
    return result


async def run_cli(args, *, comparison=False):
    from config import VECTORSTORE_DIR
    from evals.ragas_judge import JudgeSettings, RagasJudge
    from evals.ragas_pipeline import FrozenPipeline, prepare_sample_workspace

    if not args.capture_only and not args.allow_judge:
        raise ValueError("Use --allow-judge for paid semantic evaluation, or --capture-only")
    if args.index and args.prepare_samples:
        raise ValueError("Choose an existing index or sample preparation, not both")
    cases = load_cases(args.questions, None if args.no_annotations else args.annotations)
    settings = None if args.capture_only else JudgeSettings.from_environment()
    if len(cases) * (3 if comparison else 1) > (settings.max_cases if settings else 30):
        raise ValueError("Question set exceeds configured case budget")
    index = (
        prepare_sample_workspace(args.prepare_samples)
        if args.prepare_samples
        else (args.index or VECTORSTORE_DIR)
    )
    pipeline = FrozenPipeline(index, collection=args.collection)
    pipeline.start_generation()
    judge = None
    try:
        judge = await RagasJudge.create(settings) if settings else None
        if comparison:
            from evals.compare_retrieval_modes import compare_modes

            report = await compare_modes(pipeline, cases, judge=judge, output=args.output)
        else:
            report = await evaluate_mode(
                pipeline, cases, MODES[args.mode], judge=judge, output=args.output
            )
        return report
    finally:
        if judge:
            await judge.close()


def main():
    args = parser().parse_args()
    try:
        report = asyncio.run(run_cli(args))
    except Exception as exc:
        # Preserve setup failure without SDK exception messages/secrets.
        report = setup_failure_report(exc, MODES[args.mode])
        write_report(args.output, report)
        print(
            f"Evaluation setup failed ({type(exc).__name__}); check index, optional dependencies and judge settings."
        )
        return 2
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report: {args.output.resolve()}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
