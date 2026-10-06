"""Judge immutable actual outputs, without rerunning generation or retrieval."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from evals.ragas_adapter import AdaptedExecution, EvaluationSample, fingerprint
from evals.ragas_reports import aggregate_cases, version_metadata, write_report
from evals.retrieval_benchmark import LOCK, QUESTIONS, MODE_ORDER, summarize
from evals.run_ragas import load_cases


async def judge_capture(captured, cases, lock, judge, *, output=None):
    if captured["freeze"] != lock or fingerprint(cases) != lock["signature"]["question_set_hash"]:
        raise ValueError("Captured run does not match the frozen dataset")
    if captured.get("judging"):
        raise ValueError("Use the unjudged capture for an independent judge trial")
    reports = copy.deepcopy(captured["reports"])
    metadata = judge.settings.public_metadata()
    for report in reports:
        report["judge"] = metadata
    rows = {r["retrieval_mode"]: {c["id"]: c for c in r["cases"]} for r in reports}
    if set(rows) != set(MODE_ORDER) or any(
        set(r) != {c["id"] for c in cases} for r in rows.values()
    ):
        raise ValueError("Capture must contain exactly the frozen case IDs in all three modes")
    judging = {
        "capture_report_hash": fingerprint(captured),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "versions": version_metadata(),
        "policy": "same captured answers/contexts; no retrieval or generation calls",
    }
    for index, case in enumerate(cases):
        order = MODE_ORDER[index % 3 :] + MODE_ORDER[: index % 3]
        for mode in order:
            row = rows[mode][case["id"]]
            if row["status"] == "error":
                continue  # A failed pipeline is not a fabricated RAGAS sample.
            sample = EvaluationSample.model_validate(row["sample"])
            if sample.user_input != case["question"] or sample.reference != case.get("reference"):
                raise ValueError("Captured question/reference changed")
            if row["generation_kind"] in {"llm_kb", "llm_web"} and (
                not sample.retrieved_contexts or not row["provenance"].get("generation_prompt_hash")
            ):
                raise ValueError("Incomplete original generation capture")
            execution = AdaptedExecution(
                sample, row["generation_kind"], row["deterministic"], row["provenance"]
            )
            print(f"judge {index + 1}/{len(cases)} {case['id']} {mode}", flush=True)
            start = time.perf_counter()
            row["scores"] = await judge.score(execution)
            row["judge_latency_seconds"] = round(time.perf_counter() - start, 6)
            report = next(r for r in reports if r["retrieval_mode"] == mode)
            for metric, score in row["scores"].items():
                if score["status"] == "error":
                    report["failures"].append(
                        {
                            "case_id": case["id"],
                            "stage": "judge",
                            "metric": metric,
                            "error_type": score["error_type"],
                        }
                    )
            for report in reports:
                report["aggregate"] = aggregate_cases(report["cases"])
            if output:
                checkpoint = summarize(
                    reports, cases, lock, captured["preflight"] | {"judge": "in_progress"}
                )
                checkpoint["judging"] = judging
                write_report(output, checkpoint)
    result = summarize(reports, cases, lock, captured["preflight"] | {"judge": "attempted"})
    result["judging"] = judging | {"finished_at": datetime.now(timezone.utc).isoformat()}
    if output:
        write_report(output, result)
    return result


async def run(args):
    from evals.ragas_judge import JudgeSettings, RagasJudge

    if not args.allow_judge:
        raise ValueError("Paid semantic judging requires --allow-judge")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Preserve the unjudged capture in a separate file")
    captured = json.loads(args.input.read_text(encoding="utf-8"))
    cases = load_cases(args.questions, None)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    settings = JudgeSettings.from_environment()
    if len(cases) * 3 > settings.max_cases:
        raise ValueError("Captured cases exceed the explicit judge budget")
    judge = await RagasJudge.create(settings)
    try:
        return await judge_capture(captured, cases, lock, judge, output=args.output)
    finally:
        await judge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--allow-judge", action="store_true")
    result = asyncio.run(run(parser.parse_args()))
    print(
        json.dumps(
            {r["retrieval_mode"]: r["aggregate"]["metrics"] for r in result["reports"]}, indent=2
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
