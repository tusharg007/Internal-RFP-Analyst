"""Paired retrieval ablation on the same frozen cases and corpus."""

from __future__ import annotations

import asyncio
import json

from evals.ragas_adapter import MODES
from evals.ragas_reports import cohort, setup_failure_report, write_report
from evals.run_ragas import evaluate_mode, parser, run_cli


def summarize_comparison(reports):
    if len(reports) != 3 or {r["retrieval_mode"] for r in reports} != set(MODES.values()):
        raise ValueError("Comparison requires all three distinct modes")
    if any(cohort(report) != cohort(reports[0]) for report in reports[1:]):
        raise ValueError("Comparison inputs are not frozen/matched")
    return {
        "schema_version": "1.0",
        "timestamp": reports[0]["timestamp"],
        "cohort": cohort(reports[0]),
        "reports": reports,
        "failures": [
            dict(failure, retrieval_mode=report["retrieval_mode"])
            for report in reports
            for failure in report["failures"]
        ],
        "aggregate": {report["retrieval_mode"]: report["aggregate"] for report in reports},
        "warning": "Compare score means together with scored/NA/error counts and deterministic mode correctness.",
    }


async def compare_modes(pipeline, cases, *, judge=None, output=None):
    reports = []
    for mode in MODES.values():
        mode_output = output.with_name(f"{output.stem}-{mode}.json") if output else None
        reports.append(await evaluate_mode(pipeline, cases, mode, judge=judge, output=mode_output))
    result = summarize_comparison(reports)
    if output:
        write_report(output, result)
    return result


def main():
    args = parser(comparison=True).parse_args()
    try:
        report = asyncio.run(run_cli(args, comparison=True))
    except Exception as exc:
        write_report(args.output, setup_failure_report(exc, "comparison"))
        print(f"Comparison setup failed ({type(exc).__name__}); check configuration.")
        return 2
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report: {args.output.resolve()}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
