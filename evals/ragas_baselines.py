"""Calibrate measured pilot baselines and gate comparable subsequent runs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from evals.ragas_adapter import METRICS, fingerprint, metric_score_range
from evals.ragas_reports import aggregate_cases, cohort, write_report


def baseline_cohort(report):
    data = cohort(report)
    # A code upgrade is the object of a regression check. Judge/runtime versions,
    # corpus, references, question set and generation recipe must remain fixed.
    data["versions"] = {"packages": data["versions"]["packages"]}
    data["retrieval_mode"] = report["retrieval_mode"]
    return data


def validate_report(report):
    if report.get("judge", {}).get("status") == "not_run" or not report.get("judge", {}).get(
        "model"
    ):
        raise ValueError("Baseline requires a live measured judge run, not capture-only")
    rows = report.get("cases", [])
    if [row["id"] for row in rows] != report["case_ids"] or len(set(report["case_ids"])) != len(
        rows
    ):
        raise ValueError("Baseline report is incomplete or has duplicate cases")
    if report.get("failures") or any(row["status"] != "completed" for row in rows):
        raise ValueError("Failed pilot runs cannot establish a successful baseline")
    for row in rows:
        if set(row.get("scores", {})) != set(METRICS):
            raise ValueError("Report omits a metric")
        for name, value in row["scores"].items():
            lower, upper = metric_score_range(name)
            if value["status"] not in {"scored", "not_applicable"}:
                raise ValueError("Incomplete judge result")
            if value["status"] == "scored" and (
                not isinstance(value["score"], (int, float))
                or not math.isfinite(value["score"])
                or not lower <= value["score"] <= upper
            ):
                raise ValueError("Invalid baseline score")


def calibrate(reports: list[dict], *, tolerance: float) -> dict:
    """Observed lower envelope minus an explicitly approved absolute tolerance."""
    if len(reports) < 2 or len({fingerprint(r) for r in reports}) != len(reports):
        raise ValueError("Collect at least two distinct pilot reports")
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("Approve an absolute tolerance between zero and one")
    for report in reports:
        validate_report(report)
    expected = baseline_cohort(reports[0])
    if any(baseline_cohort(r) != expected for r in reports[1:]):
        raise ValueError("Pilot cohorts do not match")
    aggregates = [aggregate_cases(report["cases"]) for report in reports]
    metric_gates = {}
    for metric in METRICS:
        applicable = [
            {
                row["id"]: row["scores"][metric].get("variant")
                for row in report["cases"]
                if row["scores"][metric]["status"] == "scored"
            }
            for report in reports
        ]
        if any(ids != applicable[0] for ids in applicable[1:]):
            raise ValueError("Pilot applicability changed; investigate before calibration")
        if not applicable[0]:
            metric_gates[metric] = {
                "status": "not_calibrated",
                "reason": "no_applicable_scored_cases",
            }
            continue
        observed = [a["metrics"][metric]["mean"] for a in aggregates]
        metric_gates[metric] = {
            "status": "calibrated",
            "observed_means": observed,
            "mean_floor": max(metric_score_range(metric)[0], min(observed) - tolerance),
            "scored_cases": applicable[0],
        }
    if not any(g["status"] == "calibrated" for g in metric_gates.values()):
        raise ValueError("No semantic baseline was measured")
    deterministic_gates = {
        name: {
            "mean_floor": min(a["deterministic"][name]["mean"] for a in aggregates),
            "evaluated_floor": min(a["deterministic"][name]["evaluated"] for a in aggregates),
        }
        for name in aggregates[0]["deterministic"]
        if all(a["deterministic"][name]["mean"] is not None for a in aggregates)
    }
    return {
        "schema_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "calibrated",
        "cohort": expected,
        "approved_absolute_tolerance": tolerance,
        "method": "minimum observed pilot aggregate minus approved absolute tolerance",
        "pilot_report_hashes": [fingerprint(r) for r in reports],
        "pilot_versions": [r["versions"] for r in reports],
        "metrics": metric_gates,
        "deterministic": deterministic_gates,
    }


def check_regression(report, baseline):
    failures = []
    if baseline.get("status") != "calibrated" or baseline_cohort(report) != baseline.get("cohort"):
        return {"passed": False, "failures": ["baseline_cohort_mismatch_or_uncalibrated"]}
    try:
        validate_report(report)
    except ValueError:
        return {"passed": False, "failures": ["incomplete_or_failed_run"]}
    aggregate = aggregate_cases(report["cases"])
    for metric, gate in baseline["metrics"].items():
        if gate["status"] != "calibrated":
            continue
        current = {
            row["id"]: row["scores"][metric].get("variant")
            for row in report["cases"]
            if row["scores"][metric]["status"] == "scored"
        }
        if current != gate["scored_cases"]:
            failures.append(f"{metric}:applicability_or_coverage_changed")
        elif aggregate["metrics"][metric]["mean"] < gate["mean_floor"]:
            failures.append(f"{metric}:below_measured_floor")
    for name, gate in baseline["deterministic"].items():
        actual = aggregate["deterministic"].get(name, {})
        if (
            actual.get("mean") is None
            or actual["mean"] < gate["mean_floor"]
            or actual.get("evaluated", 0) < gate["evaluated_floor"]
        ):
            failures.append(f"{name}:deterministic_regression")
    return {"passed": not failures, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("calibrate")
    pilot.add_argument("reports", nargs="+", type=Path)
    pilot.add_argument("--tolerance", type=float, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    gate = sub.add_parser("check")
    gate.add_argument("report", type=Path)
    gate.add_argument("--baseline", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        result = (
            calibrate([read(path) for path in args.reports], tolerance=args.tolerance)
            if args.command == "calibrate"
            else check_regression(read(args.report), read(args.baseline))
        )
    except (ValueError, KeyError, OSError) as exc:
        print(
            f"Baseline operation rejected ({type(exc).__name__}); verify complete matching pilot reports."
        )
        return 2
    write_report(args.output, result)
    print(json.dumps(result if args.command == "check" else {"status": result["status"]}))
    return 1 if result.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
