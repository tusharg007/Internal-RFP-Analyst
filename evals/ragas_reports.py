"""Strict JSON reports and coverage-aware summaries (no invented semantic thresholds)."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from evals.ragas_adapter import METRICS, fingerprint


def write_report(path: Path, report: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".ragas-", suffix=".json", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def aggregate_cases(cases: list[dict]) -> dict:
    metrics = {}
    for name in METRICS:
        entries = [
            case.get("scores", {}).get(name, {"status": "error", "score": None}) for case in cases
        ]
        scored = [entry["score"] for entry in entries if entry["status"] == "scored"]
        metrics[name] = {
            "mean": sum(scored) / len(scored) if scored else None,
            "scored": len(scored),
            "total_cases": len(cases),
            "errors": sum(e["status"] == "error" for e in entries),
            "not_applicable": sum(e["status"] == "not_applicable" for e in entries),
            "not_run": sum(e["status"] == "not_run" for e in entries),
            "variants": sorted({e["variant"] for e in entries if e.get("variant")}),
        }
    checks = {}
    for name in sorted({key for case in cases for key in case.get("deterministic", {})}):
        values = [case.get("deterministic", {}).get(name) for case in cases]
        present = [value for value in values if value is not None]
        checks[name] = {
            "mean": sum(present) / len(present) if present else None,
            "evaluated": len(present),
            "total_cases": len(cases),
        }
    return {
        "metrics": metrics,
        "deterministic": checks,
        "total_cases": len(cases),
        "pipeline_failures": sum(case["status"] == "error" for case in cases),
        "pipeline_latency_seconds": sum(case.get("pipeline_latency_seconds", 0) for case in cases),
        "judge_latency_seconds": sum(case.get("judge_latency_seconds", 0) for case in cases),
    }


def version_metadata() -> dict:
    packages = {}
    for name in (
        "ragas",
        "instructor",
        "langchain-core",
        "langchain-community",
        "langgraph",
        "fastembed",
        "chromadb",
        "neo4j",
        "groq",
        "google-genai",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown"
    paths = list((root / "src").rglob("*.py")) + list((root / "evals").glob("*.py"))
    paths += [root / name for name in ("config.py", "rag_engine.py", "agent.py")]
    return {
        "git_revision": revision,
        "packages": packages,
        "source_tree_hash": fingerprint(
            {
                str(path.relative_to(root)): fingerprint(path.read_text(encoding="utf-8"))
                for path in sorted(paths)
            }
        ),
    }


def new_report(
    mode: str, *, cases: list[dict], corpus_hash: str, judge: dict, generation: dict
) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "1.0",
        "timestamp": timestamp,
        "retrieval_mode": mode,
        "question_set_hash": fingerprint(cases),
        "corpus_hash": corpus_hash,
        "case_ids": [case["id"] for case in cases],
        "judge": judge,
        "generation": generation,
        "versions": version_metadata(),
        "comparison_policy": {
            "web_search": "disabled",
            "strict_graph_only": True,
            "hybrid_fallback": "recorded_as_mode_failure",
        },
        "baseline_status": "uncalibrated",
        "cases": [],
        "failures": [],
    }


def cohort(report: dict) -> dict:
    """All non-mode inputs that must remain frozen across comparisons/baselines."""
    return {
        key: report[key]
        for key in (
            "schema_version",
            "question_set_hash",
            "corpus_hash",
            "case_ids",
            "judge",
            "generation",
            "versions",
            "comparison_policy",
        )
    }


def setup_failure_report(exc: Exception, mode: str) -> dict:
    from config import get_ragas_settings

    settings = get_ragas_settings()
    return {
        "schema_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "setup_error",
        "retrieval_mode": mode,
        "judge": {"provider": settings["provider"] or None, "model": settings["model"] or None},
        "versions": version_metadata(),
        "error_type": type(exc).__name__,
        "failures": [{"stage": "setup", "error_type": type(exc).__name__}],
        "cases": [],
        "baseline_status": "uncalibrated",
    }
