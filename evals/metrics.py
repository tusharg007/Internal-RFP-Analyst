"""Evaluation metrics for deterministic RAG checks."""

from __future__ import annotations

import re
from statistics import mean

_CITATION_PATTERN = re.compile(r"\[Source:\s*[^,\]]+\s*,\s*Page\s*\d+\]")


def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def compute_retrieval_hit_rate(results: list[dict]) -> float:
    scores = []
    for result in results:
        expected_sources = result.get("expected_sources", [])
        if not expected_sources:
            continue
        retrieved_sources = set(result.get("retrieved_sources", []))
        hit_count = sum(1 for source in expected_sources if source in retrieved_sources)
        scores.append(hit_count / len(expected_sources))
    return _safe_mean(scores)


def compute_citation_coverage(results: list[dict]) -> float:
    scores = []
    for result in results:
        if not result.get("expect_citations", False):
            continue
        answer = result.get("answer", "")
        scores.append(1.0 if _CITATION_PATTERN.search(answer) else 0.0)
    return _safe_mean(scores)


def compute_grounded_answer_score(results: list[dict]) -> float:
    scores = [1.0 if result.get("grounded", False) else 0.0 for result in results]
    return _safe_mean(scores)


def compute_average_latency(results: list[dict]) -> float:
    latencies = [float(result.get("latency_ms", 0.0)) for result in results]
    return _safe_mean(latencies)


def compute_tool_call_count(results: list[dict]) -> float:
    counts = [float(result.get("tool_call_count", 0)) for result in results]
    return _safe_mean(counts)


def compute_failure_rate(results: list[dict]) -> float:
    failures = [1.0 if not result.get("passed", False) else 0.0 for result in results]
    return _safe_mean(failures)


def build_metrics_summary(results: list[dict]) -> dict:
    average_latency_ms = compute_average_latency(results)
    return {
        "retrieval_hit_rate": compute_retrieval_hit_rate(results),
        "citation_coverage": compute_citation_coverage(results),
        "grounded_answer_score": compute_grounded_answer_score(results),
        "average_latency": average_latency_ms,
        "average_latency_ms": average_latency_ms,
        "tool_call_count": compute_tool_call_count(results),
        "failure_rate": compute_failure_rate(results),
    }
