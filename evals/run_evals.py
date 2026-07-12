"""Offline smoke evaluation using a mock corpus and deterministic synthesis."""

from __future__ import annotations

import json
import time
from pathlib import Path

from config import OFFLINE_SMOKE_EVAL_RESULTS_PATH

MOCK_CORPUS = {
    "banking": "The banking audit project used Python, Azure, and Streamlit.",
    "healthcare": "The healthcare migration project used Azure data services.",
    "insurance": "The insurance automation project focused on claims workflows.",
}

GOLDEN_CASES = [
    {
        "question": "What tech stack did we use for banking?",
        "expected_keywords": ["Python", "Azure", "Streamlit"],
        "topic": "banking",
    },
    {
        "question": "Which project used Azure services?",
        "expected_keywords": ["Azure"],
        "topic": "healthcare",
    },
    {
        "question": "What was automated in insurance?",
        "expected_keywords": ["claims", "workflows"],
        "topic": "insurance",
    },
]


def synthesize_offline_answer(topic: str) -> str:
    """Return a deterministic mock answer for the smoke evaluation."""
    return MOCK_CORPUS[topic]


def run_offline_smoke_eval(output_path: Path = OFFLINE_SMOKE_EVAL_RESULTS_PATH) -> dict:
    """Run the existing mock-style smoke evaluation and persist a labeled snapshot."""
    start = time.perf_counter()
    results = []
    correct = 0

    for case in GOLDEN_CASES:
        answer = synthesize_offline_answer(case["topic"])
        passed = all(keyword.lower() in answer.lower() for keyword in case["expected_keywords"])
        if passed:
            correct += 1
        results.append(
            {
                "question": case["question"],
                "mode": "offline_smoke_eval",
                "answer": answer,
                "passed": passed,
            }
        )

    latency_seconds = time.perf_counter() - start
    payload = {
        "evaluation_name": "Offline Smoke Evaluation",
        "evaluation_type": "offline_smoke_eval",
        "score": f"{correct}/{len(GOLDEN_CASES)}",
        "pass_rate": round(correct / len(GOLDEN_CASES), 2),
        "latency": latency_seconds,
        "latency_unit": "seconds",
        "notes": "Mock corpus and deterministic answer synthesis only. This is a smoke test, not proof of real RAG quality.",
        "cases": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


if __name__ == "__main__":
    summary = run_offline_smoke_eval()
    print(json.dumps(summary, indent=2))
