"""Helpers for rendering evaluation snapshots safely."""

import json
from pathlib import Path


def format_latency(value, unit: str | None = None) -> str:
    """Format latency consistently, converting seconds to ms when appropriate."""
    if value is None:
        return "N/A"

    numeric_value = float(value)
    normalized_unit = (unit or "").lower()
    if normalized_unit in {"s", "sec", "secs", "second", "seconds"} or (
        not normalized_unit and numeric_value < 10
    ):
        milliseconds = numeric_value * 1000
        if milliseconds < 1000:
            return f"{milliseconds:.0f} ms"
        return f"{numeric_value:.2f} s"

    return f"{numeric_value:.2f} ms"


def load_eval_snapshot(results_path: Path, missing_message: str = "No evaluation run found") -> dict:
    """Load a local evaluation snapshot if one exists."""
    if not results_path.exists():
        return {"status": "missing", "message": missing_message}

    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    latency_value = payload.get("latency")
    latency_unit = payload.get("latency_unit")
    return {
        "status": "ready",
        "message": "Evaluation snapshot loaded",
        "payload": payload,
        "latency_display": format_latency(latency_value, latency_unit),
    }
