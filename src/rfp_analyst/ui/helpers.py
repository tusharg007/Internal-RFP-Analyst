"""UI helpers for Streamlit rendering."""

from __future__ import annotations


def get_chat_avatar(role: str) -> str:
    """Return a Streamlit-safe avatar for a given chat role."""
    return "👤" if role == "user" else "🤖"


def format_latency_display(metrics: dict) -> str:
    """Render evaluation latency safely with correct units."""
    if not metrics:
        return "N/A"

    if "average_latency_ms" in metrics:
        return f"{float(metrics['average_latency_ms']):.2f} ms"

    latency_value = float(metrics.get("average_latency", 0.0))
    if latency_value >= 1:
        return f"{latency_value:.2f} ms"
    return f"{latency_value:.3f} s"
