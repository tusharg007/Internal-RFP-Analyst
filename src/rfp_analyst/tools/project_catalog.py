"""Deterministic helpers for project inventory questions.

Broad questions such as "list all projects with their timelines" are a poor fit
for free-form generation: every project must be represented and each table row
must remain independently grounded.  This module extracts the repeated timeline
fields from retrieved project documents and renders a citation-safe table.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping


_PHASE_LINE = re.compile(
    r"^(?:[-*\u2022]\s*)?(Phase\s+\d+(?:\s*\([^)]*\))?\s*:[^\n]+)$",
    re.IGNORECASE,
)
_DURATION_LINE = re.compile(r"^Total\s+Duration\s*:\s*(.+)$", re.IGNORECASE)


def _project_title(source: str) -> str:
    stem = Path(source).stem
    stem = re.sub(r"^\d+[\s_-]*", "", stem)
    return re.sub(r"[_-]+", " ", stem).strip() or source


def _page_number(document: Mapping[str, Any]) -> int:
    metadata = document.get("metadata") or {}
    raw_page = metadata.get("page", document.get("page", 0))
    try:
        return int(raw_page) + 1
    except (TypeError, ValueError):
        return 1


def _clean_cell(value: str) -> str:
    return " ".join(value.split()).replace("|", r"\|")


def _extract_timeline(documents: Iterable[Mapping[str, Any]]) -> tuple[list[str], str, int]:
    phases: list[str] = []
    duration = ""
    citation_page = 1
    citation_page_selected = False

    for document in documents:
        content = str(document.get("content", ""))
        if not citation_page_selected and re.search(
            r"Timeline\s*&\s*Milestones|Total\s+Duration", content, re.IGNORECASE
        ):
            citation_page = _page_number(document)
            citation_page_selected = True

        for raw_line in content.splitlines():
            line = " ".join(raw_line.strip().split())
            phase_match = _PHASE_LINE.match(line)
            if phase_match:
                phase = phase_match.group(1)
                if phase.casefold() not in {item.casefold() for item in phases}:
                    phases.append(phase)
                continue

            duration_match = _DURATION_LINE.match(line)
            if duration_match and not duration:
                duration = duration_match.group(1).strip()

    return phases, duration, citation_page


def build_project_timeline_catalog(
    documents: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a grounded Markdown table from timeline-bearing project chunks."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for document in documents:
        source = str(document.get("source", "Unknown"))
        grouped.setdefault(source, []).append(document)

    rows: list[str] = []
    included_sources: list[str] = []
    for source in sorted(grouped, key=str.casefold):
        source_documents = sorted(
            grouped[source],
            key=lambda item: (_page_number(item), str(item.get("chunk_id", ""))),
        )
        phases, duration, page = _extract_timeline(source_documents)
        if not phases and not duration:
            continue

        timeline = "<br>".join(_clean_cell(phase) for phase in phases)
        timeline = timeline or "Not stated in retrieved evidence"
        duration = _clean_cell(duration) if duration else "Not stated in retrieved evidence"
        citation = f"[Source: {source}, Page {page}]"
        rows.append(
            f"| {_clean_cell(_project_title(source))} | {timeline} | {duration} | {citation} |"
        )
        included_sources.append(source)

    if not rows:
        return {"answer_markdown": "", "sources": [], "project_count": 0}

    answer = "\n".join(
        [
            "## Projects & Their Timelines",
            "",
            "| Project | Timeline & Milestones | Total Duration | Source |",
            "| --- | --- | --- | --- |",
            *rows,
        ]
    )
    return {
        "answer_markdown": answer,
        "sources": included_sources,
        "project_count": len(included_sources),
    }


__all__ = ["build_project_timeline_catalog"]
