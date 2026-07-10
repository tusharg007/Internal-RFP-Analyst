"""Project-comparison tool."""

from __future__ import annotations

import re

from config import RETRIEVAL_K
from rfp_analyst.tools.search_kb import search_knowledge_base


_FIELDS = {
    "timeline": "Timeline & Milestones",
    "budget": "Budget Range",
    "tech_stack": "Technology Stack",
    "outcomes": "Key Outcomes",
}


def _extract_field(text: str, label: str) -> str:
    pattern = re.compile(rf"{re.escape(label)}[:\s-]*(.+?)(?:\n[A-Z][A-Za-z &]+[:\s]|$)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match:
        return " ".join(match.group(1).split())
    return "Not found in retrieved evidence"


def compare_projects(query: str, search_fn=None, k: int = RETRIEVAL_K) -> dict:
    """Compare projects across key delivery dimensions."""
    search_result = search_knowledge_base(query, k=k, search_fn=search_fn)
    grouped = {}
    for document in search_result["documents"]:
        source = document.metadata.get("source_file", "Unknown")
        grouped.setdefault(source, []).append(document.page_content)

    rows = []
    for source, chunks in grouped.items():
        combined = "\n".join(chunks)
        rows.append(
            {
                "source": source,
                "timeline": _extract_field(combined, _FIELDS["timeline"]),
                "budget": _extract_field(combined, _FIELDS["budget"]),
                "tech_stack": _extract_field(combined, _FIELDS["tech_stack"]),
                "outcomes": _extract_field(combined, _FIELDS["outcomes"]),
            }
        )

    if not rows:
        comparison_markdown = "No comparable project evidence found."
    else:
        header = "| Project | Timeline | Budget | Tech Stack | Outcomes |"
        separator = "| --- | --- | --- | --- | --- |"
        body = [
            f"| {row['source']} | {row['timeline']} | {row['budget']} | {row['tech_stack']} | {row['outcomes']} |"
            for row in rows
        ]
        comparison_markdown = "\n".join([header, separator, *body])

    return {
        "query": query,
        "comparison_markdown": comparison_markdown,
        "rows": rows,
        "documents": search_result["documents"],
        "sources": search_result["sources"],
    }
