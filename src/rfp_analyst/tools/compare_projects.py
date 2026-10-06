"""Project-comparison tool."""

from __future__ import annotations

import re
from pathlib import Path

from config import RETRIEVAL_K
from rfp_analyst.tools.search_kb import search_knowledge_base


_FIELDS = {
    "timeline": "Timeline & Milestones",
    "budget": "Budget Range",
    "tech_stack": "Technology Stack",
    "outcomes": "Key Outcomes",
}


def _extract_field(text: str, label: str) -> str:
    # Phase/Total Duration lines are field contents, not new section headings.
    headings = "|".join(re.escape(name) for name in (*_FIELDS.values(), "Team Composition"))
    pattern = re.compile(
        rf"{re.escape(label)}[:\s-]*(.+?)(?=\n(?:{headings})\b|$)",
        re.I | re.S,
    )
    match = pattern.search(text)
    if match:
        return " ".join(match.group(1).split())
    # A hydrated witness may start after the section heading due to chunking.
    # Recover only the explicit typed line, never infer values from phase dates.
    typed_label = {
        "Timeline & Milestones": "Total Duration",
        "Budget Range": "Estimated project cost",
    }.get(label)
    if typed_label:
        typed = re.search(rf"(?im)^\s*{re.escape(typed_label)}:\s*[^\n]+", text)
        if typed:
            return typed.group(0).strip()
    return "Not found in retrieved evidence"


def compare_projects(query: str, search_fn=None, k: int = RETRIEVAL_K) -> dict:
    """Compare projects across key delivery dimensions."""
    search_result = search_knowledge_base(query, k=k, search_fn=search_fn)
    grouped = {}
    for document in search_result["documents"]:
        source = document.metadata.get("source_file", "Unknown")
        grouped.setdefault(source, []).append(document.page_content)

    from rfp_analyst.retrieval.decisions import plan_retrieval, matches_project_reference

    plan = plan_retrieval(query).plan
    requested = tuple(field for field in ("timeline", "budget") if field in plan.fields)
    documents = list(search_result["documents"])
    # A bounded field-comparison lookup can request missing fields separately.
    # Keep only chunks from the selected source, never substitute another project.
    if plan.query_type == "project_fields" and len(plan.project_refs) == 2:
        selected = set()
        titles = {}
        for ref in plan.project_refs:
            # Short filenames need not contain the project name. Match bounded
            # first-page title lines as well, not arbitrary later-page passages.
            sources = []
            for source in grouped:
                title_lines = [
                    line.strip()
                    for doc in documents
                    if doc.metadata.get("source_file") == source and doc.metadata.get("page") == 0
                    for line in doc.page_content.splitlines()[:5]
                    if len(line.strip()) <= 200 and ":" not in line and "|" not in line
                ]
                matching_titles = [
                    title for title in title_lines if matches_project_reference(ref, title)
                ]
                if matches_project_reference(ref, source) or matching_titles:
                    sources.append(source)
                    if matching_titles:
                        titles[source] = matching_titles[0]
            if len(sources) == 1:
                selected.add(sources[0])
        grouped = {source: contents for source, contents in grouped.items() if source in selected}
        documents = [doc for doc in documents if doc.metadata.get("source_file") in selected]
        for source in list(grouped)[:2]:
            title = titles.get(source) or re.sub(r"^\d+[_ -]*", "", Path(source).stem).replace(
                "_", " "
            )
            for field in requested:
                if (
                    _extract_field("\n".join(grouped[source]), _FIELDS[field])
                    != "Not found in retrieved evidence"
                ):
                    continue
                result = search_knowledge_base(f"{field} for {title}", k=k, search_fn=search_fn)
                for doc in result["documents"]:
                    if doc.metadata.get("source_file") == source and doc not in documents:
                        documents.append(doc)
                        grouped[source].append(doc.page_content)

    rows = []
    evidence_lines = []
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
        for field in requested:
            for doc in documents:
                if doc.metadata.get("source_file") != source:
                    continue
                value = _extract_field(doc.page_content, _FIELDS[field])
                if value != "Not found in retrieved evidence":
                    page = int(doc.metadata.get("page", 0)) + 1
                    evidence_lines.append(
                        f"{source} — {field}: {value} [Source: {source}, Page {page}]"
                    )
                    break

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
        "comparison_evidence": "\n".join(evidence_lines),
        "documents": documents,
        "sources": search_result["sources"],
    }
