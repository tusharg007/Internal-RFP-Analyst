"""RFP requirement extraction and matching tools."""

from __future__ import annotations

import re

from config import RETRIEVAL_K
from rfp_analyst.tools.search_kb import search_knowledge_base


def extract_rfp_requirements(rfp_text: str) -> dict:
    """Turn raw RFP text into a simple structured requirements list."""
    lines = [line.strip(" -\t") for line in rfp_text.splitlines() if line.strip()]
    candidates = []
    for line in lines:
        segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", line) if segment.strip()]
        for segment in segments:
            lower = segment.lower()
            if any(keyword in lower for keyword in ("must", "should", "require", "need", "include", "support")):
                candidates.append(segment)

    if not candidates:
        candidates = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", rfp_text) if segment.strip()]

    requirements = []
    for index, text in enumerate(candidates, start=1):
        requirements.append({"id": f"REQ-{index:02d}", "text": text})

    return {
        "requirements": requirements,
        "summary": f"Extracted {len(requirements)} requirement(s).",
    }


def find_relevant_case_studies(requirements: list[dict], search_fn=None, k: int = RETRIEVAL_K) -> dict:
    """Find matching prior work for a set of requirements."""
    matches = {}
    supporting_documents = []
    for requirement in requirements:
        result = search_knowledge_base(requirement["text"], k=k, search_fn=search_fn)
        for source in result["sources"]:
            entry = matches.setdefault(
                source["source"],
                {"source": source["source"], "pages": set(), "matched_requirements": [], "snippets": []},
            )
            entry["pages"].add(source["page"])
            entry["matched_requirements"].append(requirement["id"])
            entry["snippets"].append(source["snippet"])
        supporting_documents.extend(result["documents"])

    normalized_matches = []
    for item in matches.values():
        normalized_matches.append(
            {
                "source": item["source"],
                "pages": sorted(item["pages"]),
                "matched_requirements": sorted(set(item["matched_requirements"])),
                "snippets": item["snippets"][:3],
            }
        )

    normalized_matches.sort(key=lambda item: (-len(item["matched_requirements"]), item["source"]))
    return {
        "matches": normalized_matches,
        "documents": supporting_documents,
    }
