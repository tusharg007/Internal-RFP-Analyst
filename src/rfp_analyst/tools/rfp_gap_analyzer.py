"""RFP requirement extraction and matching tools."""

from __future__ import annotations

import re

from config import RETRIEVAL_K
from rfp_analyst.tools.search_kb import search_knowledge_base

GAP_DIMENSIONS = {
    "scope and source systems": ("scope", "source system", "source systems"),
    "data volume": ("data volume", "records", "terabyte", "gb", "rows"),
    "integrations": ("integration", "integrations", "api", "connector"),
    "security/compliance controls": ("security", "compliance", "hipaa", "control", "controls"),
    "users and dashboard KPIs": ("user", "users", "dashboard", "kpi", "analytics"),
    "timelines and milestones": ("timeline", "milestone", "phase", "phased", "weeks", "months"),
    "SLAs and acceptance criteria": ("sla", "acceptance", "criteria", "uptime"),
    "budget and staffing": ("budget", "staffing", "team", "fte", "cost"),
    "measurable success metrics": ("metric", "metrics", "outcome", "outcomes", "success"),
    "risks and dependencies": ("risk", "risks", "dependency", "dependencies"),
}

FIT_DIMENSIONS = {
    "Azure migration": ("azure", "migration", "cloud"),
    "regulatory/HIPAA alignment": ("hipaa", "regulatory", "compliance", "controls", "audit"),
    "dashboards and analytics": ("dashboard", "dashboards", "analytics", "power bi", "reporting"),
    "phased delivery": ("phase", "phased", "timeline", "milestone", "weeks"),
    "measurable outcomes": ("outcome", "outcomes", "improved", "reduced", "increased", "measurable"),
}


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _infer_requirement_gaps(requirement_text: str) -> list[dict]:
    gaps = []
    for label, keywords in GAP_DIMENSIONS.items():
        if _contains_any(requirement_text, keywords):
            continue
        gaps.append(
            {
                "dimension": label,
                "detail": (
                    f"Inferred gap: the target requirement does not specify {label}; "
                    "treat this as absent or ambiguous information until confirmed."
                ),
                "status": "absent_or_ambiguous",
            }
        )
    return gaps


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
        requirements.append(
            {
                "id": f"REQ-{index:02d}",
                "text": text,
                "inferred_gaps": _infer_requirement_gaps(text),
            }
        )

    gaps = [
        {"requirement_id": requirement["id"], **gap}
        for requirement in requirements
        for gap in requirement.get("inferred_gaps", [])
    ]

    return {
        "requirements": requirements,
        "gaps": gaps,
        "summary": f"Extracted {len(requirements)} requirement(s).",
    }


def _score_case_study(match: dict) -> dict:
    evidence_text = " ".join(match.get("snippets", []))
    dimension_scores = {}
    missing_coverage = []
    for dimension, keywords in FIT_DIMENSIONS.items():
        score = 1 if _contains_any(evidence_text, keywords) else 0
        dimension_scores[dimension] = score
        if not score:
            missing_coverage.append(dimension)
    total_score = sum(dimension_scores.values())
    return {
        "fit_score": total_score,
        "dimension_scores": dimension_scores,
        "missing_coverage": missing_coverage,
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
                {
                    "source": source["source"],
                    "pages": set(),
                    "matched_requirements": [],
                    "snippets": [],
                    "citations": set(),
                },
            )
            entry["pages"].add(source["page"])
            entry["matched_requirements"].append(requirement["id"])
            entry["snippets"].append(source["snippet"])
            entry["citations"].add((source["source"], source["page"]))
        supporting_documents.extend(result["documents"])

    normalized_matches = []
    for item in matches.values():
        scoring = _score_case_study(item)
        normalized_matches.append(
            {
                "source": item["source"],
                "pages": sorted(item["pages"]),
                "matched_requirements": sorted(set(item["matched_requirements"])),
                "snippets": item["snippets"][:3],
                "fit_score": scoring["fit_score"],
                "dimension_scores": scoring["dimension_scores"],
                "missing_coverage": scoring["missing_coverage"],
                "citations": [
                    {"source": source, "page": page}
                    for source, page in sorted(item["citations"], key=lambda value: (value[0], value[1]))[:2]
                ],
            }
        )

    normalized_matches.sort(
        key=lambda item: (-item["fit_score"], -len(item["matched_requirements"]), item["source"])
    )
    return {
        "matches": normalized_matches[:3],
        "documents": supporting_documents,
    }
