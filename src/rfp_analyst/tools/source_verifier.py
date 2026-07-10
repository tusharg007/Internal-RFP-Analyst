"""Grounding verification helpers."""

from __future__ import annotations

import re


_CITATION_PATTERN = re.compile(r"\[Source:\s*(?P<source>[^,\]]+)\s*,\s*Page\s*(?P<page>\d+)\]")


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9]{4,}", text.lower())}


def verify_answer_grounding(answer: str, supporting_documents: list[object]) -> dict:
    """Check whether the answer's major claims are backed by cited evidence."""
    source_index = {}
    for document in supporting_documents:
        source_name = document.metadata.get("source_file", "Unknown")
        source_index.setdefault(source_name, []).append(document.page_content)

    unsupported_claims = []
    checked_claims = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if len(line) <= 20 or line.startswith("##"):
            continue

        line_citations = [match.group("source").strip() for match in _CITATION_PATTERN.finditer(line)]
        line_without_citations = _CITATION_PATTERN.sub("", line)
        sentence_candidates = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", line_without_citations) if segment.strip()]

        for claim in sentence_candidates:
            if len(claim) <= 20:
                continue
            checked_claims.append(claim)
            claim_tokens = _tokenize(claim)
            if not claim_tokens:
                continue

            candidate_sources = line_citations or list(source_index)
            supported = False
            for source_name in candidate_sources:
                combined_source_text = " ".join(source_index.get(source_name, []))
                source_tokens = _tokenize(combined_source_text)
                if len(claim_tokens.intersection(source_tokens)) >= 2:
                    supported = True
                    break

            if not supported:
                unsupported_claims.append(claim)

    return {
        "is_grounded": not unsupported_claims,
        "checked_claims": checked_claims,
        "unsupported_claims": unsupported_claims,
    }
