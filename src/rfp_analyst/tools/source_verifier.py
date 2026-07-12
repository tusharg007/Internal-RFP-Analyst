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
        if isinstance(document, dict):
            source_name = document.get("source", "Unknown")
            page = int(document.get("page", 0) or 0) + 1
            content = document.get("content", "")
        else:
            metadata = document.metadata or {}
            source_name = metadata.get("source_file", "Unknown")
            page = int(metadata.get("page", 0) or 0) + 1
            content = document.page_content
        source_index.setdefault((source_name, page), []).append(content)

    unsupported_claims = []
    checked_claims = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if len(line) <= 20 or line.startswith("##"):
            continue
        if "retrieved evidence does not support this claim" in line.lower():
            continue

        line_citations = [
            (match.group("source").strip(), int(match.group("page")))
            for match in _CITATION_PATTERN.finditer(line)
        ]
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
            for source_key in candidate_sources:
                combined_source_text = " ".join(source_index.get(source_key, []))
                source_tokens = _tokenize(combined_source_text)
                numeric_claims = set(re.findall(r"\b\d[\d,.%$-]*\b", claim))
                numeric_evidence = set(re.findall(r"\b\d[\d,.%$-]*\b", combined_source_text))
                numbers_supported = not numeric_claims or numeric_claims.issubset(numeric_evidence)
                if len(claim_tokens.intersection(source_tokens)) >= 2 and numbers_supported:
                    supported = True
                    break

            if not supported:
                unsupported_claims.append(claim)

    return {
        "is_grounded": not unsupported_claims,
        "checked_claims": checked_claims,
        "unsupported_claims": unsupported_claims,
    }
