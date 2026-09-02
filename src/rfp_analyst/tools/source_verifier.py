"""Grounding verification helpers."""

from __future__ import annotations

import re


_CITATION_PATTERN = re.compile(r"\[Source:\s*(?P<source>[^,\]]+)\s*,\s*Page\s*(?P<page>\d+)\]")
_TABLE_HEADER_CELLS = {
    "#",
    "project",
    "document",
    "timeline",
    "timeline & milestones",
    "total duration",
    "source",
    "citation",
}


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9]{4,}", text.lower())}


def _numeric_tokens(text: str) -> set[str]:
    return {
        token.rstrip(".,")
        for token in re.findall(r"\b\d[\d,.%$-]*\b", text)
        if token.rstrip(".,")
    }


def _normalize_source_name(source: str) -> str:
    """Normalize harmless Markdown escaping used in generated citations."""

    return re.sub(r"\\([_*\[\]()`])", r"\1", str(source).strip())


def _is_markdown_table_scaffolding(line: str) -> bool:
    if not (line.startswith("|") and line.endswith("|")):
        return False
    cells = [cell.strip().lower() for cell in line.strip("|").split("|")]
    if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
        return True
    return bool(cells) and all(cell in _TABLE_HEADER_CELLS for cell in cells)


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
        source_index.setdefault((_normalize_source_name(source_name), page), []).append(content)

    unsupported_claims = []
    checked_claims = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if len(line) <= 20 or line.startswith("#") or _is_markdown_table_scaffolding(line):
            continue
        if "retrieved evidence does not support this claim" in line.lower():
            continue

        line_citations = [
            (_normalize_source_name(match.group("source")), int(match.group("page")))
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
                # The exact cited filename is evidence for document/project
                # identity (including years or identifiers embedded in it).
                source_identity = re.sub(r"[_-]+", " ", source_key[0])
                combined_source_text = f"{source_identity} " + " ".join(source_index.get(source_key, []))
                source_tokens = _tokenize(combined_source_text)
                numeric_claims = _numeric_tokens(claim)
                numeric_evidence = _numeric_tokens(combined_source_text)
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
