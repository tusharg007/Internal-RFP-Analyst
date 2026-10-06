"""Capture evidence at the generation boundary, never by re-retrieving it."""

from __future__ import annotations

import hashlib


def empty_capture(kind: str = "not_generated") -> dict:
    return {
        "generation_contexts": [],
        "generation_kind": kind,
        "generation_prompt_hash": "",
        "generation_evidence": [],
        "generation_auxiliary_context_hash": "",
    }


def capture_generation(state: dict, prompt: str, *, web: bool = False) -> dict:
    context = str(state.get("web_results" if web else "retrieval_context", "") or "")
    documents = [] if web else state.get("retrieved_documents", [])
    blocks = []
    for doc in documents:
        page = doc.get("page", 0)
        page = page + 1 if isinstance(page, int) else page
        blocks.append(
            f"[Source: {doc.get('source', 'Unknown')}, Page {page}]\n{doc.get('content', '')}"
        )
    # Only split when the complete supplied string matches exactly; do not trim,
    # compact or substitute the broader candidate set for the generation context.
    contexts = (
        blocks
        if blocks and "\n\n---\n\n".join(blocks) == context
        else ([context] if context else [])
    )
    if any(block not in prompt for block in contexts):
        raise ValueError("Generation evidence does not match the supplied prompt")
    notes = str(state.get("specialized_notes", "") or "")
    return {
        "generation_contexts": contexts,
        "generation_kind": "llm_web" if web else "llm_kb",
        "generation_prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "generation_evidence": [
            {
                key: doc.get(key)
                for key in ("chunk_id", "evidence_id", "source", "page", "document_origin")
            }
            for doc in documents
        ],
        # Derived tool notes are not original evidence and are not judge contexts.
        "generation_auxiliary_context_hash": hashlib.sha256(notes.encode()).hexdigest()
        if notes
        else "",
    }
