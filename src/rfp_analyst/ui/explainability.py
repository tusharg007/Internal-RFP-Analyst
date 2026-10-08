"""Bounded operational explanations, not model reasoning or invented lineage."""

from html import escape


def trace_card_html(title: str, summary: str, result: str = "") -> str:
    """Escape source/query text before placing it inside our static UI markup."""
    output = f"<br>{escape(str(result))}" if result else ""
    return (
        '<div class="reasoning-box">'
        f"<strong>{escape(str(title))}</strong><br>"
        f"<em>{escape(str(summary))}</em>{output}</div>"
    )


def query_explanation(payload: dict) -> dict:
    """Export only recorded decisions/IDs; never prompts, text, secrets or clients."""
    paths = payload.get("graph_paths", []) or []
    evidence = payload.get("generation_evidence", []) or []
    path_keys = (
        "path_id", "query_type", "subject_ids", "entity_ids", "assertion_ids",
        "evidence_ids", "chunk_ids", "business_hops", "interpretation",
    )
    evidence_keys = ("chunk_id", "evidence_id", "source", "page", "document_origin")
    relationship_keys = ("subject_id", "predicate", "object_id", "assertion_id", "evidence_id", "modality")
    recorded_paths = []
    for path in paths[:20]:
        recorded = {k: path[k] for k in path_keys if k in path}
        recorded["relationships"] = [
            {k: relation[k] for k in relationship_keys if k in relation}
            for relation in (path.get("relationships", []) or [])[:30]
        ]
        recorded_paths.append(recorded)
    checks = [
        {"stage": step.get("tool"), "status": step.get("verification_status")}
        for step in payload.get("traces", [])
        if step.get("tool") in {"grounding_verifier", "final_grounding_verifier"}
    ]
    return {
        "requested_mode": payload.get("requested_retrieval_mode", "unknown"),
        "effective_mode": payload.get("retrieval_mode", "unknown"),
        "intent": payload.get("intent", "unknown"),
        "scope": payload.get("retrieval_scope", "unknown"),
        "graph_query_type": payload.get("graph_query_type", "unsupported"),
        "graph_version": payload.get("graph_version", ""),
        "graph_fallback_reason": payload.get("graph_fallback_reason", ""),
        "graph_path_count": len(paths),
        "graph_paths": recorded_paths,
        "graph_entities": [
            {k: entity[k] for k in ("entity_id", "kind") if k in entity}
            for entity in (payload.get("graph_entities", []) or [])[:50]
        ],
        "generation_kind": payload.get("generation_kind", "not_generated"),
        "generation_prompt_hash": payload.get("generation_prompt_hash", ""),
        "generation_evidence": [{k: e[k] for k in evidence_keys if k in e} for e in evidence[:20]],
        "kb_grade": payload.get("kb_grade", "not_evaluated"),
        "web_grade": payload.get("web_grade", "not_evaluated"),
        "retry_count": payload.get("retry_count", 0),
        "grounded": payload.get("grounded"),
        "grounding_checks": checks,
        "claim_to_chunk_links": "not_recorded",
    }
