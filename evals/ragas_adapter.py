"""Single-execution adapter and deterministic checks; no optional judge imports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

MODES = {"vector": "vector_only", "graph": "graph_only", "hybrid": "hybrid"}
METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


def metric_score_range(name: str) -> tuple[float, float]:
    # Cosine-based scores can be negative; do not silently turn them into errors.
    return {"answer_relevancy": (-1.0, 1.0), "answer_correctness": (-0.25, 1.0)}.get(name, (0.0, 1.0))


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class EvaluationSample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    user_input: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    response: str
    reference: str | None = None

    def to_ragas(self):
        """Lazy conversion, without a retriever or provider call."""
        from ragas import SingleTurnSample

        return SingleTurnSample(**self.model_dump(exclude_none=True))


@dataclass(frozen=True)
class AdaptedExecution:
    sample: EvaluationSample
    generation_kind: str
    deterministic: dict
    provenance: dict


def _citations(response: str) -> set[tuple[str, int]]:
    import re

    return {
        (source.strip(), int(page))
        for source, page in re.findall(r"\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]", response)
    }


def adapt_execution(case: dict, payload: dict, requested_mode: str) -> AdaptedExecution:
    """Use only boundary capture, never retrieved_documents as a context substitute."""
    if requested_mode not in MODES.values():
        raise ValueError("Unsupported retrieval mode")
    kind = payload.get("generation_kind", "missing_capture")
    if kind == "missing_capture":
        raise ValueError("Pipeline output lacks generation-boundary evidence capture")
    contexts = payload.get("generation_contexts", [])
    if kind in {"llm_kb", "llm_web"} and (
        not contexts or not payload.get("generation_prompt_hash")
    ):
        raise ValueError("Generated evidence answer has incomplete boundary capture")
    reference = case.get("reference")
    if reference is not None and (not isinstance(reference, str) or not reference.strip()):
        raise ValueError("Reference must be nonempty or omitted")
    sample = EvaluationSample(
        user_input=case["question"],
        retrieved_contexts=contexts,
        response=payload.get("answer", ""),
        reference=reference,
    )
    documents = payload.get("retrieved_documents", [])
    sources = {doc.get("source") for doc in documents}
    origins = {doc.get("document_origin") for doc in documents}
    tools = [event.get("tool") for event in payload.get("traces", [])]
    expected_sources = case.get("expected_sources", [])
    actual_citations = _citations(sample.response)
    evidence = payload.get("generation_evidence", [])
    available_citations = {
        (doc.get("source"), doc["page"] + 1) for doc in evidence if isinstance(doc.get("page"), int)
    }
    citation_validity = (
        (
            len(actual_citations & available_citations) / len(actual_citations)
            if actual_citations
            else 0.0
        )
        if kind == "llm_kb"
        else None
    )
    no_answer = (
        kind in {"insufficient", "not_generated"}
        or payload.get("response_mode") == "fallback"
        or any(
            marker in sample.response.lower()
            for marker in (
                "could not find",
                "no grounded evidence",
                "cannot find",
                "not ready",
                "do not have",
            )
        )
    )
    expected_intent = case.get("expected_intent")
    effective_mode = payload.get("retrieval_mode", "vector_only")
    retrieval_applicable = kind != "direct"
    events = payload.get("evaluation_retrieval_events", [])
    mode_ok = (
        effective_mode == requested_mode
        and not payload.get("graph_fallback_reason")
        and all(e.get("mode") == requested_mode and not e.get("fallback_reason") for e in events)
    )
    paths = payload.get("graph_paths", [])
    supplied_chunk_ids = {item.get("chunk_id") for item in evidence}
    supplied_evidence_ids = {item.get("evidence_id") for item in evidence}
    path_ok = (
        all(
            path.get("chunk_ids")
            and path.get("evidence_ids")
            and set(path["chunk_ids"]) <= supplied_chunk_ids
            and set(path["evidence_ids"]) <= supplied_evidence_ids
            for path in paths
        )
        if paths and kind == "llm_kb"
        else None
    )
    checks = {
        "source_coverage": (
            sum(s in sources for s in expected_sources) / len(expected_sources)
            if expected_sources
            else None
        ),
        "tool_correctness": all(tool in tools for tool in case.get("expected_tools", [])),
        "origin_correctness": all(origin in origins for origin in case.get("expected_origins", [])),
        "route_correctness": payload.get("intent") == expected_intent if expected_intent else None,
        "retrieval_mode_correctness": mode_ok if retrieval_applicable else None,
        "no_answer_behavior": no_answer == bool(case.get("expects_no_answer", False)),
        "citation_identity_coverage": citation_validity,
        "grounding_passed": payload.get("grounded"),
        "graph_path_provenance_correctness": path_ok,
        "answer_present": bool(sample.response.strip()),
    }
    return AdaptedExecution(
        sample,
        kind,
        checks,
        {
            "requested_retrieval_mode": requested_mode,
            "retrieval_mode": effective_mode,
            "generation_prompt_hash": payload.get("generation_prompt_hash", ""),
            "generation_auxiliary_context_hash": payload.get(
                "generation_auxiliary_context_hash", ""
            ),
            "generation_evidence": evidence,
            "intent": payload.get("intent"),
            "graph_version": payload.get("graph_version", ""),
            "graph_fallback_reason": payload.get("graph_fallback_reason", ""),
            "graph_paths": payload.get("graph_paths", []),
            "retry_count": payload.get("retry_count", 0),
            "executed_tools": tools,
            "retrieved_sources": sorted(s for s in sources if s),
            "retrieved_origins": sorted(o for o in origins if o),
            "retrieval_events": events,
        },
    )


def applicability(kind: str, sample: EvaluationSample, metric: str) -> str | None:
    if kind not in {"llm_kb", "llm_web"}:
        return f"generation_kind:{kind}"
    if not sample.response.strip():
        return "empty_response"
    if metric in {"context_recall", "answer_correctness"} and sample.reference is None:
        return "reference_missing"
    return None
