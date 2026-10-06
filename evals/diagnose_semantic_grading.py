"""Observe ONE unchanged public semantic_modernization execution, never judge it.

Callbacks and a transparent vector-result observer capture existing boundaries.
No prompts, provider parameters, retrieval results or grader outputs are edited.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

from evals.ragas_pipeline import FrozenPipeline
from evals.ragas_reports import write_report

SOURCE = "01_Banking_Sector_Digital_Audit_2024.pdf"
# Explicit source-verified equivalents, not a judge or new benchmark expectations.
CONCEPTS = {
    "digital maturity": r"digital (?:transformation )?maturity",
    "data governance": r"data governance|data quality, lineage, and governance",
    "peer benchmarking": r"benchmark\w*.{0,90}industry peers|benchmarking & peer analysis",
    "prioritized modernization roadmap": r"prioritized (?:transformation|modernization) roadmap",
    "ROI projections": r"roi projections",
}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def concept_check(documents):
    """Require a banking-source quote for each concept; unrelated docs cannot count."""
    witnesses = {}
    for concept, pattern in CONCEPTS.items():
        matches = []
        for document in documents:
            if document.get("source") != SOURCE:
                continue
            normalized = " ".join(document.get("content", "").split())
            match = re.search(pattern, normalized, flags=re.I)
            if match:
                matches.append({"source": SOURCE, "page_number": document.get("page", 0) + 1,
                                "chunk_id": document.get("chunk_id"), "quote": match.group()})
        witnesses[concept] = matches
    missing = [concept for concept, matches in witnesses.items() if not matches]
    return {"sufficiency": "insufficient" if missing else "sufficient",
            "missing_expected_concepts": missing, "witnesses": witnesses}


def page_rows(rows):
    return [{**copy.deepcopy(row), "page_number": row.get("page", 0) + 1} for row in rows]


class GradingObserver(BaseCallbackHandler):
    """Observe node inputs, actual model messages/responses and parser outcomes."""

    raise_error = True

    def __init__(self, output):
        self.output = output
        self.attempts = []
        self.retrievals = []
        self.rewrites = []
        self.chain_stages = {}
        self.models = {}
        self.calls = []
        self.writer_errors = []
        self._checkpoint_lock = threading.RLock()

    def checkpoint(self, *, strict=False):
        # LangGraph callbacks can run on different threads. Diagnostic file I/O
        # must never turn a successful application node into a failed node.
        with self._checkpoint_lock:
            for attempt in range(4):
                try:
                    write_report(self.output / "grading_trace.json", {
                        "attempts": self.attempts, "raw_retrievals": self.retrievals,
                        "rewrites": self.rewrites, "application_calls": self.calls,
                        "diagnostic_writer_errors": self.writer_errors,
                        "page_convention": "page is zero-based; page_number is one-based",
                        "rationale_note": "The unchanged schema/prompt requests grade only; absent rationales remain unknown.",
                    })
                    return
                except OSError as exc:
                    if isinstance(exc, PermissionError) and attempt < 3:
                        time.sleep(0.05 * (attempt + 1))
                        continue
                    self.writer_errors.append({"error_type": type(exc).__name__})
                    if strict:
                        raise
                    return  # In-memory capture survives until the final strict flush.

    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kwargs):
        stage = (metadata or {}).get("langgraph_node")
        self.chain_stages[run_id] = stage
        if stage != "grade_kb_evidence" or not isinstance(inputs, dict):
            return
        if "retrieved_documents" not in inputs or "user_query" not in inputs:
            return
        from rfp_analyst.agent.grader import _format_kb_evidence

        rows = copy.deepcopy(inputs["retrieved_documents"])
        context = _format_kb_evidence(rows)
        signature = (inputs.get("retry_count", 0), inputs.get("current_query"), sha(context))
        if self.attempts and self.attempts[-1]["state_signature"] == list(signature):
            return  # Sequence and callable can both trace the same node input.
        self.attempts.append({
            "attempt": len(self.attempts) + 1, "state_signature": list(signature),
            "original_query": inputs["user_query"], "current_query": inputs.get("current_query"),
            "resolved_query": inputs.get("resolved_query"), "retry_count": inputs.get("retry_count", 0),
            "retrieved_chunks": page_rows(rows), "grader_evidence": context,
            "evidence_hash": sha(context), "concept_check": concept_check(rows),
            "raw_model_response": None, "parsed_grade": None,
            "application_grade": None, "rationale": None, "parser_errors": [],
        })
        self.checkpoint()  # Persist BEFORE the grader model is called.

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        from evals.provider_guard import ProviderAbort

        meta = metadata or {}
        params = kwargs.get("invocation_params", {})
        identity = (serialized or {}).get("id", [])
        classname = identity[-1] if identity else "unknown"
        info = {"stage": meta.get("langgraph_node", "unknown"),
                "provider": meta.get("ls_provider", "unknown"),
                "model": meta.get("ls_model_name") or params.get("model"),
                "class": classname, "status": "started"}
        self.calls.append(info)
        self.models[run_id] = info
        if classname != "ChatGoogleGenerativeAI":
            self.checkpoint()
            raise ProviderAbort({**info, "category": "unexpected_provider", "http_status": None,
                                 "retrieval_quality_failure": False})
        if info["stage"] == "grade_kb_evidence":
            if not self.attempts:
                raise RuntimeError("Grader state boundary was not captured")
            from rfp_analyst.agent.prompts import KB_GRADER_PROMPT

            entry = self.attempts[-1]
            content = [m.content for m in messages[0]]
            if len(content) != 1 or not isinstance(content[0], str):
                raise RuntimeError("Unexpected grader message shape; cannot assert exact capture")
            prompt = content[0]
            entry.update({"model": info["model"], "provider": info["provider"],
                          "prompt_hash": sha(prompt), "prompt_chars": len(prompt),
                          "prompt_matches_captured_evidence": prompt == KB_GRADER_PROMPT.format(
                              question=entry["original_query"], context=entry["grader_evidence"])})
            info["attempt"] = entry["attempt"]
        self.checkpoint()

    def on_llm_end(self, response, *, run_id, **kwargs):
        info = self.models.pop(run_id, None)
        if not info:
            return
        info["status"] = "completed"
        if info["stage"] == "grade_kb_evidence":
            entry = self.attempts[info["attempt"] - 1]
            generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            raw_text = generation.text
            entry["raw_model_response"] = {
                "text": raw_text, "content": copy.deepcopy(getattr(message, "content", None)),
                "response_metadata": copy.deepcopy(getattr(message, "response_metadata", {})),
                "usage_metadata": copy.deepcopy(getattr(message, "usage_metadata", {})),
                "generation_info": copy.deepcopy(generation.generation_info),
            }
            try:
                raw_json = json.loads(raw_text)
            except (ValueError, TypeError):
                raw_json = None
            entry["raw_grade"] = raw_json.get("grade") if isinstance(raw_json, dict) else None
            if isinstance(raw_json, dict):
                entry["rationale"] = raw_json.get("rationale") or raw_json.get("reason")
        self.checkpoint()

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        stage = self.chain_stages.pop(run_id, None)
        if stage == "grade_kb_evidence" and self.attempts:
            if type(outputs).__name__ == "EvidenceGrade":
                self.attempts[-1]["parsed_grade"] = outputs.grade
            if isinstance(outputs, dict) and "kb_grade" in outputs:
                self.attempts[-1]["application_grade"] = outputs["kb_grade"]
        if stage == "rewrite_query" and isinstance(outputs, dict) and "current_query" in outputs:
            self.rewrites.append({"rewritten_query": outputs["current_query"],
                                  "retry_count": outputs.get("retry_count")})
        if stage in {"grade_kb_evidence", "rewrite_query"}:
            self.checkpoint()

    def on_chain_error(self, error, *, run_id, **kwargs):
        if self.chain_stages.pop(run_id, None) == "grade_kb_evidence" and self.attempts:
            self.attempts[-1].setdefault("chain_errors", []).append({"error_type": type(error).__name__})
            if type(error).__name__ in {"OutputParserException", "ValidationError"}:
                self.attempts[-1]["parser_errors"].append({"error_type": type(error).__name__})
            self.checkpoint()


_observer = ContextVar("semantic_grading_diagnostic_observer", default=None)
register_configure_hook(_observer, inheritable=True)


class ObservedPipeline(FrozenPipeline):
    """Return the identical original vector results after recording them locally."""

    def __init__(self, index, observer, **kwargs):
        super().__init__(index, **kwargs)
        self.observer = observer

    def vector_search(self, query, k, scope):
        results = super().vector_search(query, k, scope)
        self.observer.retrievals.append({
            "query": query, "k": k, "scope": scope,
            "chunks": [{"source": doc.metadata.get("source_file"),
                        "page": doc.metadata.get("page"), "page_number": doc.metadata.get("page", 0) + 1,
                        "chunk_id": doc.metadata.get("chunk_id"),
                        "document_origin": doc.metadata.get("document_origin"),
                        "metadata": copy.deepcopy(doc.metadata), "content": doc.page_content,
                        "score": float(score)} for doc, score in results],
        })
        self.observer.checkpoint()
        return results


def diagnostic_summary(observer):
    rows = observer.attempts
    classified = []
    for entry in rows:
        sufficient = entry["concept_check"]["sufficiency"] == "sufficient"
        grade = entry.get("application_grade")
        classification = (
            "RETRIEVAL EVIDENCE FAILURE" if not sufficient else
            "PROVIDER/GRADER COMPATIBILITY" if grade == "weak" else
            "SUFFICIENT EVIDENCE ACCEPTED" if grade == "good" else "INCONCLUSIVE"
        )
        classified.append({"attempt": entry["attempt"], "evidence_sufficiency": entry["concept_check"]["sufficiency"],
                           "missing_expected_concepts": entry["concept_check"]["missing_expected_concepts"],
                           "raw_grade": entry.get("raw_grade"), "parsed_grade": entry.get("parsed_grade"),
                           "application_grade": grade, "rationale": entry.get("rationale"),
                           "prompt_hash": entry.get("prompt_hash"), "evidence_hash": entry["evidence_hash"],
                           "model": entry.get("model"), "classification": classification})
    exact_same = rows[0]["evidence_hash"] == rows[1]["evidence_hash"] if len(rows) == 2 else None
    chunk_sets_same = (
        {r["chunk_id"] for r in rows[0]["retrieved_chunks"]}
        == {r["chunk_id"] for r in rows[1]["retrieved_chunks"]}
    ) if len(rows) == 2 else None
    return {"attempts": classified, "exact_grader_evidence_identical": exact_same,
            "chunk_sets_identical": chunk_sets_same,
            "rewrite_changed_evidence": not exact_same if exact_same is not None else None,
            "original_query": rows[0]["original_query"] if rows else None,
            "queries_used_for_vector_search": [r["query"] for r in observer.retrievals],
            "rationale_note": "Only grade is requested by the unchanged schema; no absent rationale was invented."}


def run(output):
    from evals import matched_environment as matched
    from evals.gemini_matched_capture import google_environment, graph_fingerprints
    from evals.provider_guard import ProviderAbort, stop_on_provider_failure

    if output.exists():
        raise ValueError("Use a fresh diagnostic directory")
    with google_environment():
        args = SimpleNamespace(corpus_id=matched.CORPUS_ID, index=matched.INDEX, collection="rfp_kb_v2",
                               output_dir=output, publish=False, capture=False)
        matched.run(args)
        write_report(output / "environment_before.json", json.loads((output / "environment.json").read_text()))
        before = graph_fingerprints()
        observer = GradingObserver(output)
        pipeline = ObservedPipeline(matched.INDEX, observer, collection="rfp_kb_v2")
        cases, lock = matched.validate_frozen(pipeline)
        case = next(c for c in cases if c["id"] == "semantic_modernization")
        pipeline.start_generation()
        report = {"case": case, "requested_mode": "vector_only", "graph_corpus_id": pipeline.corpus_id,
                  "corpus_hash": pipeline.corpus_hash, "generation": pipeline.generation_metadata,
                  "judge": "not_run", "benchmark_started": False, "provider_failures": []}
        token = _observer.set(observer)
        try:
            with stop_on_provider_failure():
                payload = pipeline.execute(case, "vector_only")
            report.update({"status": "completed", "final_answer": payload.get("answer"),
                           "generation_kind": payload.get("generation_kind"), "traces": payload.get("traces", [])})
        except ProviderAbort as exc:
            report.update({"status": "stopped_provider_failure", "provider_failures": [exc.failure]})
        except Exception as exc:
            report.update({"status": "stopped_pipeline_failure", "error_type": type(exc).__name__})
        finally:
            _observer.reset(token)
            observer.checkpoint(strict=True)
            pipeline.load_current()
            report.update(diagnostic_summary(observer))
            matched.assert_frozen(matched.QUESTIONS, cases, pipeline.load_current(), lock)
            matched.run(args)
            write_report(output / "environment_after.json", json.loads((output / "environment.json").read_text()))
            after = graph_fingerprints()
            report["corpus_fingerprints"] = {"before": before, "after": after, "unchanged": before == after}
            write_report(output / "diagnostic.json", report)
        print(json.dumps({k: report[k] for k in (
            "status", "attempts", "exact_grader_evidence_identical", "rewrite_changed_evidence", "provider_failures"
        )}, indent=2))
        return 0 if report["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args.output_dir.resolve())
    except Exception as exc:
        print(f"Diagnostic setup/verification stopped ({type(exc).__name__}); details redacted.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
