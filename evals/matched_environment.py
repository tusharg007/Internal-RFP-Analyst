"""Isolated public-index publication, exact parity, and judge-free paired capture.

This is evaluation infrastructure only. It never alters application policies,
initializes global schema, writes Chroma, or republishes the protected live corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from evals.provider_guard import ProviderAbort, stop_on_provider_failure
from evals.ragas_pipeline import FrozenPipeline
from evals.ragas_reports import aggregate_cases, new_report, write_report
from evals.retrieval_benchmark import (
    LOCK, MODE_ORDER, QUESTIONS, ROOT, assert_frozen, benchmark_checks, dataset_signature,
)
from evals.run_ragas import evaluate_mode, load_cases
from rfp_analyst.graph.ingestion import build_snapshot

INDEX = ROOT / "evals/ragas_workspace/20261005-public-pilot/vectorstore"
CORPUS_ID = "rfp-eval-frozen-641e9d4dad22"
PROTECTED = "internal-rfp"
PUBLIC_HASH = "641e9d4dad22e33140475ac40b0245dde1edb3ed6e2acc0c5fa8e75919d9d39f"


def digest_records(records):
    encoded = [json.dumps(r, sort_keys=True, ensure_ascii=False, default=str) for r in records]
    return hashlib.sha256("\n".join(sorted(encoded)).encode()).hexdigest()


class GraphInspection:
    """Fixed bounded read-only diagnostics, not a user-supplied Cypher interface."""

    def __enter__(self):
        from neo4j import GraphDatabase
        from rfp_analyst.graph.settings import GraphSettings

        self.settings = GraphSettings.from_config("reader")
        if not self.settings.configured:
            raise ValueError("Configured Neo4j reader required")
        self.driver = GraphDatabase.driver(
            self.settings.uri, auth=(self.settings.username, self.settings.password),
            connection_timeout=self.settings.connection_timeout_seconds,
            connection_acquisition_timeout=self.settings.acquisition_timeout_seconds,
            max_connection_pool_size=1, max_transaction_retry_time=0,
        )
        return self

    def __exit__(self, *args):
        self.driver.close()

    def namespace(self, corpus_id):
        # Include every historical version and inbound/outbound edge, not just counts.
        with self.driver.session(database=self.settings.database, default_access_mode="READ") as s:
            with s.begin_transaction(timeout=20) as tx:
                nodes = [dict(r) for r in tx.run(
                    "MATCH (n {corpus_id:$corpus_id}) "
                    "RETURN labels(n) AS labels, properties(n) AS properties LIMIT 2001",
                    corpus_id=corpus_id)]
                edges = [dict(r) for r in tx.run(
                    "MATCH (a)-[r]->(b) WHERE a.corpus_id=$corpus_id OR b.corpus_id=$corpus_id "
                    "RETURN type(r) AS type, properties(r) AS properties, "
                    "properties(a) AS start, properties(b) AS end LIMIT 5001",
                    corpus_id=corpus_id)]
        if len(nodes) > 2000 or len(edges) > 5000:
            raise ValueError("Inspection bound exceeded; cannot assert complete parity")
        return {"nodes": nodes, "edges": edges}


def protected_fingerprint(raw):
    """Hash locally; never export private source names, assertions, or quotations."""
    return {
        "corpus_id": PROTECTED,
        "nodes": len(raw["nodes"]), "relationships": len(raw["edges"]),
        "node_sha256": digest_records(raw["nodes"]),
        "relationship_sha256": digest_records(raw["edges"]),
        "active_versions": sorted(
            n["properties"]["active_version"] for n in raw["nodes"]
            if "RFPCorpusHead" in n["labels"]
        ),
    }


def manifest(pipeline, snapshot):
    items = {i.chunk_id: i for i in pipeline.snapshot.inputs}
    return {
        "index": str(pipeline.index), "collection": pipeline.collection,
        "indexed_digest": pipeline.corpus_hash,
        "document_count": len(snapshot.documents), "chunk_count": len(snapshot.chunks),
        "eval_target_rfp_present": any(d.source_file == "eval_target_rfp.pdf" for d in snapshot.documents),
        "graph_corpus_id": snapshot.corpus_id, "graph_version": snapshot.version,
        "extractor_version": snapshot.extractor_version,
        "normalization_version": snapshot.normalization_version,
        "documents": [d.model_dump() for d in snapshot.documents],
        "chunks": [
            {**c.model_dump(), "chunk_index": items[c.chunk_id].chunk_index}
            for c in snapshot.chunks
        ],
    }


def compare_manifest(raw, snapshot, indexed_digest):
    version = snapshot.version
    nodes = raw["nodes"]
    heads = [n["properties"] for n in nodes if "RFPCorpusHead" in n["labels"]]
    manifests = [n["properties"] for n in nodes if "RFPSnapshot" in n["labels"]]
    failures = []
    if len(heads) != 1 or heads[0].get("active_version") != version:
        failures.append("active_head")
    if len(manifests) != 1 or manifests[0].get("indexed_digest") != indexed_digest:
        failures.append("indexed_digest_or_snapshot_count")
    if any(n["properties"].get("corpus_version", version) != version for n in nodes):
        failures.append("unexpected_snapshot_version")
    actual = {}
    for label, expected in (("RFPDocument", snapshot.documents), ("RFPChunk", snapshot.chunks)):
        values = [n["properties"] for n in nodes if label in n["labels"]]
        expected_values = [v.model_dump() for v in expected]
        actual[label] = values
        if digest_records(values) != digest_records(expected_values):
            failures.append(label + "_records")
    expected_links = {(c.evidence_id, c.document_id) for c in snapshot.chunks}
    links = [
        (e["start"].get("evidence_id"), e["end"].get("document_id"))
        for e in raw["edges"] if e["type"] == "IN_DOCUMENT" and e["start"].get("evidence_id")
    ]
    if set(links) != expected_links or len(links) != len(expected_links):
        failures.append("chunk_document_relationships")
    if any(e[side].get("corpus_id") != snapshot.corpus_id
           for e in raw["edges"] for side in ("start", "end")):
        failures.append("cross_namespace_relationship")
    return {
        "parity": not failures, "failures": failures,
        "graph_document_count": len(actual["RFPDocument"]),
        "graph_chunk_count": len(actual["RFPChunk"]),
        "expected_document_sha256": digest_records([d.model_dump() for d in snapshot.documents]),
        "actual_document_sha256": digest_records(actual["RFPDocument"]),
        "expected_chunk_sha256": digest_records([c.model_dump() for c in snapshot.chunks]),
        "actual_chunk_sha256": digest_records(actual["RFPChunk"]),
        "compared_fields": "all persisted Document/Chunk properties; exact IN_DOCUMENT edges",
    }


def validate_frozen(pipeline):
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    cases = load_cases(QUESTIONS, None)
    if dataset_signature(QUESTIONS, cases, pipeline.snapshot) != lock["signature"]:
        raise ValueError("Existing freeze mismatch; no publication or capture allowed")
    if pipeline.corpus_hash != PUBLIC_HASH or len(pipeline.snapshot.inputs) != 54:
        raise ValueError("Not the approved public frozen index")
    # 'upload' here is the locked synthetic target, not arbitrary private uploads.
    if pipeline.stats["indexed_upload_files"] != ["eval_target_rfp.pdf"]:
        raise ValueError("Unexpected upload in public evaluation index")
    return cases, lock


def capture_complete(reports, expected, provider_failures, parity):
    rows = [row for r in reports.values() for row in r["cases"]]
    return bool(parity and len(rows) == expected and not provider_failures
                and all(r["status"] == "completed" and not backend_failure(r) for r in rows))


def backend_failure(row):
    """Unsupported plans are limitations, not network/integrity/provider failures."""
    prov = row.get("provenance", {})
    reasons = [prov.get("graph_fallback_reason", "")]
    reasons.extend(e.get("fallback_reason", "") for e in prov.get("retrieval_events", []))
    expected_declines = {
        "unsupported_plan", "ambiguous_project_reference", "ambiguous_target_rfp",
        "unsupported_target_constraints",
    }
    return any(reason and reason not in expected_declines for reason in reasons)


async def capture(pipeline, cases, lock, output):
    pipeline.start_generation()  # No judge instance or judge dependency initialization.
    reports = {mode: new_report(
        mode, cases=cases, corpus_hash=pipeline.corpus_hash,
        judge={"provider": None, "model": None, "status": "not_run:capture_only"},
        generation=pipeline.generation_metadata,
    ) for mode in MODE_ORDER}
    result = {
        "status": "running", "expected_executions": len(cases) * 3,
        "completed_executions": 0, "capture_valid": False,
        "graph_corpus_id": pipeline.corpus_id, "index": str(pipeline.index),
        "freeze": lock, "provider_failures": [], "reports": list(reports.values()),
        "judge": "not_run:capture_only", "retrieval_quality_conclusions": "not_established",
    }
    write_report(output, result)
    for index, case in enumerate(cases):
        order = MODE_ORDER[index % 3:] + MODE_ORDER[:index % 3]
        for mode in order:
            assert_frozen(QUESTIONS, cases, pipeline.load_current(), lock)
            print(f"{index + 1}/{len(cases)} {case['id']} {mode}", flush=True)
            try:
                # ContextVar flows into evaluate_mode's asyncio.to_thread application call.
                with stop_on_provider_failure():
                    one = await evaluate_mode(pipeline, [case], mode, judge=None)
            except ProviderAbort as exc:
                failure = {"case_id": case["id"], "requested_retrieval_mode": mode, **exc.failure}
                result["provider_failures"].append(failure)
                result["aborted_execution"] = failure
                result["status"] = "stopped_provider_failure"
                write_report(output, result)
                return result
            row = one["cases"][0]
            row["query_class"] = case["query_class"]
            row["execution_order"] = list(order)
            row["benchmark"] = benchmark_checks(case, row, pipeline.snapshot)
            reports[mode]["cases"].append(row)
            reports[mode]["failures"].extend(one["failures"])
            reports[mode]["aggregate"] = aggregate_cases(reports[mode]["cases"])
            result["completed_executions"] += row["status"] == "completed"
            write_report(output.with_name(f"{output.stem}-{mode}.json"), reports[mode])
            write_report(output, result)
            if row["status"] != "completed" or backend_failure(row):
                result["status"] = "stopped_pipeline_failure"
                write_report(output, result)
                return result
    result["status"] = "captured_pending_final_parity"
    write_report(output, result)
    return result


def run(args):
    if args.corpus_id == PROTECTED or not args.corpus_id.startswith("rfp-eval-frozen-"):
        raise ValueError("Only an isolated frozen evaluation namespace is allowed")
    output = args.output_dir.resolve()
    # Environment is process-local and restored, never written to .env or app defaults.
    old_id = os.environ.get("RFP_GRAPH_CORPUS_ID")
    os.environ["RFP_GRAPH_CORPUS_ID"] = args.corpus_id
    try:
        pipeline = FrozenPipeline(args.index, collection=args.collection)
        if pipeline.corpus_id != args.corpus_id:
            raise ValueError("Configuration precedence prevented isolated corpus selection")
        cases, lock = validate_frozen(pipeline)
        snapshot = build_snapshot(args.corpus_id, pipeline.snapshot.inputs)
        report = {"manifest": manifest(pipeline, snapshot), "publication": "not_requested"}
        with GraphInspection() as graph:
            before = protected_fingerprint(graph.namespace(PROTECTED))
            report["internal_rfp_before"] = before
            environment_path = output / "environment.json"
            # Retain the ORIGINAL pre-publication guard across later capture invocations.
            if environment_path.exists():
                prior = json.loads(environment_path.read_text(encoding="utf-8"))
                if prior["internal_rfp_before"] != before:
                    raise ValueError("Protected corpus differs from prior recorded baseline")
            write_report(environment_path, report)
            raw = graph.namespace(args.corpus_id)
            if args.publish and not raw["nodes"]:
                from rfp_analyst.graph.repository import create_ingestion_repository

                # Existing atomic, constraint-aware ingestion. NO schema initialization,
                # global deletion, live-head publication, or LLM extraction.
                with create_ingestion_repository(role="writer") as repository:
                    if not repository.health().available:
                        raise ValueError("Graph writer unavailable")
                    if repository.active_version(args.corpus_id) is not None:
                        raise ValueError("Evaluation namespace appeared concurrently")
                    status = repository.publish(snapshot, expected_version=None)
                    if status != "ready":
                        raise ValueError("Isolated publication did not complete")
                report["publication"] = "published_new_namespace"
            elif raw["nodes"]:
                report["publication"] = "reused_without_writes"
            raw = graph.namespace(args.corpus_id)
            report["comparison"] = compare_manifest(raw, snapshot, pipeline.corpus_hash)
            pipeline.load_current()
            report["frozen_index_unchanged_after_publication"] = True
            report["internal_rfp_after_publication"] = protected_fingerprint(graph.namespace(PROTECTED))
            report["internal_rfp_untouched"] = before == report["internal_rfp_after_publication"]
            write_report(environment_path, report)
            if not report["comparison"]["parity"] or not report["internal_rfp_untouched"]:
                raise ValueError("Manifest/protected-corpus verification failed; capture forbidden")
            if args.capture:
                capture_path = output / "capture.json"
                if capture_path.exists():
                    raise ValueError("Preserve prior capture; select a fresh output directory")
                try:
                    result = asyncio.run(capture(pipeline, cases, lock, capture_path))
                finally:
                    pipeline.load_current()
                    report["comparison_after_capture"] = compare_manifest(
                        graph.namespace(args.corpus_id), snapshot, pipeline.corpus_hash
                    )
                    report["internal_rfp_after_capture"] = protected_fingerprint(graph.namespace(PROTECTED))
                    report["internal_rfp_untouched"] = before == report["internal_rfp_after_capture"]
                    write_report(environment_path, report)
                result["parity_after_capture"] = report["comparison_after_capture"]["parity"]
                result["internal_rfp_untouched"] = report["internal_rfp_untouched"]
                result["capture_valid"] = capture_complete(
                    {r["retrieval_mode"]: r for r in result["reports"]}, len(cases) * 3,
                    result["provider_failures"],
                    result["parity_after_capture"] and result["internal_rfp_untouched"],
                )
                if result["capture_valid"]:
                    result["status"] = "completed_valid_capture_no_judge"
                write_report(capture_path, result)
                print(json.dumps({k: result[k] for k in (
                    "status", "capture_valid", "completed_executions", "provider_failures"
                )}, indent=2))
                return 0 if result["capture_valid"] else 1
        print(json.dumps({"parity": report["comparison"]["parity"],
                          "internal_rfp_untouched": report["internal_rfp_untouched"],
                          "publication": report["publication"], "graph_version": snapshot.version}, indent=2))
        return 0
    finally:
        if old_id is None:
            os.environ.pop("RFP_GRAPH_CORPUS_ID", None)
        else:
            os.environ["RFP_GRAPH_CORPUS_ID"] = old_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=INDEX)
    p.add_argument("--collection", default="rfp_kb_v2")
    p.add_argument("--corpus-id", default=CORPUS_ID)
    p.add_argument("--output-dir", type=Path, default=ROOT / "evals/results/matched-public-20261006")
    p.add_argument("--publish", action="store_true")
    p.add_argument("--capture", action="store_true")
    args = p.parse_args()
    try:
        return run(args)
    except Exception as exc:
        # Never print provider/driver raw exceptions, credentials or private data.
        print(f"Matched evaluation stopped during setup/verification ({type(exc).__name__}).")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
