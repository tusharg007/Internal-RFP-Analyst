"""Frozen, paired RFP retrieval experiment; never substitutes a fake graph backend."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from evals.ragas_adapter import METRICS, MODES, fingerprint
from evals.ragas_reports import aggregate_cases, cohort, new_report, setup_failure_report, version_metadata, write_report
from evals.run_ragas import evaluate_mode, load_cases

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals/retrieval_questions.yaml"
LOCK = ROOT / "evals/retrieval_benchmark.lock.json"
CLASSES = {
    "semantic_single_hop",
    "exact_factual_document",
    "relationship",
    "multi_hop",
    "cross_document_comparison",
    "requirement_matching",
    "unsupported_no_answer",
    "ambiguous",
}
MODE_ORDER = tuple(MODES.values())
PROTOCOL = {
    "version": "1.0",
    "web_search": "disabled_for_closed_corpus_ablation",
    "mode_order": "case_index_rotated_vector_graph_hybrid",
    "repetitions": 1,
    "cache": "shared_embedding_client_no_answer_cache",
    "latency": "wall_clock_pipeline_including_routing_grading_generation_repair_not_judge_or_pacing",
    "winners": "deterministic_pareto_only_when_all_requested_backends_operate; semantic_winners_require_complete_paired_scores",
    "optimal_mode": "pre_score_hypothesis_not_forced_mode_compliance",
    "graph_unavailable": "inconclusive_not_vector_win",
    "thresholds": "none_calibrated",
}


def normalize(text):
    return " ".join(str(text).casefold().split())


def frozen_text_hash(path):
    """Match the original LF freeze despite Git's Windows CRLF checkout.

    Only CRLF translation is normalized. Content, whitespace, BOMs and case
    remain byte-sensitive; parsed case hashes and source witnesses are also checked.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_cases(cases):
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    if {c.get("query_class") for c in cases} != CLASSES:
        raise ValueError("Benchmark must cover exactly the eight query classes")
    for case in cases:
        if case.get("expected_optimal_mode") not in MODE_ORDER or not case.get("mode_rationale"):
            raise ValueError("Optimal-mode hypothesis and rationale required before scoring")
        if case.get("expected_behavior") not in {"answer", "abstain", "clarify"}:
            raise ValueError("Expected answer behavior required")
        if case["expected_behavior"] == "answer" and not case.get("required_evidence"):
            raise ValueError("Answerable cases need independently checked evidence")
        for clause in case.get("required_evidence", []):
            if (
                not clause.get("source")
                or type(clause.get("page")) is not int
                or clause["page"] < 1
            ):
                raise ValueError("Gold evidence needs a source and one-based page")
            if not clause.get("anchors") or any(not normalize(a) for a in clause["anchors"]):
                raise ValueError("Gold evidence anchors cannot be empty")


def dataset_signature(questions, cases, snapshot):
    validate_cases(cases)
    witnesses = {}
    for case in cases:
        witnesses[case["id"]] = []
        for clause in case.get("required_evidence", []):
            chunks = [
                i
                for i in snapshot.inputs
                if i.source_file == clause["source"] and i.page_index + 1 == clause["page"]
            ]
            anchors = {}
            for anchor in clause["anchors"]:
                ids = sorted(i.chunk_id for i in chunks if normalize(anchor) in normalize(i.text))
                if not ids:
                    raise ValueError(
                        f"Unverified gold anchor: {case['id']} / {clause['source']} / {anchor}"
                    )
                anchors[anchor] = ids
            witnesses[case["id"]].append({**clause, "anchor_chunk_ids": anchors})
    return {
        "question_file_sha256": frozen_text_hash(questions),
        "question_set_hash": fingerprint(cases),
        "legacy_golden_sha256": frozen_text_hash(ROOT / "evals/golden_questions.yaml"),
        "corpus_hash": snapshot.fingerprint,
        "case_ids": [c["id"] for c in cases],
        "protocol": PROTOCOL,
        "source_checked_witnesses": witnesses,
    }


def freeze(questions, cases, snapshot, lock_path=LOCK):
    """Create once BEFORE execution; subsequent runs must match, never overwrite."""
    signature = dataset_signature(questions, cases, snapshot)
    path = Path(lock_path)
    if path.exists():
        lock = json.loads(path.read_text(encoding="utf-8"))
        if lock["signature"] != signature:
            raise ValueError(
                "Frozen benchmark changed; document an objective dataset bug, do not re-freeze after scores"
            )
        return lock
    lock = {
        "schema_version": "1.0",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "signature_hash": fingerprint(signature),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents two competing runs from replacing the freeze.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return lock


def assert_frozen(questions, cases, snapshot, lock):
    if dataset_signature(questions, load_cases(questions, None), snapshot) != lock["signature"]:
        raise ValueError("Dataset/corpus changed during the frozen experiment")
    if fingerprint(cases) != lock["signature"]["question_set_hash"]:
        raise ValueError("In-memory cases changed during execution")


def _coverage(clauses, records):
    """AND over gold clauses and their anchors; never gold labels from retrieved output."""
    hits = []
    for clause in clauses:
        text = " ".join(
            r["content"]
            for r in records
            if r.get("source") == clause["source"] and r.get("page") == clause["page"]
        )
        hits.append(all(normalize(a) in normalize(text) for a in clause["anchors"]))
    return sum(hits) / len(hits) if hits else None


def generation_records(contexts):
    records = []
    for context in contexts:
        matches = list(re.finditer(r"\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]\n", context))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
            records.append(
                {
                    "source": match[1].strip(),
                    "page": int(match[2]),
                    "content": context[match.end() : end],
                }
            )
    return records


def benchmark_checks(case, row, snapshot):
    from rfp_analyst.retrieval.decisions import plan_retrieval

    prov, checks = row.get("provenance", {}), row.get("deterministic", {})
    kind = row.get("generation_kind")
    evidence = prov.get("generation_evidence", [])
    indexed = {i.chunk_id: i for i in snapshot.inputs}
    valid = 0
    for item in evidence:
        original = indexed.get(item.get("chunk_id"))
        valid += bool(
            original
            and original.source_file == item.get("source")
            and original.page_index == item.get("page")
            and original.document_origin == item.get("document_origin")
            and item.get("evidence_id") in {None, original.evidence_id}
        )
    records = generation_records(row.get("sample", {}).get("retrieved_contexts", []))
    coverage = _coverage(case.get("required_evidence", []), records)
    retrieved = [
        dict(d, page=d["page"] + 1, content=d.get("content") or "")
        for d in row.get("retrieval_evidence", [])
        if type(d.get("page")) is int
    ]
    retrieval_coverage = _coverage(case.get("required_evidence", []), retrieved)
    # Exceptions do not export intermediate state: a generation failure does not
    # establish that retrieval missed the gold evidence. Keep that result unknown.
    if row["status"] == "error":
        retrieval_coverage = None
        coverage = None
    intent = prov.get("intent")
    if case["expected_behavior"] == "clarify":
        behavior = intent == "ambiguous" and not evidence
    elif case["expected_behavior"] == "abstain":
        behavior = checks.get("no_answer_behavior") is True
    else:
        behavior = kind in {"llm_kb", "deterministic"} and checks.get("no_answer_behavior") is True
    if row["status"] == "error":
        behavior = None
    # no_answer_behavior is agreement with the expectation, not the no-answer flag.
    paths = prov.get("graph_paths", [])
    actual_mode = row["requested_retrieval_mode"]
    bypass = case["expected_behavior"] == "clarify" and behavior
    unavailable = prov.get("graph_fallback_reason") or next(
        (
            e.get("fallback_reason")
            for e in prov.get("retrieval_events", [])
            if e.get("fallback_reason")
        ),
        "",
    )
    backend_valid = (bypass or checks.get("retrieval_mode_correctness") is True) and row[
        "status"
    ] == "completed"
    answer = re.sub(r"\[Source:[^\]]+\]", "", row.get("sample", {}).get("response", ""))
    anchors = case.get("answer_anchors", [])
    # Exact numeric probes are NOT semantic entailment or general answer correctness.
    anchor_hits = [
        bool(re.search(r"(?<!\w)" + re.escape(normalize(a)) + r"(?!\w)", normalize(answer)))
        for a in anchors
    ]
    return {
        "retrieval_success": retrieval_coverage == 1.0 if retrieval_coverage is not None else None,
        "retrieved_evidence_recall": retrieval_coverage,
        "generation_evidence_recall": coverage,
        "provenance_coverage": valid / len(evidence) if evidence else None,
        "answer_grounding": checks.get("grounding_passed") if kind == "llm_kb" else None,
        "route_correctness": checks.get("route_correctness"),
        "tool_correctness": checks.get("tool_correctness"),
        "expected_behavior": behavior,
        "exact_answer_probe_coverage": sum(anchor_hits) / len(anchor_hits)
        if anchor_hits and row["status"] != "error"
        else None,
        "optimal_auto_route_correctness": plan_retrieval(case["question"]).mode
        == case["expected_optimal_mode"],
        "auto_mode": plan_retrieval(case["question"]).mode,
        "requested_backend_valid": bool(backend_valid),
        "backend_reason": unavailable or ("bypassed_for_clarification" if bypass else ""),
        "graph_path_correctness": checks.get("graph_path_provenance_correctness")
        if paths
        else None,
        "graph_path_count": len(paths),
        "requested_mode": actual_mode,
    }


def comparison_cases(reports, cases):
    indexed = {r["retrieval_mode"]: {c["id"]: c for c in r["cases"]} for r in reports}
    results = []
    for case in cases:
        rows = {m: indexed[m][case["id"]] for m in MODE_ORDER}
        dimensions = (
            "retrieved_evidence_recall",
            "generation_evidence_recall",
            "provenance_coverage",
            "expected_behavior",
            "route_correctness",
            "tool_correctness",
            "answer_grounding",
            "exact_answer_probe_coverage",
        )
        vectors = {m: tuple(rows[m]["benchmark"][d] for d in dimensions) for m in MODE_ORDER}
        valid = all(r["benchmark"]["requested_backend_valid"] for r in rows.values())
        common = [
            index for index in range(len(dimensions))
            if all(vectors[mode][index] is not None for mode in MODE_ORDER)
        ]

        # Every pair uses the SAME available dimensions across all three modes.
        def dominates(left, right):
            paired = [
                (vectors[left][index], vectors[right][index]) for index in common
            ]
            return bool(paired) and all(a >= b for a, b in paired) and any(a > b for a, b in paired)

        winner = (
            [
                m
                for m in MODE_ORDER
                if all(dominates(m, other) for other in MODE_ORDER if other != m)
            ]
            if valid
            else []
        )
        same = len({vectors[m] for m in MODE_ORDER}) == 1
        category = (
            (winner[0] + "_better")
            if winner
            else ("no_benefit" if valid and same else "inconclusive")
        )
        paired_semantics = {}
        for metric in METRICS:
            scores = {m: rows[m].get("scores", {}).get(metric, {}) for m in MODE_ORDER}
            paired_semantics[metric] = (
                {m: scores[m]["score"] for m in MODE_ORDER}
                if valid and all(s.get("status") == "scored" for s in scores.values())
                else None
            )
        results.append(
            {
                "id": case["id"],
                "query_class": case["query_class"],
                "expected_optimal_mode": case["expected_optimal_mode"],
                "deterministic_category": category,
                "compared_dimensions": [dimensions[index] for index in common],
                "interpretation": "No retrieval used; clarification bypass"
                if case["expected_behavior"] == "clarify" and valid
                else "Backend failure/fallback prevents a three-way conclusion"
                if not valid
                else "One-run deterministic comparison only; not statistical or semantic superiority",
                "semantic_paired_scores": paired_semantics,
                "by_mode": {
                    m: {
                        "checks": rows[m]["benchmark"],
                        "ragas": rows[m]["scores"],
                        "pipeline_latency_seconds": rows[m]["pipeline_latency_seconds"],
                    }
                    for m in MODE_ORDER
                },
            }
        )
    return results


def summarize(reports, cases, lock, preflight):
    if len(reports) != 3 or {r["retrieval_mode"] for r in reports} != set(MODE_ORDER):
        raise ValueError("All three modes required")
    if any(cohort(r) != cohort(reports[0]) for r in reports[1:]):
        raise ValueError("Unmatched execution cohorts")
    if any(
        [r["id"] for r in report["cases"]] != lock["signature"]["case_ids"] for report in reports
    ):
        raise ValueError("Incomplete or reordered case set")
    comparisons = comparison_cases(reports, cases)
    execution_complete = all(
        row["status"] == "completed" and row["benchmark"]["requested_backend_valid"]
        for report in reports
        for row in report["cases"]
    )
    semantic_complete = all(
        all(row.get("scores", {}).get(metric, {}).get("status") == "scored" for row in rows)
        or all(
            row.get("scores", {}).get(metric, {}).get("status") == "not_applicable"
            for row in rows
        )
        for case in cases
        for rows in [[next(r for r in report["cases"] if r["id"] == case["id"]) for report in reports]]
        for metric in METRICS
    )
    by_class = {}
    for cls in sorted(CLASSES):
        by_class[cls] = {}
        for report in reports:
            rows = [r for r in report["cases"] if r["query_class"] == cls]
            latencies = [r["pipeline_latency_seconds"] for r in rows]
            numeric = {}
            for key in (
                "retrieval_success",
                "retrieved_evidence_recall",
                "generation_evidence_recall",
                "provenance_coverage",
                "answer_grounding",
                "route_correctness",
                "tool_correctness",
                "expected_behavior",
                "optimal_auto_route_correctness",
                "exact_answer_probe_coverage",
                "graph_path_correctness",
            ):
                values = [r["benchmark"][key] for r in rows if r["benchmark"][key] is not None]
                numeric[key] = {
                    "mean": statistics.mean(values) if values else None,
                    "evaluated": len(values),
                    "total": len(rows),
                }
            by_class[cls][report["retrieval_mode"]] = {
                "checks": numeric,
                "ragas": aggregate_cases(rows)["metrics"],
                "median_pipeline_latency_seconds": statistics.median(latencies),
                "backend_valid_cases": sum(r["benchmark"]["requested_backend_valid"] for r in rows),
            }
    return {
        "schema_version": "2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if execution_complete and semantic_complete else "incomplete",
        "completion": {
            "execution_complete": execution_complete,
            "semantic_complete": semantic_complete,
            "quality_failures": sum(
                failure.get("stage") == "deterministic"
                for report in reports for failure in report["failures"]
            ),
            "inconclusive_comparisons": sum(
                c["deterministic_category"] == "inconclusive" for c in comparisons
            ),
            "interpretation": "Completed execution is not a quality pass or evidence of superiority.",
        },
        "freeze": lock,
        "preflight": preflight,
        "reports": reports,
        "cases": comparisons,
        "by_query_class": by_class,
        "categories": {
            key: [c["id"] for c in comparisons if c["deterministic_category"] == key]
            for key in (
                "vector_only_better",
                "graph_only_better",
                "hybrid_better",
                "no_benefit",
                "inconclusive",
            )
        },
        "limitations": [
            "Two cases per class and one repetition are an exploratory baseline, not a significance test.",
            "Expected optimal modes are pre-score hypotheses; forced ablations do not measure an auto-routed deployment.",
            "Provenance identity and lexical/numeric grounding are not semantic entailment or claim-level provenance.",
            "Unavailable or failed RAGAS scores are null, never zero or perfect.",
            "No graph backend failure is interpreted as evidence of vector superiority.",
        ],
    }


def markdown(result):
    lines = [
        "# GraphRAG retrieval evaluation",
        "",
        f"Run: {result['timestamp']}. Status: **{result['status']}**.",
        "",
        "## Frozen protocol",
        "",
        "Sixteen independent additions cover all eight query classes (two each). The original nine golden cases remain unchanged and are tested separately. This closed-corpus ablation disables Tavily in every mode. Each case runs once in all three modes; mode order rotates by case index. No answers or retrieval results are cached. Corpus and case expectations are locked before execution, with page/anchor/chunk witnesses.",
        "",
        f"Case hash: `{result['freeze']['signature']['question_set_hash']}`.",
        "",
        f"Corpus hash: `{result['freeze']['signature']['corpus_hash']}`.",
        "",
        "Optimal-mode labels are hypotheses, not measured wins. The forced backend check and independent auto-planner check are different metrics. Timing includes the actual pipeline, but excludes judge time and deliberate pacing. Shared embedding initialization and single-run provider variability limit latency conclusions.",
        "",
        "## Availability",
        "",
        f"Graph preflight: `{result['preflight']['graph']}`. Judge: `{result['preflight']['judge']}`.",
        "",
        "## Per-case comparison",
        "",
        "Recall is coverage of independently source-checked page/anchor clauses in the exact generation evidence. A returned filename alone does not count. `—` means not applicable. `invalid` marks a requested backend that declined/fell back.",
        "",
        "| Case | Class | Expected optimal | Vector recall | Graph recall | Hybrid recall | Finding |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for case in result["cases"]:
        values = []
        for mode in MODE_ORDER:
            checks = case["by_mode"][mode]["checks"]
            value = checks["generation_evidence_recall"]
            values.append(
                ("—" if value is None else f"{value:.2f}")
                + (" (invalid)" if not checks["requested_backend_valid"] else "")
            )
        lines.append(
            f"| {case['id']} | {case['query_class']} | {case['expected_optimal_mode']} | {' | '.join(values)} | {case['deterministic_category']} |"
        )
    lines += ["", "## Where each approach helps", ""]
    for category, title in (
        ("vector_only_better", "Vector better"),
        ("graph_only_better", "Graph better"),
        ("hybrid_better", "Hybrid better"),
        ("no_benefit", "GraphRAG adds no measured benefit"),
        ("inconclusive", "Inconclusive"),
    ):
        ids = result["categories"][category]
        lines.append(f"- {title}: {', '.join(ids) if ids else 'not established in this run'}.")
    lines += [
        "",
        "The categories use deterministic Pareto dominance, with no weighted blend or hand-selected metric. Semantic scores are shown separately. Missing graph access cannot establish a vector win. No-benefit clarification cases bypass retrieval and therefore say nothing about relative retrieval accuracy.",
        "",
        "## Coverage and semantic metrics",
        "",
        "All individual outputs, exact supplied contexts, RAGAS statuses/scores, deterministic checks, package/model metadata, failures, and latencies are in `evals/results/retrieval_comparison.json`. Class-level aggregates report evaluated/total denominators. No regression thresholds have been calibrated.",
        "",
        "| Mode | Completed / total | Valid requested backend | Faithfulness scored | Relevancy scored | Precision scored | Recall scored | Correctness scored | Pipeline median (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in result["reports"]:
        rows = report["cases"]
        counts = [report["aggregate"]["metrics"][name]["scored"] for name in METRICS]
        lines.append(
            f"| {report['retrieval_mode']} | {sum(r['status'] == 'completed' for r in rows)} / {len(rows)} | {sum(r['benchmark']['requested_backend_valid'] for r in rows)} | {' | '.join(map(str, counts))} | {statistics.median(r['pipeline_latency_seconds'] for r in rows):.3f} |"
        )
    lines += [
        "",
        "## Interpretation and architecture follow-up",
        "",
        "When Neo4j is disabled/unavailable, graph-only must decline and hybrid may use vector fallback. This tests safe degradation, not graph quality. A synthetic in-memory projection is not substituted for a real Neo4j run. Connected-query improvements cannot be established until the same frozen corpus is ingested into an isolated Neo4j corpus and all three modes are rerun.",
        "",
        "The independent planner checks identify query-shape gaps even without a graph server. Inspect those mismatches and the recorded graph fallback reasons; they are diagnostic leads, not proof that a multi-hop query failed in Neo4j. Do not revise cases to use parser-friendly wording. If a valid live graph run misses a connected requirement, inspect canonicalization, project reference resolution, complete witness retention, and supporting-text fusion before rerunning the unchanged lock.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        ".venv\\Scripts\\python.exe -m evals.retrieval_benchmark --index <public-index> --collection rfp_kb_v2 --capture-only",
        ".venv\\Scripts\\python.exe -m evals.retrieval_benchmark --index <same-public-index> --collection rfp_kb_v2 --allow-judge",
        "```",
        "",
        "Configure a read-only Neo4j account and ingest the identical indexed corpus using a separate ingestion account/corpus ID before the second command. Configure `RAGAS_JUDGE_PROVIDER`, `RAGAS_JUDGE_MODEL`, and judge credentials through the existing environment settings. The second command makes paid judge calls only with explicit consent. `--freeze-only` validates and creates the immutable manifest without any generation/judge calls. Preserve earlier JSON reports when rerunning an implementation change; never replace the freeze to improve scores.",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in result["limitations"]]
    return "\n".join(lines) + "\n"


async def run(args):
    from evals.ragas_judge import JudgeSettings, RagasJudge
    from evals.ragas_pipeline import FrozenPipeline
    from rfp_analyst.graph.reader import create_graph_reader

    pipeline = FrozenPipeline(args.index, collection=args.collection)
    cases = load_cases(args.questions, None)
    lock = freeze(args.questions, cases, pipeline.snapshot, args.lock)
    if args.freeze_only:
        print(f"Frozen {len(cases)} cases: {lock['signature_hash']}")
        return None
    if not args.capture_only and not args.allow_judge:
        raise ValueError("Choose --capture-only or --allow-judge")
    reader = create_graph_reader()
    try:
        if getattr(reader, "reason", None):
            graph_status = reader.reason
        else:
            try:
                graph_status = (
                    "available"
                    if reader.current_version(pipeline.corpus_id)
                    else "missing_snapshot"
                )
            except Exception as exc:
                graph_status = f"unavailable:{type(exc).__name__}"
    finally:
        reader.close()
    settings = None if args.capture_only else JudgeSettings.from_environment()
    if settings and len(cases) * 3 > settings.max_cases:
        raise ValueError("Raise RAGAS_MAX_CASES explicitly for this 48-execution experiment")
    pipeline.start_generation()
    judge = None
    try:
        judge = await RagasJudge.create(settings) if settings else None
        judge_meta = (
            settings.public_metadata()
            if settings
            else {"provider": None, "model": None, "status": "not_run"}
        )
        reports = {
            m: new_report(
                m,
                cases=cases,
                corpus_hash=pipeline.corpus_hash,
                judge=judge_meta,
                generation=pipeline.generation_metadata,
            )
            for m in MODE_ORDER
        }
        for index, case in enumerate(cases):
            order = MODE_ORDER[index % 3 :] + MODE_ORDER[: index % 3]
            for mode in order:
                assert_frozen(args.questions, cases, pipeline.load_current(), lock)
                print(f"{index + 1}/{len(cases)} {case['id']} {mode}", flush=True)
                one = await evaluate_mode(pipeline, [case], mode, judge=judge)
                if one["versions"] != reports[mode]["versions"]:
                    raise ValueError("Implementation changed during execution")
                row = one["cases"][0]
                row["query_class"] = case["query_class"]
                row["expected_optimal_mode"] = case["expected_optimal_mode"]
                row["execution_order"] = list(order)
                row["benchmark"] = benchmark_checks(case, row, pipeline.snapshot)
                reports[mode]["cases"].append(row)
                reports[mode]["failures"].extend(one["failures"])
                reports[mode]["aggregate"] = aggregate_cases(reports[mode]["cases"])
                write_report(
                    args.output.with_name(f"{args.output.stem}-{mode}.json"), reports[mode]
                )
                if args.case_interval_seconds:
                    await asyncio.sleep(args.case_interval_seconds)
        assert_frozen(args.questions, cases, pipeline.load_current(), lock)
        if version_metadata() != reports[MODE_ORDER[0]]["versions"]:
            raise ValueError("Implementation changed during execution")
        result = summarize(
            list(reports.values()),
            cases,
            lock,
            {"graph": graph_status, "judge": "configured" if judge else "not_run:capture_only"},
        )
        write_report(args.output, result)
        return result
    finally:
        if judge:
            await judge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--collection", default="rfp_kb_v2")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evals/results/retrieval_comparison.json"
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--allow-judge", action="store_true")
    parser.add_argument("--case-interval-seconds", type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.case_interval_seconds <= 300:
        parser.error("Case interval must be in [0,300] seconds")
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        write_report(args.output, setup_failure_report(exc, "comparison"))
        print(f"Benchmark setup/execution failed ({type(exc).__name__}); details redacted.")
        return 2
    if result:
        print(
            json.dumps({"status": result["status"], "categories": result["categories"]}, indent=2)
        )
    return 0 if result is None or result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
