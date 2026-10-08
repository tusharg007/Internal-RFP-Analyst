"""Protocol and scoring tests only; no live model, Neo4j, or judge calls."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.ragas_adapter import METRICS
from evals.ragas_reports import aggregate_cases, new_report
from evals.retrieval_benchmark import (
    CLASSES,
    MODE_ORDER,
    QUESTIONS,
    assert_frozen,
    benchmark_checks,
    comparison_cases,
    freeze,
    frozen_text_hash,
    generation_records,
    markdown,
    summarize,
    validate_cases,
)
from evals.run_ragas import load_cases


def test_frozen_text_hash_accepts_checkout_eol_only(tmp_path):
    lf, crlf, changed = (tmp_path / name for name in ('lf.yaml', 'crlf.yaml', 'changed.yaml'))
    lf.write_bytes(b'question: Which projects used Azure?\nreference: original\n')
    crlf.write_bytes(lf.read_bytes().replace(b'\n', b'\r\n'))
    changed.write_bytes(b'question: Which projects used AWS?\nreference: original\n')
    assert frozen_text_hash(lf) == frozen_text_hash(crlf)
    assert frozen_text_hash(lf) != frozen_text_hash(changed)


def cases():
    return load_cases(QUESTIONS, None)


def snapshot():
    inputs = []
    for case in cases():
        for n, clause in enumerate(case.get("required_evidence", [])):
            inputs.append(
                SimpleNamespace(
                    source_file=clause["source"],
                    page_index=clause["page"] - 1,
                    chunk_id=f"{case['id']}-{n}",
                    text="\n".join(clause["anchors"]),
                    document_origin="sample",
                    evidence_id=f"e-{case['id']}-{n}",
                )
            )
    return SimpleNamespace(inputs=inputs, fingerprint="test-fixture-not-live-corpus")


def reports():
    result = []
    for mode in MODE_ORDER:
        report = new_report(
            mode,
            cases=cases(),
            corpus_hash=snapshot().fingerprint,
            judge={"status": "not_run"},
            generation={"model": "test-double"},
        )
        for case in cases():
            row = {
                "id": case["id"],
                "query_class": case["query_class"],
                "status": "completed",
                "requested_retrieval_mode": mode,
                "generation_kind": "llm_kb",
                "pipeline_latency_seconds": 1,
                "provenance": {"intent": case["expected_intent"]},
                "deterministic": {
                    "no_answer_behavior": True,
                    "grounding_passed": True,
                    "tool_correctness": True,
                    "route_correctness": True,
                    "retrieval_mode_correctness": True,
                },
                "sample": {"response": "", "retrieved_contexts": []},
                "scores": {m: {"status": "not_run", "score": None} for m in METRICS},
            }
            row["benchmark"] = benchmark_checks(case, row, snapshot())
            report["cases"].append(row)
        report["aggregate"] = aggregate_cases(report["cases"])
        result.append(report)
    return result


def test_expansion_has_eight_classes_two_cases_each_and_frozen_hypotheses():
    validate_cases(cases())
    assert len(cases()) == 16
    assert all(sum(c["query_class"] == cls for c in cases()) == 2 for cls in CLASSES)
    legacy = load_cases(Path("evals/golden_questions.yaml"), None)
    assert len(legacy) == 9
    assert not {c["id"] for c in cases()} & {c["id"] for c in legacy}


def test_freeze_verifies_each_source_page_anchor_and_refuses_overwrite(tmp_path):
    lock_path = tmp_path / "freeze.json"
    lock = freeze(QUESTIONS, cases(), snapshot(), lock_path)
    before = lock_path.read_bytes()
    assert freeze(QUESTIONS, cases(), snapshot(), lock_path) == lock
    assert lock_path.read_bytes() == before
    changed = cases()
    changed[0]["expected_optimal_mode"] = "hybrid"
    with pytest.raises(ValueError, match="Frozen benchmark changed"):
        freeze(QUESTIONS, changed, snapshot(), lock_path)
    assert lock_path.read_bytes() == before


def test_unverified_gold_rejected_before_any_pipeline_calls(tmp_path):
    changed = cases()
    changed[0]["required_evidence"][0]["page"] = 999
    with pytest.raises(ValueError, match="Unverified gold anchor"):
        freeze(QUESTIONS, changed, snapshot(), tmp_path / "freeze.json")


def test_mutation_during_run_detected(tmp_path):
    lock = freeze(QUESTIONS, cases(), snapshot(), tmp_path / "freeze.json")
    changed = cases()
    changed[1]["question"] += " changed after scoring"
    with pytest.raises(ValueError, match="In-memory"):
        assert_frozen(QUESTIONS, changed, snapshot(), lock)


def test_exact_generation_context_parser_keeps_source_page_boundaries():
    parsed = generation_records(
        ["[Source: a.pdf, Page 2]\nAzure\n\n---\n\n[Source: b.pdf, Page 3]\nHIPAA"]
    )
    assert parsed[0]["source"] == "a.pdf" and "HIPAA" not in parsed[0]["content"]
    assert parsed[1] == {"source": "b.pdf", "page": 3, "content": "HIPAA"}


def test_sources_alone_or_other_document_anchor_do_not_satisfy_gold():
    case = cases()[0]
    row = reports()[0]["cases"][0]
    clause = case["required_evidence"][0]
    row["sample"]["retrieved_contexts"] = [
        f"[Source: unrelated.pdf, Page 2]\n{' '.join(clause['anchors'])}"
    ]
    checks = benchmark_checks(case, row, snapshot())
    assert checks["generation_evidence_recall"] == 0
    assert checks["provenance_coverage"] is None  # No evidence cannot mean 100% provenance.


def test_retrieval_and_generation_recall_are_independent():
    case = cases()[0]
    row = reports()[0]["cases"][0]
    clause = case["required_evidence"][0]
    row["retrieval_evidence"] = [
        {
            "source": clause["source"],
            "page": clause["page"] - 1,
            "content": " ".join(clause["anchors"]),
        }
    ]
    checks = benchmark_checks(case, row, snapshot())
    assert checks["retrieval_success"] is True
    assert checks["generation_evidence_recall"] == 0


def test_failed_pipeline_has_unknown_retrieval_not_a_fabricated_miss():
    row = reports()[0]["cases"][0]
    row.update(status="error", error_type="RateLimitError")
    checks = benchmark_checks(cases()[0], row, snapshot())
    assert checks["retrieval_success"] is None
    assert checks["retrieved_evidence_recall"] is None
    assert checks["generation_evidence_recall"] is None
    assert checks["expected_behavior"] is None
    assert checks["requested_backend_valid"] is False


def test_numeric_probe_does_not_accept_19_or_citation_page_9_as_team_size():
    case = next(c for c in cases() if c["id"] == "factual_insurance_team")
    row = reports()[0]["cases"][3]
    row["sample"]["response"] = "Team: 19. [Source: x.pdf, Page 9]"
    assert benchmark_checks(case, row, snapshot())["exact_answer_probe_coverage"] == 0


def test_disabled_graph_is_never_counted_as_a_vector_win():
    data = reports()
    for row in data[1]["cases"]:
        row["benchmark"]["requested_backend_valid"] = False
        row["benchmark"]["backend_reason"] = "disabled"
    compared = comparison_cases(data, cases())
    assert all(c["deterministic_category"] == "inconclusive" for c in compared)
    assert all(all(s is None for s in c["semantic_paired_scores"].values()) for c in compared)


def test_pareto_dominance_not_latency_or_expected_mode_controls_win():
    data = reports()
    for row in data[0]["cases"]:
        row["benchmark"]["generation_evidence_recall"] = 1
        row["pipeline_latency_seconds"] = 500
    for report in data[1:]:
        for row in report["cases"]:
            row["benchmark"]["generation_evidence_recall"] = 0
    assert all(
        c["deterministic_category"] == "vector_only_better" for c in comparison_cases(data, cases())
    )


def test_no_benefit_is_distinct_from_semantic_superiority():
    data = reports()
    compared = comparison_cases(data, cases())
    assert all(c["deterministic_category"] == "no_benefit" for c in compared)
    assert all(c["semantic_paired_scores"]["faithfulness"] is None for c in compared)


def test_summary_requires_identical_full_cases_and_reports_denominators(tmp_path):
    lock = freeze(QUESTIONS, cases(), snapshot(), tmp_path / "freeze.json")
    data = reports()
    result = summarize(data, cases(), lock, {"graph": "test-double", "judge": "not_run"})
    assert result["status"] == "incomplete"
    assert result["by_query_class"]["semantic_single_hop"]["vector_only"]["checks"][
        "provenance_coverage"
    ] == {"mean": None, "evaluated": 0, "total": 2}
    assert "Graph better: not established" in markdown(result)
    json.dumps(result, allow_nan=False)
    broken = copy.deepcopy(data)
    broken[1]["cases"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(broken, cases(), lock, {})
    broken = copy.deepcopy(data)
    broken[1]["corpus_hash"] = "different"
    with pytest.raises(ValueError, match="Unmatched"):
        summarize(broken, cases(), lock, {})


@pytest.mark.parametrize(
    "query",
    [
        "Which pharmaceutical projects mention GDPR, and what framework is listed alongside it?",
        "Which projects use Azure and which services are associated with it?",
        "Which projects mention HIPAA and what controls are related to that?",
    ],
)
def test_local_named_relationship_reference_does_not_require_prior_history(query):
    from rfp_analyst.agent.graph import _resolve_conversational_query

    assert _resolve_conversational_query(query, [], "all") == (query, [], "not_needed")


@pytest.mark.parametrize(
    "query",
    [
        "What about that?",
        "Does it use Azure?",
        "Compare these projects with Azure.",
        "Which projects used it?",
        "What is listed alongside it?",
        "Does that project use GDPR and which standards are related to it?",
    ],
)
def test_missing_or_earlier_deictic_reference_still_requires_history(query):
    from rfp_analyst.agent.graph import _resolve_conversational_query

    assert _resolve_conversational_query(query, [], "all")[2] == "ambiguous"


def test_posthoc_judge_preserves_actual_contexts_and_never_runs_pipeline(tmp_path, monkeypatch):
    from evals.judge_retrieval_capture import judge_capture
    from evals.ragas_pipeline import FrozenPipeline

    monkeypatch.setattr(
        FrozenPipeline, "execute", lambda *_: pytest.fail("Second retrieval/generation")
    )
    lock = freeze(QUESTIONS, cases(), snapshot(), tmp_path / "freeze.json")
    data = reports()
    for report in data:
        for row, case in zip(report["cases"], cases()):
            row["sample"] = {
                "user_input": case["question"],
                "reference": case.get("reference"),
                "response": "Captured answer",
                "retrieved_contexts": ["EXACT ORIGINAL EVIDENCE"],
            }
            row["provenance"]["generation_prompt_hash"] = "captured-hash"
    captured = summarize(data, cases(), lock, {"graph": "test-double", "judge": "not_run"})
    original = copy.deepcopy(captured)

    class Judge:
        settings = SimpleNamespace(
            public_metadata=lambda: {"provider": "test-double", "model": "offline"}
        )
        calls = []

        async def score(self, execution):
            self.calls.append(execution)
            return {m: {"status": "scored", "score": 0.5} for m in METRICS}

    judge = Judge()
    result = asyncio.run(judge_capture(captured, cases(), lock, judge))
    assert len(judge.calls) == 48
    assert all(c.sample.retrieved_contexts == ["EXACT ORIGINAL EVIDENCE"] for c in judge.calls)
    assert captured == original
    assert result["judging"]["capture_report_hash"]
    for before, after in zip(captured["reports"], result["reports"]):
        for left, right in zip(before["cases"], after["cases"]):
            assert left["sample"] == right["sample"]
            assert left["pipeline_latency_seconds"] == right["pipeline_latency_seconds"]


def test_complete_tradeoff_and_quality_failures_are_not_execution_failures(tmp_path):
    data = reports()
    for report in data:
        for row in report["cases"]:
            row["scores"] = {m: {"status": "scored", "score": 0.5} for m in METRICS}
    # Tradeoff: one mode retrieves more, another routes better. No winner invented.
    data[0]["cases"][0]["benchmark"]["generation_evidence_recall"] = 1
    data[0]["cases"][0]["benchmark"]["route_correctness"] = False
    data[0]["failures"].append({"case_id": cases()[0]["id"], "stage": "deterministic"})
    lock = freeze(QUESTIONS, cases(), snapshot(), tmp_path / "freeze.json")
    result = summarize(data, cases(), lock, {})
    assert result["status"] == "completed"
    assert result["completion"]["quality_failures"] == 1
    assert result["cases"][0]["deterministic_category"] == "inconclusive"


def test_shared_not_applicable_metrics_do_not_make_run_incomplete(tmp_path):
    data = reports()
    for report in data:
        for row in report["cases"]:
            row["scores"] = {m: {"status": "scored", "score": 0.5} for m in METRICS}
        report["cases"][-1]["scores"] = {
            m: {"status": "not_applicable", "score": None, "reason": "generation_kind:direct"}
            for m in METRICS
        }
    lock = freeze(QUESTIONS, cases(), snapshot(), tmp_path / "freeze.json")
    assert summarize(data, cases(), lock, {})["status"] == "completed"
    data[1]["cases"][-1]["scores"]["faithfulness"]["status"] = "error"
    assert summarize(data, cases(), lock, {})["status"] == "incomplete"
