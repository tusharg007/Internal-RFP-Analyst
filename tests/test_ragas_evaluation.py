"""Offline evaluation contracts. No real judge or generation provider is called."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.ragas_adapter import METRICS, MODES, adapt_execution
from evals.ragas_baselines import calibrate, check_regression
from evals.ragas_judge import JudgeSettings, metric_arguments, score_execution
from evals.ragas_reports import aggregate_cases, new_report, write_report
from evals.run_ragas import evaluate_mode, load_cases, parser
from rfp_analyst.agent.generation_evidence import capture_generation, empty_capture


def case():
    return {
        "id": "actual-case",
        "question": "What technology was used?",
        "expected_sources": ["case.pdf"],
        "expected_origins": ["sample"],
        "expected_tools": ["search_knowledge_base"],
        "expected_intent": "search",
        "reference": "The project used Azure.",
    }


def payload():
    doc = {
        "source": "case.pdf",
        "page": 1,
        "content": "The project used Azure.",
        "chunk_id": "real-id",
        "document_origin": "sample",
    }
    context = "[Source: case.pdf, Page 2]\nThe project used Azure."
    state = {
        "retrieval_context": context,
        "retrieved_documents": [doc],
        "specialized_notes": "Derived analysis, not original evidence",
    }
    return (
        state
        | capture_generation(state, "Question\n" + context + "\n" + state["specialized_notes"])
        | {
            "answer": "The project used Azure. [Source: case.pdf, Page 2]",
            "intent": "search",
            "retrieval_mode": "vector_only",
            "grounded": True,
            "response_mode": "llm",
            "traces": [{"tool": "search_knowledge_base"}],
        }
    )


class FakeMetric:
    def __init__(self, value=0.8, error=None):
        self.value, self.error, self.calls = value, error, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(value=self.value)


def registry(**options):
    return {
        name: FakeMetric(**options)
        for name in (
            "faithfulness",
            "answer_relevancy",
            "context_precision_with_reference",
            "context_precision_without_reference",
            "context_recall",
            "answer_correctness",
        )
    }


def pilot(score=0.8, timestamp="first"):
    report = new_report(
        "vector_only",
        cases=[case()],
        corpus_hash="frozen-corpus",
        judge={"provider": "test-double", "model": "offline-test"},
        generation={"model": "frozen"},
    )
    report["timestamp"] = timestamp
    execution = adapt_execution(case(), payload(), "vector_only")
    report["cases"] = [
        {
            "id": case()["id"],
            "status": "completed",
            "deterministic": execution.deterministic,
            "scores": asyncio.run(score_execution(execution, registry(value=score))),
        }
    ]
    report["aggregate"] = aggregate_cases(report["cases"])
    return report


def test_adapter_uses_final_capture_not_candidates_or_another_retrieval():
    result = payload()
    result["retrieved_documents"].append({"source": "unused.pdf", "page": 4, "content": "UNUSED"})
    result["prompt"] = "Old synthesized prompt that was never sent"
    adapted = adapt_execution(case(), result, "vector_only")
    assert adapted.sample.retrieved_contexts == [
        "[Source: case.pdf, Page 2]\nThe project used Azure."
    ]
    assert "UNUSED" not in str(adapted.sample.retrieved_contexts)
    assert "Derived analysis" not in str(adapted.sample.retrieved_contexts)
    assert adapted.sample.response == result["answer"]
    assert adapted.sample.user_input == case()["question"]
    assert adapted.deterministic["citation_identity_coverage"] == 1


def test_compacted_generation_context_cannot_be_replaced_by_full_chunks():
    state = payload() | {"retrieval_context": "Exact compacted evidence"}
    capture = capture_generation(state, "Prompt\nExact compacted evidence")
    assert capture["generation_contexts"] == ["Exact compacted evidence"]


def test_multi_chunk_capture_preserves_order_page_numbers_and_whitespace():
    docs = [
        {"source": "a.pdf", "page": 0, "content": "  first\n"},
        {"source": "b.pdf", "page": "3", "content": "second"},
    ]
    context = "[Source: a.pdf, Page 1]\n  first\n\n\n---\n\n[Source: b.pdf, Page 3]\nsecond"
    capture = capture_generation(
        {"retrieved_documents": docs, "retrieval_context": context}, context
    )
    assert "\n\n---\n\n".join(capture["generation_contexts"]) == context
    assert capture["generation_contexts"][0].endswith("  first\n")


def test_capture_rejects_evidence_not_in_actual_prompt():
    with pytest.raises(ValueError, match="supplied prompt"):
        capture_generation({"retrieval_context": "evidence"}, "different prompt")


def test_web_capture_is_exact_supplied_web_string():
    context = "Title: Current result\nURL: https://example.org\nEvidence: text"
    assert capture_generation({"web_results": context}, context, web=True)[
        "generation_contexts"
    ] == [context]


@pytest.mark.parametrize("kind", ["direct", "deterministic", "insufficient", "not_generated"])
def test_nongenerative_paths_are_not_scored_as_perfect_evidence(kind):
    result = payload() | empty_capture(kind)
    fake = registry()
    scores = asyncio.run(score_execution(adapt_execution(case(), result, "vector_only"), fake))
    assert all(
        entry["status"] == "not_applicable" and entry["score"] is None for entry in scores.values()
    )
    assert not any(metric.calls for metric in fake.values())


@pytest.mark.parametrize("mutate", ["missing_capture", "empty_context", "no_hash"])
def test_adapter_never_reconstructs_missing_context(mutate):
    result = payload()
    if mutate == "missing_capture":
        result.pop("generation_kind")
    elif mutate == "empty_context":
        result["generation_contexts"] = []
    else:
        result["generation_prompt_hash"] = ""
    with pytest.raises(ValueError):
        adapt_execution(case(), result, "vector_only")


def test_missing_reference_skips_only_reference_required_metrics():
    question = case()
    question.pop("reference")
    fake = registry()
    scores = asyncio.run(score_execution(adapt_execution(question, payload(), "vector_only"), fake))
    assert scores["faithfulness"]["status"] == "scored"
    assert scores["context_precision"]["variant"] == "context_precision_without_reference"
    assert scores["context_recall"]["reason"] == "reference_missing"
    assert scores["answer_correctness"]["reason"] == "reference_missing"
    assert not fake["answer_correctness"].calls
    assert not fake["context_recall"].calls


def test_real_reference_passed_to_reference_metrics():
    fake = registry()
    scores = asyncio.run(score_execution(adapt_execution(case(), payload(), "vector_only"), fake))
    assert all(result["status"] == "scored" for result in scores.values())
    assert fake["context_precision_with_reference"].calls[0]["reference"] == case()["reference"]
    assert fake["answer_correctness"].calls[0]["reference"] == case()["reference"]
    assert fake["faithfulness"].calls[0]["retrieved_contexts"] == payload()["generation_contexts"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10])
def test_invalid_judge_values_become_explicit_failures(value):
    scores = asyncio.run(
        score_execution(adapt_execution(case(), payload(), "vector_only"), registry(value=value))
    )
    assert all(
        result["status"] == "error" and result["score"] is None for result in scores.values()
    )
    json.dumps(scores, allow_nan=False)


def test_provider_exceptions_are_redacted_not_persisted():
    scores = asyncio.run(
        score_execution(
            adapt_execution(case(), payload(), "vector_only"),
            registry(error=RuntimeError("secret-key PRIVATE CONTENT")),
        )
    )
    assert "secret-key" not in json.dumps(scores)
    assert scores["faithfulness"]["error_type"] == "RuntimeError"


def test_metric_timeouts_are_bounded():
    class SlowMetric:
        async def ascore(self, **kwargs):
            await asyncio.sleep(1)

    fake = dict.fromkeys(registry(), SlowMetric())
    scores = asyncio.run(
        score_execution(adapt_execution(case(), payload(), "vector_only"), fake, timeout=0.001)
    )
    assert all(result["error_type"] == "TimeoutError" for result in scores.values())


def test_agent_metrics_are_deterministic_and_detect_secondary_vector_fallback():
    result = payload() | {
        "retrieval_mode": "hybrid",
        "evaluation_retrieval_events": [
            {"mode": "hybrid", "fallback_reason": ""},
            {"mode": "vector_only", "fallback_reason": "unavailable"},
        ],
    }
    checks = adapt_execution(case(), result, "hybrid").deterministic
    assert checks["tool_correctness"] and checks["route_correctness"]
    assert not checks["retrieval_mode_correctness"]
    result["intent"] = "direct"
    assert not adapt_execution(case(), result, "hybrid").deterministic["route_correctness"]


def test_aggregate_keeps_failed_and_na_cases_in_denominator():
    report = pilot()
    report["cases"].extend(
        [
            {
                "id": "failure",
                "status": "error",
                "scores": {name: {"status": "error", "score": None} for name in METRICS},
            },
            {
                "id": "abstention",
                "status": "completed",
                "scores": {name: {"status": "not_applicable", "score": None} for name in METRICS},
            },
        ]
    )
    summary = aggregate_cases(report["cases"])
    assert summary["metrics"]["faithfulness"] == {
        "mean": 0.8,
        "scored": 1,
        "total_cases": 3,
        "errors": 1,
        "not_applicable": 1,
        "not_run": 0,
        "variants": ["faithfulness"],
    }
    assert summary["pipeline_failures"] == 1


def test_runner_executes_each_pipeline_case_once_and_checkpoints(tmp_path):
    class Pipeline:
        corpus_hash = "frozen"
        generation_metadata = {"model": "injected"}
        calls = []

        def execute(self, question, mode):
            self.calls.append((question["id"], mode))
            return payload()

    pipeline = Pipeline()
    path = tmp_path / "results.json"
    result = asyncio.run(evaluate_mode(pipeline, [case()], "vector_only", output=path))
    assert pipeline.calls == [(case()["id"], "vector_only")]
    assert json.loads(path.read_text(encoding="utf-8")) == result
    assert result["cases"][0]["sample"]["response"] == payload()["answer"]
    assert result["judge"]["status"] == "not_run"


def test_pipeline_failure_is_recorded_without_secrets():
    class Pipeline:
        corpus_hash = "frozen"
        generation_metadata = {}

        def execute(self, *args):
            raise ConnectionError("secret-key")

    report = asyncio.run(evaluate_mode(Pipeline(), [case()], "graph_only"))
    assert report["aggregate"]["pipeline_failures"] == 1
    assert len(report["cases"]) == 1
    assert "secret-key" not in json.dumps(report)


def test_comparison_reuses_exact_same_cases_and_corpus():
    from evals.compare_retrieval_modes import compare_modes

    class Pipeline:
        corpus_hash = "frozen"
        generation_metadata = {"model": "same"}
        calls = []

        def execute(self, question, mode):
            self.calls.append((question["id"], mode))
            return payload() | {"retrieval_mode": mode}

    pipeline = Pipeline()
    report = asyncio.run(compare_modes(pipeline, [case()]))
    assert [mode for _, mode in pipeline.calls] == list(MODES.values())
    assert len({r["question_set_hash"] for r in report["reports"]}) == 1
    assert len({r["corpus_hash"] for r in report["reports"]}) == 1


def test_comparison_rejects_changed_inputs():
    from evals.compare_retrieval_modes import summarize_comparison

    reports = [
        pilot(timestamp=str(i)) | {"retrieval_mode": mode} for i, mode in enumerate(MODES.values())
    ]
    reports[1]["corpus_hash"] = "changed"
    with pytest.raises(ValueError, match="frozen"):
        summarize_comparison(reports)


def test_thresholds_are_calibrated_from_measured_runs_not_fixed_defaults():
    baseline = calibrate([pilot(0.8, "one"), pilot(0.75, "two")], tolerance=0.02)
    assert baseline["metrics"]["faithfulness"]["mean_floor"] == 0.73
    assert check_regression(pilot(0.74), baseline)["passed"]
    assert not check_regression(pilot(0.72), baseline)["passed"]


@pytest.mark.parametrize(
    "bad", ["one_run", "capture_only", "failure", "incomplete", "different_corpus", "nan_tolerance"]
)
def test_invalid_pilots_cannot_create_thresholds(bad):
    reports = [pilot(timestamp="one"), pilot(timestamp="two")]
    tolerance = 0.02
    if bad == "one_run":
        reports = reports[:1]
    elif bad == "capture_only":
        reports[0]["judge"]["status"] = "not_run"
    elif bad == "failure":
        reports[0]["failures"] = [{"error_type": "TimeoutError"}]
    elif bad == "incomplete":
        reports[0]["cases"] = []
    elif bad == "different_corpus":
        reports[0]["corpus_hash"] = "other"
    elif bad == "nan_tolerance":
        tolerance = float("nan")
    with pytest.raises(ValueError):
        calibrate(reports, tolerance=tolerance)


def test_abstention_cannot_raise_mean_by_dropping_scored_case():
    baseline = calibrate([pilot(timestamp="one"), pilot(timestamp="two")], tolerance=0.02)
    run = pilot()
    run["cases"][0]["scores"]["faithfulness"] = {"status": "not_applicable", "score": None}
    assert (
        "faithfulness:applicability_or_coverage_changed"
        in check_regression(run, baseline)["failures"]
    )


def test_code_revision_can_change_but_judge_version_requires_recalibration():
    baseline = calibrate([pilot(timestamp="one"), pilot(timestamp="two")], tolerance=0.02)
    run = pilot()
    run["versions"]["source_tree_hash"] = "new-code"
    assert check_regression(run, baseline)["passed"]
    run["versions"]["packages"]["ragas"] = "another-version"
    assert not check_regression(run, baseline)["passed"]


def test_settings_require_explicit_judge_and_do_not_expose_credentials():
    with pytest.raises(ValueError):
        JudgeSettings("groq", "", "credential", embedding_model="local")
    settings = JudgeSettings("groq", "configured-model", "secret-key", embedding_model="local")
    assert "secret-key" not in repr(settings)
    assert "secret-key" not in json.dumps(settings.public_metadata())
    from evals.ragas_reports import setup_failure_report

    report = setup_failure_report(ValueError("secret-key"), "vector_only")
    assert report["timestamp"] and report["versions"]
    assert "secret-key" not in json.dumps(report)


def test_golden_cases_unchanged_annotations_are_only_reference_expectations():
    path = Path(__file__).resolve().parents[1] / "evals/golden_questions.yaml"
    raw = load_cases(path, None)
    annotated = load_cases(path)
    assert len(raw) == len(annotated) == 9
    assert [c["question"] for c in raw] == [c["question"] for c in annotated]
    assert sum("reference" in c for c in annotated) == 2


def test_cli_help_parsing_does_not_construct_a_judge():
    args = parser().parse_args(["--mode", "graph", "--capture-only"])
    assert args.mode == "graph" and args.capture_only


def test_json_writer_rejects_nan_and_cleans_temporary_file(tmp_path):
    with pytest.raises(ValueError):
        write_report(tmp_path / "invalid.json", {"score": float("nan")})
    assert not list(tmp_path.iterdir())


def test_metric_argument_shapes_match_modern_ragas():
    execution = adapt_execution(case(), payload(), "vector_only")
    assert set(metric_arguments("answer_relevancy", execution.sample)[1]) == {
        "user_input",
        "response",
    }
    assert set(metric_arguments("context_recall", execution.sample)[1]) == {
        "user_input",
        "reference",
        "retrieved_contexts",
    }


def test_optional_ragas_sample_contract_without_external_calls():
    pytest.importorskip("ragas")
    sample = adapt_execution(case(), payload(), "vector_only").sample
    assert sample.to_ragas().model_dump(exclude_none=True) == sample.model_dump(exclude_none=True)


@pytest.mark.parametrize("provider", ["groq", "google"])
def test_optional_provider_adapters_construct_async_clients_without_network(provider):
    pytest.importorskip("ragas")
    from evals.ragas_judge import build_judge_llm

    settings = JudgeSettings(
        provider, "test-not-a-live-model", "test-not-a-real-key", embedding_model="test"
    )
    client, llm = build_judge_llm(settings)
    assert llm.is_async
    if provider == "groq":
        asyncio.run(client.close())
    else:
        asyncio.run(client.aio.aclose())
        client.close()


def test_generation_capture_records_actual_invocation_and_repaired_answer(monkeypatch):
    from rfp_analyst.agent import graph

    calls = []

    class LLM:
        def invoke(self, messages):
            calls.append(messages[0].content)
            return SimpleNamespace(content="original answer")

    monkeypatch.setattr(
        graph, "_verify_generated_answer", lambda p, a: ("repaired answer", {"is_grounded": True})
    )
    state = payload() | {"llm": LLM()}
    result = graph.generate_from_kb(state)
    assert result["answer"] == "repaired answer"
    assert len(calls) == 1
    assert all(context in calls[0] for context in result["generation_contexts"])
    assert result["generation_prompt_hash"] != ""


def test_web_ablation_never_calls_tavily(monkeypatch):
    from rfp_analyst.tools import web_search

    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "test-key")
    assert web_search.search_web({"allow_web_search": False}) == {
        "web_results": "",
        "source_used": "web",
    }


def test_insufficient_node_discards_previous_attempt_evidence():
    from rfp_analyst.agent.graph import answer_insufficient

    result = answer_insufficient(payload())
    assert result["generation_contexts"] == []
    assert result["generation_kind"] == "insufficient"


def test_frozen_vector_search_uses_matching_client_settings_and_never_creates(monkeypatch):
    from evals.ragas_pipeline import FrozenPipeline
    import chromadb
    import langchain_chroma
    from rfp_analyst.retrieval import vector_store

    calls = {}

    def client(**kwargs):
        calls["client"] = kwargs
        return "existing-client"

    class Store:
        def __init__(self, **kwargs):
            calls["store"] = kwargs

        def similarity_search_with_relevance_scores(self, question, **kwargs):
            calls["query"] = kwargs
            return []

    monkeypatch.setattr(chromadb, "PersistentClient", client)
    monkeypatch.setattr(langchain_chroma, "Chroma", Store)
    monkeypatch.setattr(vector_store, "get_embeddings", lambda: "local-embedding")
    pipeline = object.__new__(FrozenPipeline)
    pipeline.index, pipeline.collection, pipeline.manager = Path("indexed"), "existing", None
    assert pipeline.vector_search("q", 6, "upload") == []
    assert calls["client"]["settings"].anonymized_telemetry is False
    assert calls["store"]["create_collection_if_not_exists"] is False
    assert calls["query"] == {"k": 6, "filter": {"document_origin": "upload"}}


def test_graph_witness_metrics_use_generation_capture_not_broader_candidates():
    result = payload()
    result["graph_paths"] = [{"chunk_ids": ["real-id"], "evidence_ids": ["evidence-real"]}]
    result["generation_evidence"][0]["evidence_id"] = "evidence-real"
    assert adapt_execution(case(), result, "vector_only").deterministic[
        "graph_path_provenance_correctness"
    ]
    result["graph_paths"][0]["chunk_ids"].append("unused-candidate")
    assert not adapt_execution(case(), result, "vector_only").deterministic[
        "graph_path_provenance_correctness"
    ]


def test_wrong_mode_is_persisted_as_a_failure_not_success():
    class Pipeline:
        corpus_hash = "frozen"
        generation_metadata = {}

        def execute(self, *args):
            return payload()

    report = asyncio.run(evaluate_mode(Pipeline(), [case()], "hybrid"))
    assert {
        "case_id": case()["id"],
        "stage": "deterministic",
        "check": "retrieval_mode_correctness",
    } in report["failures"]


def test_judge_initialization_failure_restores_logging_and_closes_client(monkeypatch):
    import logging
    from evals.ragas_judge import RagasJudge

    closed = []
    settings = JudgeSettings("groq", "test", "not-real", embedding_model="test")
    level = logging.getLogger("instructor").level

    def initialize(self):
        from evals.ragas_judge import _acquire_redacted_logging

        _acquire_redacted_logging()
        self._logging_acquired = True

        class Client:
            async def close(self):
                closed.append(True)

        self.client = Client()
        raise ValueError("initialization failed")

    monkeypatch.setattr(RagasJudge, "_initialize", initialize)
    with pytest.raises(ValueError):
        asyncio.run(RagasJudge.create(settings))
    assert closed == [True]
    assert logging.getLogger("instructor").level == level


def test_pacing_and_retry_configuration_is_recorded():
    settings = JudgeSettings(
        "groq",
        "test",
        "not-real",
        embedding_model="local",
        max_retries=0,
        min_call_interval_seconds=20,
        max_tokens=2048,
    )
    assert settings.public_metadata()["min_call_interval_seconds"] == 20
    assert settings.public_metadata()["max_tokens"] == 2048


def test_generated_abstention_uses_existing_no_answer_markers():
    result = payload() | {"answer": "I could not find relevant evidence."}
    question = case() | {"expects_no_answer": True}
    checks = adapt_execution(question, result, "vector_only").deterministic
    assert checks["no_answer_behavior"]
    assert checks["citation_identity_coverage"] == 0


def test_negative_cosine_scores_are_preserved_not_reported_as_judge_failures():
    fake = registry()
    fake["answer_relevancy"] = FakeMetric(value=-0.1)
    scores = asyncio.run(score_execution(adapt_execution(case(), payload(), "vector_only"), fake))
    assert scores["answer_relevancy"]["status"] == "scored"
    assert scores["answer_relevancy"]["score"] == -0.1
