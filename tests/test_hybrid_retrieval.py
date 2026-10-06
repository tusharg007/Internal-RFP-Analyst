"""Secure graph projections and hybrid workflow tests; no model or network."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock
import re

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from rfp_analyst.agent import graph as workflow
from rfp_analyst.graph.ingestion import IndexedChunk, build_snapshot, digest
from rfp_analyst.graph.reader import (
    CHECK_HEAD,
    HEADER,
    PROJECT_SEEDS,
    REQUIREMENT_SEEDS,
    READ_TEMPLATES,
    GraphReadFailure,
    Neo4jGraphReader,
    create_graph_reader,
)
from rfp_analyst.graph.store import GraphPermissionError
from rfp_analyst.retrieval.decisions import GraphPlan, RetrievalDecision, plan_retrieval
from rfp_analyst.retrieval.hybrid import (
    CorpusSnapshot,
    HybridRetrievalProvider,
    fuse_evidence,
    projection_paths,
)
from tests.test_graph_ingestion import IngestionDriver, IngestionSession, IngestionTransaction
from tests.test_graph_store import FakeResult, settings


def case(name, title, industry, tech, framework="HIPAA"):
    texts = [
        f"CONFIDENTIAL | CASE STUDY | {industry}\n{title}\nDocument Type: Case Study\nIndustry: {industry}",
        f"Technology Stack\n- Cloud: {tech}\n- BI: Power BI\n- Compliance: {framework}\nTimeline & Milestones\nTotal Duration: 16 weeks",
        "Budget Range\nEstimated project cost: $100,000 - $200,000 USD\nKey Outcomes\n- Reduced fraud investigation time by 35%\n- Projected savings of 40%",
    ]
    file_hash = digest(texts)
    return [
        IndexedChunk(
            chunk_id=digest([name, index, text]),
            source_file=name,
            document_origin="sample",
            file_hash=file_hash,
            page_index=index,
            chunk_index=index,
            text=text,
        )
        for index, text in enumerate(texts)
    ]


def corpus():
    target = "Client RFP Requirements\nThe solution must support Azure and HIPAA controls."
    return [
        *case(
            "healthcare.pdf",
            "Healthcare Migration",
            "Healthcare & Life Sciences",
            "Microsoft Azure",
        ),
        *case("insurance.pdf", "Insurance Automation", "Insurance", "AWS", "SOC 2"),
        *case("banking.pdf", "Banking Audit", "Banking & Financial Services", "Azure"),
        IndexedChunk(
            chunk_id=digest(target),
            source_file="target.pdf",
            document_origin="upload",
            file_hash=digest("target"),
            page_index=0,
            chunk_index=0,
            text=target,
        ),
    ]


class ReadTransaction(IngestionTransaction):
    def run(self, statement, **params):
        if statement not in READ_TEMPLATES:
            return super().run(statement, **params)
        self.driver.calls.append((statement, params))
        if statement == self.driver.fail_statement:
            raise RuntimeError("super-secret neo4j://secret-host private text")
        snapshot = self.driver.snapshot
        if statement == HEADER:
            if (
                params["corpus_id"] != snapshot.corpus_id
                or params["indexed_digest"] != self.driver.indexed_digest
            ):
                return FakeResult()
            return FakeResult(
                [
                    dict(
                        version=snapshot.version,
                        indexed_digest=self.driver.indexed_digest,
                        documents=len(snapshot.documents),
                        chunks=len(snapshot.chunks),
                        entities=len(snapshot.entities),
                        assertions=len(snapshot.assertions),
                    )
                ]
            )
        if statement == CHECK_HEAD:
            return FakeResult([{"version": self.driver.head}])
        entities = {e.entity_id: e for e in snapshot.entities}
        chunks = {c.evidence_id: c for c in snapshot.chunks}
        docs = {d.document_id: d for d in snapshot.documents}
        if statement in {PROJECT_SEEDS, REQUIREMENT_SEEDS}:
            kind = "Project" if statement == PROJECT_SEEDS else "Requirement"
            rows = []
            for identity in snapshot.identities:
                subject = entities[identity.entity_id]
                source = chunks[identity.evidence_id]
                document = docs[source.document_id]
                if (
                    subject.kind != kind
                    or subject.document_id not in params["document_ids"]
                    or source.document_origin not in params["origins"]
                ):
                    continue
                rows.append(
                    dict(
                        subject=subject.model_dump()
                        | {"corpus_id": snapshot.corpus_id, "corpus_version": snapshot.version},
                        chunk=source.model_dump(),
                        document=document.model_dump(),
                        start=identity.start,
                        end=identity.end,
                        span_scope="chunk",
                    )
                )
            return FakeResult(rows[: params["seed_limit"]])
        rows = []
        for assertion in snapshot.assertions:
            if (
                assertion.subject_id != params["subject_id"]
                or assertion.fact.predicate not in params["predicates"]
            ):
                continue
            source = chunks[assertion.evidence_id]
            if (
                source.document_id not in params["document_ids"]
                or source.document_origin not in params["origins"]
            ):
                continue
            properties = (
                assertion.model_dump(exclude={"fact"})
                | assertion.fact.model_dump()
                | {
                    "corpus_id": snapshot.corpus_id,
                    "corpus_version": snapshot.version,
                    "review_status": "validated_source_span",
                    "extractor_version": snapshot.extractor_version,
                }
            )
            obj = entities.get(assertion.object_id)
            rows.append(
                dict(
                    assertion=properties,
                    object=(
                        obj.model_dump()
                        | {"corpus_id": snapshot.corpus_id, "corpus_version": snapshot.version}
                    )
                    if obj
                    else None,
                    chunk=source.model_dump(),
                    document=docs[source.document_id].model_dump(),
                    start=assertion.fact.start,
                    end=assertion.fact.end,
                    span_scope="chunk",
                )
            )
        if self.driver.mutate:
            rows = self.driver.mutate(rows)
        return FakeResult(rows[: params["fact_limit"]])


class ReadSession(IngestionSession):
    def begin_transaction(self, **kwargs):
        self.driver.transaction_options.append(kwargs)
        return ReadTransaction(self.driver)


class ReadDriver(IngestionDriver):
    def __init__(self, snapshot):
        super().__init__()
        self.snapshot = snapshot
        self.head = snapshot.version
        self.indexed_digest = CorpusSnapshot(snapshot.inputs).fingerprint
        self.mutate = None

    def session(self, **kwargs):
        self.session_options.append(kwargs)
        return ReadSession(self)


@pytest.fixture(autouse=True)
def no_providers(monkeypatch):
    monkeypatch.setattr("rfp_analyst.agent.router._create_router_llm", lambda: None)
    monkeypatch.setattr("rfp_analyst.agent.grader._create_grader_llm", lambda: None)
    monkeypatch.setattr("rfp_analyst.agent.query_rewriter._create_rewriter_llm", lambda: None)
    monkeypatch.setattr("rfp_analyst.tools.web_search.TAVILY_API_KEY", "")


@pytest.fixture
def environment():
    snapshot = build_snapshot("synthetic", corpus())
    driver = ReadDriver(snapshot)
    reader = Neo4jGraphReader(settings(role="reader"), driver=driver)
    captured = CorpusSnapshot(snapshot.inputs)
    vector = Mock(
        side_effect=lambda query, k, scope="all": [
            (
                Document(
                    page_content=i.text,
                    metadata={
                        "source_file": i.source_file,
                        "page": i.page_index,
                        "document_origin": i.document_origin,
                        "chunk_id": i.chunk_id,
                    },
                ),
                0.91,
            )
            for i in captured.inputs
            if scope == "all" or i.document_origin == scope
        ][:k]
    )
    yield snapshot, driver, reader, captured, vector
    reader.close()


def provider(environment, *, policy="auto", strict=False):
    snapshot, _, reader, captured, _ = environment
    return HybridRetrievalProvider(
        reader, lambda: captured, corpus_id=snapshot.corpus_id, policy=policy, strict=strict
    )


def ready(environment):
    captured = environment[3]
    return {
        "status": "ready",
        "total_documents": 4,
        "total_chunks": len(captured.inputs),
        "scope_chunk_counts": {
            scope: len([i for i in captured.inputs if scope == "all" or i.document_origin == scope])
            for scope in ("all", "sample", "upload")
        },
        "indexed_upload_document_count": 1,
        "indexed_upload_files": ["target.pdf"],
        "document_names": ["healthcare.pdf", "insurance.pdf", "banking.pdf", "target.pdf"],
    }


@pytest.mark.parametrize(
    "query,expected",
    [
        ("summarize what Project X says about security", "vector_only"),
        ("what does this RFP say about timeline?", "vector_only"),
        ("find passages about Azure migration", "vector_only"),
        ("which healthcare projects used Azure and had compliance requirements?", "hybrid"),
        ("which previous projects match these three target RFP capabilities?", "hybrid"),
        ("what technologies are shared by Project A and Project B?", "hybrid"),
        ("which projects delivered fraud for banking?", "hybrid"),
        ("Which projects used Azure?", "graph_only"),
        ("What is the timeline for Healthcare Migration?", "graph_only"),
        ("What is my resume tech stack?", "vector_only"),
    ],
)
def test_explicit_query_characteristic_routing(query, expected):
    assert plan_retrieval(query).mode == expected


@pytest.mark.parametrize(
    "mutation",
    [
        {"query_type": "MATCH (n) RETURN n"},
        {"cypher": "DELETE n"},
        {"relationship": "SATISFIES"},
        {"labels": ["Secret"]},
        {"technology_groups": (("InventedTech",),)},
        {"project_refs": ("a", "b", "c")},
        {"frameworks": ("not-a-framework",)},
    ],
)
def test_plan_schema_rejects_unrestricted_query_syntax(mutation):
    with pytest.raises(ValidationError):
        GraphPlan.model_validate(mutation)
    with pytest.raises(ValidationError):
        RetrievalDecision.model_validate({"mode": "anything", "plan": {}})


def test_templates_have_fixed_depth_labels_and_no_writes():
    forbidden = r"\b(?:CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|APOC|LOAD|UNION)\b"
    allowed = {
        "RFPCorpusHead",
        "RFPSnapshot",
        "RFPDomainEntity",
        "RFPAssertion",
        "RFPChunk",
        "RFPDocument",
        "SUPPORTED_BY",
        "IN_DOCUMENT",
        "HAS_ASSERTION",
        "OBJECT",
    }
    for statement in READ_TEMPLATES:
        assert not re.search(forbidden, statement, re.I)
        assert "LIMIT" in statement
        assert not re.search(r"\[.*\*", statement)
        assert set(re.findall(r":([A-Z][A-Za-z_]*)\b", statement)) <= allowed


def test_graph_only_returns_original_text_and_provenance_without_vector_search(environment):
    service = provider(environment, policy="graph_only", strict=True)
    vectors = Mock(side_effect=AssertionError("Graph-only issued a vector search"))
    result = service.retrieve(
        "Which healthcare projects used Azure and had compliance requirements?",
        k=6,
        scope="sample",
        vector_supplier=vectors,
    )
    assert result.effective_mode == "graph_only" and result.paths
    vectors.assert_not_called()
    assert {d["source"] for d in result.documents} == {"healthcare.pdf"}
    assert all(d["score"] is None for d in result.documents)
    assert all(
        d["content"] == next(i.text for i in environment[3].inputs if i.chunk_id == d["chunk_id"])
        for d in result.documents
    )
    assert all(p["assertion_ids"] for p in result.paths)
    assert all(p["file_hash"] and p["document_id"] and p["evidence_id"] for p in result.provenance)
    assert all(o["default_access_mode"] == "READ" for o in environment[1].session_options)


def test_shared_technology_paths_have_both_projects_and_supports(environment):
    service = provider(environment, policy="graph_only", strict=True)
    result = service.retrieve(
        "What technologies are shared by healthcare.pdf and insurance.pdf?",
        k=1,
        scope="sample",
        vector_supplier=Mock(),
    )
    assert len(result.paths) == 1
    path = result.paths[0]
    assert len(path["subject_ids"]) == 2 and len(path["assertion_ids"]) == 2
    assert path["business_hops"] == 2
    assert set(path["chunk_ids"]) <= {d["chunk_id"] for d in result.documents}
    assert len(result.documents) == 4  # k=1 never cuts a mandatory witness group.


def test_ambiguous_project_reference_declines_graph_not_arbitrary_seed(environment):
    result = provider(environment, policy="graph_only", strict=True).retrieve(
        "What technologies are shared by nonexistent project and insurance.pdf?",
        k=6,
        scope="all",
        vector_supplier=Mock(),
    )
    assert not result.documents and result.fallback_reason == "ambiguous_project_reference"


def test_uploaded_requirements_and_case_candidates_include_both_witness_sets(environment):
    result = provider(environment, policy="graph_only", strict=True).retrieve(
        "Which previous projects match target RFP requirements?",
        k=6,
        scope="all",
        vector_supplier=Mock(),
    )
    assert result.paths
    assert {p["document_origin"] for p in result.provenance} == {"sample", "upload"}
    assert all(
        p["interpretation"] == "candidate_relationship_not_contractual_satisfaction"
        for p in result.paths
    )


@pytest.mark.parametrize("target", [False, True])
def test_multiple_uploaded_rfps_are_not_silently_joined_as_one_target(target):
    inputs = corpus()
    second = inputs[-1].model_copy(
        update={
            "source_file": "other_target.pdf",
            "file_hash": digest("other-target"),
            "chunk_id": digest("other-chunk"),
        }
    )
    snapshot = build_snapshot("synthetic", [*inputs, second])
    captured = CorpusSnapshot(snapshot.inputs)
    reader = Neo4jGraphReader(settings(role="reader"), driver=ReadDriver(snapshot))
    service = HybridRetrievalProvider(
        reader, lambda: captured, corpus_id="synthetic", policy="graph_only", strict=True
    )
    try:
        result = service.retrieve(
            "Which previous projects match target RFP requirements?",
            k=6,
            scope="upload" if target else "all",
            target=target,
            vector_supplier=Mock(),
        )
        assert result.fallback_reason == "ambiguous_target_rfp" and not result.paths
    finally:
        reader.close()


@pytest.mark.parametrize("scope", ["sample", "upload"])
def test_scope_is_enforced_on_every_hydrated_witness(environment, scope):
    result = provider(environment, policy="graph_only", strict=True).retrieve(
        "Which previous projects match target RFP requirements?",
        k=6,
        scope=scope,
        vector_supplier=Mock(),
    )
    assert all(d["document_origin"] == scope for d in result.documents)
    assert not result.paths  # Cannot cross the requested scope for a join.


def test_hybrid_deduplicates_and_keeps_native_similarity_separate(environment):
    service = provider(environment, policy="hybrid")
    source = next(
        i for i in environment[3].inputs if i.source_file == "healthcare.pdf" and i.page_index == 1
    )
    row = dict(
        source=source.source_file,
        page=source.page_index,
        document_origin=source.document_origin,
        chunk_id=source.chunk_id,
        content=source.text,
        score=0.73,
        raw_score=0.73,
    )
    result = service.retrieve(
        "Which healthcare projects used Azure and had compliance requirements?",
        k=6,
        scope="sample",
        vector_supplier=lambda: [row],
    )
    assert len({d["chunk_id"] for d in result.documents}) == len(result.documents)
    merged = next(d for d in result.documents if d["chunk_id"] == source.chunk_id)
    assert merged["retrieval_channel"] == "hybrid" and merged["score"] == 0.73
    assert merged["fusion_score"] != merged["score"] and merged["graph_assertion_ids"]


def test_graph_only_no_results_does_not_hide_vector_in_strict_ablation(environment):
    vectors = Mock(return_value=[{"content": "fallback"}])
    service = provider(environment, policy="graph_only", strict=True)
    result = service.retrieve(
        "Which projects used Snowflake?", k=6, scope="sample", vector_supplier=vectors
    )
    assert not result.documents and result.fallback_reason == "no_graph_evidence"
    vectors.assert_not_called()


@pytest.mark.parametrize(
    "reason", ["unavailable", "timeout", "unsynchronized_snapshot", "disabled"]
)
def test_graph_failure_production_falls_back_to_vector(environment, reason):
    service = provider(environment, policy="graph_only")
    service.reader = Mock()
    service.reader.fetch.side_effect = GraphReadFailure(reason)
    vectors = Mock(return_value=[{"content": "fallback"}])
    result = service.retrieve(
        "Which projects used Azure?", k=6, scope="sample", vector_supplier=vectors
    )
    assert result.effective_mode == "vector_only" and result.fallback_reason == reason
    vectors.assert_called_once()


@pytest.mark.parametrize(
    "field,value",
    [
        ("text_hash", "f" * 64),
        ("page_index", 999),
        ("document_origin", "upload"),
        ("source_file", "wrong.pdf"),
        ("evidence_id", "f" * 64),
        ("span_scope", "page"),
    ],
)
def test_corrupt_graph_provenance_is_discarded(environment, field, value):
    def mutate(rows):
        if rows:
            rows[0]["chunk"][field] = value
        return rows

    environment[1].mutate = mutate
    result = provider(environment, policy="graph_only", strict=True).retrieve(
        "Which projects used Azure?", k=6, scope="sample", vector_supplier=Mock()
    )
    assert not result.paths and not result.provenance and not result.documents


def test_quote_and_relationship_offset_tampering_are_rejected(environment):
    def mutate(rows):
        if rows:
            rows[0]["start"] += 1
        return rows

    environment[1].mutate = mutate
    result = provider(environment, policy="graph_only", strict=True).retrieve(
        "Which projects used Azure?", k=6, scope="all", vector_supplier=Mock()
    )
    assert result.fallback_reason == "invalid_assertion_support"


def test_stale_snapshot_and_deletion_never_hydrate_old_graph(environment):
    environment[1].indexed_digest = "f" * 64
    result = provider(environment, policy="graph_only", strict=True).retrieve(
        "Which projects used Azure?", k=6, scope="all", vector_supplier=Mock()
    )
    assert not result.documents and result.fallback_reason == "unsynchronized_snapshot"


def test_read_credentials_roles_and_forbidden_operations():
    for role in ("writer", "admin"):
        with pytest.raises(GraphPermissionError):
            create_graph_reader(settings(role=role), driver=Mock())
    reader = create_graph_reader(settings(role="reader"), driver=Mock())
    with pytest.raises(GraphPermissionError):
        reader.initialize_schema()
    with pytest.raises(GraphPermissionError):
        reader.ingest_provenance([], [])


def test_oversized_seed_parameters_are_rejected_before_driver(environment):
    with pytest.raises(ValueError):
        environment[2].fetch(
            "synthetic",
            environment[3].fingerprint,
            [digest(i) for i in range(1001)],
            GraphPlan(query_type="project_constraints", has_framework=True),
            scope="all",
        )
    assert not environment[1].calls


def test_injection_like_project_values_stay_parameters(environment):
    plan = GraphPlan(
        query_type="shared_technologies", project_refs=("x' DELETE n //", "insurance.pdf")
    )
    # Reference values are resolved locally, never inserted into any statement.
    environment[2].fetch(
        "synthetic", environment[3].fingerprint, environment[3].documents("all"), plan, scope="all"
    )
    assert all("x' DELETE" not in statement for statement, _ in environment[1].calls)
    assert all(statement in READ_TEMPLATES for statement, _ in environment[1].calls)


def test_query_timeouts_are_bound_and_driver_failures_do_not_leak_secrets(environment, caplog):
    environment[1].fail_statement = PROJECT_SEEDS
    with pytest.raises(GraphReadFailure, match="unavailable"):
        environment[2].fetch(
            "synthetic",
            environment[3].fingerprint,
            environment[3].documents("all"),
            GraphPlan(query_type="project_constraints", has_framework=True),
            scope="all",
        )
    assert "super-secret" not in caplog.text and "secret-host" not in caplog.text
    assert all(0 < o["timeout"] <= 2 for o in environment[1].transaction_options)
    assert all(
        o["metadata"]["operation"] == "rfp_graph_read" for o in environment[1].transaction_options
    )


def test_vector_only_is_differentially_backward_compatible(environment):
    from tests.test_langgraph_agent import fake_retrieval, ready_stats

    state = dict(
        kb_ready=True,
        planned_tools=["search_knowledge_base"],
        intent="search",
        user_query="What technology was used?",
        retrieval_fn=fake_retrieval,
        vectorstore_stats=ready_stats(),
        traces=[],
    )
    expected = workflow._execute_vector_retrieval(state)
    state["retrieval_provider"] = provider(environment, policy="vector_only")
    actual = workflow.execute_retrieval(state)
    assert actual == expected and not environment[1].calls


def test_auto_narrative_vector_decision_is_recorded_without_graph_reads(environment):
    from tests.test_langgraph_agent import fake_retrieval, ready_stats

    state = dict(
        kb_ready=True,
        planned_tools=["search_knowledge_base"],
        intent="search",
        user_query="summarize what Project X says about security",
        retrieval_fn=fake_retrieval,
        vectorstore_stats=ready_stats(),
        traces=[],
        retrieval_provider=provider(environment),
    )
    result = workflow.execute_retrieval(state)
    decision = RetrievalDecision.model_validate(result["retrieval_decision"])
    assert decision.mode == "vector_only" and decision.reason == "narrative_evidence"
    assert not environment[1].calls


def test_primary_graph_workflow_grades_generates_and_verifies_original_sources(environment):
    class LLM:
        def invoke(self, messages):
            return type(
                "Answer",
                (),
                {
                    "content": "Healthcare Migration used Microsoft Azure. [Source: healthcare.pdf, Page 2]"
                },
            )()

    vector = Mock(side_effect=AssertionError("Graph-only vector call"))
    payload = workflow.prepare_query_payload(
        "Which projects used Azure?",
        retrieval_provider=provider(environment, policy="graph_only", strict=True),
        retrieval_fn=vector,
        vectorstore_stats=ready(environment),
        llm=LLM(),
    )
    assert payload["retrieval_mode"] == "graph_only" and payload["graph_paths"]
    assert payload["kb_grade"] == "good" and payload["source_used"] == "private_kb"
    assert payload["generation_kind"] == "llm_kb"
    assert payload["generation_contexts"]
    assert "\n\n---\n\n".join(payload["generation_contexts"]) == payload["retrieval_context"]
    assert {item["chunk_id"] for item in payload["generation_evidence"]} == {
        item["chunk_id"] for item in payload["retrieved_documents"]
    }
    assert any(t["tool"] == "final_grounding_verifier" for t in payload["traces"])
    assert "retrieval_provider" not in payload
    vector.assert_not_called()


def test_secondary_tools_use_same_graph_only_provider(environment):
    service = provider(environment, policy="graph_only", strict=True)
    vector = Mock(side_effect=AssertionError("Secondary tool leaked a vector search"))
    state = dict(
        retrieval_provider=service, retrieval_fn=vector, retrieval_scope="all", retrieval_k=6
    )
    results = workflow._scoped_search(state, "sample")(
        "The solution must support Azure and HIPAA.", k=6
    )
    assert results and all(score is None for _, score in results)
    vector.assert_not_called()


def test_snapshot_change_before_generation_is_detected(environment):
    service = provider(environment, policy="graph_only", strict=True)
    result = service.retrieve(
        "Which projects used Azure?", k=6, scope="all", vector_supplier=Mock()
    )
    assert result.paths and service.validate_current()
    environment[1].head = "f" * 64
    assert not service.validate_current()


def graph_state(environment, *, strict=True):
    service = provider(environment, policy="graph_only", strict=strict)
    llm = Mock()
    llm.invoke.return_value = type(
        "Answer", (), {"content": "Microsoft Azure. [Source: healthcare.pdf, Page 2]"}
    )()
    state = dict(
        kb_ready=True,
        planned_tools=["search_knowledge_base"],
        intent="search",
        user_query="Which projects used Azure?",
        current_query="Which projects used Azure?",
        retrieval_fn=environment[4],
        retrieval_scope="all",
        retrieval_k=6,
        vectorstore_stats=ready(environment),
        traces=[],
        llm=llm,
        retrieval_provider=service,
    )
    state.update(workflow.execute_retrieval(state))
    assert state["graph_paths"]
    return state


@pytest.mark.parametrize("change", ["head", "index", "service"])
def test_strict_generation_rejects_changed_or_unavailable_snapshot(environment, change):
    state = graph_state(environment)
    if change == "head":
        environment[1].head = "f" * 64
    elif change == "index":
        state["retrieval_provider"].corpus_loader = lambda: CorpusSnapshot(
            environment[3].inputs[:-1]
        )
    else:
        state["retrieval_provider"].reader = Mock()
        state["retrieval_provider"].reader.current_version.side_effect = GraphReadFailure(
            "unavailable"
        )
    result = workflow.generate_from_kb(state)
    assert result["response_mode"] == "fallback" and not result["graph_paths"]
    assert not result["graph_provenance"] and not result["retrieved_documents"]
    state["llm"].invoke.assert_not_called()
    environment[4].assert_not_called()


def test_generation_snapshot_failure_production_retrieves_fresh_vectors(environment):
    from tests.test_langgraph_agent import fake_retrieval

    state = graph_state(environment, strict=False)
    state["retrieval_fn"] = Mock(side_effect=fake_retrieval)
    state["specialized_notes"] = "STALE GRAPH MARKER"
    state["tool_outputs"] = {"project_catalog": {"answer_markdown": "STALE GRAPH MARKER"}}
    environment[1].head = "f" * 64
    result = workflow.generate_from_kb(state)
    assert result["retrieval_mode"] == "vector_only" and not result["graph_paths"]
    assert result["retrieval_provider"] is None
    assert not result["graph_provenance"] and not result["tool_outputs"]
    state["retrieval_fn"].assert_called()
    prompt = state["llm"].invoke.call_args[0][0][0].content
    assert "STALE GRAPH MARKER" not in prompt and "banking_case_study.pdf" in prompt


def test_generation_budget_covers_actual_invoked_prompt_and_original_witnesses(environment):
    state = graph_state(environment)
    state["specialized_notes"] = "UNVERIFIED TOOL NOTES" * 10000
    result = workflow.generate_from_kb(state)
    prompt = state["llm"].invoke.call_args[0][0][0].content
    assert prompt == result["prompt"]
    assert result["prompt_budget"]["estimated_tokens"] == workflow._estimate_tokens(prompt)
    assert workflow._estimate_tokens(prompt) <= workflow.MAX_PROMPT_TOKENS
    assert "UNVERIFIED TOOL NOTES" not in prompt
    assert all(d["content"] in prompt for d in result["retrieved_documents"])
    assert all(
        set(p["chunk_ids"]) <= {d["chunk_id"] for d in result["retrieved_documents"]}
        for p in result["graph_paths"]
    )
    assert any(t["tool"] == "final_grounding_verifier" for t in result["traces"])


def test_existing_ui_source_labels_and_stream_adapter_accept_graph_payload(environment):
    import agent as ui_agent

    state = graph_state(environment)
    payload = workflow.generate_from_kb(state)
    trace = ui_agent._payload_to_reasoning_trace(payload)
    assert any(t.get("source_used") == "Private KB" for t in trace)
    assert any(t.get("snippet") == "graph-backed original evidence" for t in trace)
    assert "".join(ui_agent.query_agent_stream(state["llm"], payload)) == payload["answer"]
    state["llm"].invoke.assert_called_once()
    state["llm"].stream.assert_not_called()


def test_tiny_prompt_budget_declines_instead_of_slicing_mandatory_graph_sources(
    environment, monkeypatch
):
    state = graph_state(environment)
    monkeypatch.setattr(workflow, "MAX_PROMPT_TOKENS", 100)
    result = workflow.generate_from_kb(state)
    assert not result["graph_paths"] and result["response_mode"] == "fallback"
    state["llm"].invoke.assert_not_called()


@pytest.mark.parametrize("stage", ["grading", "generation"])
def test_snapshot_changes_during_llm_calls_discard_answer_and_prompt(
    environment, monkeypatch, stage
):
    state = graph_state(environment)
    if stage == "grading":

        def grader(_state):
            environment[1].head = "f" * 64
            return {"kb_grade": "good"}

        monkeypatch.setattr(workflow, "grade_kb_evidence", grader)
    else:

        def invoke(_messages):
            environment[1].head = "f" * 64
            return type("Answer", (), {"content": "STALE GENERATED ANSWER"})()

        state["llm"].invoke.side_effect = invoke
    result = workflow.generate_from_kb(state)
    assert result["response_mode"] == "fallback" and not result["graph_paths"]
    assert result["prompt"] == "" and not result["retrieved_documents"]
    assert "STALE GENERATED ANSWER" not in result["answer"]
    if stage == "grading":
        state["llm"].invoke.assert_not_called()


def test_graph_witness_budget_discards_whole_paths_and_their_assertions(environment, monkeypatch):
    import rfp_analyst.retrieval.hybrid as hybrid

    plan = GraphPlan(
        query_type="shared_technologies", project_refs=("healthcare.pdf", "insurance.pdf")
    )
    projection = environment[2].fetch(
        "synthetic",
        environment[3].fingerprint,
        environment[3].documents("sample"),
        plan,
        scope="sample",
    )
    paths = projection_paths(projection, environment[3], "synthetic", plan, scope="sample")
    documents, retained = fuse_evidence([], paths, k=1)
    assert retained and len(documents) == 4
    monkeypatch.setattr(hybrid, "MAX_CHUNKS", 3)
    assert fuse_evidence([], paths, k=1) == ([], [])
    assert hybrid.select_generation_evidence(
        documents, [p.metadata() for p in paths], max_chars=16000
    ) == ([], [])
    monkeypatch.setattr(hybrid, "MAX_CHUNKS", 20)
    monkeypatch.setattr(hybrid, "MAX_ASSERTIONS", 1)
    assert fuse_evidence([], paths) == ([], [])
    assert hybrid.select_generation_evidence(
        documents, [p.metadata() for p in paths], max_chars=16000
    ) == ([], [])


def test_per_run_outage_circuit_avoids_repeated_connection_attempts(environment):
    service = provider(environment, policy="hybrid")
    service.reader = Mock()
    service.reader.fetch.side_effect = GraphReadFailure("unavailable")
    vectors = Mock(return_value=[])
    for _ in range(3):
        assert (
            service.retrieve(
                "Which projects used Azure?", k=6, scope="sample", vector_supplier=vectors
            ).fallback_reason
            == "unavailable"
        )
    service.reader.fetch.assert_called_once()
    assert vectors.call_count == 3


@pytest.mark.parametrize("failure", [False, True])
def test_vector_failure_is_not_hidden_or_retried_by_graph_fallback(environment, failure):
    service = provider(environment, policy="hybrid")
    if failure:
        service.reader = Mock()
        service.reader.fetch.side_effect = GraphReadFailure("unavailable")
    vectors = Mock(side_effect=LookupError("vector failure"))
    with pytest.raises(LookupError, match="vector failure"):
        service.retrieve("Which projects used Azure?", k=6, scope="all", vector_supplier=vectors)
    vectors.assert_called_once()


def test_disabled_graph_skips_index_export_and_driver_creation(environment):
    from rfp_analyst.graph.reader import UnavailableGraphReader

    loader = Mock(side_effect=AssertionError("Disabled graph touched the index"))
    service = HybridRetrievalProvider(UnavailableGraphReader(), loader, policy="auto")
    vectors = Mock(return_value=[])
    result = service.retrieve(
        "Which projects used Azure?", k=6, scope="all", vector_supplier=vectors
    )
    assert result.effective_mode == "vector_only" and result.fallback_reason == "disabled"
    loader.assert_not_called()
    vectors.assert_called_once()


@pytest.mark.parametrize("owned", [False, True])
def test_provider_reader_lifecycle_ownership(environment, owned):
    reader = Mock()
    service = HybridRetrievalProvider(reader, lambda: environment[3], owns_reader=owned)
    service.close()
    if owned:
        reader.close.assert_called_once()
    else:
        reader.close.assert_not_called()


def test_workflow_owned_provider_closes_even_on_exception(environment, monkeypatch):
    service = provider(environment, policy="hybrid")
    service.close = Mock()
    monkeypatch.setattr(
        "rfp_analyst.retrieval.hybrid.create_retrieval_provider", lambda **kwargs: service
    )
    graph = Mock()
    graph.invoke.side_effect = LookupError("workflow failed")
    monkeypatch.setattr(workflow, "compile_query_graph", lambda: graph)
    with pytest.raises(LookupError):
        workflow.prepare_query_payload("q", retrieval_mode="hybrid")
    service.close.assert_called_once()


def test_workflow_does_not_close_injected_provider(environment):
    service = provider(environment, policy="auto")
    service.close = Mock()
    workflow.prepare_query_payload(
        "hello", retrieval_provider=service, vectorstore_stats=ready(environment)
    )
    service.close.assert_not_called()
    assert not environment[1].calls


@pytest.mark.parametrize(
    "budget,reason",
    [
        ("seeds", "seed_budget_exceeded"),
        ("facts", "fact_budget_exceeded"),
        ("records", "read_budget_exceeded"),
        ("deadline", "timeout"),
    ],
)
def test_projection_bounds_fail_closed(environment, monkeypatch, budget, reason):
    import rfp_analyst.graph.reader as reader_module

    if budget == "seeds":
        monkeypatch.setattr(reader_module, "MAX_PROJECTS", 1)
    elif budget == "facts":
        environment[1].mutate = lambda rows: (rows[:1] * 31) if rows else rows
    elif budget == "records":
        monkeypatch.setattr(reader_module, "MAX_READ_RECORDS", 3)
    else:
        times = iter([0, 0, 4])
        monkeypatch.setattr(reader_module, "monotonic", lambda: next(times))
    result = provider(environment, policy="graph_only", strict=True).retrieve(
        "Which projects used Azure?", k=6, scope="all", vector_supplier=Mock()
    )
    assert not result.documents and result.fallback_reason == reason


def test_head_changes_during_read_are_rejected(environment):
    environment[1].head = "f" * 64
    result = provider(environment, policy="graph_only", strict=True).retrieve(
        "Which projects used Azure?", k=6, scope="all", vector_supplier=Mock()
    )
    assert not result.documents and result.fallback_reason == "snapshot_changed"


def test_excess_constraints_are_not_silently_dropped():
    from rfp_analyst.graph.extraction import TECHNOLOGIES

    query = "Which projects used " + " and ".join(list(TECHNOLOGIES)[:8]) + "?"
    decision = plan_retrieval(query)
    assert decision.mode == "vector_only" and decision.reason == "unsupported_ontology"


def test_secondary_hybrid_prefilters_vector_cosine_without_filtering_graph_witnesses(environment):
    state = dict(
        retrieval_provider=provider(environment, policy="hybrid"),
        retrieval_fn=Mock(
            return_value=[
                (Document(page_content="UNRELATED VECTOR", metadata={"chunk_id": "unknown"}), 0.01)
            ]
        ),
    )
    documents = workflow._scoped_search(state, "sample")("Which projects used Azure?")
    assert documents and all(score is None for _, score in documents)
    assert all("UNRELATED VECTOR" not in d.page_content for d, _ in documents)


def test_full_rfp_tool_chain_preserves_graph_only_through_secondary_searches(environment):
    vector = Mock(side_effect=AssertionError("RFP tool chain issued a vector search"))
    payload = workflow.prepare_query_payload(
        "Extract requirements, find case studies, compare fit, and create proposal outline",
        retrieval_provider=provider(environment, policy="graph_only", strict=True),
        retrieval_fn=vector,
        vectorstore_stats=ready(environment),
    )
    assert payload["intent"] == "rfp_analysis" and payload["retrieval_mode"] == "graph_only"
    assert all(
        name in payload["tool_outputs"]
        for name in (
            "extract_rfp_requirements",
            "find_relevant_case_studies",
            "compare_projects",
            "proposal_writer",
        )
    )
    assert {p["document_origin"] for p in payload["graph_provenance"]} == {"sample", "upload"}
    assert all(
        set(p["chunk_ids"]) <= {d["chunk_id"] for d in payload["retrieved_documents"]}
        for p in payload["graph_paths"]
    )
    vector.assert_not_called()


@pytest.mark.parametrize("kind", ["case_study", "project_outline", "proposal", "rfp_response"])
def test_all_actual_public_document_kinds_support_graph_project_queries(kind):
    texts = case("project.pdf", "Migration Project", "Healthcare & Life Sciences", "Azure")
    label = {
        "case_study": "Case Study",
        "project_outline": "Project Outline",
        "proposal": "Proposal",
        "rfp_response": "RFP Response",
    }[kind]
    inputs = [
        i.model_copy(
            update={"text": i.text.replace("Document Type: Case Study", f"Document Type: {label}")}
        )
        for i in texts
    ]
    snapshot = build_snapshot("synthetic", inputs)
    captured = CorpusSnapshot(snapshot.inputs)
    reader = Neo4jGraphReader(settings(role="reader"), driver=ReadDriver(snapshot))
    service = HybridRetrievalProvider(
        reader, lambda: captured, corpus_id="synthetic", policy="graph_only", strict=True
    )
    try:
        result = service.retrieve(
            "Which projects used Azure?", k=6, scope="sample", vector_supplier=Mock()
        )
        assert result.paths and not result.fallback_reason
    finally:
        reader.close()


def test_graph_returned_entities_cannot_change_snapshot_namespace(environment):
    def mutate(rows):
        rows = deepcopy(rows)
        for row in rows:
            if row["object"]:
                row["object"]["corpus_id"] = "another-corpus"
                break
        return rows

    environment[1].mutate = mutate
    result = provider(environment, policy="graph_only", strict=True).retrieve(
        "Which projects used Azure?", k=6, scope="all", vector_supplier=Mock()
    )
    assert result.fallback_reason == "cross_version_entity" and not result.paths


@pytest.mark.parametrize(
    "query",
    [
        "What is the timeline for " + "x" * 257,
        "What technologies are shared by " + "x" * 257 + " and Project B?",
    ],
)
def test_unbounded_entity_reference_declines_routing_without_crashing(query):
    decision = plan_retrieval(query)
    assert decision.mode == "vector_only" and decision.reason == "unsupported_ontology"


def test_target_rfp_candidates_keep_additional_explicit_query_constraints(environment):
    result = provider(environment, policy="graph_only", strict=True).retrieve(
        "Which healthcare projects match target RFP requirements?",
        k=6,
        scope="all",
        vector_supplier=Mock(),
    )
    assert result.paths and all(p["query_type"] == "requirement_candidates" for p in result.paths)
    case_sources = {d["source"] for d in result.documents if d["document_origin"] == "sample"}
    assert case_sources == {"healthcare.pdf"}
    assert all(
        set(r["assertion_id"] for r in p["relationships"]) == set(p["assertion_ids"])
        for p in result.paths
    )


def test_real_public_pdf_graph_joins_have_exact_original_page_support(tmp_path, monkeypatch):
    import document_generator
    from rfp_analyst.ingestion.chunking import chunk_loaded_sources
    from rfp_analyst.ingestion.loaders import load_pdf_sources

    directory = tmp_path / "public-synthetic-pdfs"
    monkeypatch.setattr(document_generator, "DATA_DIR", directory)
    document_generator.generate_all_documents()
    inputs = [
        IndexedChunk.from_document(d) for d in chunk_loaded_sources(load_pdf_sources(directory))
    ]
    snapshot = build_snapshot("public-synthetic", inputs)
    captured = CorpusSnapshot(snapshot.inputs)
    reader = Neo4jGraphReader(settings(role="reader"), driver=ReadDriver(snapshot))
    service = HybridRetrievalProvider(
        reader, lambda: captured, corpus_id=snapshot.corpus_id, policy="graph_only", strict=True
    )
    try:
        result = service.retrieve(
            "Which healthcare projects used Azure and had compliance requirements?",
            k=6,
            scope="sample",
            vector_supplier=Mock(),
        )
        assert result.paths and not result.fallback_reason
        assert {d["source"] for d in result.documents} == {
            "02_Healthcare_Data_Migration_to_Azure_Cloud.pdf"
        }
        assert all(
            d["content"] == next(i.text for i in inputs if i.chunk_id == d["chunk_id"])
            for d in result.documents
        )
    finally:
        reader.close()
