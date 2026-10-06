"""Opt-in real-server tests, constrained to the disposable local Docker port."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from rfp_analyst.graph import GraphIntegrityError, GraphSettings, create_graph_store
from tests.test_graph_store import chunk, document

pytestmark = [
    pytest.mark.neo4j_integration,
    pytest.mark.skipif(
        os.getenv("RFP_ANALYST_NEO4J_INTEGRATION") != "1",
        reason="Set RFP_ANALYST_NEO4J_INTEGRATION=1 with the local Docker test service",
    ),
]


@pytest.fixture
def real_store():
    from neo4j import GraphDatabase

    # Never load .env, application URI or private data for integration tests.
    password = os.getenv("NEO4J_TEST_PASSWORD", "")
    if not password:
        pytest.fail("NEO4J_TEST_PASSWORD must be supplied for the disposable local test service")
    uri = "bolt://localhost:17687"
    settings = GraphSettings(
        enabled=True, uri=uri, username="neo4j", password=password, role="admin"
    )
    corpus_id = f"test-neo4j-foundation-{uuid4().hex}"
    driver = GraphDatabase.driver(
        uri, auth=("neo4j", password), connection_timeout=1, max_transaction_retry_time=0
    )
    try:
        with create_graph_store(settings, driver=driver) as store:
            assert store.health().available, (
                "Local Neo4j test service must be healthy when integration tests are enabled"
            )
            assert store.initialize_schema().ok
            yield store, driver, corpus_id
    finally:
        try:
            # Delete only this fixture's generated test namespace, not schema or other corpora.
            assert corpus_id.startswith("test-neo4j-foundation-")
            driver.execute_query(
                "MATCH (n) WHERE n.corpus_id = $corpus_id DETACH DELETE n",
                corpus_id=corpus_id,
                database_="neo4j",
            )
        finally:
            driver.close()


def test_real_schema_and_repeat_ingestion(real_store):
    store, driver, corpus_id = real_store
    assert store.initialize_schema().ok
    doc = document(corpus_id=corpus_id)
    evidence = chunk(corpus_id=corpus_id)
    assert store.ingest_provenance([doc], [evidence]).ok
    assert store.ingest_provenance([doc], [evidence]).ok
    records, _, _ = driver.execute_query(
        "MATCH (c:RFPChunk {corpus_id: $corpus_id})-[:IN_DOCUMENT]->(d:RFPDocument) "
        "RETURN properties(c) AS chunk, properties(d) AS document",
        corpus_id=corpus_id,
        database_="neo4j",
        routing_="r",
    )
    assert len(records) == 1
    assert records[0]["chunk"] == evidence.model_dump()
    assert records[0]["document"] == doc.model_dump()


def test_real_orphan_rolls_back_and_frozen_mapping_is_enforced(real_store):
    store, driver, corpus_id = real_store
    doc = document(corpus_id=corpus_id)
    with pytest.raises(GraphIntegrityError):
        store.ingest_provenance([doc], [chunk(corpus_id=corpus_id, document_id="missing")])
    records, _, _ = driver.execute_query(
        "MATCH (d:RFPDocument {corpus_id: $corpus_id}) RETURN count(d) AS count",
        corpus_id=corpus_id,
        database_="neo4j",
        routing_="r",
    )
    assert records[0]["count"] == 0
    store.ingest_provenance([doc], [chunk(corpus_id=corpus_id)])
    with pytest.raises(GraphIntegrityError):
        store.ingest_provenance([], [chunk(corpus_id=corpus_id, evidence_id="conflicting-alias")])


def test_real_domain_snapshot_aliases_supports_and_deletion(real_store):
    from rfp_analyst.graph.ingestion import build_snapshot
    from rfp_analyst.graph.repository import Neo4jIngestionRepository
    from tests.test_graph_ingestion import project

    foundation, driver, corpus_id = real_store
    with Neo4jIngestionRepository(foundation.settings, driver=driver) as repository:
        assert repository.initialize_schema().ok
        original = build_snapshot(
            corpus_id, project("one.pdf", "Azure") + project("two.pdf", "MS Azure")
        )
        assert repository.publish(original, expected_version=None) == "ready"
        assert repository.publish(original, expected_version=None) == "ready"
        records, _, _ = driver.execute_query(
            "MATCH (h:RFPCorpusHead {corpus_id: $corpus_id}) "
            "MATCH (s:RFPDomainEntity {corpus_id: $corpus_id, corpus_version: h.active_version})"
            "-[:HAS_ASSERTION]->(a:RFPAssertion)-[:OBJECT]->(o:RFPDomainEntity {name: 'Microsoft Azure'}) "
            "MATCH (a)-[r:SUPPORTED_BY]->(c:RFPChunk)-[:IN_DOCUMENT]->(d:RFPDocument) "
            "RETURN count(DISTINCT o) AS entities, count(DISTINCT s) AS projects, "
            "collect(DISTINCT d.source_file) AS files, collect(DISTINCT r.span_scope) AS scopes",
            corpus_id=corpus_id,
            database_="neo4j",
            routing_="r",
        )
        assert records[0]["entities"] == 1 and records[0]["projects"] == 2
        assert set(records[0]["files"]) == {"one.pdf", "two.pdf"}
        assert records[0]["scopes"] == ["chunk"]
        reduced = build_snapshot(corpus_id, project("one.pdf"))
        assert repository.publish(reduced, expected_version=original.version) == "ready"
        assert repository.active_version(corpus_id) == reduced.version
        records, _, _ = driver.execute_query(
            "MATCH (d:RFPDocument {corpus_id: $corpus_id, corpus_version: $version}) "
            "RETURN collect(d.source_file) AS files",
            corpus_id=corpus_id,
            version=reduced.version,
            database_="neo4j",
            routing_="r",
        )
        assert records[0]["files"] == ["one.pdf"]


def test_real_domain_transaction_rolls_back_before_head_publication(real_store):
    from rfp_analyst.graph.ingestion import build_snapshot
    from rfp_analyst.graph.repository import Neo4jIngestionRepository
    from tests.test_graph_ingestion import project

    foundation, driver, corpus_id = real_store
    with Neo4jIngestionRepository(foundation.settings, driver=driver) as repository:
        assert repository.initialize_schema().ok
        snapshot = build_snapshot(corpus_id, project())
        # Test-fixture corruption: frozen chunk hash conflicts only after the
        # document write. The complete publication transaction must roll back.
        conflicting = snapshot.chunks[0].model_copy(update={"text_hash": "f" * 64})
        assert repository.ingest_provenance(snapshot.documents, [conflicting]).ok
        with pytest.raises(GraphIntegrityError):
            repository.publish(snapshot, expected_version=None)
        assert repository.active_version(corpus_id) is None
        records, _, _ = driver.execute_query(
            "MATCH (n:RFPDomainEntity {corpus_id: $corpus_id}) RETURN count(n) AS count",
            corpus_id=corpus_id,
            database_="neo4j",
            routing_="r",
        )
        assert records[0]["count"] == 0


def test_real_graph_reader_templates_hydrate_original_source_paths(real_store):
    """Community test validates Cypher, not production reader RBAC."""
    from dataclasses import replace
    from rfp_analyst.graph.ingestion import build_snapshot
    from rfp_analyst.graph.reader import Neo4jGraphReader
    from rfp_analyst.graph.repository import Neo4jIngestionRepository
    from rfp_analyst.retrieval.hybrid import CorpusSnapshot, HybridRetrievalProvider
    from tests.test_hybrid_retrieval import corpus

    foundation, driver, corpus_id = real_store
    snapshot = build_snapshot(corpus_id, corpus())
    with Neo4jIngestionRepository(foundation.settings, driver=driver) as repository:
        assert repository.initialize_schema().ok
        assert repository.publish(snapshot, expected_version=None) == "ready"
    reader = Neo4jGraphReader(replace(foundation.settings, role="reader"), driver=driver)
    captured = CorpusSnapshot(snapshot.inputs)
    service = HybridRetrievalProvider(
        reader, lambda: captured, corpus_id=corpus_id, policy="graph_only", strict=True
    )
    try:
        for query in (
            "Which healthcare projects used Azure and had compliance requirements?",
            "What technologies are shared by healthcare.pdf and insurance.pdf?",
            "Which previous projects match target RFP requirements?",
            "What is the timeline for Healthcare Migration?",
        ):

            def forbidden_vector():
                raise AssertionError("Graph-only template test issued a vector search")

            result = service.retrieve(query, k=6, scope="all", vector_supplier=forbidden_vector)
            assert result.paths and not result.fallback_reason
            assert all(
                set(p["chunk_ids"]) <= {d["chunk_id"] for d in result.documents}
                for p in result.paths
            )
            assert all(
                d["content"] == next(i.text for i in captured.inputs if i.chunk_id == d["chunk_id"])
                for d in result.documents
            )
    finally:
        reader.close()
