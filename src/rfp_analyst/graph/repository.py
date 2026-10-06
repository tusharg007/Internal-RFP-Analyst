"""Fixed, bounded Neo4j ingestion templates with atomic snapshot publication."""

from __future__ import annotations

from .ingestion import GraphSnapshot
from .schema import CHECK_CONSTRAINTS, CHECK_INDEXES, UPSERT_CHUNKS, UPSERT_DOCUMENTS
from .settings import GraphSettings
from .store import (
    GraphIntegrityError,
    GraphPermissionError,
    GraphSchemaError,
    GraphWriteResult,
    Neo4jGraphStore,
    NullGraphStore,
)

DOMAIN_CONSTRAINTS = {
    "rfp_entity_identity": ("RFPDomainEntity", ["corpus_id", "corpus_version", "entity_id"]),
    "rfp_assertion_identity": ("RFPAssertion", ["corpus_id", "corpus_version", "assertion_id"]),
    "rfp_snapshot_identity": ("RFPSnapshot", ["corpus_id", "corpus_version"]),
    "rfp_corpus_head": ("RFPCorpusHead", ["corpus_id"]),
}
DOMAIN_INDEXES = {
    "rfp_entity_kind": ("RFPDomainEntity", ["corpus_id", "corpus_version", "kind"]),
    "rfp_assertion_predicate": ("RFPAssertion", ["corpus_id", "corpus_version", "predicate"]),
}
DOMAIN_STATEMENTS = tuple(
    f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE "
    + (
        "(" + ", ".join(f"n.{p}" for p in properties) + ")"
        if len(properties) > 1
        else f"n.{properties[0]}"
    )
    + " IS UNIQUE"
    for name, (label, properties) in DOMAIN_CONSTRAINTS.items()
) + tuple(
    f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON ("
    + ", ".join(f"n.{p}" for p in properties)
    + ")"
    for name, (label, properties) in DOMAIN_INDEXES.items()
)

READ_HEAD = "MATCH (h:RFPCorpusHead {corpus_id: $corpus_id}) RETURN h.active_version AS version"
LOCK_HEAD = """
MERGE (h:RFPCorpusHead {corpus_id: $corpus_id})
ON CREATE SET h.active_version = ''
SET h.active_version = coalesce(h.active_version, '')
RETURN h.active_version AS version
"""
UPSERT_ENTITIES = """
UNWIND $rows AS row
MERGE (e:RFPDomainEntity {corpus_id: row.corpus_id, corpus_version: row.corpus_version, entity_id: row.entity_id})
ON CREATE SET e += row
RETURN properties(e) AS properties
"""
UPSERT_IDENTITIES = """
UNWIND $rows AS row
MATCH (e:RFPDomainEntity {corpus_id: row.corpus_id, corpus_version: row.corpus_version, entity_id: row.entity_id})
MATCH (c:RFPChunk {corpus_id: row.corpus_id, corpus_version: row.corpus_version, evidence_id: row.evidence_id})
WHERE e.document_id = c.document_id
MATCH (d:RFPDocument {corpus_id: row.corpus_id, corpus_version: row.corpus_version, document_id: e.document_id})
MERGE (e)-[:IN_DOCUMENT]->(d)
MERGE (e)-[r:SUPPORTED_BY {start: row.start, end: row.end, span_scope: 'chunk'}]->(c)
RETURN count(r) AS count
"""
UPSERT_ASSERTIONS = """
UNWIND $rows AS row
MATCH (s:RFPDomainEntity {corpus_id: row.corpus_id, corpus_version: row.corpus_version, entity_id: row.subject_id})
MATCH (c:RFPChunk {corpus_id: row.corpus_id, corpus_version: row.corpus_version, evidence_id: row.evidence_id})
WHERE s.document_id = c.document_id AND s.kind = row.subject_kind
MERGE (a:RFPAssertion {corpus_id: row.corpus_id, corpus_version: row.corpus_version, assertion_id: row.assertion_id})
ON CREATE SET a += row
MERGE (s)-[:HAS_ASSERTION]->(a)
MERGE (a)-[:SUPPORTED_BY {start: row.start, end: row.end, span_scope: 'chunk'}]->(c)
RETURN properties(a) AS properties
"""
UPSERT_OBJECTS = """
UNWIND $rows AS row
MATCH (a:RFPAssertion {corpus_id: row.corpus_id, corpus_version: row.corpus_version, assertion_id: row.assertion_id})
MATCH (o:RFPDomainEntity {corpus_id: row.corpus_id, corpus_version: row.corpus_version, entity_id: row.object_id})
MERGE (a)-[r:OBJECT]->(o)
RETURN count(r) AS count
"""
UPSERT_SNAPSHOT = """
MERGE (n:RFPSnapshot {corpus_id: $row.corpus_id, corpus_version: $row.corpus_version})
ON CREATE SET n += $row, n.ingested_at = datetime()
RETURN properties(n) AS properties
"""
PUBLISH_HEAD = """
MATCH (h:RFPCorpusHead {corpus_id: $corpus_id})
MATCH (n:RFPSnapshot {corpus_id: $corpus_id, corpus_version: $version})
SET h.active_version = $version
RETURN h.active_version AS version
"""


class GraphServiceUnavailable(RuntimeError):
    """No diagnostic source text, connection information or credentials exposed."""


class Neo4jIngestionRepository(Neo4jGraphStore):
    """A small bounded corpus is staged and published in ONE data transaction.

    Past snapshots are audit history, not current facts. Active readers (a later
    phase) MUST filter by head version and verify the indexed corpus manifest.
    Large-corpus partitioned staging is deliberately not silently substituted.
    """

    def _require_domain_schema(self, driver, *, indexes=False):
        with driver.session(database=self.settings.database, default_access_mode="READ") as session:
            with session.begin_transaction(timeout=self.settings.transaction_timeout_seconds) as tx:
                checks = [(CHECK_CONSTRAINTS, DOMAIN_CONSTRAINTS)]
                if indexes:
                    checks.append((CHECK_INDEXES, DOMAIN_INDEXES))
                for statement, specs in checks:
                    actual = {r["name"]: r for r in tx.run(statement, names=list(specs))}
                    for name, (label, properties) in specs.items():
                        row = actual.get(name)
                        valid_types = (
                            {"RANGE"}
                            if statement == CHECK_INDEXES
                            else {"UNIQUENESS", "NODE_PROPERTY_UNIQUENESS"}
                        )
                        if (
                            not row
                            or row["type"] not in valid_types
                            or row["entityType"] != "NODE"
                            or row["labelsOrTypes"] != [label]
                            or row["properties"] != properties
                        ):
                            raise GraphSchemaError(
                                "Initialize/repair the domain graph migration as admin"
                            )
                tx.commit()

    def initialize_schema(self):
        result = super().initialize_schema()
        if not result.ok:
            return result
        with self._lock:
            try:
                driver = self._get_driver()
                with driver.session(
                    database=self.settings.database, default_access_mode="WRITE"
                ) as session:
                    for statement in DOMAIN_STATEMENTS:
                        with session.begin_transaction(
                            timeout=self.settings.transaction_timeout_seconds
                        ) as tx:
                            tx.run(statement).consume()
                            tx.commit()
                self._require_domain_schema(driver, indexes=True)
                return GraphWriteResult("ready")
            except GraphSchemaError:
                raise
            except Exception as error:
                return GraphWriteResult(self._failed("domain schema initialization", error))

    def active_version(self, corpus_id):
        with self._lock:
            if self._inactive_status():
                raise GraphServiceUnavailable("Graph repository inactive")
            try:
                with self._get_driver().session(
                    database=self.settings.database, default_access_mode="READ"
                ) as session:
                    with session.begin_transaction(
                        timeout=self.settings.transaction_timeout_seconds
                    ) as tx:
                        row = tx.run(READ_HEAD, corpus_id=corpus_id).single()
                        tx.commit()
                        return (row["version"] or None) if row else None
            except Exception as error:
                self._failed("snapshot lookup", error)
                raise GraphServiceUnavailable("Graph snapshot lookup unavailable") from None

    def publish(self, snapshot: GraphSnapshot, *, expected_version: str | None):
        with self._lock:
            inactive = self._inactive_status()
            if inactive:
                return inactive
            if self.settings.role not in {"writer", "admin"}:
                raise GraphPermissionError("Domain ingestion requires a writer repository")
            snapshot = snapshot.validate_integrity()
            if snapshot.row_count > self.settings.max_batch_size:
                raise ValueError("Snapshot exceeds the configured atomic ingestion row bound")
            base = {"corpus_id": snapshot.corpus_id, "corpus_version": snapshot.version}
            entities = [base | item.model_dump() for item in snapshot.entities]
            identities = [base | item.model_dump() for item in snapshot.identities]
            assertions = []
            for item in snapshot.assertions:
                row = item.model_dump(exclude={"fact"}) | item.fact.model_dump() | base
                row |= {
                    "extractor_version": snapshot.extractor_version,
                    "review_status": "validated_source_span",
                }
                assertions.append(row)
            manifest = base | {
                "extractor_version": snapshot.extractor_version,
                "normalization_version": snapshot.normalization_version,
                "indexed_digest": digest_inputs(snapshot),
                "format_version": "rfp-graph-v1",
                "chunker_version": "existing-index-metadata",
                "embedding_version": "not_recorded_in_legacy_index",
                "coverage": "conservative_partial",
                "documents": len(snapshot.documents),
                "chunks": len(snapshot.chunks),
                "entities": len(snapshot.entities),
                "assertions": len(snapshot.assertions),
            }
            try:
                driver = self._get_driver()
                self._require_schema(driver)
                self._require_domain_schema(driver)
                with driver.session(
                    database=self.settings.database, default_access_mode="WRITE"
                ) as session:
                    with session.begin_transaction(
                        timeout=self.settings.transaction_timeout_seconds
                    ) as tx:
                        head = tx.run(LOCK_HEAD, corpus_id=snapshot.corpus_id).single()
                        actual = head["version"] or None
                        # The same committed snapshot can be retried after a lost acknowledgement.
                        if actual != snapshot.version and actual != expected_version:
                            raise GraphIntegrityError(
                                "Concurrent graph publication; reconcile before retry"
                            )
                        for statement, rows, identity in (
                            (
                                UPSERT_DOCUMENTS,
                                [d.model_dump() for d in snapshot.documents],
                                "document_id",
                            ),
                            (
                                UPSERT_CHUNKS,
                                [c.model_dump() for c in snapshot.chunks],
                                "evidence_id",
                            ),
                            (UPSERT_ENTITIES, entities, "entity_id"),
                            (UPSERT_ASSERTIONS, assertions, "assertion_id"),
                        ):
                            if rows:
                                self._validate_persisted(
                                    tx.run(statement, rows=rows), rows, identity
                                )
                        for statement, rows in (
                            (UPSERT_IDENTITIES, identities),
                            (UPSERT_OBJECTS, [row for row in assertions if row["object_id"]]),
                        ):
                            if rows and tx.run(statement, rows=rows).single()["count"] != len(rows):
                                raise GraphIntegrityError(
                                    "Incomplete graph support/object relationships"
                                )
                        persisted = tx.run(UPSERT_SNAPSHOT, row=manifest).single()["properties"]
                        if any(persisted.get(key) != value for key, value in manifest.items()):
                            raise GraphIntegrityError("Frozen snapshot manifest conflict")
                        if (
                            tx.run(
                                PUBLISH_HEAD, corpus_id=snapshot.corpus_id, version=snapshot.version
                            ).single()["version"]
                            != snapshot.version
                        ):
                            raise GraphIntegrityError("Graph publication failed integrity check")
                        tx.commit()
                return "ready"
            except (GraphIntegrityError, GraphSchemaError):
                raise
            except Exception as error:
                if str(getattr(error, "code", "")).startswith("Neo.ClientError.Schema.Constraint"):
                    raise GraphIntegrityError(
                        "Graph uniqueness constraint rejected the snapshot"
                    ) from None
                return self._failed("domain ingestion", error)


def digest_inputs(snapshot):
    from .ingestion import digest

    return digest([item.model_dump() for item in sorted(snapshot.inputs, key=lambda i: i.chunk_id)])


class NullIngestionRepository(NullGraphStore):
    def active_version(self, corpus_id):
        raise GraphServiceUnavailable("Graph repository inactive")

    def publish(self, snapshot, *, expected_version):
        return self.health().status


def create_ingestion_repository(settings=None, *, role="writer", **driver_options):
    if settings is None:
        try:
            settings = GraphSettings.from_config(role)
        except (ValueError, TypeError):
            return NullIngestionRepository("invalid_configuration")
    if not settings.enabled:
        return NullIngestionRepository()
    if not settings.configured and driver_options.get("driver") is None:
        return NullIngestionRepository("not_configured")
    return Neo4jIngestionRepository(settings, **driver_options)
