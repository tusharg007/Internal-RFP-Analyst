"""Finite read-only graph projection templates, not text-to-Cypher.

Intersections and two-project joins run over this bounded, source-bearing
projection. No query string, label, procedure, or relationship comes from a user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from time import monotonic
from typing import Protocol

from pydantic import Field

from rfp_analyst.graph.extraction import StrictModel
from rfp_analyst.graph.models import Identifier, Sha256
from rfp_analyst.retrieval.decisions import GraphPlan

from .settings import GraphSettings
from .store import GraphPermissionError, Neo4jGraphStore

logger = logging.getLogger(__name__)
MAX_PROJECTS = 100
MAX_FACTS_PER_SUBJECT = 30
MAX_READ_RECORDS = 500
GRAPH_DEADLINE_SECONDS = 3.0

HEADER = """
MATCH (h:RFPCorpusHead {corpus_id: $corpus_id})
MATCH (n:RFPSnapshot {corpus_id: $corpus_id, corpus_version: h.active_version})
WHERE n.indexed_digest = $indexed_digest
RETURN n.corpus_version AS version, n.indexed_digest AS indexed_digest,
       n.documents AS documents, n.chunks AS chunks, n.entities AS entities, n.assertions AS assertions
LIMIT 1
"""
CHECK_HEAD = """
MATCH (h:RFPCorpusHead {corpus_id: $corpus_id})
RETURN h.active_version AS version LIMIT 1
"""
PROJECT_SEEDS = """
MATCH (p:RFPDomainEntity {corpus_id: $corpus_id, corpus_version: $version, kind: 'Project'})
WHERE p.document_id IN $document_ids
WITH p ORDER BY p.entity_id LIMIT $seed_limit
MATCH (p)-[r:SUPPORTED_BY]->(c:RFPChunk {corpus_id: $corpus_id, corpus_version: $version})
MATCH (c)-[:IN_DOCUMENT]->(d:RFPDocument {corpus_id: $corpus_id, corpus_version: $version})
WHERE d.document_id = p.document_id AND d.document_origin IN $origins
      AND d.access_partition = 'internal'
RETURN properties(p) AS subject, properties(c) AS chunk, properties(d) AS document,
       r.start AS start, r.end AS end, r.span_scope AS span_scope
ORDER BY p.entity_id LIMIT $seed_limit
"""
REQUIREMENT_SEEDS = PROJECT_SEEDS.replace("kind: 'Project'", "kind: 'Requirement'")
SUBJECT_FACTS = """
MATCH (p:RFPDomainEntity {corpus_id: $corpus_id, corpus_version: $version, entity_id: $subject_id})
MATCH (p)-[:HAS_ASSERTION]->(a:RFPAssertion {corpus_id: $corpus_id, corpus_version: $version})
WHERE a.predicate IN $predicates
WITH a ORDER BY a.assertion_id LIMIT $fact_limit
MATCH (a)-[r:SUPPORTED_BY]->(c:RFPChunk {corpus_id: $corpus_id, corpus_version: $version})
MATCH (c)-[:IN_DOCUMENT]->(d:RFPDocument {corpus_id: $corpus_id, corpus_version: $version})
WHERE d.document_id IN $document_ids AND d.document_origin IN $origins
      AND d.access_partition = 'internal'
OPTIONAL MATCH (a)-[:OBJECT]->(o:RFPDomainEntity {corpus_id: $corpus_id, corpus_version: $version})
RETURN properties(a) AS assertion, properties(o) AS object, properties(c) AS chunk,
       properties(d) AS document, r.start AS start, r.end AS end, r.span_scope AS span_scope
ORDER BY a.assertion_id LIMIT $fact_limit
"""
# Registry is immutable reviewed source, not a generic run(query) API.
READ_TEMPLATES = (HEADER, CHECK_HEAD, PROJECT_SEEDS, REQUIREMENT_SEEDS, SUBJECT_FACTS)


class SnapshotHeader(StrictModel):
    version: Sha256
    indexed_digest: Sha256
    documents: int = Field(ge=0, le=1000, strict=True)
    chunks: int = Field(ge=0, le=1000, strict=True)
    entities: int = Field(ge=0, le=1000, strict=True)
    assertions: int = Field(ge=0, le=1000, strict=True)


class GraphReadFailure(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class GraphProjection:
    header: SnapshotHeader
    subjects: tuple[dict, ...]
    facts: dict[str, tuple[dict, ...]]
    requirements: tuple[dict, ...] = ()


class GraphReader(Protocol):
    def fetch(
        self,
        corpus_id: str,
        indexed_digest: str,
        document_ids: list[str],
        plan: GraphPlan,
        *,
        scope: str,
        target: bool = False,
    ) -> GraphProjection: ...

    def current_version(self, corpus_id: str) -> str | None: ...

    def close(self) -> None: ...


def _predicates(plan, requirement=False):
    if requirement:
        return ["requires_technology", "requires_framework"]
    if plan.query_type == "shared_technologies":
        return ["uses_technology"]
    if plan.query_type == "project_fields":
        return list(plan.fields)
    return ["uses_technology", "in_industry", "mentions_framework", "timeline", "outcome"]


class Neo4jGraphReader(Neo4jGraphStore):
    """Uses ONLY reader settings and READ sessions; owns/borrows driver as before.

    Server-side privileges must also be deployed as read-only. An injected test
    driver does not prove database RBAC. Inherited write operations reject this role.
    """

    def __init__(self, settings: GraphSettings, **kwargs):
        if settings.role != "reader":
            raise GraphPermissionError("Graph queries require the dedicated reader identity")
        bounded = replace(
            settings,
            connection_timeout_seconds=min(settings.connection_timeout_seconds, 1),
            acquisition_timeout_seconds=min(settings.acquisition_timeout_seconds, 1),
            transaction_timeout_seconds=min(settings.transaction_timeout_seconds, 2),
        )
        super().__init__(bounded, **kwargs)

    def current_version(self, corpus_id):
        from pydantic import TypeAdapter

        corpus_id = TypeAdapter(Identifier).validate_python(corpus_id)
        with self._lock:
            if self._inactive_status():
                raise GraphReadFailure(self._inactive_status())
            try:
                with self._get_driver().session(
                    database=self.settings.database, default_access_mode="READ"
                ) as session:
                    with session.begin_transaction(
                        timeout=self.settings.transaction_timeout_seconds
                    ) as tx:
                        row = tx.run(CHECK_HEAD, corpus_id=corpus_id).single()
                        tx.commit()
                        return row["version"] if row else None
            except Exception as error:
                self._failed("snapshot currentness", error)
                raise GraphReadFailure("unavailable") from None

    def fetch(self, corpus_id, indexed_digest, document_ids, plan, *, scope="all", target=False):
        plan = GraphPlan.model_validate(plan.model_dump())
        # Validate identifiers and bounds before reaching the driver.
        from pydantic import TypeAdapter

        corpus_id = TypeAdapter(Identifier).validate_python(corpus_id)
        indexed_digest = TypeAdapter(Sha256).validate_python(indexed_digest)
        if (
            scope not in {"all", "sample", "upload"}
            or len(document_ids) > 1000
            or len(set(document_ids)) != len(document_ids)
        ):
            raise ValueError("Invalid graph scope or document seed bound")
        for document_id in document_ids:
            TypeAdapter(Sha256).validate_python(document_id)
        if plan.query_type == "unsupported":
            raise GraphReadFailure("unsupported_plan")
        started = monotonic()
        deadline = started + GRAPH_DEADLINE_SECONDS
        with self._lock:
            inactive = self._inactive_status()
            if inactive:
                raise GraphReadFailure(inactive)
            try:
                driver = self._get_driver()
                with driver.session(
                    database=self.settings.database, default_access_mode="READ"
                ) as session:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise GraphReadFailure("timeout")
                    with session.begin_transaction(
                        timeout=min(self.settings.transaction_timeout_seconds, remaining),
                        metadata={"operation": "rfp_graph_read", "template": plan.query_type},
                    ) as tx:
                        parameters = {"corpus_id": corpus_id, "indexed_digest": indexed_digest}
                        record = tx.run(HEADER, **parameters).single()
                        if not record:
                            raise GraphReadFailure("unsynchronized_snapshot")
                        header = SnapshotHeader.model_validate(dict(record))
                        parameters |= {
                            "version": header.version,
                            "document_ids": document_ids,
                            "origins": [scope] if scope != "all" else ["sample", "upload"],
                            "seed_limit": MAX_PROJECTS + 1,
                        }
                        total = 0

                        def bounded(statement, **params):
                            nonlocal total
                            if monotonic() >= deadline:
                                raise GraphReadFailure("timeout")
                            rows = []
                            for row in tx.run(statement, **params):
                                total += 1
                                if monotonic() >= deadline:
                                    raise GraphReadFailure("timeout")
                                if total > MAX_READ_RECORDS:
                                    raise GraphReadFailure("read_budget_exceeded")
                                rows.append(dict(row))
                            return tuple(rows)

                        subjects = () if target else bounded(PROJECT_SEEDS, **parameters)
                        requirements = (
                            bounded(REQUIREMENT_SEEDS, **parameters)
                            if target or plan.query_type == "requirement_candidates"
                            else ()
                        )
                        if len(subjects) > MAX_PROJECTS or len(requirements) > MAX_PROJECTS:
                            raise GraphReadFailure("seed_budget_exceeded")
                        facts = {}
                        for subject in (*subjects, *requirements):
                            entity_id = subject["subject"]["entity_id"]
                            rows = bounded(
                                SUBJECT_FACTS,
                                **parameters,
                                subject_id=entity_id,
                                predicates=_predicates(
                                    plan, subject["subject"]["kind"] == "Requirement"
                                ),
                                fact_limit=MAX_FACTS_PER_SUBJECT + 1,
                            )
                            if len(rows) > MAX_FACTS_PER_SUBJECT:
                                raise GraphReadFailure("fact_budget_exceeded")
                            if entity_id in facts:
                                raise GraphReadFailure("duplicate_subject_witness")
                            facts[entity_id] = rows
                        last = tx.run(CHECK_HEAD, corpus_id=corpus_id).single()
                        if not last or last["version"] != header.version:
                            raise GraphReadFailure("snapshot_changed")
                        if monotonic() >= deadline:
                            raise GraphReadFailure("timeout")
                        tx.commit()
                logger.info(
                    "Graph read template=%s records=%d elapsed_ms=%d",
                    plan.query_type,
                    total,
                    int((monotonic() - started) * 1000),
                )
                return GraphProjection(header, subjects, facts, requirements)
            except GraphReadFailure:
                raise
            except Exception as error:
                self._failed("read projection", error)
                raise GraphReadFailure("unavailable") from None


class UnavailableGraphReader:
    def __init__(self, reason="disabled"):
        self.reason = reason

    def fetch(self, *args, **kwargs):
        raise GraphReadFailure(self.reason)

    def current_version(self, corpus_id):
        raise GraphReadFailure(self.reason)

    def close(self):
        pass


def create_graph_reader(settings=None, **driver_options):
    if settings is None:
        try:
            settings = GraphSettings.from_config("reader")
        except (ValueError, TypeError):
            return UnavailableGraphReader("invalid_configuration")
    if settings.role != "reader":
        raise GraphPermissionError("Graph queries never accept writer/admin settings")
    if not settings.enabled:
        return UnavailableGraphReader()
    if not settings.configured and driver_options.get("driver") is None:
        return UnavailableGraphReader("not_configured")
    return Neo4jGraphReader(settings, **driver_options)
