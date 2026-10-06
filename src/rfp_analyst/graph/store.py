"""Small graph persistence boundary, deliberately independent of LangGraph.

No arbitrary query API, graph retrieval, extraction, or application startup hook.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal, Protocol, runtime_checkable

from rfp_analyst.exceptions import RFPAnalystError

from .models import GraphChunk, GraphDocument
from .schema import (
    CHECK_CONSTRAINTS,
    CHECK_INDEXES,
    SCHEMA_CONSTRAINTS,
    SCHEMA_INDEXES,
    SCHEMA_STATEMENTS,
    UPSERT_CHUNKS,
    UPSERT_DOCUMENTS,
)
from .settings import GraphRole, GraphSettings

logger = logging.getLogger(__name__)
GraphStatus = Literal[
    "disabled",
    "not_configured",
    "invalid_configuration",
    "dependency_missing",
    "ready",
    "unavailable",
    "closed",
]


class GraphPermissionError(RFPAnalystError):
    """A reader attempted ingestion, or a non-admin attempted schema migration."""


class GraphIntegrityError(RFPAnalystError):
    """A frozen identity changed or a chunk lacks a matching provenance parent."""


class GraphSchemaError(RFPAnalystError):
    """The prerequisite uniqueness constraints/indexes are absent or incompatible."""


class _DependencyMissingError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraphHealth:
    status: GraphStatus

    @property
    def available(self) -> bool:
        return self.status == "ready"


@dataclass(frozen=True)
class GraphWriteResult:
    status: GraphStatus
    documents_processed: int = 0
    chunks_processed: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ready"


@runtime_checkable
class GraphStore(Protocol):
    def health(self) -> GraphHealth: ...

    def initialize_schema(self) -> GraphWriteResult: ...

    def ingest_provenance(
        self, documents: Sequence[GraphDocument], chunks: Sequence[GraphChunk]
    ) -> GraphWriteResult: ...

    def close(self) -> None: ...

    def __enter__(self) -> GraphStore: ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...


class NullGraphStore:
    """Explicit disabled/unconfigured result, never pretends to persist data."""

    def __init__(self, status: GraphStatus = "disabled"):
        if status not in {
            "disabled",
            "not_configured",
            "invalid_configuration",
            "dependency_missing",
            "unavailable",
        }:
            raise ValueError("A null graph store cannot claim persistence readiness")
        self._status = status
        self._closed = False

    def health(self) -> GraphHealth:
        return GraphHealth("closed" if self._closed else self._status)

    def initialize_schema(self) -> GraphWriteResult:
        return GraphWriteResult(self.health().status)

    def ingest_provenance(
        self, documents: Sequence[GraphDocument], chunks: Sequence[GraphChunk]
    ) -> GraphWriteResult:
        return GraphWriteResult(self.health().status)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> NullGraphStore:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _default_driver_factory(settings: GraphSettings):
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise _DependencyMissingError("Install the optional graph dependency") from None
    return GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
        connection_timeout=settings.connection_timeout_seconds,
        connection_acquisition_timeout=settings.acquisition_timeout_seconds,
        max_connection_pool_size=settings.max_connection_pool_size,
        max_transaction_retry_time=0,
    )


class Neo4jGraphStore:
    """Instance-owned lazy driver; injected drivers are borrowed by default.

    Application roles are an additional guard, not database authorization.
    Explicit transactions disable hidden managed-transaction retry loops.
    The per-instance lock makes lazy creation/close safe; each call owns its
    session/transaction. Close only after the caller has finished all work.
    """

    def __init__(
        self,
        settings: GraphSettings,
        *,
        driver=None,
        driver_factory: Callable[[GraphSettings], Any] | None = None,
        owns_driver: bool = False,
    ):
        self.settings = settings
        self._driver = driver
        self._driver_factory = driver_factory or _default_driver_factory
        self._owns_driver = driver is None or owns_driver
        self._closed = False
        self._lock = RLock()

    def _inactive_status(self) -> GraphStatus | None:
        if self._closed:
            return "closed"
        if not self.settings.enabled:
            return "disabled"
        if self._driver is None and not self.settings.configured:
            return "not_configured"
        return None

    def _get_driver(self):
        if self._driver is None:
            self._driver = self._driver_factory(self.settings)
        return self._driver

    def _close_driver(self) -> None:
        driver = self._driver
        if driver is not None and self._owns_driver:
            self._driver = None
            try:
                driver.close()
            except Exception:
                logger.warning("Neo4j driver close failed; details redacted")

    def _failed(self, operation: str, error: Exception) -> GraphStatus:
        # Never log the driver's exception text, URI, credentials or row data.
        logger.warning("Neo4j %s unavailable; details redacted", operation)
        self._close_driver()
        return "dependency_missing" if isinstance(error, _DependencyMissingError) else "unavailable"

    def health(self) -> GraphHealth:
        with self._lock:
            inactive = self._inactive_status()
            if inactive:
                return GraphHealth(inactive)
            try:
                driver = self._get_driver()
                driver.verify_connectivity()
                with driver.session(
                    database=self.settings.database, default_access_mode="READ"
                ) as session:
                    with session.begin_transaction(
                        timeout=self.settings.transaction_timeout_seconds
                    ) as tx:
                        record = tx.run("RETURN 1 AS ok").single()
                        if not record or record["ok"] != 1:
                            raise RuntimeError("Graph health probe failed")
                        tx.commit()
                return GraphHealth("ready")
            except Exception as error:
                return GraphHealth(self._failed("health check", error))

    def _require_schema(self, driver, *, check_indexes: bool = False) -> None:
        with driver.session(database=self.settings.database, default_access_mode="READ") as session:
            with session.begin_transaction(timeout=self.settings.transaction_timeout_seconds) as tx:
                checks = [(CHECK_CONSTRAINTS, SCHEMA_CONSTRAINTS)]
                if check_indexes:
                    checks.append((CHECK_INDEXES, SCHEMA_INDEXES))
                for statement, specs in checks:
                    records = {
                        record["name"]: record for record in tx.run(statement, names=list(specs))
                    }
                    for name, (label, properties) in specs.items():
                        record = records.get(name)
                        if (
                            not record
                            or record["entityType"] != "NODE"
                            or record["labelsOrTypes"] != [label]
                            or record["properties"] != properties
                            or (
                                statement == CHECK_CONSTRAINTS
                                and record["type"] not in {"UNIQUENESS", "NODE_PROPERTY_UNIQUENESS"}
                            )
                            or (statement == CHECK_INDEXES and record["type"] != "RANGE")
                        ):
                            raise GraphSchemaError(
                                "Initialize/repair the graph schema with an admin repository"
                            )
                tx.commit()

    def initialize_schema(self) -> GraphWriteResult:
        with self._lock:
            inactive = self._inactive_status()
            if inactive:
                return GraphWriteResult(inactive)
            if self.settings.role != "admin":
                raise GraphPermissionError(
                    "Schema initialization requires an explicit admin repository"
                )
            try:
                driver = self._get_driver()
                with driver.session(
                    database=self.settings.database, default_access_mode="WRITE"
                ) as session:
                    # Schema changes and data writes deliberately use separate transactions.
                    for statement in SCHEMA_STATEMENTS:
                        with session.begin_transaction(
                            timeout=self.settings.transaction_timeout_seconds
                        ) as tx:
                            tx.run(statement).consume()
                            tx.commit()
                self._require_schema(driver, check_indexes=True)
                return GraphWriteResult("ready")
            except GraphSchemaError:
                raise
            except Exception as error:
                return GraphWriteResult(self._failed("schema initialization", error))

    @staticmethod
    def _rows(records, identity_field: str, model_type) -> list[dict]:
        unique = {}
        for record in records:
            # Pydantic model_copy/model_construct can bypass validation. Recheck
            # serialized values at the persistence boundary, not just isinstance.
            row = model_type.model_validate(record.model_dump()).model_dump()
            key = (row["corpus_id"], row["corpus_version"], row[identity_field])
            if key in unique and unique[key] != row:
                raise GraphIntegrityError("Conflicting provenance records in one batch")
            unique[key] = row
        return list(unique.values())

    @staticmethod
    def _validate_persisted(result, rows: list[dict], identity_field: str) -> None:
        expected = {
            (row["corpus_id"], row["corpus_version"], row[identity_field]): row for row in rows
        }
        seen = set()
        for record in result:
            properties = record["properties"]
            key = (
                properties["corpus_id"],
                properties["corpus_version"],
                properties[identity_field],
            )
            wanted = expected.get(key)
            if wanted is None or any(
                properties.get(name) != value for name, value in wanted.items()
            ):
                raise GraphIntegrityError("Frozen provenance identity conflicts with stored data")
            seen.add(key)
        if seen != set(expected):
            raise GraphIntegrityError("A chunk has no matching document provenance parent")

    def ingest_provenance(
        self, documents: Sequence[GraphDocument], chunks: Sequence[GraphChunk]
    ) -> GraphWriteResult:
        with self._lock:
            inactive = self._inactive_status()
            if inactive:
                return GraphWriteResult(inactive)
            if self.settings.role not in {"writer", "admin"}:
                raise GraphPermissionError(
                    "Provenance ingestion requires an explicit writer repository"
                )
            if len(documents) + len(chunks) > self.settings.max_batch_size:
                raise ValueError("Graph ingestion batch exceeds its configured bound")
            if any(not isinstance(item, GraphDocument) for item in documents) or any(
                not isinstance(item, GraphChunk) for item in chunks
            ):
                raise TypeError("Graph ingestion requires validated provenance models")
            doc_rows = self._rows(documents, "document_id", GraphDocument)
            chunk_rows = self._rows(chunks, "evidence_id", GraphChunk)
            # A portable mapping must not reassign one legacy chunk ID within a version.
            legacy_rows = self._rows(chunks, "chunk_id", GraphChunk)
            if len(legacy_rows) != len(chunk_rows):
                raise GraphIntegrityError("Conflicting legacy chunk identity mapping")
            try:
                driver = self._get_driver()
                self._require_schema(driver)
                with driver.session(
                    database=self.settings.database, default_access_mode="WRITE"
                ) as session:
                    with session.begin_transaction(
                        timeout=self.settings.transaction_timeout_seconds
                    ) as tx:
                        if doc_rows:
                            self._validate_persisted(
                                tx.run(UPSERT_DOCUMENTS, rows=doc_rows), doc_rows, "document_id"
                            )
                        if chunk_rows:
                            self._validate_persisted(
                                tx.run(UPSERT_CHUNKS, rows=chunk_rows), chunk_rows, "evidence_id"
                            )
                        tx.commit()
                return GraphWriteResult("ready", len(doc_rows), len(chunk_rows))
            except (GraphIntegrityError, GraphSchemaError):
                # Transaction context rolls back all data changes, including new documents.
                raise
            except Exception as error:
                if str(getattr(error, "code", "")).startswith("Neo.ClientError.Schema.Constraint"):
                    raise GraphIntegrityError(
                        "Graph uniqueness constraints rejected the provenance batch"
                    ) from None
                return GraphWriteResult(self._failed("provenance ingestion", error))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._close_driver()

    def __enter__(self) -> Neo4jGraphStore:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def create_graph_store(
    settings: GraphSettings | None = None,
    *,
    role: GraphRole = "reader",
    driver=None,
    driver_factory: Callable[[GraphSettings], Any] | None = None,
    owns_driver: bool = False,
) -> GraphStore:
    """Safe composition boundary; never connects or initializes schema at startup."""
    if settings is None:
        try:
            settings = GraphSettings.from_config(role)
        except (ValueError, TypeError):
            logger.warning("Neo4j configuration invalid; graph persistence disabled")
            return NullGraphStore("invalid_configuration")
    if not settings.enabled:
        return NullGraphStore()
    if driver is None and not settings.configured:
        return NullGraphStore("not_configured")
    return Neo4jGraphStore(
        settings, driver=driver, driver_factory=driver_factory, owns_driver=owns_driver
    )
