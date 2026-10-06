"""Neo4j foundation unit tests: no server, embeddings, credentials or network."""

from __future__ import annotations

import builtins
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from rfp_analyst.graph import (
    GraphChunk,
    GraphDocument,
    GraphIntegrityError,
    GraphPermissionError,
    GraphSchemaError,
    GraphSettings,
    GraphStore,
    Neo4jGraphStore,
    NullGraphStore,
    create_graph_store,
)
from rfp_analyst.graph.schema import (
    CHECK_CONSTRAINTS,
    CHECK_INDEXES,
    SCHEMA_CONSTRAINTS,
    SCHEMA_INDEXES,
    SCHEMA_STATEMENTS,
    UPSERT_CHUNKS,
    UPSERT_DOCUMENTS,
)


def settings(**changes):
    return replace(
        GraphSettings(
            enabled=True, uri="bolt://localhost:17687", username="unit-user", password="unit-secret"
        ),
        **changes,
    )


def document(**changes):
    return GraphDocument(
        **{
            "corpus_id": "synthetic",
            "corpus_version": "v1",
            "document_id": "doc-1",
            "source_file": "synthetic.pdf",
            "document_origin": "sample",
            "file_hash": "a" * 64,
            **changes,
        }
    )


def chunk(**changes):
    fields = document().model_dump()
    for name in ("kind", "revision", "access_partition"):
        fields.pop(name)
    return GraphChunk(
        **{
            **fields,
            "evidence_id": "evidence-1",
            "chunk_id": "existing-chroma-id",
            "page_index": 0,
            "text_hash": "b" * 64,
            "span_start": 0,
            "span_end": 50,
            **changes,
        }
    )


def schema_records(specs):
    return [
        {
            "name": name,
            "type": "UNIQUENESS" if specs is SCHEMA_CONSTRAINTS else "RANGE",
            "entityType": "NODE",
            "labelsOrTypes": [label],
            "properties": props,
        }
        for name, (label, props) in specs.items()
    ]


class FakeConstraintError(RuntimeError):
    code = "Neo.ClientError.Schema.ConstraintValidationFailed"


class FakeResult(list):
    def single(self):
        return self[0] if self else None

    def consume(self):
        return None


class FakeTransaction:
    def __init__(self, driver):
        self.driver = driver
        self.state = deepcopy(driver.state)
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.driver.transactions_closed += 1
        if not self.committed:
            self.driver.rollbacks += 1

    def commit(self):
        self.driver.state = self.state
        self.committed = True

    def run(self, statement, **parameters):
        self.driver.calls.append((statement, parameters))
        if statement == self.driver.fail_statement:
            raise RuntimeError("sensitive driver payload: unit-secret")
        if statement == "RETURN 1 AS ok":
            return FakeResult([{"ok": 1}])
        if statement in SCHEMA_STATEMENTS:
            for category, specs in (
                ("constraints", SCHEMA_CONSTRAINTS),
                ("indexes", SCHEMA_INDEXES),
            ):
                for record in schema_records(specs):
                    if f" {record['name']} " in statement and not any(
                        existing["name"] == record["name"] for existing in self.state[category]
                    ):
                        self.state[category].append(record)
            return FakeResult()
        if statement == CHECK_CONSTRAINTS:
            return FakeResult(self.state["constraints"])
        if statement == CHECK_INDEXES:
            return FakeResult(self.state["indexes"])
        rows = parameters["rows"]
        if statement == UPSERT_DOCUMENTS:
            result = []
            for row in rows:
                key = (row["corpus_id"], row["corpus_version"], row["document_id"])
                stored = self.state["documents"].setdefault(key, deepcopy(row))
                result.append({"properties": stored})
            return FakeResult(result)
        if statement == UPSERT_CHUNKS:
            result = []
            for row in rows:
                parent_key = (row["corpus_id"], row["corpus_version"], row["document_id"])
                parent = self.state["documents"].get(parent_key)
                if not parent or any(
                    parent[name] != row[name]
                    for name in ("file_hash", "source_file", "document_origin")
                ):
                    continue
                key = (row["corpus_id"], row["corpus_version"], row["evidence_id"])
                for other_key, other in self.state["chunks"].items():
                    if (
                        other_key != key
                        and other_key[:2] == key[:2]
                        and other["chunk_id"] == row["chunk_id"]
                    ):
                        raise FakeConstraintError("conflicting mapping")
                stored = self.state["chunks"].setdefault(key, deepcopy(row))
                self.state["relationships"].add((key, parent_key))
                result.append({"properties": stored})
            return FakeResult(result)
        raise AssertionError("Unexpected non-allowlisted statement")


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.driver.sessions_closed += 1

    def begin_transaction(self, **kwargs):
        self.driver.transaction_options.append(kwargs)
        return FakeTransaction(self.driver)


class FakeDriver:
    def __init__(self):
        self.state = {
            "documents": {},
            "chunks": {},
            "relationships": set(),
            "constraints": schema_records(SCHEMA_CONSTRAINTS),
            "indexes": schema_records(SCHEMA_INDEXES),
        }
        self.calls = []
        self.session_options = []
        self.transaction_options = []
        self.sessions_closed = 0
        self.transactions_closed = 0
        self.rollbacks = 0
        self.close_count = 0
        self.fail_statement = None
        self.verify_error = None

    def verify_connectivity(self):
        if self.verify_error:
            raise self.verify_error

    def session(self, **kwargs):
        self.session_options.append(kwargs)
        return FakeSession(self)

    def close(self):
        self.close_count += 1


@pytest.fixture
def driver():
    return FakeDriver()


def test_disabled_store_is_noop_without_driver_import(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "neo4j" or name.startswith("neo4j."):
            raise AssertionError("Disabled graph must not import the optional driver")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    factory = Mock(side_effect=AssertionError("Unexpected connection"))
    with create_graph_store(GraphSettings(), driver_factory=factory) as store:
        assert isinstance(store, GraphStore)
        assert store.health().status == "disabled"
        assert not store.initialize_schema().ok
        assert not store.ingest_provenance([document()], [chunk()]).ok
    assert store.health().status == "closed"
    factory.assert_not_called()


def test_enabled_without_credentials_is_explicit_noop():
    factory = Mock()
    store = create_graph_store(GraphSettings(enabled=True), driver_factory=factory)
    assert isinstance(store, NullGraphStore)
    assert store.health().status == "not_configured"
    factory.assert_not_called()


def test_null_store_cannot_claim_persistence_success():
    with pytest.raises(ValueError):
        NullGraphStore("ready")


def test_missing_dependency_is_safe(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "neo4j":
            raise ImportError("missing optional package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    store = create_graph_store(settings())
    assert store.health().status == "dependency_missing"


def test_driver_creation_is_lazy_and_instance_local(driver):
    other_driver = FakeDriver()
    factory = Mock(side_effect=[driver, other_driver])
    first = create_graph_store(settings(), driver_factory=factory)
    second = create_graph_store(settings(), driver_factory=factory)
    factory.assert_not_called()
    assert first.health().available
    assert factory.call_count == 1
    assert second.health().available
    assert factory.call_count == 2
    first.close()
    second.close()
    assert driver.close_count == other_driver.close_count == 1


def test_default_driver_has_bounded_pool_and_no_hidden_retry(monkeypatch, driver):
    factory = Mock(return_value=driver)
    monkeypatch.setitem(
        sys.modules,
        "neo4j",
        SimpleNamespace(GraphDatabase=SimpleNamespace(driver=factory)),
    )
    with create_graph_store(settings()) as store:
        assert store.health().available
    factory.assert_called_once_with(
        "bolt://localhost:17687",
        auth=("unit-user", "unit-secret"),
        connection_timeout=1.0,
        connection_acquisition_timeout=2.0,
        max_connection_pool_size=10,
        max_transaction_retry_time=0,
    )
    assert driver.close_count == 1


def test_health_checks_database_read_mode_and_transaction_timeout(driver):
    store = Neo4jGraphStore(settings(database="synthetic"), driver=driver)
    assert store.health().available
    assert driver.session_options == [{"database": "synthetic", "default_access_mode": "READ"}]
    assert driver.transaction_options == [{"timeout": 2.0}]
    assert driver.sessions_closed == driver.transactions_closed == 1


def test_unavailable_driver_is_redacted_and_borrowed_not_closed(driver, caplog):
    driver.verify_error = RuntimeError("unit-secret bolt://unit-user:unit-secret@host/private")
    store = Neo4jGraphStore(settings(), driver=driver)
    assert store.health().status == "unavailable"
    assert "unit-secret" not in caplog.text
    assert "bolt://" not in caplog.text
    assert driver.close_count == 0
    driver.verify_error = None
    assert store.health().available


def test_factory_failure_degrades_safely(caplog):
    factory = Mock(side_effect=RuntimeError("unit-secret"))
    store = create_graph_store(settings(), driver_factory=factory)
    assert store.health().status == "unavailable"
    assert "unit-secret" not in caplog.text


def test_owned_driver_closed_on_failure_and_can_recover(driver):
    driver.verify_error = RuntimeError("offline")
    healthy = FakeDriver()
    factory = Mock(side_effect=[driver, healthy])
    store = create_graph_store(settings(), driver_factory=factory)
    assert store.health().status == "unavailable"
    assert driver.close_count == 1
    assert store.health().available
    store.close()
    assert healthy.close_count == 1


@pytest.mark.parametrize("owned,expected_closes", [(False, 0), (True, 1)])
def test_close_is_idempotent_and_obeys_ownership(driver, owned, expected_closes):
    store = Neo4jGraphStore(settings(role="admin"), driver=driver, owns_driver=owned)
    with store:
        assert store.health().available
    store.close()
    assert driver.close_count == expected_closes
    assert store.health().status == "closed"
    assert store.initialize_schema().status == "closed"
    assert store.ingest_provenance([document()], [chunk()]).status == "closed"


def test_close_before_connection_does_not_create_driver():
    factory = Mock()
    store = create_graph_store(settings(), driver_factory=factory)
    store.close()
    assert store.health().status == "closed"
    factory.assert_not_called()


@pytest.mark.parametrize("role", ["reader", "writer"])
def test_only_explicit_admin_can_initialize_schema(driver, role):
    store = Neo4jGraphStore(settings(role=role), driver=driver)
    with pytest.raises(GraphPermissionError):
        store.initialize_schema()
    assert not driver.calls


def test_reader_cannot_ingest(driver):
    store = Neo4jGraphStore(settings(), driver=driver)
    with pytest.raises(GraphPermissionError):
        store.ingest_provenance([document()], [chunk()])
    assert not driver.calls


def test_schema_initialization_is_fixed_repeatable_and_verified(driver):
    store = Neo4jGraphStore(settings(role="admin"), driver=driver)
    assert store.initialize_schema().ok
    assert store.initialize_schema().ok
    ddl = [statement for statement, _ in driver.calls if statement in SCHEMA_STATEMENTS]
    assert ddl == list(SCHEMA_STATEMENTS) * 2
    assert all("IF NOT EXISTS" in statement for statement in ddl)
    assert driver.sessions_closed == len(driver.session_options)
    assert driver.transactions_closed == len(driver.transaction_options)


def test_schema_failure_is_safe_and_does_not_report_success(driver, caplog):
    driver.fail_statement = SCHEMA_STATEMENTS[1]
    store = Neo4jGraphStore(settings(role="admin"), driver=driver)
    assert store.initialize_schema().status == "unavailable"
    assert "unit-secret" not in caplog.text


def test_schema_can_initialize_from_empty_database(driver):
    driver.state["constraints"] = []
    driver.state["indexes"] = []
    store = Neo4jGraphStore(settings(role="admin"), driver=driver)
    assert store.initialize_schema().ok
    assert len(driver.state["constraints"]) == len(driver.state["indexes"]) == 3


def test_existing_incompatible_index_is_not_silently_accepted(driver):
    driver.state["indexes"][0]["type"] = "FULLTEXT"
    store = Neo4jGraphStore(settings(role="admin"), driver=driver)
    with pytest.raises(GraphSchemaError):
        store.initialize_schema()
    assert driver.state["indexes"][0]["type"] == "FULLTEXT"


@pytest.mark.parametrize("change", ["missing", "wrong_label", "wrong_properties", "wrong_type"])
def test_ingestion_requires_actual_unique_constraints(driver, change):
    if change == "missing":
        driver.state["constraints"] = []
    else:
        key, value = {
            "wrong_label": ("labelsOrTypes", ["Other"]),
            "wrong_properties": ("properties", ["document_id"]),
            "wrong_type": ("type", "EXISTENCE"),
        }[change]
        driver.state["constraints"][0][key] = value
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    with pytest.raises(GraphSchemaError):
        store.ingest_provenance([document()], [chunk()])
    assert not driver.state["documents"]


def test_ingestion_is_idempotent_and_preserves_all_provenance(driver):
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    first = store.ingest_provenance([document(), document()], [chunk(), chunk()])
    second = store.ingest_provenance([document()], [chunk()])
    assert first == second
    assert first.ok and first.documents_processed == first.chunks_processed == 1
    assert (
        len(driver.state["documents"])
        == len(driver.state["chunks"])
        == len(driver.state["relationships"])
        == 1
    )
    assert next(iter(driver.state["chunks"].values())) == chunk().model_dump()


def test_later_chunks_can_reference_an_existing_parent(driver):
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    assert store.ingest_provenance([document()], []).ok
    assert store.ingest_provenance([], [chunk()]).ok


@pytest.mark.parametrize(
    "changes", [{"document_id": "missing"}, {"file_hash": "c" * 64}, {"document_origin": "upload"}]
)
def test_orphan_or_mismatched_parent_rolls_back_entire_batch(driver, changes):
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    with pytest.raises(GraphIntegrityError):
        store.ingest_provenance([document()], [chunk(**changes)])
    assert not driver.state["documents"]
    assert not driver.state["chunks"]
    assert driver.rollbacks == 1


@pytest.mark.parametrize("record_type", ["document", "chunk"])
def test_changed_frozen_identity_is_rejected_not_overwritten(driver, record_type):
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    store.ingest_provenance([document()], [chunk()])
    before = deepcopy(driver.state)
    with pytest.raises(GraphIntegrityError):
        if record_type == "document":
            store.ingest_provenance([document(kind="proposal")], [])
        else:
            store.ingest_provenance([], [chunk(span_end=51)])
    assert driver.state == before


def test_conflicting_legacy_mapping_is_rejected(driver):
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    with pytest.raises(GraphIntegrityError):
        store.ingest_provenance([document()], [chunk(), chunk(evidence_id="another")])
    assert not driver.calls
    store.ingest_provenance([document()], [chunk()])
    with pytest.raises(GraphIntegrityError):
        store.ingest_provenance([], [chunk(evidence_id="another")])
    assert len(driver.state["chunks"]) == 1


def test_conflicting_duplicate_input_is_rejected_before_database_access(driver):
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    with pytest.raises(GraphIntegrityError):
        store.ingest_provenance([document(), document(kind="proposal")], [])
    assert not driver.calls


def test_corpus_versions_and_origins_are_not_merged(driver):
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    store.ingest_provenance([document()], [chunk()])
    store.ingest_provenance(
        [document(corpus_version="v2", document_origin="upload")],
        [chunk(corpus_version="v2", document_origin="upload")],
    )
    assert len(driver.state["documents"]) == len(driver.state["chunks"]) == 2


def test_bounded_batch_rejects_before_database_access(driver):
    store = Neo4jGraphStore(settings(role="writer", max_batch_size=1), driver=driver)
    with pytest.raises(ValueError, match="bound"):
        store.ingest_provenance([document()], [chunk()])
    assert not driver.calls


def test_models_required_at_ingestion_boundary(driver):
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    with pytest.raises(TypeError):
        store.ingest_provenance([document().model_dump()], [])
    assert not driver.calls


@pytest.mark.parametrize("record_type", ["document", "chunk"])
def test_bypassed_model_validation_is_rechecked_before_writes(driver, record_type):
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    with pytest.raises(ValidationError):
        if record_type == "document":
            store.ingest_provenance([document().model_copy(update={"file_hash": "invalid"})], [])
        else:
            store.ingest_provenance([], [chunk().model_copy(update={"page_index": -1})])
    assert not driver.calls


def test_ingestion_failure_rolls_back_and_reports_unavailable(driver, caplog):
    driver.fail_statement = UPSERT_CHUNKS
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    result = store.ingest_provenance([document()], [chunk()])
    assert result.status == "unavailable" and not result.ok
    assert result.documents_processed == result.chunks_processed == 0
    assert not driver.state["documents"]
    assert "unit-secret" not in caplog.text


def test_input_values_are_parameters_never_query_text(driver):
    malicious_id = "id'}) DETACH DELETE n //"
    store = Neo4jGraphStore(settings(role="writer"), driver=driver)
    assert store.ingest_provenance(
        [document(document_id=malicious_id)], [chunk(document_id=malicious_id)]
    ).ok
    assert all(malicious_id not in statement for statement, _ in driver.calls)
    assert any(
        malicious_id == row["document_id"]
        for _, params in driver.calls
        for row in params.get("rows", [])
    )
    assert not hasattr(store, "query") and not hasattr(store, "run_cypher")


@pytest.mark.parametrize(
    "fields",
    [
        {"page_index": -1},
        {"page_index": True},
        {"span_end": 0},
        {"span_start": 10, "span_end": 9},
        {"source_file": "../private.pdf"},
        {"source_file": "C:\\private.pdf"},
        {"file_hash": "not-sha256"},
        {"document_origin": "web"},
        {"corpus_id": " "},
    ],
)
def test_invalid_provenance_is_rejected(fields):
    with pytest.raises(ValidationError):
        chunk(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"enabled": "false"},
        {"role": "root"},
        {"connection_timeout_seconds": 0},
        {"transaction_timeout_seconds": float("nan")},
        {"max_batch_size": 1001},
        {"max_connection_pool_size": 0},
        {"database": " "},
        {"uri": "https://localhost"},
        {"uri": "bolt://remote.example:7687"},
        {"uri": "neo4j+ssc://remote.example"},
        {"uri": "neo4j+s://user:secret@remote.example"},
    ],
)
def test_unsafe_or_unbounded_settings_are_rejected(fields):
    with pytest.raises(ValueError):
        settings(**fields)


def test_settings_repr_never_contains_connection_secrets():
    assert "unit-secret" not in repr(settings())
    assert "unit-user" not in repr(settings())
    assert "bolt://" not in repr(settings())
    assert settings(uri="neo4j+s://remote.example").configured


def test_config_resolves_roles_separately_and_dynamically(monkeypatch):
    import config

    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets={}))
    monkeypatch.setenv("NEO4J_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:17687")
    monkeypatch.setenv("NEO4J_USERNAME", "reader")
    monkeypatch.setenv("NEO4J_PASSWORD", "reader-secret")
    monkeypatch.delenv("NEO4J_INGEST_USERNAME", raising=False)
    monkeypatch.delenv("NEO4J_INGEST_PASSWORD", raising=False)
    assert config.get_neo4j_settings()["password"] == "reader-secret"
    assert not config.get_neo4j_settings("writer")["password"]
    assert create_graph_store(role="writer").health().status == "not_configured"
    monkeypatch.setenv("NEO4J_INGEST_USERNAME", "worker")
    monkeypatch.setenv("NEO4J_INGEST_PASSWORD", "worker-secret")
    assert config.get_neo4j_settings("writer")["password"] == "worker-secret"
    monkeypatch.setenv("NEO4J_ENABLED", "false")
    assert config.get_neo4j_settings()["password"] == ""
    assert create_graph_store().health().status == "disabled"


def test_config_secret_precedence_and_invalid_config_degrade(monkeypatch):
    import config

    monkeypatch.setenv("NEO4J_PASSWORD", "env-secret")
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        SimpleNamespace(
            secrets={
                "NEO4J_ENABLED": "true",
                "NEO4J_PASSWORD": "secret-priority",
                "NEO4J_URI": "bolt://localhost:17687",
                "NEO4J_USERNAME": "reader",
            }
        ),
    )
    assert config.get_neo4j_settings()["password"] == "secret-priority"
    monkeypatch.setenv("NEO4J_TRANSACTION_TIMEOUT_SECONDS", "nan")
    assert create_graph_store().health().status == "invalid_configuration"


@pytest.mark.parametrize("enabled", ["false", "true"])
def test_existing_app_starts_without_constructing_graph_driver(monkeypatch, enabled):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NEO4J_ENABLED", enabled)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:17687")
    monkeypatch.setenv("NEO4J_USERNAME", "reader")
    monkeypatch.setenv("NEO4J_PASSWORD", "not-a-real-password")
    with (
        patch(
            "rfp_analyst.graph.store._default_driver_factory",
            side_effect=AssertionError("Unexpected graph dependency"),
        ) as factory,
        patch("config.get_api_keys", return_value=("", "")),
        patch("rag_engine.get_vectorstore_stats", return_value={"status": "not_initialized"}),
    ):
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=10)
    assert not at.exception
    factory.assert_not_called()
