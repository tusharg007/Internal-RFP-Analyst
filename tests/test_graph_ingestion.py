"""Domain ingestion, provenance, frozen snapshots and transaction safety."""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from rfp_analyst.graph.extraction import (
    ChunkExtraction,
    SectionExtractor,
    StructuredLLMExtractor,
    canonicalize,
)
from rfp_analyst.graph.ingestion import (
    ExtractionRejected,
    IndexedChunk,
    build_snapshot,
    digest,
    enqueue_graph_sync,
    read_indexed_corpus,
    rebuild_graph,
)
from rfp_analyst.graph.repository import (
    DOMAIN_CONSTRAINTS,
    DOMAIN_INDEXES,
    DOMAIN_STATEMENTS,
    LOCK_HEAD,
    PUBLISH_HEAD,
    READ_HEAD,
    UPSERT_ASSERTIONS,
    UPSERT_ENTITIES,
    UPSERT_IDENTITIES,
    UPSERT_OBJECTS,
    UPSERT_SNAPSHOT,
    Neo4jIngestionRepository,
    create_ingestion_repository,
)
from rfp_analyst.graph.schema import CHECK_CONSTRAINTS, CHECK_INDEXES
from rfp_analyst.graph.store import GraphIntegrityError, GraphPermissionError, GraphSchemaError
from tests.test_graph_store import FakeDriver, FakeResult, FakeSession, FakeTransaction, settings


def indexed(text, *, name="case.pdf", page=0, index=0, origin="sample"):
    return IndexedChunk(
        chunk_id=digest([name, page, index, text, origin]),
        source_file=name,
        document_origin=origin,
        file_hash=digest(name),
        page_index=page,
        chunk_index=index,
        text=text,
    )


def project(name="case.pdf", tech="Microsoft Azure", *, origin="sample"):
    return [
        indexed(
            "CONFIDENTIAL | CASE STUDY | Banking\nDigital Audit\nDocument Type: Case Study\nIndustry: Banking",
            name=name,
            origin=origin,
        ),
        indexed(
            f"Technology Stack\n- Cloud: {tech}\n- BI: Power BI\nTimeline & Milestones\nTotal Duration: 16 weeks",
            name=name,
            page=1,
            index=1,
            origin=origin,
        ),
        indexed(
            "Budget Range\nEstimated project cost: $850,000 - $1,100,000 USD\nKey Outcomes\n- Reduced audit cycle by 35%\n- Projected savings of 40%",
            name=name,
            page=2,
            index=2,
            origin=origin,
        ),
    ]


def domain_records(specs, category):
    return [
        dict(
            name=name,
            type="UNIQUENESS" if category == "constraints" else "RANGE",
            entityType="NODE",
            labelsOrTypes=[label],
            properties=properties,
        )
        for name, (label, properties) in specs.items()
    ]


class IngestionTransaction(FakeTransaction):
    def run(self, statement, **parameters):
        if statement == self.driver.fail_statement:
            raise RuntimeError("sensitive provider details")
        if statement in DOMAIN_STATEMENTS:
            self.driver.calls.append((statement, parameters))
            for category, specs in (
                ("constraints", DOMAIN_CONSTRAINTS),
                ("indexes", DOMAIN_INDEXES),
            ):
                for record in domain_records(specs, category):
                    if f" {record['name']} " in statement and not any(
                        r["name"] == record["name"] for r in self.state[category]
                    ):
                        self.state[category].append(record)
            return FakeResult()
        if statement in {CHECK_CONSTRAINTS, CHECK_INDEXES}:
            self.driver.calls.append((statement, parameters))
            category = "constraints" if statement == CHECK_CONSTRAINTS else "indexes"
            return FakeResult(
                [row for row in self.state[category] if row["name"] in parameters["names"]]
            )
        if statement in {READ_HEAD, LOCK_HEAD, PUBLISH_HEAD}:
            self.driver.calls.append((statement, parameters))
            corpus_id = parameters["corpus_id"]
            if statement == READ_HEAD:
                return FakeResult(
                    [{"version": self.state["heads"][corpus_id]}]
                    if corpus_id in self.state["heads"]
                    else []
                )
            if statement == LOCK_HEAD:
                return FakeResult([{"version": self.state["heads"].setdefault(corpus_id, "")}])
            self.state["heads"][corpus_id] = parameters["version"]
            return FakeResult([{"version": parameters["version"]}])
        if statement == UPSERT_SNAPSHOT:
            self.driver.calls.append((statement, parameters))
            row = parameters["row"]
            key = (row["corpus_id"], row["corpus_version"])
            return FakeResult(
                [{"properties": self.state["snapshots"].setdefault(key, deepcopy(row))}]
            )
        if statement in {UPSERT_ENTITIES, UPSERT_ASSERTIONS}:
            self.driver.calls.append((statement, parameters))
            kind = "entities" if statement == UPSERT_ENTITIES else "assertions"
            identity = "entity_id" if statement == UPSERT_ENTITIES else "assertion_id"
            result = []
            for row in parameters["rows"]:
                key = (row["corpus_id"], row["corpus_version"], row[identity])
                if kind == "assertions":
                    source_key = key[:2] + (row["evidence_id"],)
                    subject_key = key[:2] + (row["subject_id"],)
                    if (
                        source_key not in self.state["chunks"]
                        or subject_key not in self.state["entities"]
                    ):
                        continue
                result.append({"properties": self.state[kind].setdefault(key, deepcopy(row))})
            return FakeResult(result)
        if statement in {UPSERT_IDENTITIES, UPSERT_OBJECTS}:
            self.driver.calls.append((statement, parameters))
            for row in parameters["rows"]:
                self.state["domain_edges"].add((statement, tuple(sorted(row.items()))))
            return FakeResult([{"count": len(parameters["rows"])}])
        return super().run(statement, **parameters)


class IngestionSession(FakeSession):
    def begin_transaction(self, **kwargs):
        self.driver.transaction_options.append(kwargs)
        return IngestionTransaction(self.driver)


class IngestionDriver(FakeDriver):
    def __init__(self):
        super().__init__()
        self.state |= {
            "heads": {},
            "snapshots": {},
            "entities": {},
            "assertions": {},
            "domain_edges": set(),
        }
        self.state["constraints"] += domain_records(DOMAIN_CONSTRAINTS, "constraints")
        self.state["indexes"] += domain_records(DOMAIN_INDEXES, "indexes")

    def session(self, **kwargs):
        self.session_options.append(kwargs)
        return IngestionSession(self)


@pytest.fixture
def repository():
    driver = IngestionDriver()
    with Neo4jIngestionRepository(settings(role="writer"), driver=driver) as repo:
        yield repo, driver


def test_repeat_ingestion_is_idempotent_and_retry_safe(repository):
    repo, driver = repository
    snapshot = build_snapshot("synthetic", project())
    assert repo.publish(snapshot, expected_version=None) == "ready"
    before = deepcopy(driver.state)
    # Simulates retry with the original head after an ambiguous commit acknowledgement.
    assert repo.publish(snapshot, expected_version=None) == "ready"
    assert driver.state == before
    assert repo.active_version("synthetic") == snapshot.version
    assert (
        build_snapshot("synthetic", list(reversed(project())) + project()).version
        == snapshot.version
    )


@pytest.mark.parametrize("alias", ["Azure", "MS Azure", "Microsoft Azure", " microsoft   azure "])
def test_alias_normalization(alias):
    assert canonicalize("Technology", alias) == "Microsoft Azure"


def test_duplicate_entities_and_same_entity_across_documents(repository):
    snapshot = build_snapshot(
        "synthetic", project("one.pdf", "MS Azure") + project("two.pdf", "Azure")
    )
    azure = [entity for entity in snapshot.entities if entity.name == "Microsoft Azure"]
    assert len(azure) == 1
    assert len([entity for entity in snapshot.entities if entity.kind == "Project"]) == 2
    assert (
        len({a.subject_id for a in snapshot.assertions if a.object_id == azure[0].entity_id}) == 2
    )
    assert repository[0].publish(snapshot, expected_version=None) == "ready"


def test_products_do_not_imply_provider_or_other_products():
    snapshot = build_snapshot("synthetic", project(tech="Azure SQL Database"))
    names = {entity.name for entity in snapshot.entities}
    assert "Azure SQL Database" in names and "Microsoft Azure" not in names
    assert "Azure Data Factory" not in names


def test_provenance_and_ingestion_metadata_are_preserved(repository):
    inputs = project(origin="upload")
    snapshot = build_snapshot("synthetic", inputs)
    assert {c.chunk_id for c in snapshot.chunks} == {c.chunk_id for c in inputs}
    assert all(c.document_origin == "upload" and c.span_scope == "chunk" for c in snapshot.chunks)
    assert {c.page_index for c in snapshot.chunks} == {0, 1, 2}
    assert snapshot.documents[0].revision == inputs[0].file_hash
    for assertion in snapshot.assertions:
        source = next(i for i in inputs if i.evidence_id == assertion.evidence_id)
        assert source.text[assertion.fact.start : assertion.fact.end] == assertion.fact.quote
    repo, driver = repository
    assert repo.publish(snapshot, expected_version=None) == "ready"
    manifest = driver.state["snapshots"][("synthetic", snapshot.version)]
    assert manifest["extractor_version"] == snapshot.extractor_version
    assert manifest["normalization_version"] == snapshot.normalization_version
    assert manifest["embedding_version"] == "not_recorded_in_legacy_index"
    outcomes = {a.fact.modality for a in snapshot.assertions if a.fact.predicate == "outcome"}
    assert outcomes == {"achieved", "projected"}


@pytest.mark.parametrize(
    "mutation",
    [
        {"predicate": "SATISFIES"},
        {"label": "Client"},
        {"confidence": 0.2},
        {"quote": "invented"},
        {"object_name": "Invented DB"},
        {"end": 10000},
        {"object_name": "Power BI"},
        {"subject_kind": "Requirement"},
    ],
)
def test_malformed_llm_extraction_is_quarantined_without_any_write(repository, mutation):
    repo, driver = repository
    model = Mock()

    def respond(prompt):
        text = json.loads(prompt.split("UNTRUSTED CHUNK (JSON string): ", 1)[1])
        raw = (
            SectionExtractor().extract(text, project=True, target_rfp=False).model_dump(mode="json")
        )
        for fact in raw["facts"]:
            if fact["predicate"] == "uses_technology" and fact["object_name"] == "Microsoft Azure":
                fact.update(mutation)
                break
        return raw

    model.with_structured_output.return_value.invoke.side_effect = respond
    extractor = StructuredLLMExtractor(model, model_version="mock-v1")
    with pytest.raises(ExtractionRejected) as error:
        rebuild_graph(repo, "synthetic", project(tech="Azure"), extractor=extractor)
    assert error.value.issues
    assert driver.state["heads"] == {} and driver.state["chunks"] == {}
    model.with_structured_output.assert_called_once_with(ChunkExtraction, method="json_mode")


def test_valid_structured_llm_data_is_accepted_only_with_verified_spans(repository):
    model = Mock()

    def respond(prompt):
        text = json.loads(prompt.split("UNTRUSTED CHUNK (JSON string): ", 1)[1])
        return (
            SectionExtractor().extract(text, project=True, target_rfp=False).model_dump(mode="json")
        )

    model.with_structured_output.return_value.invoke.side_effect = respond
    extractor = StructuredLLMExtractor(model, model_version="test-zero-temperature-model-v1")
    result = rebuild_graph(repository[0], "synthetic", project(), extractor=extractor)
    assert result["published"]
    assert all(
        a["extraction_method"] == "structured_llm"
        for a in repository[1].state["assertions"].values()
    )


def test_malformed_json_and_llm_failure_are_not_silently_accepted(repository):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("private prompt")
    with pytest.raises(ExtractionRejected) as error:
        rebuild_graph(
            repository[0],
            "synthetic",
            project(),
            extractor=StructuredLLMExtractor(llm, model_version="test"),
        )
    assert "private prompt" not in str(error.value)
    assert not repository[1].state["snapshots"]


def test_missing_service_short_circuits_extraction():
    extractor = Mock()
    repo = create_ingestion_repository(settings(enabled=False))
    assert rebuild_graph(repo, "synthetic", project(), extractor=extractor) == {
        "status": "disabled",
        "published": False,
    }
    extractor.extract.assert_not_called()
    driver = IngestionDriver()
    driver.verify_error = RuntimeError("secret")
    repo = Neo4jIngestionRepository(settings(role="writer"), driver=driver)
    assert (
        rebuild_graph(repo, "synthetic", project(), extractor=extractor)["status"] == "unavailable"
    )
    extractor.extract.assert_not_called()


def test_service_loss_after_health_does_not_publish(repository):
    repo, driver = repository
    driver.fail_statement = READ_HEAD
    assert rebuild_graph(repo, "synthetic", project())["status"] == "unavailable"
    assert not driver.state["heads"]


def test_deletion_and_reingestion_replace_active_snapshot(repository):
    repo, driver = repository
    original = build_snapshot("synthetic", project("one.pdf") + project("two.pdf"))
    reduced = build_snapshot("synthetic", project("one.pdf"))
    empty = build_snapshot("synthetic", [])
    assert repo.publish(original, expected_version=None) == "ready"
    assert repo.publish(reduced, expected_version=original.version) == "ready"
    active_docs = [
        d
        for d in driver.state["documents"].values()
        if d["corpus_version"] == repo.active_version("synthetic")
    ]
    assert [d["source_file"] for d in active_docs] == ["one.pdf"]
    assert repo.publish(empty, expected_version=reduced.version) == "ready"
    assert not [
        d
        for d in driver.state["documents"].values()
        if d["corpus_version"] == repo.active_version("synthetic")
    ]
    assert repo.publish(original, expected_version=empty.version) == "ready"
    assert len(driver.state["snapshots"]) == 3  # Historic versions retained, not active facts.


@pytest.mark.parametrize(
    "failed_statement",
    [
        UPSERT_ENTITIES,
        UPSERT_ASSERTIONS,
        UPSERT_IDENTITIES,
        UPSERT_OBJECTS,
        UPSERT_SNAPSHOT,
        PUBLISH_HEAD,
    ],
)
def test_write_failure_rolls_back_entire_snapshot_and_retry_succeeds(repository, failed_statement):
    repo, driver = repository
    snapshot = build_snapshot("synthetic", project())
    before = deepcopy(driver.state)
    driver.fail_statement = failed_statement
    assert repo.publish(snapshot, expected_version=None) == "unavailable"
    assert driver.state == before
    driver.fail_statement = None
    assert repo.publish(snapshot, expected_version=None) == "ready"


def test_concurrent_stale_publication_is_rejected(repository):
    repo, driver = repository
    first = build_snapshot("synthetic", project("first.pdf"))
    second = build_snapshot("synthetic", project("second.pdf"))
    repo.publish(first, expected_version=None)
    before = deepcopy(driver.state)
    with pytest.raises(GraphIntegrityError, match="Concurrent"):
        repo.publish(second, expected_version=None)
    assert driver.state == before


def test_schema_validation_and_explicit_admin_migration(repository):
    repo, driver = repository
    driver.state["constraints"] = [
        row for row in driver.state["constraints"] if row["name"] != "rfp_assertion_identity"
    ]
    with pytest.raises(GraphSchemaError):
        repo.publish(build_snapshot("synthetic", project()), expected_version=None)
    assert not driver.state["heads"]
    with pytest.raises(GraphPermissionError):
        repo.initialize_schema()
    with Neo4jIngestionRepository(settings(role="admin"), driver=driver) as admin:
        assert admin.initialize_schema().ok
        assert admin.initialize_schema().ok
    assert repo.publish(build_snapshot("synthetic", project()), expected_version=None) == "ready"


def test_readers_cannot_publish_and_row_bound_is_enforced():
    snapshot = build_snapshot("synthetic", project())
    driver = IngestionDriver()
    with Neo4jIngestionRepository(settings(role="reader"), driver=driver) as reader:
        with pytest.raises(GraphPermissionError):
            reader.publish(snapshot, expected_version=None)
    with Neo4jIngestionRepository(
        settings(role="writer", max_batch_size=1), driver=driver
    ) as bounded:
        with pytest.raises(ValueError, match="bound"):
            bounded.publish(snapshot, expected_version=None)
    assert not driver.state["heads"]


def test_invalid_provenance_or_version_cannot_bypass_snapshot_validation(repository):
    snapshot = build_snapshot("synthetic", project())
    forged = snapshot.model_copy(update={"version": "f" * 64})
    with pytest.raises(GraphIntegrityError):
        repository[0].publish(forged, expected_version=None)
    with pytest.raises(ValidationError):
        IndexedChunk.model_validate(
            indexed("test").model_dump() | {"source_file": "../private.pdf"}
        )
    assert not repository[1].state["heads"]


def test_graph_job_failure_does_not_regress_successful_vector_ingestion(
    tmp_path, monkeypatch, caplog
):
    from tests.test_document_scope import test_successful_ingestion_cleans_temp_dirs

    def fail(*args, **kwargs):
        raise RuntimeError("sensitive file contents")

    monkeypatch.setattr("rfp_analyst.graph.ingestion.enqueue_graph_sync", fail)
    test_successful_ingestion_cleans_temp_dirs(monkeypatch, tmp_path)
    assert "Graph sync job could not be recorded" in caplog.text
    assert "sensitive file contents" not in caplog.text


def test_graph_job_is_not_enqueued_when_vector_publication_fails(tmp_path, monkeypatch):
    import rag_engine
    from langchain_core.documents import Document
    from rfp_analyst.exceptions import IngestionError
    from tests.test_document_scope import FakeIngestionManager, make_loaded_source

    job = Mock()
    monkeypatch.setattr("rfp_analyst.graph.ingestion.enqueue_graph_sync", job)
    monkeypatch.setattr(
        rag_engine, "_load_sources", lambda **kwargs: [make_loaded_source("sample")]
    )
    monkeypatch.setattr(
        rag_engine,
        "chunk_loaded_sources",
        lambda _: [Document(page_content="test", metadata={"chunk_id": "test"})],
    )
    monkeypatch.setattr(rag_engine, "VectorStoreManager", FakeIngestionManager)
    monkeypatch.setattr(
        rag_engine, "_swap_vectorstore", Mock(side_effect=RuntimeError("swap failed"))
    )
    with pytest.raises(IngestionError):
        rag_engine.ingest_documents(
            sample_dir=tmp_path, uploads_dir=tmp_path, persist_dir=tmp_path / "vectorstore"
        )
    job.assert_not_called()


def test_wrong_domain_constraint_shape_blocks_publication(repository):
    repo, driver = repository
    row = next(r for r in driver.state["constraints"] if r["name"] == "rfp_assertion_identity")
    row["properties"] = ["assertion_id"]  # Name alone is not sufficient.
    with pytest.raises(GraphSchemaError):
        repo.publish(build_snapshot("synthetic", project()), expected_version=None)
    assert not driver.state["heads"]


def test_conflicting_indexed_identity_is_rejected():
    first = indexed("one")
    second = first.model_copy(update={"text": "two"})
    with pytest.raises(GraphIntegrityError):
        build_snapshot("synthetic", [first, second])


def test_page_or_origin_change_creates_new_evidence_and_snapshot():
    first = indexed("same text")
    second = indexed("same text", page=1)
    third = indexed("same text", origin="upload")
    assert len({i.evidence_id for i in [first, second, third]}) == 3
    assert len({build_snapshot("synthetic", [i]).version for i in [first, second, third]}) == 3


def test_cli_disabled_graph_does_not_open_an_index(monkeypatch, capsys):
    from rfp_analyst.graph import __main__ as cli

    monkeypatch.setattr(
        cli,
        "create_ingestion_repository",
        lambda **kwargs: create_ingestion_repository(settings(enabled=False)),
    )
    assert cli.main(["rebuild", "--persist-dir", "does-not-exist"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "disabled"


def test_target_rfp_requirements_have_own_support_and_no_project():
    source = indexed(
        "Client RFP Requirements\nThe solution must support Azure migration and HIPAA controls.",
        origin="upload",
    )
    snapshot = build_snapshot("synthetic", [source])
    assert snapshot.documents[0].kind == "target_rfp"
    assert not any(e.kind == "Project" for e in snapshot.entities)
    assert {a.fact.predicate for a in snapshot.assertions} == {
        "requires_technology",
        "requires_framework",
    }
    assert all(a.evidence_id == source.evidence_id for a in snapshot.assertions)
    assert all(
        a.fact.modality == "proposed_control"
        for a in snapshot.assertions
        if a.fact.predicate == "requires_framework"
    )


def test_unrecognized_upload_does_not_become_a_project():
    snapshot = build_snapshot(
        "synthetic", [indexed("My resume\nSkills: Azure, Power BI", origin="upload")]
    )
    assert len(snapshot.chunks) == 1
    assert (
        snapshot.documents[0].kind == "other" and not snapshot.entities and not snapshot.assertions
    )


class Collection:
    def __init__(self, inputs):
        self.inputs = inputs
        self.calls = 0

    def count(self):
        return len(self.inputs)

    def get(self, **kwargs):
        assert kwargs["include"] == ["documents", "metadatas"]
        self.calls += 1
        return {
            "ids": [i.chunk_id for i in self.inputs],
            "documents": [i.text for i in self.inputs],
            "metadatas": [
                dict(
                    chunk_id=i.chunk_id,
                    source_file=i.source_file,
                    document_origin=i.document_origin,
                    file_hash=i.file_hash,
                    page=i.page_index,
                    chunk_index=i.chunk_index,
                )
                for i in self.inputs
            ],
        }


def test_rebuild_reads_existing_index_without_reembedding(repository):
    collection = Collection(project())
    inputs = read_indexed_corpus(collection)
    assert collection.calls == 2
    assert rebuild_graph(repository[0], "synthetic", inputs)["published"]
    with pytest.raises(ValueError, match="bound"):
        read_indexed_corpus(collection, max_chunks=1)


def test_real_chroma_collection_exports_existing_metadata_without_embeddings(
    tmp_path, repository, monkeypatch, capsys
):
    from rfp_analyst.graph import __main__ as cli
    from chromadb import PersistentClient
    from chromadb.config import Settings
    from rag_engine import _release_chroma_resources

    client = PersistentClient(
        path=str(tmp_path / "isolated-index"), settings=Settings(anonymized_telemetry=False)
    )
    try:
        collection = client.create_collection("synthetic-graph-ingestion", embedding_function=None)
        inputs = project()
        raw = Collection(inputs).get(include=["documents", "metadatas"])
        # Fixture vectors only: no model, network, PDF re-indexing or production index.
        collection.add(
            ids=raw["ids"],
            documents=raw["documents"],
            metadatas=raw["metadatas"],
            embeddings=[[0.1, 0.2] for _ in inputs],
        )
        before = collection.get(include=["documents", "metadatas", "embeddings"])
        exported = read_indexed_corpus(collection)
        assert rebuild_graph(repository[0], "synthetic", exported)["published"]
        monkeypatch.setattr(
            cli,
            "create_ingestion_repository",
            lambda **kwargs: create_ingestion_repository(settings(enabled=False)),
        )
        assert (
            cli.main(
                [
                    "rebuild",
                    "--dry-run",
                    "--persist-dir",
                    str(tmp_path / "isolated-index"),
                    "--collection",
                    "synthetic-graph-ingestion",
                ]
            )
            == 0
        )
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "validated" and not report["published"]
        assert report["documents"] == 1 and report["chunks"] == 3
        after = collection.get(include=["documents", "metadatas", "embeddings"])
        for key in ("ids", "documents", "metadatas"):
            assert before[key] == after[key]
        assert (before["embeddings"] == after["embeddings"]).all()
    finally:
        _release_chroma_resources(SimpleNamespace(_client=client))


def test_index_changes_during_export_are_rejected():
    collection = Collection(project())
    original_get = collection.get

    def changing(**kwargs):
        raw = original_get(**kwargs)
        if collection.calls == 2:
            raw["documents"][0] += " changed"
        return raw

    collection.get = changing
    with pytest.raises(GraphIntegrityError, match="changed"):
        read_indexed_corpus(collection)


def test_graph_job_is_opt_in_contains_no_document_text_and_is_repeatable(tmp_path, monkeypatch):
    import config

    documents = [
        SimpleNamespace(
            page_content=i.text,
            metadata={
                "chunk_id": i.chunk_id,
                "source_file": i.source_file,
                "document_origin": i.document_origin,
                "file_hash": i.file_hash,
                "page": i.page_index,
                "chunk_index": i.chunk_index,
            },
        )
        for i in project()
    ]
    monkeypatch.setattr(config, "get_neo4j_settings", lambda role: {"enabled": False})
    enqueue_graph_sync(documents, tmp_path / "vectorstore")
    assert not (tmp_path / "graph_sync").exists()
    monkeypatch.setattr(config, "get_neo4j_settings", lambda role: {"enabled": True})
    enqueue_graph_sync(documents, tmp_path / "vectorstore")
    path = tmp_path / "graph_sync" / "pending.json"
    first = path.read_text()
    assert "Digital Audit" not in first and "source_file" not in first
    enqueue_graph_sync(list(reversed(documents)), tmp_path / "vectorstore")
    assert path.read_text() == first
    assert len(list(path.parent.iterdir())) == 1


def test_real_public_sample_sections_extract_ten_projects_and_field_provenance(
    tmp_path, monkeypatch
):
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
    assert len(snapshot.documents) == 10
    assert len([e for e in snapshot.entities if e.kind == "Project"]) == 10
    for predicate in ("timeline", "budget", "in_industry"):
        assertions = [a for a in snapshot.assertions if a.fact.predicate == predicate]
        assert len(assertions) == 10 and len({a.subject_id for a in assertions}) == 10
    assert snapshot.row_count <= 500
    assert build_snapshot("public-synthetic", list(reversed(inputs))).version == snapshot.version
