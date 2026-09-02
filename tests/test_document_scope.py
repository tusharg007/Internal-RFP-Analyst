from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

import agent
import document_generator
import rag_engine
from rfp_analyst.agent.graph import prepare_query_payload
from rfp_analyst.uploads import is_uploaded_pdf_unchanged, persist_uploaded_pdf
from rfp_analyst.exceptions import IngestionError
from rfp_analyst.ingestion.chunking import build_chunk_id, chunk_loaded_sources
from rfp_analyst.ingestion import loaders
from rfp_analyst.retrieval.vector_store import VectorStoreManager
from rfp_analyst.schemas import LoadedSource


class FakeUpload:
    def __init__(self, name: str, data: bytes = b"pdf-bytes", file_type: str = "application/pdf"):
        self.name = name
        self._data = data
        self.type = file_type
        self.size = len(data)

    def getbuffer(self):
        return self._data


def test_unchanged_uploader_value_is_idempotent_after_persistence(tmp_path):
    upload = FakeUpload("resume.pdf", b"same-pdf-bytes")

    assert is_uploaded_pdf_unchanged(upload, uploads_dir=tmp_path) is False
    persist_uploaded_pdf(upload, uploads_dir=tmp_path)
    assert is_uploaded_pdf_unchanged(upload, uploads_dir=tmp_path) is True
    assert is_uploaded_pdf_unchanged(
        FakeUpload("resume.pdf", b"changed-pdf-bytes"), uploads_dir=tmp_path
    ) is False


class FakeLoader:
    def __init__(self, _path: str):
        self.path = _path

    def load(self):
        return [Document(page_content="hello", metadata={"page": 0})]


class FakeVectorStoreManager:
    def similarity_search(self, _query: str, k: int = 6, scope: str = "all"):
        docs = [
            (Document(page_content="sample", metadata={"document_origin": "sample", "source_file": "sample.pdf", "page": 0, "chunk_id": "s1"}), 0.9),
            (Document(page_content="upload", metadata={"document_origin": "upload", "source_file": "upload.pdf", "page": 0, "chunk_id": "u1"}), 0.8),
        ]
        return docs[:k]


class FakeIngestionManager:
    seen_documents = []

    def __init__(self, persist_dir, collection_name):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def upsert_documents(self, documents):
        self.__class__.seen_documents = list(documents)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        (self.persist_dir / "manifest.txt").write_text(str(len(documents)), encoding="utf-8")
        return object()


def make_loaded_source(origin: str = "sample"):
    return LoadedSource(
        source_file=f"{origin}.pdf",
        source_path=f"C:/tmp/{origin}.pdf",
        file_hash=f"hash-{origin}",
        page_count=1,
        document_type="Proposal",
        documents=[Document(page_content=f"{origin} content", metadata={"page": 0})],
        document_origin=origin,
    )


def test_identical_content_from_different_origins_has_different_chunk_ids():
    sample = LoadedSource(
        source_file="client_profile.pdf",
        source_path="C:/samples/client_profile.pdf",
        file_hash="same-hash",
        page_count=1,
        document_type=None,
        documents=[Document(page_content="same content", metadata={"page": 0})],
        document_origin="sample",
    )
    upload = LoadedSource(
        source_file="client_profile.pdf",
        source_path="C:/uploads/client_profile.pdf",
        file_hash="same-hash",
        page_count=1,
        document_type=None,
        documents=[Document(page_content="same content", metadata={"page": 0})],
        document_origin="upload",
    )

    chunks = chunk_loaded_sources([sample, upload], chunk_size=100, chunk_overlap=0)
    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == 2
    assert len(set(chunk_ids)) == 2


def test_chunk_ids_differ_by_page_and_chunk_index():
    first = build_chunk_id(
        document_origin="upload",
        source_path="C:/uploads/client.pdf",
        source_file="client.pdf",
        file_hash="hash",
        page=0,
        chunk_index=0,
        content="same",
    )
    different_page = build_chunk_id(
        document_origin="upload",
        source_path="C:/uploads/client.pdf",
        source_file="client.pdf",
        file_hash="hash",
        page=1,
        chunk_index=0,
        content="same",
    )
    different_index = build_chunk_id(
        document_origin="upload",
        source_path="C:/uploads/client.pdf",
        source_file="client.pdf",
        file_hash="hash",
        page=0,
        chunk_index=1,
        content="same",
    )

    assert first != different_page
    assert first != different_index


def test_uploaded_pdfs_are_saved_to_uploads_dir(tmp_path: Path):
    uploads_dir = tmp_path / "uploads"
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()

    saved_path = persist_uploaded_pdf(FakeUpload("Client Stack.pdf"), uploads_dir=uploads_dir)

    assert saved_path.parent == uploads_dir
    assert saved_path.exists()
    assert not list(sample_dir.glob("*.pdf"))


def test_sample_generator_writes_only_to_sample_docs_dir(monkeypatch, tmp_path: Path):
    sample_dir = tmp_path / "samples"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(document_generator, "DATA_DIR", sample_dir)

    document_generator.generate_all_documents()

    assert list(sample_dir.glob("*.pdf"))
    assert not list(upload_dir.glob("*.pdf"))


@pytest.mark.parametrize("origin", ["upload", "sample"])
def test_loaded_docs_get_document_origin_metadata(monkeypatch, tmp_path: Path, origin: str):
    pdf_path = tmp_path / f"{origin}.pdf"
    pdf_path.write_bytes(b"fake")

    monkeypatch.setattr(loaders, "ensure_safe_pdf_path", lambda path: path)
    monkeypatch.setattr(loaders, "validate_pdf", lambda _path: 1)
    monkeypatch.setattr(loaders, "sha256_file", lambda _path: f"hash-{origin}")
    monkeypatch.setattr(loaders, "PyMuPDFLoader", FakeLoader)

    sources = loaders.load_pdf_sources(tmp_path, document_origin=origin)

    assert len(sources) == 1
    assert sources[0].document_origin == origin
    assert sources[0].documents[0].metadata["document_origin"] == origin


def test_retrieval_scope_upload_excludes_sample_docs(monkeypatch):
    monkeypatch.setattr(rag_engine, "VectorStoreManager", lambda **_kwargs: FakeVectorStoreManager())

    results = rag_engine.similarity_search("tech stack", scope="upload")

    assert results
    assert all(doc.metadata["document_origin"] == "upload" for doc, _score in results)


def test_retrieval_scope_sample_excludes_upload_docs(monkeypatch):
    monkeypatch.setattr(rag_engine, "VectorStoreManager", lambda **_kwargs: FakeVectorStoreManager())

    results = rag_engine.similarity_search("tech stack", scope="sample")

    assert results
    assert all(doc.metadata["document_origin"] == "sample" for doc, _score in results)


def test_source_trace_is_deduplicated_by_source_page_and_chunk_id():
    payload = {
        "user_query": "what is my stack",
        "traces": [
            {
                "tool": "search_knowledge_base",
                "input": {"query": "what is my stack", "scope": "upload"},
                "documents": [
                    {"source": "upload.pdf", "page": 0, "score": "0.91", "chunk_id": "abc", "document_origin": "upload"},
                    {"source": "upload.pdf", "page": 0, "score": "0.91", "chunk_id": "abc", "document_origin": "upload"},
                ],
            }
        ]
    }

    trace = agent._payload_to_reasoning_trace(payload)

    source_entries = [item for item in trace if item.get("tool_response")]
    assert len(source_entries) == 1


def test_scope_with_zero_chunks_returns_clean_message():
    payload = prepare_query_payload(
        user_query="what is my stack",
        retrieval_scope="upload",
        vectorstore_stats={
            "status": "ready",
            "total_chunks": 5,
            "total_documents": 1,
            "scope_chunk_counts": {"upload": 0, "sample": 5, "all": 5},
            "document_names": ["sample.pdf"],
        },
        retrieval_fn=lambda _query, _k, _scope: [],
    )

    assert payload["response_mode"] == "fallback"
    assert payload["answer"] == rag_engine.NO_SCOPE_DOCUMENTS_MESSAGE


def test_all_documents_scope_still_returns_results():
    captured = {}

    def retrieval_fn(_query, _k, scope):
        captured["scope"] = scope
        return [
            (Document(page_content="sample", metadata={"document_origin": "sample", "source_file": "sample.pdf", "page": 0, "chunk_id": "s1"}), 0.9),
            (Document(page_content="upload", metadata={"document_origin": "upload", "source_file": "upload.pdf", "page": 0, "chunk_id": "u1"}), 0.8),
        ]

    payload = prepare_query_payload(
        user_query="what is my stack",
        retrieval_scope="all",
        vectorstore_stats={
            "status": "ready",
            "total_chunks": 2,
            "total_documents": 2,
            "scope_chunk_counts": {"upload": 1, "sample": 1, "all": 2},
            "document_names": ["sample.pdf", "upload.pdf"],
        },
        retrieval_fn=retrieval_fn,
    )

    assert captured["scope"] == "all"
    assert len(payload["retrieved_documents"]) == 2


def test_pending_uploads_warn_before_answering():
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    pending_stats = {
        "status": "ready",
        "total_documents": 1,
        "total_chunks": 3,
        "document_names": ["sample.pdf"],
        "indexed_sample_document_count": 1,
        "indexed_upload_document_count": 0,
        "pending_upload_files": ["client.pdf"],
        "scope_chunk_counts": {"sample": 3, "upload": 0, "all": 3},
    }

    with (
        patch("config.get_api_keys", return_value=("groq", "")),
        patch("rag_engine.get_vectorstore_stats", return_value=pending_stats),
        patch("rag_engine.ingest_documents"),
    ):
        at = AppTest.from_file(str(app_path)).run(timeout=10)

    assert not at.exception
    assert any("pending indexing" in str(element.value).lower() for element in at.warning)
    assert at.chat_input[0].disabled is True


def test_ingestion_success_clears_in_progress_and_pending_state():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    not_ready = {
        "status": "not_initialized",
        "total_documents": 1,
        "total_chunks": 0,
        "document_names": [],
        "indexed_sample_document_count": 0,
        "indexed_upload_document_count": 0,
        "pending_upload_files": ["client.pdf"],
        "scope_chunk_counts": {"sample": 0, "upload": 0, "all": 0},
    }
    ready = {
        "status": "ready",
        "total_documents": 2,
        "total_chunks": 4,
        "document_names": ["sample.pdf", "client.pdf"],
        "indexed_sample_document_count": 1,
        "indexed_upload_document_count": 1,
        "pending_upload_files": [],
        "scope_chunk_counts": {"sample": 2, "upload": 2, "all": 4},
    }
    stats_calls = {"count": 0}
    seen_flag = {"value": None}

    def fake_stats(*_args, **_kwargs):
        stats_calls["count"] += 1
        return not_ready if stats_calls["count"] < 2 else ready

    def fake_ingest(*_args, **_kwargs):
        import streamlit as st

        seen_flag["value"] = st.session_state.ingestion_in_progress
        return None

    with (
        patch("config.get_api_keys", return_value=("groq", "")),
        patch("rag_engine.get_vectorstore_stats", side_effect=fake_stats),
        patch("rag_engine.ingest_documents", side_effect=fake_ingest),
    ):
        at = AppTest.from_file(str(app_path)).run(timeout=10)
        at.button[7].click().run(timeout=10)

    assert seen_flag["value"] is True
    assert at.session_state.filtered_state["ingestion_in_progress"] is False
    assert at.session_state.filtered_state["pending_uploads"] is False
    assert at.session_state.filtered_state["last_ingestion_error"] == ""


def test_windows_lock_failure_is_friendly_and_cleans_temp_dirs(monkeypatch, tmp_path: Path):
    sample_dir = tmp_path / "sample"
    upload_dir = tmp_path / "uploads"
    persist_dir = tmp_path / "vectorstore"
    sample_dir.mkdir()
    upload_dir.mkdir()

    monkeypatch.setattr(rag_engine, "_load_sources", lambda **_kwargs: [make_loaded_source("sample")])
    monkeypatch.setattr(
        rag_engine,
        "chunk_loaded_sources",
        lambda _sources: [Document(page_content="chunk", metadata={"chunk_id": "abc", "source_file": "sample.pdf", "document_origin": "sample", "file_hash": "hash-sample", "page": 0})],
    )
    monkeypatch.setattr(rag_engine, "VectorStoreManager", FakeIngestionManager)
    monkeypatch.setattr(
        rag_engine,
        "_swap_vectorstore",
        lambda _temp, _persist: (_ for _ in ()).throw(PermissionError(5, "Access is denied")),
    )

    with pytest.raises(IngestionError, match="Vectorstore files are locked"):
        rag_engine.ingest_documents(sample_dir=sample_dir, uploads_dir=upload_dir, persist_dir=persist_dir)

    assert not list(tmp_path.glob("vectorstore_build_*"))


def test_successful_ingestion_cleans_temp_dirs(monkeypatch, tmp_path: Path):
    sample_dir = tmp_path / "sample"
    upload_dir = tmp_path / "uploads"
    persist_dir = tmp_path / "vectorstore"
    sample_dir.mkdir()
    upload_dir.mkdir()

    monkeypatch.setattr(rag_engine, "_load_sources", lambda **_kwargs: [make_loaded_source("sample")])
    monkeypatch.setattr(
        rag_engine,
        "chunk_loaded_sources",
        lambda _sources: [Document(page_content="chunk", metadata={"chunk_id": "abc", "source_file": "sample.pdf", "document_origin": "sample", "file_hash": "hash-sample", "page": 0})],
    )
    monkeypatch.setattr(rag_engine, "VectorStoreManager", FakeIngestionManager)
    monkeypatch.setattr(
        rag_engine,
        "get_vectorstore_stats",
        lambda **_kwargs: {
            "status": "ready",
            "total_documents": 1,
            "total_chunks": 1,
            "document_names": ["sample.pdf"],
            "indexed_sample_document_count": 1,
            "indexed_upload_document_count": 0,
            "pending_upload_files": [],
            "scope_chunk_counts": {"sample": 1, "upload": 0, "all": 1},
        },
    )

    result = rag_engine.ingest_documents(sample_dir=sample_dir, uploads_dir=upload_dir, persist_dir=persist_dir)

    assert result["total_chunks"] == 1
    assert not list(tmp_path.glob("vectorstore_build_*"))


def test_duplicate_file_hash_prefers_upload_deterministically():
    sample = make_loaded_source("sample")
    upload = make_loaded_source("upload")
    sample = LoadedSource(**{**sample.__dict__, "file_hash": "same-hash"})
    upload = LoadedSource(**{**upload.__dict__, "file_hash": "same-hash"})

    unique_sources, duplicate_files = rag_engine._deduplicate_loaded_sources([sample, upload])

    assert [source.document_origin for source in unique_sources] == ["upload"]
    assert duplicate_files[0]["skipped_origin"] == "sample"
    assert duplicate_files[0]["kept_origin"] == "upload"


def test_same_pdf_in_sample_and_upload_does_not_crash_ingestion(monkeypatch, tmp_path: Path):
    sample_dir = tmp_path / "sample"
    upload_dir = tmp_path / "uploads"
    persist_dir = tmp_path / "vectorstore"
    sample_dir.mkdir()
    upload_dir.mkdir()

    sample = make_loaded_source("sample")
    upload = make_loaded_source("upload")
    sample = LoadedSource(**{**sample.__dict__, "source_file": "client_profile.pdf", "file_hash": "same-hash"})
    upload = LoadedSource(**{**upload.__dict__, "source_file": "client_profile.pdf", "file_hash": "same-hash"})
    FakeIngestionManager.seen_documents = []

    monkeypatch.setattr(rag_engine, "_load_sources", lambda **_kwargs: [sample, upload])
    monkeypatch.setattr(rag_engine, "VectorStoreManager", FakeIngestionManager)
    monkeypatch.setattr(
        rag_engine,
        "get_vectorstore_stats",
        lambda **_kwargs: {
            "status": "ready",
            "total_documents": 1,
            "total_chunks": len(FakeIngestionManager.seen_documents),
            "document_names": ["client_profile.pdf"],
            "indexed_sample_document_count": 0,
            "indexed_upload_document_count": 1,
            "pending_upload_files": [],
            "scope_chunk_counts": {
                "sample": 0,
                "upload": len(FakeIngestionManager.seen_documents),
                "all": len(FakeIngestionManager.seen_documents),
            },
        },
    )

    result = rag_engine.ingest_documents(sample_dir=sample_dir, uploads_dir=upload_dir, persist_dir=persist_dir)

    assert result["files_discovered"] == 2
    assert result["unique_files"] == 1
    assert result["duplicate_files_skipped"][0]["skipped_origin"] == "sample"
    assert result["sample_chunks"] == 0
    assert result["upload_chunks"] > 0
    assert all(doc.metadata["document_origin"] == "upload" for doc in FakeIngestionManager.seen_documents)


def test_vectorstore_upsert_sends_unique_ids_to_chroma(monkeypatch, tmp_path: Path):
    class FakeCollection:
        def __init__(self):
            self.ids = []

        def get(self):
            return {"ids": self.ids, "metadatas": []}

        def count(self):
            return len(self.ids)

    class FakeChroma:
        def __init__(self):
            self._collection = FakeCollection()

        def add_documents(self, documents, ids):
            assert len(ids) == len(set(ids))
            self._collection.ids.extend(ids)

    fake_chroma = FakeChroma()
    monkeypatch.setattr(VectorStoreManager, "load", lambda self, create_if_missing=True: fake_chroma)
    manager = VectorStoreManager(persist_dir=tmp_path, embedding_function=object())
    docs = [
        Document(page_content="first", metadata={"chunk_id": "duplicate"}),
        Document(page_content="second", metadata={"chunk_id": "duplicate"}),
        Document(page_content="third", metadata={"chunk_id": "unique"}),
    ]

    manager.upsert_documents(docs)

    assert fake_chroma._collection.ids == ["duplicate", "unique"]
