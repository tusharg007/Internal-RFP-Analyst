from pathlib import Path

from langchain_core.documents import Document

from rfp_analyst.ingestion.chunking import chunk_loaded_sources
from rfp_analyst.ingestion.pipeline import IngestionPipeline
from rfp_analyst.ingestion.registry import IngestionRegistry
from rfp_analyst.schemas import LoadedSource


class FakeCollection:
    def __init__(self):
        self.documents = {}

    def get(self):
        return {
            "ids": list(self.documents),
            "metadatas": [doc.metadata for doc in self.documents.values()],
        }

    def count(self):
        return len(self.documents)


class FakeVectorStore:
    def __init__(self):
        self._collection = FakeCollection()

    def add_documents(self, documents, ids):
        for doc, chunk_id in zip(documents, ids):
            self._collection.documents[chunk_id] = doc


class FakeVectorStoreManager:
    def __init__(self):
        self.vectorstore = FakeVectorStore()

    def upsert_documents(self, documents):
        existing_ids = set(self.vectorstore._collection.get()["ids"])
        new_docs = []
        new_ids = []
        for document in documents:
            chunk_id = document.metadata["chunk_id"]
            if chunk_id in existing_ids:
                continue
            new_docs.append(document)
            new_ids.append(chunk_id)
        if new_docs:
            self.vectorstore.add_documents(new_docs, new_ids)
        return self.vectorstore

    def get_stats(self):
        metadatas = self.vectorstore._collection.get()["metadatas"]
        source_files = {metadata["source_file"] for metadata in metadatas}
        return {
            "total_chunks": self.vectorstore._collection.count(),
            "total_documents": len(source_files),
            "document_names": sorted(source_files),
            "status": "ready" if metadatas else "not_initialized",
        }


def make_loaded_source(name="Client Proposal.pdf", file_hash="abc123"):
    return LoadedSource(
        source_file=name,
        source_path=str(Path("C:/tmp") / name),
        file_hash=file_hash,
        page_count=1,
        document_type="Proposal",
        documents=[
            Document(
                page_content="Alpha beta gamma delta epsilon zeta eta theta.",
                metadata={"page": 0},
            )
        ],
    )


def test_chunking_adds_required_metadata():
    chunks = chunk_loaded_sources(
        [make_loaded_source()],
        chunk_size=20,
        chunk_overlap=0,
    )

    assert chunks
    assert all(chunk.metadata["source_file"] == "Client Proposal.pdf" for chunk in chunks)
    assert all(chunk.metadata["file_hash"] == "abc123" for chunk in chunks)
    assert all(chunk.metadata["document_type"] == "Proposal" for chunk in chunks)
    assert all(chunk.metadata["chunk_id"] for chunk in chunks)
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)


def test_registry_prevents_duplicate_ingestion(monkeypatch, tmp_path):
    registry = IngestionRegistry(tmp_path / "ingestion_registry.json")
    vector_store_manager = FakeVectorStoreManager()
    source = make_loaded_source()

    monkeypatch.setattr(
        "rfp_analyst.ingestion.pipeline.load_pdf_sources",
        lambda _doc_dir: [source],
    )

    pipeline = IngestionPipeline(
        doc_dir=tmp_path,
        persist_dir=tmp_path / "vectorstore",
        registry=registry,
        vector_store_manager=vector_store_manager,
    )

    pipeline.run()
    first_count = vector_store_manager.get_stats()["total_chunks"]

    pipeline.run()
    second_count = vector_store_manager.get_stats()["total_chunks"]

    assert first_count > 0
    assert second_count == first_count
    assert registry.contains_hash(source.file_hash)


def test_repeated_ingestion_does_not_double_chunk_count(monkeypatch, tmp_path):
    registry = IngestionRegistry(tmp_path / "ingestion_registry.json")
    vector_store_manager = FakeVectorStoreManager()
    source = make_loaded_source(file_hash="hash-repeat")

    monkeypatch.setattr(
        "rfp_analyst.ingestion.pipeline.load_pdf_sources",
        lambda _doc_dir: [source],
    )

    pipeline = IngestionPipeline(
        doc_dir=tmp_path,
        persist_dir=tmp_path / "vectorstore",
        registry=registry,
        vector_store_manager=vector_store_manager,
    )

    pipeline.run()
    stats_after_first_run = vector_store_manager.get_stats()
    pipeline.run()
    stats_after_second_run = vector_store_manager.get_stats()

    assert stats_after_first_run["total_chunks"] == stats_after_second_run["total_chunks"]
    assert stats_after_first_run["total_documents"] == stats_after_second_run["total_documents"]
