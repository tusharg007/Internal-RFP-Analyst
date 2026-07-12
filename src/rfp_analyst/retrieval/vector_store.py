"""Vector store management for ingestion and retrieval."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document

from config import COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_K, VECTORSTORE_DIR
from rfp_analyst.exceptions import KnowledgeBaseNotReadyError

VALID_SCOPES = {"all", "sample", "upload"}


def get_embeddings():
    """Initialize local embeddings for ingestion and search."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def deduplicate_documents_by_chunk_id(documents: list[Document]) -> tuple[list[Document], list[str]]:
    """Return documents with unique chunk IDs, preserving first occurrence."""
    unique_chunks_by_id: dict[str, Document] = {}
    duplicate_ids: list[str] = []
    for document in documents:
        chunk_id = document.metadata["chunk_id"]
        if chunk_id in unique_chunks_by_id:
            duplicate_ids.append(chunk_id)
            continue
        unique_chunks_by_id[chunk_id] = document
    return list(unique_chunks_by_id.values()), duplicate_ids


class VectorStoreManager:
    """Encapsulate Chroma persistence and deduplicated upserts."""

    def __init__(
        self,
        persist_dir: Path = VECTORSTORE_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_function=None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_function = embedding_function or get_embeddings()

    def load(self, create_if_missing: bool = True) -> Chroma:
        """Load or initialize the Chroma collection."""
        if not self.persist_dir.exists():
            if not create_if_missing:
                raise KnowledgeBaseNotReadyError(
                    "Knowledge base is not ready. Generate or upload PDFs and click Ingest Documents."
                )
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=str(self.persist_dir),
        )

    def upsert_documents(self, documents: list[Document]) -> Chroma:
        """Add only new chunk IDs into the collection."""
        vectorstore = self.load(create_if_missing=True)
        existing_ids = set(vectorstore._collection.get().get("ids", []))
        unique_documents, duplicate_ids = deduplicate_documents_by_chunk_id(documents)
        if duplicate_ids:
            print(f"Skipped {len(duplicate_ids)} duplicate chunk ID(s) before Chroma upsert")

        new_documents = []
        new_ids = []
        for document in unique_documents:
            chunk_id = document.metadata["chunk_id"]
            if chunk_id in existing_ids:
                continue
            new_documents.append(document)
            new_ids.append(chunk_id)

        if new_documents:
            if len(new_ids) != len(set(new_ids)):
                raise ValueError("Duplicate chunk IDs detected before Chroma upsert")
            vectorstore.add_documents(new_documents, ids=new_ids)

        print(f"Vector store contains {vectorstore._collection.count()} vectors")
        print(f"Persisted to: {self.persist_dir}")
        return vectorstore

    def get_retriever(self, k: int = RETRIEVAL_K, scope: str = "all"):
        """Get a retriever for similarity search."""
        search_kwargs = {"k": k}
        if scope in {"sample", "upload"}:
            search_kwargs["filter"] = {"document_origin": scope}
        return self.load(create_if_missing=False).as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    def similarity_search(self, query: str, k: int = RETRIEVAL_K, scope: str = "all"):
        """Run similarity search with relevance scores."""
        vectorstore = self.load(create_if_missing=False)
        kwargs = {"k": k}
        if scope in {"sample", "upload"}:
            kwargs["filter"] = {"document_origin": scope}
        try:
            return vectorstore.similarity_search_with_relevance_scores(query, **kwargs)
        except TypeError:
            return vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def get_stats(self) -> dict:
        """Return collection stats for the UI."""
        try:
            vectorstore = self.load(create_if_missing=False)
            count = vectorstore._collection.count()
            all_metadata = vectorstore._collection.get().get("metadatas", [])
            sources = {
                metadata["source_file"]
                for metadata in all_metadata
                if metadata and metadata.get("source_file")
            }
            return {
                "total_chunks": count,
                "total_documents": len(sources),
                "document_names": sorted(sources),
                "status": "ready" if count else "not_initialized",
            }
        except Exception:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "document_names": [],
                "status": "not_initialized",
            }
