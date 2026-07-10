"""RAG Engine - Document Ingestion, Embedding & Retrieval Pipeline.
Backward-compatible wrappers for the production ingestion package.
"""

from pathlib import Path

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    RETRIEVAL_K,
    VECTORSTORE_DIR,
)
from rfp_analyst.exceptions import IngestionError, KnowledgeBaseNotReadyError, RetrievalError
from rfp_analyst.ingestion.chunking import chunk_loaded_sources
from rfp_analyst.ingestion.loaders import load_pdf_sources
from rfp_analyst.ingestion.pipeline import IngestionPipeline
from rfp_analyst.retrieval.vector_store import VectorStoreManager
from rfp_analyst.retrieval.vector_store import get_embeddings as _get_embeddings
from rfp_analyst.schemas import LoadedSource


def get_embeddings():
    """Backward-compatible embeddings wrapper."""
    return _get_embeddings()


def load_pdfs(doc_dir: Path = DATA_DIR):
    """Backward-compatible PDF loading wrapper."""
    loaded_sources = load_pdf_sources(doc_dir)
    all_docs = []
    for source in loaded_sources:
        all_docs.extend(source.documents)
    return all_docs


def chunk_documents(documents):
    """Backward-compatible chunking wrapper."""
    if not documents:
        return []

    grouped_sources = {}
    for document in documents:
        file_hash = document.metadata.get("file_hash", "legacy")
        grouped_sources.setdefault(file_hash, []).append(document)

    loaded_sources = []
    for file_hash, source_documents in grouped_sources.items():
        first = source_documents[0]
        loaded_sources.append(
            LoadedSource(
                source_file=first.metadata.get("source_file", "unknown.pdf"),
                source_path=first.metadata.get("source_path", ""),
                file_hash=file_hash,
                page_count=len(source_documents),
                document_type=first.metadata.get("document_type"),
                documents=source_documents,
            )
        )

    return chunk_loaded_sources(
        loaded_sources,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def create_vectorstore(chunks, persist_dir: Path = VECTORSTORE_DIR):
    """Backward-compatible vector store wrapper."""
    manager = VectorStoreManager(
        persist_dir=persist_dir,
        collection_name=COLLECTION_NAME,
    )
    return manager.upsert_documents(chunks)


def load_vectorstore(persist_dir: Path = VECTORSTORE_DIR):
    """Load an existing ChromaDB vector store from disk."""
    manager = VectorStoreManager(
        persist_dir=persist_dir,
        collection_name=COLLECTION_NAME,
    )
    vectorstore = manager.load(create_if_missing=False)
    count = vectorstore._collection.count()
    print(f"Loaded vector store with {count} vectors")
    return vectorstore


def get_retriever(k: int = RETRIEVAL_K):
    """Get a LangChain retriever from the persisted vector store."""
    manager = VectorStoreManager(
        persist_dir=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME,
    )
    return manager.get_retriever(k=k)


def similarity_search(query: str, k: int = RETRIEVAL_K):
    """Direct similarity search returning documents with scores."""
    manager = VectorStoreManager(
        persist_dir=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME,
    )
    try:
        return manager.similarity_search(query, k=k)
    except KnowledgeBaseNotReadyError:
        raise
    except Exception as error:
        raise RetrievalError(str(error)) from error


def get_vectorstore_stats():
    """Get statistics about the current vector store."""
    manager = VectorStoreManager(
        persist_dir=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME,
    )
    return manager.get_stats()


def ingest_documents(doc_dir: Path = DATA_DIR):
    """Full ingestion pipeline wrapper."""
    pipeline = IngestionPipeline(
        doc_dir=doc_dir,
        persist_dir=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME,
    )
    try:
        return pipeline.run()
    except Exception as error:
        if isinstance(error, IngestionError):
            raise
        raise IngestionError(str(error)) from error


if __name__ == "__main__":
    ingest_documents()
