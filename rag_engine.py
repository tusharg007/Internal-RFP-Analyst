"""RAG engine for document ingestion, embedding, and retrieval."""

from __future__ import annotations

import gc
import shutil
import tempfile
from pathlib import Path

from config import (
    COLLECTION_NAME,
    DATA_DIR,
    RETRIEVAL_K,
    SAMPLE_DOCS_DIR,
    UPLOADS_DIR,
    VECTORSTORE_DIR,
)
from rfp_analyst.exceptions import (
    IngestionError,
    KnowledgeBaseNotReadyError,
    NoDocumentsFoundError,
    RetrievalError,
)
from rfp_analyst.ingestion.chunking import chunk_loaded_sources
from rfp_analyst.ingestion.loaders import load_pdf_sources, sha256_file
from rfp_analyst.retrieval.vector_store import (
    VectorStoreManager,
    deduplicate_documents_by_chunk_id,
)
from rfp_analyst.schemas import LoadedSource

VALID_RETRIEVAL_SCOPES = {"all", "sample", "upload"}
NO_SCOPE_DOCUMENTS_MESSAGE = "No indexed documents found for this scope."
KB_NOT_READY_MESSAGE = (
    "Knowledge base is not ready. Generate or upload PDFs and click Ingest Documents."
)
VECTORSTORE_LOCKED_MESSAGE = (
    "Vectorstore files are locked. Stop the app and retry ingestion, or delete vectorstore manually."
)


def _normalize_scope(scope: str = "all") -> str:
    normalized = (scope or "all").strip().lower()
    return normalized if normalized in VALID_RETRIEVAL_SCOPES else "all"


def _source_directories(
    sample_dir: Path = SAMPLE_DOCS_DIR,
    uploads_dir: Path = UPLOADS_DIR,
) -> dict[str, Path]:
    return {
        "sample": Path(sample_dir),
        "upload": Path(uploads_dir),
    }


def _list_pdf_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.pdf")) if directory.exists() else []


def _available_files_by_origin(
    sample_dir: Path = SAMPLE_DOCS_DIR,
    uploads_dir: Path = UPLOADS_DIR,
) -> dict[str, list[Path]]:
    directories = _source_directories(sample_dir=sample_dir, uploads_dir=uploads_dir)
    return {origin: _list_pdf_files(path) for origin, path in directories.items()}


def _load_sources(
    sample_dir: Path = SAMPLE_DOCS_DIR,
    uploads_dir: Path = UPLOADS_DIR,
):
    loaded_sources = []
    for origin, directory in _source_directories(sample_dir=sample_dir, uploads_dir=uploads_dir).items():
        if not directory.exists():
            continue
        pdf_files = _list_pdf_files(directory)
        if not pdf_files:
            continue
        loaded_sources.extend(load_pdf_sources(directory, document_origin=origin))
    return loaded_sources


def _empty_ingestion_report() -> dict:
    return {
        "files_discovered": 0,
        "unique_files": 0,
        "duplicate_files_skipped": [],
        "chunks_created": 0,
        "duplicate_chunks_skipped": 0,
        "chunks_indexed": 0,
        "sample_chunks": 0,
        "upload_chunks": 0,
    }


def _deduplicate_loaded_sources(
    loaded_sources: list[LoadedSource],
) -> tuple[list[LoadedSource], list[dict]]:
    """Deduplicate exact same files across corpora, preferring uploads."""
    unique_by_hash: dict[str, LoadedSource] = {}
    duplicate_files: list[dict] = []

    for source in loaded_sources:
        existing = unique_by_hash.get(source.file_hash)
        if existing is None:
            unique_by_hash[source.file_hash] = source
            continue

        prefer_new_upload = (
            existing.document_origin == "sample" and source.document_origin == "upload"
        )
        skipped = existing if prefer_new_upload else source
        kept = source if prefer_new_upload else existing
        unique_by_hash[source.file_hash] = kept
        duplicate_files.append(
            {
                "file_hash": source.file_hash,
                "skipped_source_file": skipped.source_file,
                "skipped_origin": skipped.document_origin,
                "kept_source_file": kept.source_file,
                "kept_origin": kept.document_origin,
            }
        )

    return list(unique_by_hash.values()), duplicate_files


def _build_ingestion_report(
    loaded_sources: list[LoadedSource],
    unique_sources: list[LoadedSource],
    duplicate_files: list[dict],
    chunks,
    unique_chunks,
    duplicate_chunk_ids: list[str],
) -> dict:
    sample_chunks = sum(
        1 for chunk in unique_chunks if chunk.metadata.get("document_origin") == "sample"
    )
    upload_chunks = sum(
        1 for chunk in unique_chunks if chunk.metadata.get("document_origin") == "upload"
    )
    return {
        "files_discovered": len(loaded_sources),
        "unique_files": len(unique_sources),
        "duplicate_files_skipped": duplicate_files,
        "chunks_created": len(chunks),
        "duplicate_chunks_skipped": len(duplicate_chunk_ids),
        "chunks_indexed": len(unique_chunks),
        "sample_chunks": sample_chunks,
        "upload_chunks": upload_chunks,
    }


def _filter_results_for_scope(results, scope: str):
    normalized_scope = _normalize_scope(scope)
    if normalized_scope == "all":
        return results

    filtered = []
    for document, score in results:
        origin = (getattr(document, "metadata", {}) or {}).get("document_origin", "sample")
        if origin == normalized_scope:
            filtered.append((document, score))
    return filtered


def _build_scope_snapshot(all_metadata: list[dict], available_files: dict[str, list[Path]]) -> dict:
    unique_docs_by_origin = {"sample": {}, "upload": {}}
    chunk_counts = {"sample": 0, "upload": 0}
    indexed_hashes = {"sample": set(), "upload": set()}

    for metadata in all_metadata:
        if not metadata:
            continue
        origin = metadata.get("document_origin", "sample")
        if origin not in unique_docs_by_origin:
            continue
        chunk_counts[origin] += 1
        file_hash = metadata.get("file_hash")
        if file_hash:
            indexed_hashes[origin].add(file_hash)
        source_file = metadata.get("source_file")
        if source_file:
            unique_docs_by_origin[origin][source_file] = True

    pending_upload_files = []
    for upload_path in available_files["upload"]:
        try:
            if sha256_file(upload_path) not in indexed_hashes["upload"]:
                pending_upload_files.append(upload_path.name)
        except Exception:
            pending_upload_files.append(upload_path.name)

    indexed_sample_files = sorted(unique_docs_by_origin["sample"].keys())
    indexed_upload_files = sorted(unique_docs_by_origin["upload"].keys())
    available_sample_files = [path.name for path in available_files["sample"]]
    available_upload_files = [path.name for path in available_files["upload"]]

    return {
        "indexed_sample_document_count": len(indexed_sample_files),
        "indexed_upload_document_count": len(indexed_upload_files),
        "indexed_sample_files": indexed_sample_files,
        "indexed_upload_files": indexed_upload_files,
        "available_sample_documents": available_sample_files,
        "available_upload_documents": available_upload_files,
        "available_documents": available_sample_files + available_upload_files,
        "pending_upload_files": pending_upload_files,
        "scope_chunk_counts": {
            "sample": chunk_counts["sample"],
            "upload": chunk_counts["upload"],
            "all": chunk_counts["sample"] + chunk_counts["upload"],
        },
    }


def _is_windows_lock_error(error: Exception) -> bool:
    winerror = getattr(error, "winerror", None)
    if winerror == 5:
        return True
    message = str(error).lower()
    return "access is denied" in message or "used by another process" in message


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)


def _release_chroma_resources(vectorstore) -> None:
    """Best-effort release of Chroma resources before moving/deleting directories."""
    if vectorstore is None:
        return
    client = getattr(vectorstore, "_client", None)
    system = getattr(client, "_system", None)
    stop = getattr(system, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception:
            pass
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def _cleanup_temp_build_dirs(parent_dir: Path) -> None:
    for build_dir in parent_dir.glob("vectorstore_build_*"):
        _safe_rmtree(build_dir)


def load_pdfs(doc_dir: Path = DATA_DIR):
    """Backward-compatible page-document loader for a single directory."""
    loaded_sources = load_pdf_sources(doc_dir, document_origin="sample")
    all_docs = []
    for source in loaded_sources:
        all_docs.extend(source.documents)
    return all_docs


def chunk_documents(documents):
    """Backward-compatible chunking wrapper."""
    from langchain_core.documents import Document
    from rfp_analyst.schemas import LoadedSource

    if not documents:
        return []

    grouped_sources: dict[str, list[Document]] = {}
    source_metadata: dict[str, dict] = {}
    for document in documents:
        metadata = document.metadata or {}
        file_hash = metadata.get("file_hash", "legacy")
        grouped_sources.setdefault(file_hash, []).append(document)
        source_metadata.setdefault(
            file_hash,
            {
                "source_file": metadata.get("source_file", "unknown.pdf"),
                "source_path": metadata.get("source_path", ""),
                "page_count": len(grouped_sources[file_hash]),
                "document_type": metadata.get("document_type"),
                "document_origin": metadata.get("document_origin", "sample"),
            },
        )

    loaded_sources = []
    for file_hash, source_documents in grouped_sources.items():
        metadata = source_metadata[file_hash]
        loaded_sources.append(
            LoadedSource(
                source_file=metadata["source_file"],
                source_path=metadata["source_path"],
                file_hash=file_hash,
                page_count=len(source_documents),
                document_type=metadata["document_type"],
                documents=source_documents,
                document_origin=metadata["document_origin"],
            )
        )
    return chunk_loaded_sources(loaded_sources)


def create_vectorstore(chunks, persist_dir: Path = VECTORSTORE_DIR):
    """Embed chunks and store them in ChromaDB."""
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
    if count <= 0:
        raise KnowledgeBaseNotReadyError(KB_NOT_READY_MESSAGE)
    return vectorstore


def get_retriever(k: int = RETRIEVAL_K, scope: str = "all"):
    """Get a LangChain retriever from the persisted vector store."""
    manager = VectorStoreManager(
        persist_dir=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME,
    )
    return manager.get_retriever(k=k, scope=_normalize_scope(scope))


def similarity_search(query: str, k: int = RETRIEVAL_K, scope: str = "all"):
    """Direct similarity search returning documents with scores."""
    try:
        normalized_scope = _normalize_scope(scope)
        manager = VectorStoreManager(
            persist_dir=VECTORSTORE_DIR,
            collection_name=COLLECTION_NAME,
        )
        raw_results = manager.similarity_search(query, k=k, scope=normalized_scope)
        return _filter_results_for_scope(raw_results, normalized_scope)
    except KnowledgeBaseNotReadyError:
        raise
    except Exception as exc:
        raise RetrievalError("Failed to retrieve knowledge base results.") from exc


def get_vectorstore_stats(
    persist_dir: Path = VECTORSTORE_DIR,
    sample_dir: Path = SAMPLE_DOCS_DIR,
    uploads_dir: Path = UPLOADS_DIR,
):
    """Get statistics about the current vector store and corpus scope."""
    available_files = _available_files_by_origin(sample_dir=sample_dir, uploads_dir=uploads_dir)
    vectorstore = None
    try:
        vectorstore = load_vectorstore(persist_dir=persist_dir)
        count = vectorstore._collection.count()
        all_metadata = vectorstore._collection.get().get("metadatas", [])
        sources = sorted(
            {
                metadata.get("source_file")
                for metadata in all_metadata
                if metadata and metadata.get("source_file")
            }
        )
        scope_snapshot = _build_scope_snapshot(all_metadata, available_files)
        status = "ready" if count > 0 else "not_initialized"
        return {
            "total_chunks": count,
            "total_documents": len(sources),
            "document_names": sources,
            "status": status,
            **scope_snapshot,
        }
    except KnowledgeBaseNotReadyError as exc:
        available_sample_files = [path.name for path in available_files["sample"]]
        available_upload_files = [path.name for path in available_files["upload"]]
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "document_names": [],
            "available_documents": available_sample_files + available_upload_files,
            "available_sample_documents": available_sample_files,
            "available_upload_documents": available_upload_files,
            "indexed_sample_document_count": 0,
            "indexed_upload_document_count": 0,
            "indexed_sample_files": [],
            "indexed_upload_files": [],
            "pending_upload_files": available_upload_files,
            "scope_chunk_counts": {"sample": 0, "upload": 0, "all": 0},
            "status": "not_initialized",
            "error": str(exc),
        }
    except Exception as exc:
        available_sample_files = [path.name for path in available_files["sample"]]
        available_upload_files = [path.name for path in available_files["upload"]]
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "document_names": [],
            "available_documents": available_sample_files + available_upload_files,
            "available_sample_documents": available_sample_files,
            "available_upload_documents": available_upload_files,
            "indexed_sample_document_count": 0,
            "indexed_upload_document_count": 0,
            "indexed_sample_files": [],
            "indexed_upload_files": [],
            "pending_upload_files": available_upload_files,
            "scope_chunk_counts": {"sample": 0, "upload": 0, "all": 0},
            "status": "error",
            "error": str(exc),
        }
    finally:
        _release_chroma_resources(vectorstore)
        del vectorstore
        gc.collect()


def _swap_vectorstore(temp_dir: Path, persist_dir: Path) -> None:
    backup_dir = persist_dir.with_name(f"{persist_dir.name}_backup")
    _safe_rmtree(backup_dir)

    try:
        if persist_dir.exists():
            persist_dir.replace(backup_dir)
        temp_dir.replace(persist_dir)
        _safe_rmtree(backup_dir)
    except Exception as exc:
        if not persist_dir.exists() and backup_dir.exists():
            backup_dir.replace(persist_dir)
        if _is_windows_lock_error(exc):
            raise PermissionError(VECTORSTORE_LOCKED_MESSAGE) from exc
        raise


def ingest_documents(
    sample_dir: Path = SAMPLE_DOCS_DIR,
    uploads_dir: Path = UPLOADS_DIR,
    persist_dir: Path = VECTORSTORE_DIR,
):
    """Atomically ingest sample and uploaded PDFs into a single scoped KB."""
    print("=" * 60)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    persist_dir = Path(persist_dir)
    persist_dir.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_temp_build_dirs(persist_dir.parent)

    try:
        print("\n[1/3] Loading PDFs...")
        loaded_sources = _load_sources(sample_dir=sample_dir, uploads_dir=uploads_dir)
        if not loaded_sources:
            raise NoDocumentsFoundError(
                "No PDF files found. Generate sample PDFs or upload PDFs first."
            )
        unique_sources, duplicate_files = _deduplicate_loaded_sources(loaded_sources)
        if duplicate_files:
            print(f"Skipped {len(duplicate_files)} duplicate file(s) by SHA256")

        print("\n[2/3] Chunking documents...")
        chunks = chunk_loaded_sources(unique_sources)
        if not chunks:
            raise NoDocumentsFoundError("No document content was available for indexing.")
        unique_chunks, duplicate_chunk_ids = deduplicate_documents_by_chunk_id(chunks)
        if duplicate_chunk_ids:
            print(f"Skipped {len(duplicate_chunk_ids)} duplicate chunk(s) by chunk_id")
        ingestion_report = _build_ingestion_report(
            loaded_sources=loaded_sources,
            unique_sources=unique_sources,
            duplicate_files=duplicate_files,
            chunks=chunks,
            unique_chunks=unique_chunks,
            duplicate_chunk_ids=duplicate_chunk_ids,
        )

        print("\n[3/3] Embedding and storing in ChromaDB...")
        temp_root = Path(tempfile.mkdtemp(prefix="vectorstore_build_", dir=str(persist_dir.parent)))
        temp_persist_dir = temp_root / persist_dir.name
        try:
            manager = VectorStoreManager(
                persist_dir=temp_persist_dir,
                collection_name=COLLECTION_NAME,
            )
            vectorstore = manager.upsert_documents(unique_chunks)
            _release_chroma_resources(vectorstore)
            del vectorstore
            del manager
            gc.collect()
            _swap_vectorstore(temp_persist_dir, persist_dir)
        finally:
            _safe_rmtree(temp_root)
            _cleanup_temp_build_dirs(persist_dir.parent)

        stats = get_vectorstore_stats(
            persist_dir=persist_dir,
            sample_dir=sample_dir,
            uploads_dir=uploads_dir,
        )
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print(f"  Documents: {stats['total_documents']}")
        print(f"  Chunks:    {stats['total_chunks']}")
        print("=" * 60)
        return {**stats, **ingestion_report}
    except (KnowledgeBaseNotReadyError, NoDocumentsFoundError):
        raise
    except PermissionError as exc:
        if _is_windows_lock_error(exc):
            raise IngestionError(VECTORSTORE_LOCKED_MESSAGE) from exc
        raise IngestionError(f"Document ingestion failed: {exc}") from exc
    except Exception as exc:
        if _is_windows_lock_error(exc):
            raise IngestionError(VECTORSTORE_LOCKED_MESSAGE) from exc
        raise IngestionError(f"Document ingestion failed: {exc}") from exc
    finally:
        _cleanup_temp_build_dirs(persist_dir.parent)


if __name__ == "__main__":
    ingest_documents()
