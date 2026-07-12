"""Deterministic document chunking."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from rfp_analyst.schemas import LoadedSource


def _normalize_source_namespace(source_path: str, source_file: str) -> str:
    """Return a stable path namespace for deterministic chunk IDs."""
    if source_path:
        return Path(source_path).as_posix().lower()
    return source_file.lower()


def build_chunk_id(
    document_origin: str,
    source_path: str,
    source_file: str,
    file_hash: str,
    page: int,
    chunk_index: int,
    content: str,
) -> str:
    """Build a deterministic chunk ID from source identity and chunk position."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    fingerprint = "|".join(
        [
            document_origin,
            _normalize_source_namespace(source_path, source_file),
            file_hash,
            source_file,
            str(page),
            str(chunk_index),
            content_hash,
        ]
    ).encode("utf-8")
    return hashlib.sha256(fingerprint).hexdigest()


def chunk_loaded_sources(
    loaded_sources: list[LoadedSource],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split source documents into chunks with deterministic metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


    chunks: list[Document] = []
    for source in loaded_sources:
        source_chunks = splitter.split_documents(source.documents)
        for chunk_index, chunk in enumerate(source_chunks):
            page = int(chunk.metadata.get("page", 0))
            chunk_id = build_chunk_id(
                document_origin=source.document_origin,
                source_path=source.source_path,
                source_file=source.source_file,
                file_hash=source.file_hash,
                page=page,
                chunk_index=chunk_index,
                content=chunk.page_content,
            )
            chunk.metadata.update(
                {
                    "source_file": source.source_file,
                    "source_path": source.source_path,
                    "file_hash": source.file_hash,
                    "page": page,
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "document_type": source.document_type,
                    "document_origin": source.document_origin,
                }
            )
            chunks.append(chunk)

    print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
