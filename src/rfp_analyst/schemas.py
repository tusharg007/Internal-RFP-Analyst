"""Shared schemas for ingestion and retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from langchain_core.documents import Document


@dataclass(frozen=True)
class IngestionRecord:
    """Persistent ingestion metadata used for duplicate prevention."""

    source_file: str
    source_path: str
    file_hash: str
    page_count: int
    document_type: str | None = None
    document_origin: str = "sample"
    chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "IngestionRecord":
        return cls(**payload)


@dataclass(frozen=True)
class LoadedSource:
    """Represents a validated source PDF and its extracted page documents."""

    source_file: str
    source_path: str
    file_hash: str
    page_count: int
    document_type: str | None
    documents: list[Document]
    document_origin: str = "sample"
