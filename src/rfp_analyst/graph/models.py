"""Validated, immutable provenance anchors; no extraction or domain inference."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=256)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _ProvenanceRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    corpus_id: Identifier
    corpus_version: Identifier
    document_id: Identifier
    source_file: Annotated[str, Field(min_length=1, max_length=255)]
    document_origin: Literal["sample", "upload"]
    file_hash: Sha256

    @field_validator("source_file")
    @classmethod
    def validate_source_filename(cls, value: str) -> str:
        if value in {".", ".."} or any(char in value for char in "/\\:\x00\r\n"):
            raise ValueError("source_file must be a filename, not a path")
        return value


class GraphDocument(_ProvenanceRecord):
    """An explicit caller-supplied document revision, not a parsed PDF."""

    kind: Literal[
        "case_study", "proposal", "project_outline", "rfp_response", "target_rfp", "other"
    ] = "other"
    revision: Identifier = "1"
    access_partition: Identifier = "internal"


class GraphChunk(_ProvenanceRecord):
    """A portable evidence reference mapping to an unchanged Chroma chunk ID.

    Offsets are caller-supplied half-open coordinates in span_scope (page by
    default; chunk for legacy indexed evidence without source-page offsets).
    This foundation validates structure, not quote accuracy or semantic support.
    """

    evidence_id: Identifier
    chunk_id: Identifier
    page_index: Annotated[int, Field(strict=True, ge=0)]
    text_hash: Sha256
    span_start: Annotated[int, Field(strict=True, ge=0)]
    span_end: Annotated[int, Field(strict=True, gt=0)]
    # Legacy chunks have no source-page offsets. Never fabricate them.
    span_scope: Literal["page", "chunk"] = "page"

    @model_validator(mode="after")
    def validate_span(self) -> GraphChunk:
        if self.span_end <= self.span_start:
            raise ValueError("span_end must be greater than span_start")
        return self
