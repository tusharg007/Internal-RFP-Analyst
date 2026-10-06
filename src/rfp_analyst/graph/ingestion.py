"""Frozen indexed-corpus -> validated domain snapshot, independent of querying."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from .extraction import (
    NORMALIZATION_VERSION,
    SECTIONS,
    EntityKind,
    ExtractedFact,
    Extractor,
    SectionExtractor,
    StrictModel,
    canonicalize,
    normalized,
    object_kind,
    validate_extraction,
    ChunkExtraction,
)
from .models import GraphChunk, GraphDocument, Identifier, Sha256
from .store import GraphHealth, GraphIntegrityError

logger = logging.getLogger(__name__)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IndexedChunk(StrictModel):
    chunk_id: Identifier
    source_file: Annotated[str, Field(min_length=1, max_length=255)]
    document_origin: Literal["sample", "upload"]
    file_hash: Sha256
    page_index: Annotated[int, Field(strict=True, ge=0)]
    chunk_index: Annotated[int, Field(strict=True, ge=0)]
    text: Annotated[str, Field(min_length=1, max_length=10000)]

    @model_validator(mode="after")
    def filename(self):
        GraphDocument.validate_source_filename(self.source_file)
        return self

    @classmethod
    def from_document(cls, document):
        metadata = document.metadata
        return cls(
            chunk_id=metadata["chunk_id"],
            source_file=metadata["source_file"],
            document_origin=metadata["document_origin"],
            file_hash=metadata["file_hash"],
            page_index=metadata["page"],
            chunk_index=metadata["chunk_index"],
            text=document.page_content,
        )

    @property
    def document_id(self):
        return digest([self.document_origin, self.source_file, self.file_hash])

    @property
    def evidence_id(self):
        return digest([self.document_id, self.page_index, self.chunk_index, text_hash(self.text)])


class DomainEntity(StrictModel):
    entity_id: Sha256
    kind: EntityKind
    name: Annotated[str, Field(min_length=1, max_length=2000)]
    document_id: str = ""  # Project/Requirement only; shared entities are corpus-scoped.


class DomainAssertion(StrictModel):
    assertion_id: Sha256
    subject_id: Sha256
    object_id: str = ""
    evidence_id: Sha256
    fact: ExtractedFact
    extraction_method: Literal["deterministic", "structured_llm", "identity"]


class IdentitySupport(StrictModel):
    entity_id: Sha256
    evidence_id: Sha256
    start: Annotated[int, Field(strict=True, ge=0)]
    end: Annotated[int, Field(strict=True, gt=0)]


class GraphSnapshot(StrictModel):
    corpus_id: Identifier
    version: Sha256
    extractor_version: Identifier
    normalization_version: Literal["curated-aliases-v1"] = NORMALIZATION_VERSION
    inputs: tuple[IndexedChunk, ...]
    documents: tuple[GraphDocument, ...]
    chunks: tuple[GraphChunk, ...]
    entities: tuple[DomainEntity, ...]
    assertions: tuple[DomainAssertion, ...]
    identities: tuple[IdentitySupport, ...]

    def fingerprint(self):
        payload = self.model_dump(exclude={"version"})
        for category in ("documents", "chunks"):
            for row in payload[category]:
                row["corpus_version"] = "pending"
        return digest(payload)

    def validate_integrity(self):
        # Revalidate model_copy/model_construct at the database boundary too.
        validated = GraphSnapshot.model_validate(self.model_dump())
        if validated.fingerprint() != self.version:
            raise GraphIntegrityError("Snapshot content/version mismatch")
        source = {item.evidence_id: item for item in self.inputs}
        chunks = {item.evidence_id: item for item in self.chunks}
        entities = {item.entity_id: item for item in self.entities}
        docs = {item.document_id: item for item in self.documents}
        if (
            len(source) != len(self.inputs)
            or len(chunks) != len(self.chunks)
            or set(source) != set(chunks)
        ):
            raise GraphIntegrityError("Duplicate/missing evidence mapping")
        if len(entities) != len(self.entities) or len(docs) != len(self.documents):
            raise GraphIntegrityError("Duplicate graph identities")
        if set(docs) != {item.document_id for item in self.inputs}:
            raise GraphIntegrityError("Document set does not match indexed corpus")
        if len({item.chunk_id for item in self.inputs}) != len(self.inputs):
            raise GraphIntegrityError("Legacy chunk ID collision")
        for entity in self.entities:
            if entity.kind == "Project":
                expected_id = digest(["Project", entity.document_id])
                if docs.get(entity.document_id) is None or docs[entity.document_id].kind not in {
                    "project_outline",
                    "case_study",
                    "proposal",
                    "rfp_response",
                }:
                    raise GraphIntegrityError(
                        "Project must belong to a recognized project document"
                    )
            elif entity.kind == "Requirement":
                # Requirement identity is validated with its specific page below.
                if (
                    docs.get(entity.document_id) is None
                    or docs[entity.document_id].kind != "target_rfp"
                ):
                    raise GraphIntegrityError("Requirement must belong to a target RFP")
                continue
            else:
                if entity.document_id:
                    raise GraphIntegrityError(
                        "Canonical entities must be shared, not document-scoped"
                    )
                expected_id = digest(
                    [entity.kind, normalized(canonicalize(entity.kind, entity.name))]
                )
            if entity.entity_id != expected_id:
                raise GraphIntegrityError("Non-deterministic entity identity")
        supported = set()
        for chunk in self.chunks:
            item = source[chunk.evidence_id]
            parent = docs.get(chunk.document_id)
            if (
                not parent
                or chunk.corpus_id != self.corpus_id
                or chunk.corpus_version != self.version
                or chunk.document_id != item.document_id
                or chunk.chunk_id != item.chunk_id
                or chunk.text_hash != text_hash(item.text)
                or chunk.span_scope != "chunk"
                or chunk.span_start != 0
                or chunk.span_end != len(item.text)
                or any(
                    getattr(chunk, name) != getattr(item, name)
                    for name in ("source_file", "document_origin", "file_hash", "page_index")
                )
                or any(
                    getattr(parent, name) != getattr(item, name)
                    for name in ("source_file", "document_origin", "file_hash")
                )
                or parent.corpus_id != self.corpus_id
                or parent.corpus_version != self.version
            ):
                raise GraphIntegrityError("Invalid document/chunk provenance")
        for identity in self.identities:
            entity = entities.get(identity.entity_id)
            item = source.get(identity.evidence_id)
            if (
                not entity
                or not item
                or entity.kind not in {"Project", "Requirement"}
                or entity.document_id != item.document_id
                or identity.end <= identity.start
                or identity.end > len(item.text)
                or normalized(item.text[identity.start : identity.end]) != normalized(entity.name)
            ):
                raise GraphIntegrityError("Subject identity lacks exact document support")
            if entity.kind == "Requirement" and entity.entity_id != digest(
                ["Requirement", entity.document_id, item.page_index, entity.name]
            ):
                raise GraphIntegrityError("Non-deterministic requirement identity")
            supported.add(entity.entity_id)
        for assertion in self.assertions:
            item = source.get(assertion.evidence_id)
            subject = entities.get(assertion.subject_id)
            obj = entities.get(assertion.object_id)
            if (
                not item
                or not subject
                or subject.document_id != item.document_id
                or subject.kind != assertion.fact.subject_kind
            ):
                raise GraphIntegrityError("Cross-document or missing assertion support")
            validate_extraction(ChunkExtraction(facts=(assertion.fact,)), item.text)
            if subject.kind == "Requirement" and subject.name != assertion.fact.requirement:
                raise GraphIntegrityError("Requirement assertion changed its subject span")
            if assertion.assertion_id != digest(
                [
                    subject.entity_id,
                    assertion.fact.predicate,
                    assertion.object_id,
                    assertion.fact.value,
                    assertion.fact.modality,
                    item.page_index,
                    assertion.fact.quote,
                ]
            ):
                raise GraphIntegrityError("Non-deterministic assertion identity")
            if assertion.fact.object_name:
                kind = object_kind(assertion.fact.predicate)
                if (
                    not obj
                    or obj.kind != kind
                    or normalized(obj.name)
                    != normalized(canonicalize(kind, assertion.fact.object_name))
                ):
                    raise GraphIntegrityError("Invalid assertion object")
                supported.add(obj.entity_id)
            elif assertion.object_id:
                raise GraphIntegrityError("Scalar assertion has an object")
            if subject.entity_id not in supported:
                raise GraphIntegrityError("Assertion subject has no identity witness")
        if supported != set(entities):
            raise GraphIntegrityError("Unsupported entity would be published")
        if len({a.assertion_id for a in self.assertions}) != len(self.assertions):
            raise GraphIntegrityError("Duplicate assertions")
        return validated

    @property
    def row_count(self):
        return (
            len(self.documents)
            + len(self.chunks)
            + len(self.entities)
            + len(self.assertions)
            + len(self.identities)
        )


@dataclass(frozen=True)
class QuarantineIssue:
    chunk_id: str
    reason: str


class ExtractionRejected(ValueError):
    def __init__(self, issues: Sequence[QuarantineIssue]):
        self.issues = tuple(issues)
        super().__init__(f"Graph snapshot rejected: {len(issues)} quarantined extraction(s)")


def build_snapshot(
    corpus_id: str, inputs: Sequence[IndexedChunk], *, extractor: Extractor | None = None
):
    extractor = extractor or SectionExtractor()
    unique = {}
    for item in inputs:
        item = IndexedChunk.model_validate(item.model_dump())
        if item.chunk_id in unique and unique[item.chunk_id] != item:
            raise GraphIntegrityError("Conflicting indexed chunk IDs")
        unique[item.chunk_id] = item
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda item: (item.document_id, item.page_index, item.chunk_index, item.chunk_id),
        )
    )
    documents, chunks, entities, assertions, identities, issues = {}, [], {}, {}, {}, []
    grouped = {}
    for item in ordered:
        grouped.setdefault(item.document_id, []).append(item)
    for doc_id, items in grouped.items():
        cover = next(
            (item for item in items if item.page_index == 0 and "Document Type:" in item.text), None
        )
        match = (
            re.search(
                r"Document Type:\s*(Project Outline|Proposal|Case Study|RFP Response)",
                cover.text,
                re.I,
            )
            if cover
            else None
        )
        kind = match.group(1).lower().replace(" ", "_") if match else "other"
        target = not match and any(
            re.search(r"(?:Client )?RFP Requirements", item.text, re.I) for item in items
        )
        if target:
            kind = "target_rfp"
        title = re.search(r"^[^\n]*\n(.+?)\nDocument Type:", cover.text, re.S) if match else None
        project_id = ""
        if title:
            name = " ".join(title.group(1).split())
            project_id = digest(["Project", doc_id])
            entities[project_id] = DomainEntity(
                entity_id=project_id, kind="Project", name=name, document_id=doc_id
            )
            identities[project_id] = IdentitySupport(
                entity_id=project_id,
                evidence_id=cover.evidence_id,
                start=title.start(1),
                end=title.end(1),
            )
        first = items[0]
        documents[doc_id] = GraphDocument(
            corpus_id=corpus_id,
            corpus_version="pending",
            document_id=doc_id,
            source_file=first.source_file,
            document_origin=first.document_origin,
            file_hash=first.file_hash,
            revision=first.file_hash,
            kind=kind,
        )
        section = ""
        previous_page = -1
        for item in items:
            if previous_page != item.page_index:
                section = ""
            previous_page = item.page_index
            chunks.append(
                GraphChunk(
                    corpus_id=corpus_id,
                    corpus_version="pending",
                    document_id=doc_id,
                    source_file=item.source_file,
                    document_origin=item.document_origin,
                    file_hash=item.file_hash,
                    evidence_id=item.evidence_id,
                    chunk_id=item.chunk_id,
                    page_index=item.page_index,
                    text_hash=text_hash(item.text),
                    span_start=0,
                    span_end=len(item.text),
                    span_scope="chunk",
                )
            )
            try:
                result = validate_extraction(
                    extractor.extract(
                        item.text, project=bool(project_id), target_rfp=target, section=section
                    ),
                    item.text,
                )
                for fact in result.facts:
                    if (fact.subject_kind == "Project" and not project_id) or (
                        fact.subject_kind == "Requirement" and not target
                    ):
                        raise ValueError("Unrecognized document cannot create this subject")
                    subject_id = project_id
                    if fact.subject_kind == "Requirement":
                        subject_id = digest(
                            ["Requirement", doc_id, item.page_index, fact.requirement]
                        )
                        entities[subject_id] = DomainEntity(
                            entity_id=subject_id,
                            kind="Requirement",
                            name=fact.requirement,
                            document_id=doc_id,
                        )
                        start = item.text.find(fact.requirement, fact.start, fact.end)
                        identities.setdefault(
                            subject_id,
                            IdentitySupport(
                                entity_id=subject_id,
                                evidence_id=item.evidence_id,
                                start=start,
                                end=start + len(fact.requirement),
                            ),
                        )
                    object_id = ""
                    if fact.object_name:
                        entity_kind = object_kind(fact.predicate)
                        name = canonicalize(entity_kind, fact.object_name)
                        if entity_kind == "Industry":
                            name = normalized(name)
                        object_id = digest([entity_kind, normalized(name)])
                        entities[object_id] = DomainEntity(
                            entity_id=object_id, kind=entity_kind, name=name
                        )
                    assertion_id = digest(
                        [
                            subject_id,
                            fact.predicate,
                            object_id,
                            fact.value,
                            fact.modality,
                            item.page_index,
                            fact.quote,
                        ]
                    )
                    assertions.setdefault(
                        assertion_id,
                        DomainAssertion(
                            assertion_id=assertion_id,
                            subject_id=subject_id,
                            object_id=object_id,
                            evidence_id=item.evidence_id,
                            fact=fact,
                            extraction_method="deterministic"
                            if isinstance(extractor, SectionExtractor)
                            else "structured_llm",
                        ),
                    )
            except Exception:
                # Do not emit model output, prompts, source contents or driver secrets.
                issues.append(QuarantineIssue(item.chunk_id, "malformed_or_unsupported_extraction"))
            for line in item.text.splitlines():
                if line.strip() in SECTIONS:
                    section = line.strip()
    if issues:
        raise ExtractionRejected(issues)
    snapshot = GraphSnapshot(
        corpus_id=corpus_id,
        version="0" * 64,
        extractor_version=extractor.version,
        inputs=ordered,
        documents=tuple(documents.values()),
        chunks=tuple(chunks),
        entities=tuple(sorted(entities.values(), key=lambda e: e.entity_id)),
        assertions=tuple(sorted(assertions.values(), key=lambda a: a.assertion_id)),
        identities=tuple(sorted(identities.values(), key=lambda i: i.entity_id)),
    )
    version = snapshot.fingerprint()
    snapshot = snapshot.model_copy(
        update={
            "version": version,
            "documents": tuple(
                d.model_copy(update={"corpus_version": version}) for d in snapshot.documents
            ),
            "chunks": tuple(
                c.model_copy(update={"corpus_version": version}) for c in snapshot.chunks
            ),
        }
    )
    return snapshot.validate_integrity()


class IngestionRepository(Protocol):
    def health(self) -> GraphHealth: ...

    def publish(self, snapshot: GraphSnapshot, *, expected_version: str | None) -> str: ...

    def active_version(self, corpus_id: str) -> str | None: ...


def rebuild_graph(
    repository: IngestionRepository,
    corpus_id: str,
    inputs: Sequence[IndexedChunk],
    *,
    extractor: Extractor | None = None,
):
    health = repository.health()
    if not health.available:
        return {"status": health.status, "published": False}
    from .repository import GraphServiceUnavailable

    try:
        expected = repository.active_version(corpus_id)
        snapshot = build_snapshot(corpus_id, inputs, extractor=extractor)
        status = repository.publish(snapshot, expected_version=expected)
    except GraphServiceUnavailable:
        return {"status": "unavailable", "published": False}
    return {
        "status": status,
        "published": status == "ready",
        "version": snapshot.version,
        "documents": len(snapshot.documents),
        "chunks": len(snapshot.chunks),
        "entities": len(snapshot.entities),
        "assertions": len(snapshot.assertions),
    }


def read_indexed_corpus(collection, *, max_chunks: int = 1000):
    """Collection.get only: no embedding, PDF reload, update or similarity search.

    Export twice and compare content digests to detect common concurrent updates.
    Operators must quiesce vector ingestion for a guaranteed consistent rebuild.
    """
    if type(max_chunks) is not int or not 1 <= max_chunks <= 1000:
        raise ValueError("Indexed corpus export bound must be between 1 and 1000")

    def capture():
        count = collection.count()
        if count > max_chunks:
            raise ValueError("Indexed corpus exceeds the bounded rebuild limit")
        raw = collection.get(limit=max_chunks + 1, include=["documents", "metadatas"])
        ids, texts, metadata = raw["ids"], raw["documents"], raw["metadatas"]
        if not (len(ids) == len(texts) == len(metadata) == count) or len(set(ids)) != count:
            raise GraphIntegrityError("Incomplete or duplicate indexed snapshot")
        result = []
        for chunk_id, text, meta in zip(ids, texts, metadata, strict=True):
            if not meta or meta.get("chunk_id") != chunk_id:
                raise GraphIntegrityError("Indexed legacy chunk mapping is missing or inconsistent")
            result.append(
                IndexedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_file=meta["source_file"],
                    document_origin=meta["document_origin"],
                    file_hash=meta["file_hash"],
                    page_index=meta["page"],
                    chunk_index=meta["chunk_index"],
                )
            )
        return tuple(sorted(result, key=lambda item: item.chunk_id))

    first, second = capture(), capture()
    if first != second:
        raise GraphIntegrityError("Indexed corpus changed during export; retry when quiescent")
    return first


def enqueue_graph_sync(chunks, persist_dir: Path):
    """Post-publication manifest only, no extraction/connection in the UI thread.

    A rebuild always reconciles Chroma; a lost/pending manifest is not truth.
    Atomic replace is last-writer-wins, deliberately not a distributed queue.
    """
    import os
    import tempfile

    from config import get_neo4j_settings

    if not get_neo4j_settings("reader")["enabled"]:
        return
    inputs = sorted(
        (IndexedChunk.from_document(chunk) for chunk in chunks), key=lambda item: item.chunk_id
    )
    manifest = {
        "status": "pending",
        "indexed_digest": digest([i.model_dump() for i in inputs]),
        "chunk_ids": [i.chunk_id for i in inputs],
        "format_version": "graph-sync-v1",
    }
    directory = Path(persist_dir).parent / "graph_sync"
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="pending-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)
        os.replace(temporary, directory / "pending.json")
    finally:
        Path(temporary).unlink(missing_ok=True)
