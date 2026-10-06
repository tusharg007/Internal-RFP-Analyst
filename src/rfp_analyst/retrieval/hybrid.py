"""Provenance-validated graph joins, original-text hydration and rank fusion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rfp_analyst.graph.extraction import (
    ChunkExtraction,
    ExtractedFact,
    normalized,
    validate_extraction,
)
from rfp_analyst.graph.ingestion import (
    DomainAssertion,
    DomainEntity,
    IndexedChunk,
    digest,
    read_indexed_corpus,
    text_hash,
)
from rfp_analyst.graph.models import GraphChunk, GraphDocument
from rfp_analyst.graph.reader import (
    GraphReadFailure,
    GraphReader,
    UnavailableGraphReader,
    create_graph_reader,
)
from rfp_analyst.retrieval.decisions import (
    GraphPlan,
    RetrievalDecision,
    plan_retrieval,
    technology_seed_group,
    matches_project_reference,
)

MAX_PATHS = 20
MAX_ASSERTIONS = 30
MAX_CHUNKS = 20
MAX_CONTEXT_CHARS = 16000
RRF_CONSTANT = 60


@dataclass(frozen=True)
class CorpusSnapshot:
    inputs: tuple[IndexedChunk, ...]

    def __post_init__(self):
        if len(self.inputs) > 1000 or len({i.chunk_id for i in self.inputs}) != len(self.inputs):
            raise ValueError("Invalid frozen corpus bound/identity")
        for item in self.inputs:
            IndexedChunk.model_validate(item.model_dump())

    @property
    def fingerprint(self):
        return digest([i.model_dump() for i in sorted(self.inputs, key=lambda i: i.chunk_id)])

    def documents(self, scope):
        return sorted(
            {i.document_id for i in self.inputs if scope == "all" or i.document_origin == scope}
        )

    def evidence(self, chunk: GraphChunk, document: GraphDocument, *, corpus_id, version, scope):
        item = next((i for i in self.inputs if i.chunk_id == chunk.chunk_id), None)
        if (
            item is None
            or item.evidence_id != chunk.evidence_id
            or item.document_id != chunk.document_id
            or chunk.corpus_id != corpus_id
            or chunk.corpus_version != version
            or document.corpus_id != corpus_id
            or document.corpus_version != version
            or document.document_id != chunk.document_id
            or document.access_partition != "internal"
            or chunk.span_scope != "chunk"
            or chunk.span_start != 0
            or chunk.span_end != len(item.text)
            or chunk.text_hash != text_hash(item.text)
            or any(
                getattr(chunk, field) != getattr(item, field)
                for field in ("source_file", "document_origin", "file_hash", "page_index")
            )
            or any(
                getattr(document, field) != getattr(item, field)
                for field in ("source_file", "document_origin", "file_hash")
            )
            or (scope != "all" and item.document_origin != scope)
        ):
            raise GraphReadFailure("invalid_provenance")
        return item


def load_indexed_snapshot(persist_dir: Path | None = None, collection_name: str | None = None):
    """No embedding/model download, no upsert; don't stop shared Chroma systems."""
    from config import COLLECTION_NAME, VECTORSTORE_DIR
    from chromadb import PersistentClient
    from chromadb.config import Settings

    directory = persist_dir or VECTORSTORE_DIR
    if not (directory / "chroma.sqlite3").is_file():
        raise GraphReadFailure("index_unavailable")
    client = PersistentClient(path=str(directory), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(collection_name or COLLECTION_NAME, embedding_function=None)
    return CorpusSnapshot(read_indexed_corpus(collection))


@dataclass(frozen=True)
class FactWitness:
    assertion: DomainAssertion
    obj: DomainEntity | None
    source: IndexedChunk


@dataclass(frozen=True)
class SubjectWitness:
    entity: DomainEntity
    identity: IndexedChunk
    facts: tuple[FactWitness, ...]


@dataclass(frozen=True)
class EvidencePath:
    subjects: tuple[SubjectWitness, ...]
    facts: tuple[FactWitness, ...]
    query_type: str

    @property
    def sources(self):
        unique = {s.identity.chunk_id: s.identity for s in self.subjects}
        unique.update({fact.source.chunk_id: fact.source for fact in self.facts})
        return tuple(
            sorted(unique.values(), key=lambda i: (i.document_id, i.page_index, i.chunk_index))
        )

    @property
    def assertion_ids(self):
        return sorted({f.assertion.assertion_id for f in self.facts})

    def metadata(self):
        subject_ids = sorted({s.entity.entity_id for s in self.subjects})
        entities = sorted({f.obj.entity_id for f in self.facts if f.obj})
        evidence_ids = [i.evidence_id for i in self.sources]
        return {
            "path_id": digest([self.query_type, subject_ids, self.assertion_ids]),
            "query_type": self.query_type,
            "subject_ids": subject_ids,
            "entity_ids": entities,
            "assertion_ids": self.assertion_ids,
            "relationships": [
                {
                    "subject_id": f.assertion.subject_id,
                    "predicate": f.assertion.fact.predicate,
                    "object_id": f.assertion.object_id,
                    "assertion_id": f.assertion.assertion_id,
                    "evidence_id": f.source.evidence_id,
                    "modality": f.assertion.fact.modality,
                }
                for f in self.facts
            ],
            "evidence_ids": evidence_ids,
            "chunk_ids": [i.chunk_id for i in self.sources],
            "business_hops": 2 if len(subject_ids) > 1 else 1,
            "interpretation": "candidate_relationship_not_contractual_satisfaction",
        }


def _domain_entity(properties, corpus_id, version):
    if properties.get("corpus_id") != corpus_id or properties.get("corpus_version") != version:
        raise GraphReadFailure("cross_version_entity")
    return DomainEntity.model_validate(
        {
            key: value
            for key, value in properties.items()
            if key not in {"corpus_id", "corpus_version"}
        }
    )


def _validate_subject(row, facts, corpus, corpus_id, version, scope):
    entity = _domain_entity(row["subject"], corpus_id, version)
    chunk, doc = (
        GraphChunk.model_validate(row["chunk"]),
        GraphDocument.model_validate(row["document"]),
    )
    identity = corpus.evidence(chunk, doc, corpus_id=corpus_id, version=version, scope=scope)
    start, end = row["start"], row["end"]
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(identity.text)
        or row["span_scope"] != "chunk"
    ):
        raise GraphReadFailure("invalid_identity_span")
    if (
        entity.kind not in {"Project", "Requirement"}
        or entity.document_id != identity.document_id
        or normalized(identity.text[start:end]) != normalized(entity.name)
    ):
        raise GraphReadFailure("invalid_subject_identity")
    expected = (
        digest(["Project", identity.document_id])
        if entity.kind == "Project"
        else digest(["Requirement", identity.document_id, identity.page_index, entity.name])
    )
    if expected != entity.entity_id or (entity.kind == "Requirement" and doc.kind != "target_rfp"):
        raise GraphReadFailure("invalid_subject_identity")
    if entity.kind == "Project" and doc.kind not in {
        "case_study",
        "project_outline",
        "proposal",
        "rfp_response",
    }:
        raise GraphReadFailure("invalid_subject_identity")
    result = []
    seen = set()
    for record in facts:
        properties = record["assertion"]
        fact = ExtractedFact.model_validate({k: properties[k] for k in ExtractedFact.model_fields})
        assertion = DomainAssertion(
            assertion_id=properties["assertion_id"],
            subject_id=properties["subject_id"],
            object_id=properties["object_id"],
            evidence_id=properties["evidence_id"],
            fact=fact,
            extraction_method=properties["extraction_method"],
        )
        source = corpus.evidence(
            GraphChunk.model_validate(record["chunk"]),
            GraphDocument.model_validate(record["document"]),
            corpus_id=corpus_id,
            version=version,
            scope=scope,
        )
        if (
            properties["corpus_id"] != corpus_id
            or properties["corpus_version"] != version
            or properties["review_status"] != "validated_source_span"
            or assertion.subject_id != entity.entity_id
            or assertion.evidence_id != source.evidence_id
            or source.document_id != entity.document_id
            or fact.subject_kind != entity.kind
            or record["span_scope"] != "chunk"
            or record["start"] != fact.start
            or record["end"] != fact.end
            or (entity.kind == "Requirement" and entity.name != fact.requirement)
        ):
            raise GraphReadFailure("invalid_assertion_support")
        validate_extraction(ChunkExtraction(facts=(fact,)), source.text)
        obj = _domain_entity(record["object"], corpus_id, version) if record["object"] else None
        from rfp_analyst.graph.extraction import canonicalize, object_kind

        if fact.object_name:
            kind = object_kind(fact.predicate)
            if (
                not obj
                or obj.document_id
                or obj.kind != kind
                or normalized(obj.name) != normalized(canonicalize(kind, fact.object_name))
                or obj.entity_id != digest([kind, normalized(obj.name)])
            ):
                raise GraphReadFailure("invalid_assertion_object")
            if assertion.object_id != obj.entity_id:
                raise GraphReadFailure("invalid_assertion_object")
        elif obj is not None or assertion.object_id:
            raise GraphReadFailure("invalid_scalar_object")
        if assertion.assertion_id != digest(
            [
                entity.entity_id,
                fact.predicate,
                assertion.object_id,
                fact.value,
                fact.modality,
                source.page_index,
                fact.quote,
            ]
        ):
            raise GraphReadFailure("invalid_assertion_identity")
        if assertion.assertion_id in seen:
            raise GraphReadFailure("duplicate_assertion_witness")
        seen.add(assertion.assertion_id)
        result.append(FactWitness(assertion, obj, source))
    return SubjectWitness(entity, identity, tuple(result))


def _resolve_projects(refs, subjects):
    resolved = []
    for ref in refs:
        candidates = [
            s
            for s in subjects
            if matches_project_reference(ref, s.entity.name)
            or matches_project_reference(ref, s.identity.source_file)
        ]
        if len(candidates) != 1 or candidates[0] in resolved:
            raise GraphReadFailure("ambiguous_project_reference")
        resolved.append(candidates[0])
    return tuple(resolved)


def _matches(subject, plan):
    selected = []
    for group in plan.technology_groups:
        matching = [
            f
            for f in subject.facts
            if f.assertion.fact.predicate == "uses_technology" and f.obj and f.obj.name in group
        ]
        if not matching:
            return ()
        selected.extend(matching[:1])
    for name in plan.frameworks:
        matching = [
            f
            for f in subject.facts
            if f.assertion.fact.predicate == "mentions_framework" and f.obj and f.obj.name == name
        ]
        if not matching:
            return ()
        selected.extend(matching[:1])
    if plan.has_framework:
        matching = [f for f in subject.facts if f.assertion.fact.predicate == "mentions_framework"]
        if not matching:
            return ()
        selected.extend(matching[:1])
    if plan.industries:
        matching = [
            f
            for f in subject.facts
            if f.obj and f.obj.kind == "Industry" and normalized(f.obj.name) in plan.industries
        ]
        if not matching:
            return ()
        selected.extend(matching[:1])
    if plan.delivered_keyword:
        matching = [
            f
            for f in subject.facts
            if f.assertion.fact.predicate == "outcome"
            and f.assertion.fact.modality == "achieved"
            and plan.delivered_keyword in normalized(f.assertion.fact.value)
        ]
        if not matching:
            return ()
        selected.extend(matching[:1])
    # Requested scalar fields are source witnesses, not invented satisfaction
    # edges. In particular, achieved-outcome questions need original outcomes.
    selected.extend(f for f in subject.facts if f.assertion.fact.predicate in plan.fields)
    return tuple({f.assertion.assertion_id: f for f in selected}.values())


def projection_paths(projection, corpus, corpus_id, plan, *, scope, target=False):
    query_type = plan.query_type
    if projection.header.indexed_digest != corpus.fingerprint:
        raise GraphReadFailure("unsynchronized_snapshot")
    subjects = tuple(
        _validate_subject(
            row,
            projection.facts.get(row["subject"]["entity_id"], ()),
            corpus,
            corpus_id,
            projection.header.version,
            scope,
        )
        for row in projection.subjects
    )
    requirements = tuple(
        _validate_subject(
            row,
            projection.facts.get(row["subject"]["entity_id"], ()),
            corpus,
            corpus_id,
            projection.header.version,
            scope,
        )
        for row in projection.requirements
    )
    if len({s.entity.entity_id for s in (*subjects, *requirements)}) != len(subjects) + len(
        requirements
    ):
        raise GraphReadFailure("duplicate_subject_witness")
    if (target or plan.query_type == "requirement_candidates") and len(
        {s.entity.document_id for s in requirements}
    ) > 1:
        # Multiple uploaded RFPs must not become one invented conjunctive target.
        raise GraphReadFailure("ambiguous_target_rfp")
    if target:
        return [EvidencePath((s,), s.facts, plan.query_type) for s in requirements if s.facts]
    if plan.query_type == "shared_technologies":
        first, second = _resolve_projects(plan.project_refs, subjects)
        paths = []
        for left in first.facts:
            if not left.obj or left.obj.kind != "Technology":
                continue
            right = next(
                (f for f in second.facts if f.obj and f.obj.entity_id == left.obj.entity_id), None
            )
            if right:
                paths.append(EvidencePath((first, second), (left, right), plan.query_type))
        return paths
    if plan.query_type == "project_fields":
        matches = _resolve_projects(plan.project_refs, subjects)
        return [
            EvidencePath(
                (s,),
                tuple(f for f in s.facts if f.assertion.fact.predicate in plan.fields),
                plan.query_type,
            )
            for s in matches
            if any(f.assertion.fact.predicate in plan.fields for f in s.facts)
        ]
    target_facts = ()
    if plan.query_type == "requirement_candidates":
        # Requirements are source-backed constraints, not inferred SATISFIES links.
        target_facts = tuple(f for s in requirements for f in s.facts)
        tech = tuple(
            dict.fromkeys(f.obj.name for f in target_facts if f.obj and f.obj.kind == "Technology")
        )
        frames = tuple(
            dict.fromkeys(
                f.obj.name for f in target_facts if f.obj and f.obj.kind == "ComplianceFramework"
            )
        )
        if len(tech) + len(frames) > 5 or not (tech or frames):
            raise GraphReadFailure("unsupported_target_constraints")
        plan = GraphPlan(
            query_type="project_constraints",
            technology_groups=tuple(
                dict.fromkeys((*plan.technology_groups, *(technology_seed_group(n) for n in tech)))
            ),
            frameworks=tuple(dict.fromkeys((*plan.frameworks, *frames))),
            industries=plan.industries,
            has_framework=plan.has_framework,
            delivered_keyword=plan.delivered_keyword,
            fields=plan.fields,
        )
    paths = []
    for subject in subjects:
        matching = _matches(subject, plan)
        if matching:
            paths.append(
                EvidencePath((subject, *requirements), (*matching, *target_facts), query_type)
            )
    return paths


def _source_document(source, assertion_ids):
    return {
        "source": source.source_file,
        "page": source.page_index,
        "score": None,
        "raw_score": None,
        "content": source.text,
        "chunk_id": source.chunk_id,
        "document_origin": source.document_origin,
        "file_hash": source.file_hash,
        "document_id": source.document_id,
        "evidence_id": source.evidence_id,
        "retrieval_channel": "graph",
        "graph_assertion_ids": assertion_ids,
    }


def fuse_evidence(vectors, paths, *, k=6):
    """RRF orders lists; complete graph witnesses are reserved, never sliced."""
    graph_docs, selected_paths, assertions = {}, [], set()
    for path in paths:
        next_ids = set(path.assertion_ids)
        additions = {i.chunk_id: i for i in path.sources if i.chunk_id not in graph_docs}
        if (
            len(selected_paths) >= MAX_PATHS
            or len(assertions | next_ids) > MAX_ASSERTIONS
            or len(graph_docs) + len(additions) > MAX_CHUNKS
            or sum(len(d["content"]) for d in graph_docs.values())
            + sum(len(i.text) for i in additions.values())
            > MAX_CONTEXT_CHARS
        ):
            continue
        selected_paths.append(path)
        assertions |= next_ids
        for item in path.sources:
            row = graph_docs.setdefault(item.chunk_id, _source_document(item, []))
            row["graph_assertion_ids"] = sorted(
                set(row["graph_assertion_ids"])
                | {
                    f.assertion.assertion_id
                    for f in path.facts
                    if f.source.chunk_id == item.chunk_id
                }
            )
    ranks, rows = {}, {}
    for rank, row in enumerate(vectors, 1):
        key = row.get("chunk_id") or digest(
            [row["source"], row["page"], row["document_origin"], row["content"]]
        )
        if key not in rows:
            rows[key] = dict(row) | {"retrieval_channel": "vector"}
            ranks[key] = 1 / (RRF_CONSTANT + rank)
    for rank, (key, graph) in enumerate(graph_docs.items(), 1):
        if key in rows:
            vector = rows[key]
            if any(
                vector.get(field) != graph[field]
                for field in ("source", "page", "document_origin", "content")
            ):
                raise GraphReadFailure("vector_snapshot_mismatch")
            rows[key] = (
                vector
                | {
                    name: value
                    for name, value in graph.items()
                    if name not in {"score", "raw_score"}
                }
                | {"retrieval_channel": "hybrid"}
            )
        else:
            rows[key] = graph
        ranks[key] = ranks.get(key, 0) + 1 / (RRF_CONSTANT + rank)
    ordered = sorted(rows, key=lambda key: (-ranks[key], key))
    keep = set(graph_docs)
    char_count = sum(len(rows[key]["content"]) for key in keep)
    for key in ordered:
        if len(keep) >= max(min(k, MAX_CHUNKS), len(graph_docs)):
            break
        if key not in keep and char_count + len(rows[key]["content"]) <= MAX_CONTEXT_CHARS:
            keep.add(key)
            char_count += len(rows[key]["content"])
    selected = [rows[key] | {"fusion_score": ranks[key]} for key in ordered if key in keep]
    return selected, selected_paths


def select_generation_evidence(documents, paths, *, max_chars):
    """Keep whole witness groups across primary and secondary tool retrievals."""
    by_id = {d["chunk_id"]: d for d in documents}
    selected, approved, assertions = set(), [], set()
    for path in paths:
        needed = set(path["chunk_ids"])
        new_assertions = assertions | set(path["assertion_ids"])
        if (
            not needed <= set(by_id)
            or len(approved) >= MAX_PATHS
            or len(new_assertions) > MAX_ASSERTIONS
            or len(selected | needed) > MAX_CHUNKS
            or sum(
                len(by_id[key]["content"]) + len(by_id[key]["source"]) + 40
                for key in selected | needed
            )
            > max_chars
        ):
            continue
        selected |= needed
        assertions = new_assertions
        approved.append(path)
    for doc in documents:
        if doc.get("evidence_id") or len(selected) >= MAX_CHUNKS:
            continue
        key = doc["chunk_id"]
        if (
            sum(len(by_id[k]["content"]) + len(by_id[k]["source"]) + 40 for k in selected | {key})
            <= max_chars
        ):
            selected.add(key)
    # Export only assertions belonging to retained, complete witness groups.
    kept = []
    seen = set()
    for doc in documents:
        if doc["chunk_id"] not in selected or doc["chunk_id"] in seen:
            continue
        seen.add(doc["chunk_id"])
        row = dict(doc)
        if "graph_assertion_ids" in row:
            row["graph_assertion_ids"] = sorted(set(row["graph_assertion_ids"]) & assertions)
        kept.append(row)
    return kept, approved


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[dict]
    decision: RetrievalDecision
    effective_mode: str
    fallback_reason: str = ""
    paths: tuple[dict, ...] = ()
    provenance: tuple[dict, ...] = ()
    entities: tuple[dict, ...] = ()
    version: str = ""
    partial: bool = False

    def trace(self):
        return {
            "step": "hybrid_retrieval",
            "tool": "graph_retrieval",
            "requested_retrieval_mode": self.decision.mode,
            "retrieval_mode": self.effective_mode,
            "graph_query_type": self.decision.plan.query_type,
            "graph_paths": list(self.paths),
            "graph_provenance": list(self.provenance),
            "graph_entities": list(self.entities),
            "graph_version": self.version,
            "fallback_reason": self.fallback_reason,
            "partial": self.partial,
            "coverage": "conservative_partial",
            "input_summary": "Bounded, parameterized relationship retrieval",
            "output_summary": f"Mode={self.effective_mode}; chunks={len(self.documents)}; paths={len(self.paths)}"
            + (f"; graph fallback={self.fallback_reason}" if self.fallback_reason else ""),
        }


class HybridRetrievalProvider:
    """One caller-owned provider per workflow; no mutable global retrieval mode."""

    def __init__(
        self,
        reader: GraphReader,
        corpus_loader: Callable[[], CorpusSnapshot],
        *,
        corpus_id="internal-rfp",
        policy="auto",
        strict=False,
        owns_reader=False,
    ):
        if policy not in {"auto", "vector_only", "graph_only", "hybrid"}:
            raise ValueError("Unsupported retrieval policy")
        self.reader, self.corpus_loader, self.corpus_id = reader, corpus_loader, corpus_id
        self.policy, self.strict, self.owns_reader = policy, strict, owns_reader
        self.events: list[RetrievalResult] = []
        self._corpus: CorpusSnapshot | None = None
        self._version = ""
        self._unavailable_reason = (
            reader.reason if isinstance(reader, UnavailableGraphReader) else ""
        )

    def decision(self, query, *, target=False):
        if target and self.policy == "auto":
            return plan_retrieval(query, policy="vector_only")
        decision = plan_retrieval(query, policy=self.policy)
        if target and decision.mode != "vector_only":
            decision = decision.model_copy(
                update={"plan": GraphPlan(query_type="requirement_candidates")}
            )
        return decision

    def reset_transient_failure(self):
        """Allow the orchestration's explicit bounded retry, not hidden driver retries."""
        if self._unavailable_reason in {"timeout", "unavailable"}:
            self._unavailable_reason = ""

    def retrieve(self, query, *, k, scope, vector_supplier, target=False):
        try:
            return self._retrieve(
                query, k=k, scope=scope, vector_supplier=vector_supplier, target=target
            )
        except _VectorRetrievalFailure as error:
            raise error.original from None

    def _retrieve(self, query, *, k, scope, vector_supplier, target=False):
        if scope not in {"all", "sample", "upload"} or type(k) is not int or not 1 <= k <= 1000:
            raise ValueError("Invalid retrieval scope/bound")
        decision = self.decision(query, target=target)
        if decision.mode == "vector_only":
            result = RetrievalResult(vector_supplier(), decision, "vector_only")
            self.events.append(result)
            return result

        vector_cache = None

        def vectors_once():
            nonlocal vector_cache
            if vector_cache is None:
                try:
                    vector_cache = vector_supplier()
                except Exception as error:
                    raise _VectorRetrievalFailure(error) from error
            return vector_cache

        def fallback(reason):
            documents = [] if self.strict and decision.mode == "graph_only" else vectors_once()
            mode = "graph_only" if self.strict and decision.mode == "graph_only" else "vector_only"
            result = RetrievalResult(documents, decision, mode, fallback_reason=reason)
            self.events.append(result)
            return result

        if decision.plan.query_type == "unsupported":
            return fallback("unsupported_plan")
        if self._unavailable_reason:
            return fallback(self._unavailable_reason)
        try:
            corpus = self._corpus or self.corpus_loader()
            projection = self.reader.fetch(
                self.corpus_id,
                corpus.fingerprint,
                corpus.documents(scope),
                decision.plan,
                scope=scope,
                target=target,
            )
            if self._version and self._version != projection.header.version:
                raise GraphReadFailure("snapshot_changed")
            paths = projection_paths(
                projection, corpus, self.corpus_id, decision.plan, scope=scope, target=target
            )
            if not paths:
                return fallback("no_graph_evidence")
            vectors = vectors_once() if decision.mode == "hybrid" else []
            # Graph and semantic candidates must describe the same captured index.
            source_map = {i.chunk_id: i for i in corpus.inputs}
            for vector in vectors:
                item = source_map.get(vector.get("chunk_id"))
                if (
                    item is None
                    or vector["content"] != item.text
                    or vector["source"] != item.source_file
                    or vector["page"] != item.page_index
                    or vector["document_origin"] != item.document_origin
                ):
                    raise GraphReadFailure("vector_snapshot_mismatch")
            documents, selected = fuse_evidence(vectors, paths, k=k)
            if not selected:
                return fallback("witness_budget_exceeded")
            provenance = tuple(
                {
                    "chunk_id": row["chunk_id"],
                    "evidence_id": row["evidence_id"],
                    "document_id": row["document_id"],
                    "source": row["source"],
                    "page_index": row["page"],
                    "document_origin": row["document_origin"],
                    "file_hash": row["file_hash"],
                    "assertion_ids": row["graph_assertion_ids"],
                }
                for row in documents
                if row.get("evidence_id")
            )
            entities = {
                s.entity.entity_id: {"entity_id": s.entity.entity_id, "kind": s.entity.kind}
                for path in selected
                for s in path.subjects
            }
            entities.update(
                {
                    f.obj.entity_id: {"entity_id": f.obj.entity_id, "kind": f.obj.kind}
                    for path in selected
                    for f in path.facts
                    if f.obj
                }
            )
            self._corpus, self._version = corpus, projection.header.version
            result = RetrievalResult(
                documents,
                decision,
                decision.mode,
                paths=tuple(p.metadata() for p in selected),
                provenance=provenance,
                entities=tuple(entities.values()),
                version=self._version,
                partial=len(selected) < len(paths),
            )
            self.events.append(result)
            return result
        except _VectorRetrievalFailure as error:
            raise error.original
        except GraphReadFailure as error:
            if error.reason in {
                "disabled",
                "not_configured",
                "invalid_configuration",
                "dependency_missing",
                "unavailable",
                "timeout",
            }:
                self._unavailable_reason = error.reason
            return fallback(error.reason)
        except Exception:
            return fallback("invalid_or_unavailable_graph")

    def validate_current(self):
        if not self._version:
            return True
        try:
            return (
                self.reader.current_version(self.corpus_id) == self._version
                and self.corpus_loader().fingerprint == self._corpus.fingerprint
            )
        except Exception:
            return False

    def close(self):
        if self.owns_reader:
            self.reader.close()


class _VectorRetrievalFailure(Exception):
    def __init__(self, original):
        self.original = original


def create_retrieval_provider(*, policy=None):
    from config import get_retrieval_settings

    settings = get_retrieval_settings()
    return HybridRetrievalProvider(
        create_graph_reader(),
        load_indexed_snapshot,
        corpus_id=settings["corpus_id"],
        policy=policy or settings["policy"],
        owns_reader=True,
    )
