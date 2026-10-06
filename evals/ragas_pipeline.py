"""Execute the real application once per case against one frozen indexed corpus."""

from __future__ import annotations

from pathlib import Path


class FrozenPipeline:
    def __init__(self, index: Path, *, collection: str | None = None):
        from config import COLLECTION_NAME, get_retrieval_settings
        from rfp_analyst.retrieval.hybrid import load_indexed_snapshot

        self.index = Path(index).resolve()
        self.collection = collection or COLLECTION_NAME
        self.snapshot = load_indexed_snapshot(self.index, self.collection)
        self.corpus_hash = self.snapshot.fingerprint
        self.corpus_id = get_retrieval_settings()["corpus_id"]
        self.manager = None
        self.llm = None
        self.generation_metadata = {}
        self.stats = self._stats()

    def _stats(self):
        inputs = self.snapshot.inputs
        by_origin = {
            origin: sorted({i.source_file for i in inputs if i.document_origin == origin})
            for origin in ("sample", "upload")
        }
        return {
            "status": "ready" if inputs else "not_initialized",
            "total_chunks": len(inputs),
            "total_documents": len({i.document_id for i in inputs}),
            "document_names": sorted({i.source_file for i in inputs}),
            "indexed_sample_files": by_origin["sample"],
            "indexed_upload_files": by_origin["upload"],
            "indexed_sample_document_count": len(by_origin["sample"]),
            "indexed_upload_document_count": len(by_origin["upload"]),
            "pending_upload_files": [],
            "scope_chunk_counts": {
                "all": len(inputs),
                **{
                    origin: sum(i.document_origin == origin for i in inputs) for origin in by_origin
                },
            },
        }

    def start_generation(self):
        import agent
        import config

        self.llm = agent.get_llm()
        self.generation_metadata = {
            "class": type(self.llm).__name__,
            "model": getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None),
            "temperature": config.GENERATION_TEMPERATURE,
            "control_temperature": config.GRADING_TEMPERATURE,
            "retrieval_k": config.RETRIEVAL_K,
            "min_relevance_score": config.MIN_RELEVANCE_SCORE,
            "max_query_retries": config.MAX_QUERY_RETRIES,
            "max_prompt_tokens": config.MAX_PROMPT_TOKENS,
            "max_context_chars_per_chunk": config.MAX_CONTEXT_CHARS_PER_CHUNK,
            "embedding_model": config.EMBEDDING_MODEL,
        }

    def load_current(self):
        from rfp_analyst.retrieval.hybrid import load_indexed_snapshot

        snapshot = load_indexed_snapshot(self.index, self.collection)
        if snapshot.fingerprint != self.corpus_hash:
            raise ValueError("Frozen evaluation corpus changed")
        return snapshot

    def vector_search(self, query, k, scope):
        from rag_engine import _filter_results_for_scope
        from rfp_analyst.retrieval.vector_store import get_embeddings

        if self.manager is None:
            from chromadb import PersistentClient
            from chromadb.config import Settings
            from langchain_chroma import Chroma

            # Snapshot reading and semantic search must share identical client settings.
            # No create/upsert: this is an already-validated, existing collection.
            client = PersistentClient(
                path=str(self.index), settings=Settings(anonymized_telemetry=False)
            )
            self.manager = Chroma(
                client=client,
                collection_name=self.collection,
                embedding_function=get_embeddings(),
                create_collection_if_not_exists=False,
            )
        kwargs = {"k": k}
        if scope in {"sample", "upload"}:
            kwargs["filter"] = {"document_origin": scope}
        return _filter_results_for_scope(
            self.manager.similarity_search_with_relevance_scores(query, **kwargs), scope
        )

    def execute(self, case, mode):
        from rfp_analyst.agent.graph import prepare_query_payload
        from rfp_analyst.graph.reader import create_graph_reader
        from rfp_analyst.retrieval.hybrid import HybridRetrievalProvider

        self.load_current()
        provider = None
        try:
            if mode != "vector_only":
                provider = HybridRetrievalProvider(
                    create_graph_reader(),
                    self.load_current,
                    corpus_id=self.corpus_id,
                    policy=mode,
                    strict=True,
                    owns_reader=True,
                )
            # Exactly one complete graph execution: captures its final repaired answer.
            payload = prepare_query_payload(
                case["question"],
                chat_history=case.get("chat_history", []),
                vectorstore_stats=self.stats,
                retrieval_fn=self.vector_search,
                retrieval_scope=case.get("retrieval_scope", "all"),
                llm=self.llm,
                retrieval_mode=mode,
                retrieval_provider=provider,
                strict_retrieval_mode=True,
                allow_web_search=False,
            )
            self.load_current()
            if provider:
                payload["evaluation_retrieval_events"] = [
                    {"mode": e.effective_mode, "fallback_reason": e.fallback_reason}
                    for e in provider.events
                ]
            return payload
        finally:
            if provider:
                provider.close()


def prepare_sample_workspace(directory: Path):
    """Explicit preparation only, never rebuilding a corpus between retrieval modes."""
    from evals.run_kb_evals import ensure_sample_documents_ready

    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("Sample workspace must be new; reuse its index on subsequent runs")
    directory.mkdir(parents=True)
    ensure_sample_documents_ready(directory / "vectorstore", directory / "uploads")
    return directory / "vectorstore"
