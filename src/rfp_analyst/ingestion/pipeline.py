"""High-level ingestion pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from config import COLLECTION_NAME, DATA_DIR, VECTORSTORE_DIR
from rfp_analyst.ingestion.chunking import chunk_loaded_sources
from rfp_analyst.ingestion.loaders import load_pdf_sources
from rfp_analyst.ingestion.registry import IngestionRegistry
from rfp_analyst.retrieval.vector_store import VectorStoreManager
from rfp_analyst.schemas import IngestionRecord


class IngestionPipeline:
    """Production-grade ingestion pipeline with duplicate prevention."""

    def __init__(
        self,
        doc_dir: Path = DATA_DIR,
        persist_dir: Path = VECTORSTORE_DIR,
        collection_name: str = COLLECTION_NAME,
        registry: IngestionRegistry | None = None,
        vector_store_manager: VectorStoreManager | None = None,
    ):
        self.doc_dir = Path(doc_dir)
        self.persist_dir = Path(persist_dir)
        self.registry = registry or IngestionRegistry(self.persist_dir / "ingestion_registry.json")
        self.vector_store_manager = vector_store_manager or VectorStoreManager(
            persist_dir=self.persist_dir,
            collection_name=collection_name,
        )

    def run(self):
        """Load, dedupe, chunk, and persist documents."""
        print("=" * 60)
        print("DOCUMENT INGESTION PIPELINE")
        print("=" * 60)

        print("\n[1/3] Loading PDFs...")
        loaded_sources = load_pdf_sources(self.doc_dir)

        new_sources = [
            source
            for source in loaded_sources
            if not self.registry.contains_hash(source.file_hash)
        ]

        skipped_count = len(loaded_sources) - len(new_sources)
        if skipped_count:
            print(f"Skipped {skipped_count} previously ingested file(s)")

        print("\n[2/3] Chunking documents...")
        chunks = chunk_loaded_sources(new_sources) if new_sources else []

        print("\n[3/3] Embedding & storing in ChromaDB (local embeddings)...")
        vectorstore = self.vector_store_manager.upsert_documents(chunks)

        if new_sources:
            chunk_ids_by_hash: dict[str, list[str]] = {}
            for chunk in chunks:
                chunk_ids_by_hash.setdefault(chunk.metadata["file_hash"], []).append(
                    chunk.metadata["chunk_id"]
                )

            for source in new_sources:
                self.registry.register(
                    IngestionRecord(
                        source_file=source.source_file,
                        source_path=source.source_path,
                        file_hash=source.file_hash,
                        page_count=source.page_count,
                        document_type=source.document_type,
                        chunk_ids=chunk_ids_by_hash.get(source.file_hash, []),
                    )
                )

        stats = self.vector_store_manager.get_stats()
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print(f"  Documents: {stats['total_documents']}")
        print(f"  Chunks:    {stats['total_chunks']}")
        print("=" * 60)
        return vectorstore
