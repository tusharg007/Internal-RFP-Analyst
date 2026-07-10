"""App health checks."""

from __future__ import annotations

from pathlib import Path

from config import ASSETS_DIR, DATA_DIR, VECTORSTORE_DIR
from rfp_analyst.retrieval.vector_store import VectorStoreManager


REQUIRED_DIRECTORIES = {
    "data_dir": DATA_DIR,
    "vectorstore_dir": VECTORSTORE_DIR,
    "assets_dir": ASSETS_DIR,
}


def get_app_health(llm_provider_name: str = "Not configured") -> dict:
    """Return high-level application health without raising UI-breaking errors."""
    directories = {
        name: {"path": str(path), "exists": Path(path).exists()}
        for name, path in REQUIRED_DIRECTORIES.items()
    }

    stats = VectorStoreManager(persist_dir=VECTORSTORE_DIR).get_stats()
    return {
        "vectorstore_ready": stats.get("status") == "ready",
        "document_count": stats.get("total_documents", 0),
        "chunk_count": stats.get("total_chunks", 0),
        "llm_provider_configured": llm_provider_name != "Not configured",
        "llm_provider_name": llm_provider_name,
        "required_directories": directories,
    }
