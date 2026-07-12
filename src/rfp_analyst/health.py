"""Health checks for app readiness and configuration."""

from pathlib import Path

from config import DATA_DIR, UPLOADS_DIR, VECTORSTORE_DIR
from rfp_analyst.retrieval.vector_store import VectorStoreManager

REQUIRED_DIRECTORIES = {
    "data_dir": DATA_DIR,
    "vectorstore_dir": VECTORSTORE_DIR,
}


def get_provider_status(groq_api_key: str = "", google_api_key: str = "") -> dict:
    """Return provider readiness and display information."""
    if groq_api_key:
        return {"configured": True, "provider": "Groq"}
    if google_api_key:
        return {"configured": True, "provider": "Gemini"}
    return {"configured": False, "provider": "Not configured"}


def _legacy_health_snapshot(provider_name: str) -> dict:
    stats = VectorStoreManager().get_stats()
    required_directories = {
        name: {"path": str(path), "exists": Path(path).exists()}
        for name, path in REQUIRED_DIRECTORIES.items()
    }
    chunk_count = int(stats.get("total_chunks", 0) or 0)
    document_count = int(stats.get("total_documents", 0) or 0)
    return {
        "vectorstore_ready": stats.get("status") == "ready" and chunk_count > 0,
        "document_count": document_count,
        "indexed_document_count": document_count,
        "chunk_count": chunk_count,
        "llm_provider_configured": provider_name != "Not configured",
        "provider_name": provider_name,
        "required_directories": required_directories,
        "missing_directories": [
            name for name, entry in required_directories.items() if not entry["exists"]
        ],
        "vectorstore_status": stats.get("status", "not_initialized"),
        "document_names": stats.get("document_names", []),
    }


def get_app_health(
    vectorstore_stats: dict | str | None = None,
    data_dir: Path | None = None,
    vectorstore_dir: Path | None = None,
    assets_dir: Path | None = None,
    uploads_dir: Path | None = None,
    groq_api_key: str = "",
    google_api_key: str = "",
) -> dict:
    """Build a central snapshot of app health for the UI and query flow."""
    if isinstance(vectorstore_stats, str):
        return _legacy_health_snapshot(vectorstore_stats)

    stats = vectorstore_stats or {}
    provider_status = get_provider_status(groq_api_key, google_api_key)
    data_path = data_dir or REQUIRED_DIRECTORIES["data_dir"]
    vectorstore_path = vectorstore_dir or REQUIRED_DIRECTORIES["vectorstore_dir"]
    uploads_path = uploads_dir or UPLOADS_DIR
    sample_files = sorted(data_path.glob("*.pdf")) if data_path.exists() else []
    upload_files = sorted(uploads_path.glob("*.pdf")) if uploads_path.exists() else []
    required_directories = {
        "documents": data_path.exists(),
        "vectorstore": vectorstore_path.exists(),
    }
    chunk_count = int(stats.get("total_chunks", 0) or 0)
    indexed_document_count = int(stats.get("total_documents", 0) or 0)
    vectorstore_ready = stats.get("status") == "ready" and chunk_count > 0

    return {
        "vectorstore_ready": vectorstore_ready,
        "document_count": max(indexed_document_count, len(sample_files) + len(upload_files)),
        "indexed_document_count": indexed_document_count,
        "chunk_count": chunk_count,
        "llm_provider_configured": provider_status["configured"],
        "provider_name": provider_status["provider"],
        "required_directories": required_directories,
        "missing_directories": [
            name for name, exists in required_directories.items() if not exists
        ],
        "vectorstore_status": stats.get("status", "not_initialized"),
        "document_names": stats.get("document_names", []),
        "sample_document_count": len(sample_files),
        "upload_document_count": len(upload_files),
        "indexed_sample_document_count": int(stats.get("indexed_sample_document_count", 0) or 0),
        "indexed_upload_document_count": int(stats.get("indexed_upload_document_count", 0) or 0),
        "pending_upload_files": stats.get("pending_upload_files", []),
        "pending_upload_count": len(stats.get("pending_upload_files", [])),
    }
