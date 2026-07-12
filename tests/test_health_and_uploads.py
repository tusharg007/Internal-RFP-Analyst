from pathlib import Path

import pytest

import rag_engine
from rfp_analyst.exceptions import KnowledgeBaseNotReadyError, UnsupportedFileError
from rfp_analyst.health import get_app_health
from rfp_analyst.uploads import sanitize_uploaded_filename, validate_uploaded_pdf


class FakeUpload:
    def __init__(self, name: str, mime_type: str = "application/pdf", size: int = 128):
        self.name = name
        self.type = mime_type
        self.size = size

    def getbuffer(self):
        return b"x" * self.size


def test_missing_vectorstore_is_graceful(tmp_path: Path):
    missing_dir = tmp_path / "missing-vectorstore"

    with pytest.raises(KnowledgeBaseNotReadyError):
        rag_engine.load_vectorstore(missing_dir)

    stats = rag_engine.get_vectorstore_stats()
    assert stats["status"] in {"ready", "not_initialized", "error"}


def test_invalid_upload_filename_sanitization():
    assert sanitize_uploaded_filename("../../bad name!!.pdf") == "bad_name.pdf"
    with pytest.raises(UnsupportedFileError):
        sanitize_uploaded_filename("malware.exe")


def test_invalid_upload_mime_rejected():
    with pytest.raises(UnsupportedFileError):
        validate_uploaded_pdf(FakeUpload("notes.pdf", mime_type="text/plain"))


def test_app_health_check_function(tmp_path: Path):
    data_dir = tmp_path / "documents"
    vectorstore_dir = tmp_path / "vectorstore"
    assets_dir = tmp_path / "assets"
    data_dir.mkdir()
    assets_dir.mkdir()
    (data_dir / "sample.pdf").write_bytes(b"pdf")

    health = get_app_health(
        vectorstore_stats={
            "status": "not_initialized",
            "total_documents": 0,
            "total_chunks": 0,
            "document_names": [],
        },
        data_dir=data_dir,
        vectorstore_dir=vectorstore_dir,
        assets_dir=assets_dir,
        uploads_dir=tmp_path / "uploads",
        groq_api_key="",
        google_api_key="",
    )

    assert health["vectorstore_ready"] is False
    assert health["document_count"] == 1
    assert health["chunk_count"] == 0
    assert health["llm_provider_configured"] is False
    assert health["required_directories"]["documents"] is True
    assert health["required_directories"]["vectorstore"] is False


def test_vectorstore_is_not_ready_when_chunk_count_is_zero():
    health = get_app_health(
        vectorstore_stats={
            "status": "ready",
            "total_documents": 4,
            "total_chunks": 0,
            "document_names": ["a.pdf"],
        },
        groq_api_key="",
        google_api_key="",
    )

    assert health["vectorstore_ready"] is False
