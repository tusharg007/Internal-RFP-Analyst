from pathlib import Path

import pytest

from rfp_analyst.exceptions import KnowledgeBaseNotReadyError, UnsupportedFileError
from rfp_analyst.health import get_app_health
from rfp_analyst.retrieval.vector_store import VectorStoreManager
from rfp_analyst.ui.helpers import format_latency_display, get_chat_avatar
from rfp_analyst.uploads import validate_uploaded_pdf


class FakeUpload:
    def __init__(self, name: str, file_type: str = "application/pdf", size: int = 10):
        self.name = name
        self.type = file_type
        self.size = size


class FakeVectorStoreManager:
    def __init__(self, *_args, **_kwargs):
        pass

    def get_stats(self):
        return {
            "status": "ready",
            "total_documents": 3,
            "total_chunks": 9,
            "document_names": ["a.pdf"],
        }


def test_invalid_avatar_helper_values():
    assert get_chat_avatar("user") == "👤"
    assert get_chat_avatar("assistant") == "🤖"
    assert get_chat_avatar("user") not in {"User", "AI"}
    assert get_chat_avatar("assistant") not in {"User", "AI"}


def test_missing_vectorstore_raises_graceful_error(tmp_path):
    manager = VectorStoreManager(persist_dir=tmp_path / "missing_vectorstore")
    with pytest.raises(KnowledgeBaseNotReadyError):
        manager.load(create_if_missing=False)


def test_invalid_upload_filename_sanitization():
    safe_name = validate_uploaded_pdf(FakeUpload("../My bad file!!.pdf"))
    assert safe_name == "My_bad_file.pdf"


def test_invalid_upload_extension_raises():
    with pytest.raises(UnsupportedFileError):
        validate_uploaded_pdf(FakeUpload("report.txt"))


def test_invalid_upload_mime_type_raises():
    with pytest.raises(UnsupportedFileError):
        validate_uploaded_pdf(FakeUpload("report.pdf", file_type="text/plain"))


def test_eval_latency_formatting():
    assert format_latency_display({"average_latency": 0.23}) == "0.230 s"
    assert format_latency_display({"average_latency_ms": 12.5}) == "12.50 ms"


def test_app_health_check(monkeypatch, tmp_path):
    monkeypatch.setattr("rfp_analyst.health.REQUIRED_DIRECTORIES", {
        "data_dir": tmp_path / "data",
        "vectorstore_dir": tmp_path / "vectorstore",
        "assets_dir": tmp_path / "assets",
    })
    for path in (tmp_path / "data", tmp_path / "vectorstore", tmp_path / "assets"):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("rfp_analyst.health.VectorStoreManager", FakeVectorStoreManager)

    health = get_app_health("Not configured")

    assert health["vectorstore_ready"] is True
    assert health["document_count"] == 3
    assert health["chunk_count"] == 9
    assert health["llm_provider_configured"] is False
    assert all(entry["exists"] for entry in health["required_directories"].values())
