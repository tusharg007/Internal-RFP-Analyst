from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def ready_stats():
    return {
        "status": "ready",
        "total_documents": 1,
        "total_chunks": 2,
        "document_names": ["sample.pdf"],
        "available_documents": ["sample.pdf"],
        "indexed_sample_document_count": 1,
        "indexed_upload_document_count": 0,
        "pending_upload_files": [],
        "scope_chunk_counts": {"sample": 2, "upload": 0, "all": 2},
    }


def not_ready_stats():
    return {
        "status": "not_initialized",
        "total_documents": 0,
        "total_chunks": 0,
        "document_names": [],
        "available_documents": [],
        "indexed_sample_document_count": 0,
        "indexed_upload_document_count": 0,
        "pending_upload_files": [],
        "scope_chunk_counts": {"sample": 0, "upload": 0, "all": 0},
    }


def test_app_smoke_has_no_unhandled_streamlit_exception():
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    with (
        patch("config.get_api_keys", return_value=("", "")),
        patch("rag_engine.get_vectorstore_stats", return_value=ready_stats()),
        patch("rag_engine.ingest_documents"),
        patch("document_generator.generate_all_documents"),
        patch("agent.create_agent"),
        patch("agent.query_agent_stream", return_value=iter([])),
    ):
        at = AppTest.from_file(str(app_path)).run(timeout=10)

    assert not at.exception


def test_public_walkthrough_disables_document_mutations():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    with (
        patch("config.PUBLIC_DEMO_READ_ONLY", True, create=True),
        patch("config.get_api_keys", return_value=("", "")),
        patch("rag_engine.get_vectorstore_stats", return_value=ready_stats()),
        patch("rag_engine.ingest_documents") as ingest,
        patch("document_generator.generate_all_documents") as generate,
    ):
        at = AppTest.from_file(str(app_path)).run(timeout=10)
    assert not at.exception
    for button in at.button:
        if button.label in {"Generate Sample PDFs", "Ingest Documents"}:
            assert button.disabled
    assert not ingest.called and not generate.called
    assert any("Public walkthrough" in str(item.value) for item in at.info)


def test_app_does_not_auto_ingest_when_kb_is_missing():
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    with (
        patch("config.get_api_keys", return_value=("", "")),
        patch("rag_engine.get_vectorstore_stats", return_value=not_ready_stats()),
        patch("rag_engine.ingest_documents") as ingest_documents,
        patch("document_generator.generate_all_documents") as generate_documents,
    ):
        at = AppTest.from_file(str(app_path))
        at.run(timeout=10)
        at.run(timeout=10)

    assert not at.exception
    assert ingest_documents.call_count == 0
    assert generate_documents.call_count == 0
    assert any("Knowledge base is not ready" in str(item.value) for item in at.info)


def test_failed_ingestion_does_not_retry_on_next_rerun():
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    with (
        patch("config.get_api_keys", return_value=("groq", "")),
        patch("rag_engine.get_vectorstore_stats", return_value=not_ready_stats()),
        patch("rag_engine.ingest_documents", side_effect=RuntimeError("boom")) as ingest_documents,
    ):
        at = AppTest.from_file(str(app_path)).run(timeout=10)
        at.button[7].click().run(timeout=10)
        at.run(timeout=10)

    assert ingest_documents.call_count == 1
    assert at.session_state["ingestion_in_progress"] is False
    assert "Document ingestion failed" in at.session_state["last_ingestion_error"]
    assert "boom" not in at.session_state["last_ingestion_error"]
