"""Upload validation helpers for user-provided PDFs."""

from __future__ import annotations

import re
from pathlib import Path

from config import MAX_UPLOAD_SIZE_MB, UPLOADS_DIR
from .exceptions import UnsupportedFileError

MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_MIME_TYPES = {"application/pdf", "application/x-pdf"}


def sanitize_uploaded_filename(filename: str) -> str:
    """Normalize uploaded filenames and strip unsafe path segments."""
    raw_name = Path(filename or "").name.strip()
    if not raw_name:
        raise UnsupportedFileError("Uploaded file must have a filename.")

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(raw_name).stem).strip("._")
    suffix = Path(raw_name).suffix.lower()
    if not stem:
        stem = "document"
    if suffix != ".pdf":
        raise UnsupportedFileError("Only PDF files are supported.")
    return f"{stem}{suffix}"


def validate_uploaded_pdf(uploaded_file) -> str:
    """Validate a Streamlit uploaded file and return a safe filename."""
    sanitized_name = sanitize_uploaded_filename(getattr(uploaded_file, "name", ""))
    mime_type = getattr(uploaded_file, "type", "")
    if mime_type and mime_type not in ALLOWED_MIME_TYPES:
        raise UnsupportedFileError("Uploaded file must be a PDF.")

    file_size = getattr(uploaded_file, "size", None)
    if file_size is None:
        file_size = len(uploaded_file.getbuffer())
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise UnsupportedFileError(
            f"Uploaded file exceeds the {MAX_UPLOAD_SIZE_MB} MB limit."
        )

    return sanitized_name


def persist_uploaded_pdf(uploaded_file, uploads_dir: Path = UPLOADS_DIR) -> Path:
    """Persist a validated uploaded file into the dedicated uploads directory."""
    sanitized_name = validate_uploaded_pdf(uploaded_file)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target_path = uploads_dir / sanitized_name
    with target_path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return target_path


def is_uploaded_pdf_unchanged(uploaded_file, uploads_dir: Path = UPLOADS_DIR) -> bool:
    """Return whether the uploader value already exists byte-for-byte on disk."""

    sanitized_name = validate_uploaded_pdf(uploaded_file)
    target_path = uploads_dir / sanitized_name
    return target_path.exists() and target_path.read_bytes() == bytes(uploaded_file.getbuffer())
