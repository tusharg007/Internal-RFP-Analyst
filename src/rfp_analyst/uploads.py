"""Upload validation helpers."""

from __future__ import annotations

from pathlib import Path

from config import MAX_UPLOAD_FILE_SIZE_BYTES
from rfp_analyst.exceptions import UnsupportedFileError
from rfp_analyst.ingestion.loaders import sanitize_filename

VALID_PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}


def validate_uploaded_pdf(uploaded_file) -> str:
    """Validate upload metadata and return a sanitized filename."""
    original_suffix = Path(uploaded_file.name).suffix.lower()
    if original_suffix != ".pdf":
        raise UnsupportedFileError("Only PDF files are supported.")

    sanitized_name = sanitize_filename(uploaded_file.name)
    file_type = getattr(uploaded_file, "type", "") or ""
    if file_type and file_type not in VALID_PDF_MIME_TYPES:
        raise UnsupportedFileError("The uploaded file does not appear to be a valid PDF.")

    file_size = getattr(uploaded_file, "size", None)
    if file_size is not None and int(file_size) > MAX_UPLOAD_FILE_SIZE_BYTES:
        raise UnsupportedFileError(
            f"Uploaded PDF exceeds the max size of {MAX_UPLOAD_FILE_SIZE_BYTES} bytes."
        )

    return sanitized_name
