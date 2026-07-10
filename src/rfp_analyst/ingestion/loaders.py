"""Validated document loading utilities."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz
from langchain_community.document_loaders import PyMuPDFLoader

from config import DATA_DIR, MAX_UPLOAD_FILE_SIZE_BYTES, MAX_UPLOAD_PAGE_COUNT
from rfp_analyst.exceptions import IngestionError, NoDocumentsFoundError
from rfp_analyst.schemas import LoadedSource

SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    """Normalize uploaded filenames to a safe PDF filename."""
    original = Path(filename)
    stem = SAFE_FILENAME_PATTERN.sub("_", original.stem).strip("._") or "document"
    suffix = original.suffix.lower() if original.suffix else ".pdf"
    if suffix != ".pdf":
        suffix = ".pdf"
    return f"{stem}{suffix}"


def ensure_safe_pdf_path(pdf_path: Path) -> Path:
    """Rename files with unsafe names before ingestion."""
    safe_name = sanitize_filename(pdf_path.name)
    target_path = pdf_path.with_name(safe_name)

    if target_path == pdf_path:
        return pdf_path

    if target_path.exists():
        short_hash = hashlib.sha256(pdf_path.name.encode("utf-8")).hexdigest()[:8]
        target_path = pdf_path.with_name(f"{Path(safe_name).stem}_{short_hash}.pdf")

    pdf_path.rename(target_path)
    return target_path


def sha256_file(file_path: Path) -> str:
    """Return the SHA256 hash for the file contents."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_document_type(pdf_path: Path) -> str | None:
    """Infer document type from the filename when possible."""
    normalized = pdf_path.stem.replace("_", " ").lower()
    candidates = ("proposal", "project outline", "case study", "rfp response")
    for candidate in candidates:
        if candidate in normalized:
            return candidate.title()
    return None


def validate_pdf(pdf_path: Path) -> int:
    """Validate file size and page count before loading."""
    file_size = pdf_path.stat().st_size
    if file_size > MAX_UPLOAD_FILE_SIZE_BYTES:
        raise IngestionError(
            f"{pdf_path.name} exceeds max file size of {MAX_UPLOAD_FILE_SIZE_BYTES} bytes"
        )

    with fitz.open(pdf_path) as pdf_document:
        page_count = pdf_document.page_count

    if page_count > MAX_UPLOAD_PAGE_COUNT:
        raise IngestionError(
            f"{pdf_path.name} exceeds max page count of {MAX_UPLOAD_PAGE_COUNT}"
        )
    return page_count


def load_pdf_sources(doc_dir: Path = DATA_DIR) -> list[LoadedSource]:
    """Load, validate, and enrich all PDFs in a directory."""
    pdf_files = sorted(doc_dir.glob("*.pdf"))
    if not pdf_files:
        raise NoDocumentsFoundError(f"No PDF files found in {doc_dir}")

    loaded_sources: list[LoadedSource] = []
    try:
        for raw_pdf_path in pdf_files:
            pdf_path = ensure_safe_pdf_path(raw_pdf_path)
            page_count = validate_pdf(pdf_path)
            file_hash = sha256_file(pdf_path)
            document_type = infer_document_type(pdf_path)

            loader = PyMuPDFLoader(str(pdf_path))
            documents = loader.load()

            for page_index, document in enumerate(documents):
                document.metadata["source_file"] = pdf_path.name
                document.metadata["source_path"] = str(pdf_path.resolve())
                document.metadata["file_hash"] = file_hash
                document.metadata["page"] = int(document.metadata.get("page", page_index))
                document.metadata["document_type"] = document_type

            loaded_sources.append(
                LoadedSource(
                    source_file=pdf_path.name,
                    source_path=str(pdf_path.resolve()),
                    file_hash=file_hash,
                    page_count=page_count,
                    document_type=document_type,
                    documents=documents,
                )
            )
            print(f"  Loaded: {pdf_path.name} ({page_count} pages)")
    except NoDocumentsFoundError:
        raise
    except Exception as error:
        raise IngestionError(str(error)) from error

    print(f"Total pages loaded: {sum(source.page_count for source in loaded_sources)}")
    return loaded_sources
