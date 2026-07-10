"""Project-specific exceptions."""

from __future__ import annotations


class RFPAnalystError(Exception):
    """Base application exception."""


class LLMProviderNotConfiguredError(RFPAnalystError):
    """Raised when no LLM provider credentials are configured."""


class KnowledgeBaseNotReadyError(RFPAnalystError):
    """Raised when retrieval is attempted before the knowledge base is ready."""


class NoDocumentsFoundError(RFPAnalystError):
    """Raised when no source documents are available for ingestion."""


class IngestionError(RFPAnalystError):
    """Raised when document ingestion fails."""


class RetrievalError(RFPAnalystError):
    """Raised when retrieval fails."""


class UnsupportedFileError(RFPAnalystError):
    """Raised when an uploaded file is invalid or unsupported."""
