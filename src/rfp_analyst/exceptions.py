"""Custom exceptions for the Internal RFP Analyst app."""


class RFPAnalystError(Exception):
    """Base exception for app-specific failures."""


class LLMProviderNotConfiguredError(RFPAnalystError):
    """Raised when no supported LLM provider is configured."""


class KnowledgeBaseNotReadyError(RFPAnalystError):
    """Raised when retrieval is attempted before the vector store is ready."""


class NoDocumentsFoundError(RFPAnalystError):
    """Raised when ingestion is attempted without any source PDFs."""


class IngestionError(RFPAnalystError):
    """Raised when document ingestion fails."""


class RetrievalError(RFPAnalystError):
    """Raised when knowledge base retrieval fails."""


class UnsupportedFileError(RFPAnalystError):
    """Raised when an uploaded file is invalid or unsupported."""
