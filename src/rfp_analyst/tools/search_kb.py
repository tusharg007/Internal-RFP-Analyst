"""Knowledge-base search tool."""

from __future__ import annotations

from typing import Callable

from config import RETRIEVAL_K
from rfp_analyst.retrieval.vector_store import VectorStoreManager


SearchFn = Callable[[str, int], list[tuple[object, float]]]


def _default_search(query: str, k: int = RETRIEVAL_K):
    return VectorStoreManager().similarity_search(query, k=k)


def search_knowledge_base(query: str, k: int = RETRIEVAL_K, search_fn: SearchFn | None = None) -> dict:
    """Retrieve relevant chunks and normalize them for downstream tools."""
    search = search_fn or _default_search
    try:
        results = search(query, k=k)
    except Exception:
        return {
            "query": query,
            "documents": [],
            "sources": [],
            "context_parts": [],
            "context": "No relevant documents found.",
        }

    context_parts = []
    sources = []
    documents = []
    for doc, score in results:
        source = doc.metadata.get("source_file", "Unknown")
        page = int(doc.metadata.get("page", 0))
        snippet = doc.page_content.strip()
        context_parts.append(f"[Source: {source}, Page {page + 1}]\n{snippet}")
        sources.append(
            {
                "source": source,
                "page": page,
                "score": f"{score:.2f}" if score is not None else "n/a (graph witness)",
                "snippet": snippet[:220],
            }
        )
        documents.append(doc)

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."
    return {
        "query": query,
        "documents": documents,
        "sources": sources,
        "context_parts": context_parts,
        "context": context,
    }
