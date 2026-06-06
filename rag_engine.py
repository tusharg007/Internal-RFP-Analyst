"""
RAG Engine — Document Ingestion, Embedding & Retrieval Pipeline.
Uses PyMuPDF for PDF loading, ChromaDB for vector storage, and Google Gemini embeddings.
Includes rate-limit handling with exponential backoff for Gemini free-tier quotas.
"""

import os
import time
import logging
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    DATA_DIR,
    VECTORSTORE_DIR,
    RETRIEVAL_K,
)

logger = logging.getLogger(__name__)

# ─── Rate-Limit Constants ─────────────────────────────────────────────────────
MAX_RETRIES = 5              # max retry attempts on rate-limit errors
INITIAL_BACKOFF = 10         # initial backoff in seconds
BACKOFF_MULTIPLIER = 2       # exponential multiplier
INGEST_BATCH_SIZE = 20       # chunks per batch during ingestion
INTER_BATCH_DELAY = 5        # seconds to wait between batches


def _retry_on_rate_limit(func, *args, **kwargs):
    """Execute a function with exponential backoff on 429 / RESOURCE_EXHAUSTED errors."""
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = (
                "429" in error_str
                or "resource_exhausted" in error_str
                or "quota" in error_str
            )
            if is_rate_limit and attempt < MAX_RETRIES:
                wait = backoff + (attempt * 2)  # add jitter
                logger.warning(
                    f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {wait}s..."
                )
                print(f"⏳ Rate limit reached — waiting {wait}s before retry ({attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
                backoff *= BACKOFF_MULTIPLIER
            else:
                raise


class RateLimitedEmbeddings(GoogleGenerativeAIEmbeddings):
    """Wrapper around GoogleGenerativeAIEmbeddings with automatic rate-limit retry."""

    def embed_documents(self, texts, **kwargs):
        return _retry_on_rate_limit(super().embed_documents, texts, **kwargs)

    def embed_query(self, text, **kwargs):
        return _retry_on_rate_limit(super().embed_query, text, **kwargs)


def get_embeddings():
    """Initialize Google Gemini embedding model with rate-limit protection."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not set. "
            "Get a free key at https://aistudio.google.com/apikey "
            "and add it to your .env file."
        )
    return RateLimitedEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def load_pdfs(doc_dir: Path = DATA_DIR):
    """Load all PDFs from the given directory using PyMuPDF."""
    pdf_files = sorted(doc_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {doc_dir}")

    all_docs = []
    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        # Enrich metadata
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name
            doc.metadata["source_path"] = str(pdf_path)
        all_docs.extend(docs)
        print(f"  Loaded: {pdf_path.name} ({len(docs)} pages)")

    print(f"Total pages loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(documents):
    """Split documents into chunks with overlap for context continuity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def create_vectorstore(chunks, persist_dir: Path = VECTORSTORE_DIR):
    """Embed chunks and store in ChromaDB with persistence.
    
    Processes chunks in small batches with delays to avoid exceeding
    the Gemini free-tier embedding quota (100 requests/minute).
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()

    total = len(chunks)
    batch_size = INGEST_BATCH_SIZE
    vectorstore = None

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vectorstore is None:
            # First batch — create the vector store
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(persist_dir),
            )
        else:
            # Subsequent batches — add to existing store
            vectorstore.add_documents(batch)

        # Delay between batches to stay under rate limits
        if i + batch_size < total:
            print(f"  ⏳ Waiting {INTER_BATCH_DELAY}s to respect rate limits...")
            time.sleep(INTER_BATCH_DELAY)

    count = vectorstore._collection.count() if vectorstore else 0
    print(f"Vector store created with {count} vectors")
    print(f"Persisted to: {persist_dir}")
    return vectorstore


def load_vectorstore(persist_dir: Path = VECTORSTORE_DIR):
    """Load an existing ChromaDB vector store from disk."""
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. Run ingestion first."
        )
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    count = vectorstore._collection.count()
    print(f"Loaded vector store with {count} vectors")
    return vectorstore


def get_retriever(k: int = RETRIEVAL_K):
    """Get a LangChain retriever from the persisted vector store."""
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def similarity_search(query: str, k: int = RETRIEVAL_K):
    """Direct similarity search returning documents with scores."""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return results


def get_vectorstore_stats():
    """Get statistics about the current vector store."""
    try:
        vectorstore = load_vectorstore()
        count = vectorstore._collection.count()
        # Get unique sources
        all_metadata = vectorstore._collection.get()["metadatas"]
        sources = set()
        for m in all_metadata:
            if "source_file" in m:
                sources.add(m["source_file"])
        return {
            "total_chunks": count,
            "total_documents": len(sources),
            "document_names": sorted(sources),
            "status": "ready",
        }
    except Exception:
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "document_names": [],
            "status": "not_initialized",
        }


def ingest_documents(doc_dir: Path = DATA_DIR):
    """Full ingestion pipeline: load → chunk → embed → store."""
    print("=" * 60)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    print("\n[1/3] Loading PDFs...")
    documents = load_pdfs(doc_dir)

    print("\n[2/3] Chunking documents...")
    chunks = chunk_documents(documents)

    print("\n[3/3] Embedding & storing in ChromaDB...")
    vectorstore = create_vectorstore(chunks)

    stats = get_vectorstore_stats()
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Chunks:    {stats['total_chunks']}")
    print("=" * 60)
    return vectorstore


if __name__ == "__main__":
    ingest_documents()
