"""Central configuration for the Internal RFP Analyst app."""

import os
import sys
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"


def _is_pytest_runtime() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def _allow_env_file_values() -> bool:
    if os.getenv("RFP_ANALYST_ENABLE_DOTENV_IN_TESTS", "").lower() in {"1", "true", "yes"}:
        return True
    return not _is_pytest_runtime()


if _allow_env_file_values():
    load_dotenv(ENV_PATH)

DATA_DIR = BASE_DIR / "data" / "documents"
SAMPLE_DOCS_DIR = DATA_DIR
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
ASSETS_DIR = BASE_DIR / "assets"
EVALS_DIR = BASE_DIR / "evals"
OFFLINE_SMOKE_EVAL_RESULTS_PATH = EVALS_DIR / "offline_smoke_results.json"
REAL_KB_EVAL_RESULTS_PATH = EVALS_DIR / "real_kb_results.json"
EVAL_RESULTS_PATH = OFFLINE_SMOKE_EVAL_RESULTS_PATH


def _get_streamlit_secret(name: str) -> str:
    try:
        import streamlit as st

        return str(st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""


def _get_env_file_value(name: str) -> str:
    if not _allow_env_file_values():
        return ""
    values = dotenv_values(ENV_PATH)
    return str(values.get(name, "") or "").strip()


def _resolve_key(name: str) -> str:
    """Resolve an API key from Streamlit secrets, the environment, or ``.env``."""
    return (
        _get_streamlit_secret(name)
        or os.getenv(name, "")
        or _get_env_file_value(name)
    ).strip()


def _get_int_setting(name: str, default: int) -> int:
    raw_value = os.getenv(name, "") or _get_env_file_value(name)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _get_bool_setting(name: str, default: bool = False) -> bool:
    raw_value = _resolve_key(name).lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return default


def _get_float_setting(name: str, default: float) -> float:
    try:
        return float(_resolve_key(name) or default)
    except ValueError:
        return default


def get_neo4j_settings(role: str = "reader") -> dict:
    """Resolve optional graph settings without constructing a connection.

    Writer/admin credentials are only resolved when explicitly requested by a
    worker/migration caller. Neither role falls back to runtime reader secrets.
    """
    credential_prefixes = {
        "reader": "NEO4J",
        "writer": "NEO4J_INGEST",
        "admin": "NEO4J_ADMIN",
    }
    if role not in credential_prefixes:
        raise ValueError("Unsupported graph access role")
    enabled = _get_bool_setting("NEO4J_ENABLED")
    prefix = credential_prefixes[role]
    return {
        "enabled": enabled,
        "uri": _resolve_key("NEO4J_URI") if enabled else "",
        "username": _resolve_key(f"{prefix}_USERNAME") if enabled else "",
        "password": _resolve_key(f"{prefix}_PASSWORD") if enabled else "",
        "database": _resolve_key("NEO4J_DATABASE") or "neo4j",
        "role": role,
        "connection_timeout_seconds": _get_float_setting("NEO4J_CONNECTION_TIMEOUT_SECONDS", 1.0),
        "acquisition_timeout_seconds": _get_float_setting("NEO4J_ACQUISITION_TIMEOUT_SECONDS", 2.0),
        "transaction_timeout_seconds": _get_float_setting("NEO4J_TRANSACTION_TIMEOUT_SECONDS", 2.0),
        "max_connection_pool_size": _get_int_setting("NEO4J_MAX_CONNECTION_POOL_SIZE", 10),
        "max_batch_size": _get_int_setting("NEO4J_MAX_BATCH_SIZE", 500),
    }


# Persistence/ingestion only: this flag does not enable GraphRAG or change KB retrieval.
NEO4J_ENABLED = _get_bool_setting("NEO4J_ENABLED")


def get_retrieval_settings() -> dict:
    """Opt-in retrieval policy; invalid settings safely retain vector-only behavior."""
    policy = (_resolve_key("RFP_RETRIEVAL_MODE") or "vector_only").lower()
    return {
        "policy": policy if policy in {"auto", "vector_only", "graph_only", "hybrid"} else "vector_only",
        "corpus_id": _resolve_key("RFP_GRAPH_CORPUS_ID") or "internal-rfp",
    }


def get_api_keys() -> tuple[str, str]:
    """Resolve API keys dynamically so Streamlit reruns pick up .env changes."""
    return _resolve_key("GROQ_API_KEY"), _resolve_key("GOOGLE_API_KEY")


GROQ_API_KEY, GOOGLE_API_KEY = get_api_keys()


def get_ragas_settings() -> dict:
    """Optional judge settings; no connection or RAGAS import at startup."""
    provider = _resolve_key("RAGAS_JUDGE_PROVIDER").lower()
    provider_key = {"groq": "GROQ_API_KEY", "google": "GOOGLE_API_KEY"}.get(provider, "")
    return {
        "provider": provider,
        "model": _resolve_key("RAGAS_JUDGE_MODEL"),
        "api_key": _resolve_key("RAGAS_JUDGE_API_KEY") or (_resolve_key(provider_key) if provider_key else ""),
        "timeout_seconds": _get_float_setting("RAGAS_METRIC_TIMEOUT_SECONDS", 90.0),
        "max_retries": _get_int_setting("RAGAS_JUDGE_MAX_RETRIES", 1),
        "max_cases": _get_int_setting("RAGAS_MAX_CASES", 30),
        "embedding_model": _resolve_key("RAGAS_EMBEDDING_MODEL") or EMBEDDING_MODEL,
        "max_tokens": _get_int_setting("RAGAS_JUDGE_MAX_TOKENS", 4096),
        "min_call_interval_seconds": _get_float_setting("RAGAS_JUDGE_MIN_CALL_INTERVAL_SECONDS", 0.0),
    }

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.8-flash"

GENERATION_TEMPERATURE = 0.3
# Backward-compatible alias for existing callers; new code should use the name above.
LLM_TEMPERATURE = GENERATION_TEMPERATURE
LLM_MAX_TOKENS = 2048

TAVILY_API_KEY = _resolve_key("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = 5
MAX_QUERY_RETRIES = 1
GRADING_TEMPERATURE = 0.0

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
COLLECTION_NAME = "rfp_kb_v2"
RETRIEVAL_K = 6
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.50"))
MAX_PROMPT_TOKENS = _get_int_setting("MAX_PROMPT_TOKENS", 6500)
RFP_ANALYSIS_MAX_OUTPUT_TOKENS = _get_int_setting("RFP_ANALYSIS_MAX_OUTPUT_TOKENS", 1200)
MAX_CONTEXT_CHARS_PER_CHUNK = _get_int_setting("MAX_CONTEXT_CHARS_PER_CHUNK", 1000)
MAX_HISTORY_MESSAGES = _get_int_setting("MAX_HISTORY_MESSAGES", 3)
MAX_TARGET_CHUNKS = _get_int_setting("MAX_TARGET_CHUNKS", 4)
MAX_CASE_STUDIES = _get_int_setting("MAX_CASE_STUDIES", 3)
MAX_CHUNKS_PER_CASE_STUDY = _get_int_setting("MAX_CHUNKS_PER_CASE_STUDY", 2)
MAX_UPLOAD_SIZE_MB = _get_int_setting("MAX_UPLOAD_SIZE_MB", 25)
MAX_UPLOAD_FILE_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_UPLOAD_PAGE_COUNT = _get_int_setting("MAX_UPLOAD_PAGE_COUNT", 250)
AGENT_MODE = (
    os.getenv("AGENT_MODE", "") or _get_env_file_value("AGENT_MODE") or "agentic"
).strip().lower() or "agentic"

AGENT_SYSTEM_PROMPT = """You are the Internal RFP Analyst, an AI-powered knowledge agent
for a global fintech consulting firm. Your job is to help internal teams quickly find
information from past proposals, project outlines, RFP responses, and case studies.

RULES:
1. Always ground your answers in the retrieved documents. Never fabricate information.
2. Cite your sources clearly using [Source: <document name>, Page <number>] format.
3. If you cannot find relevant information, say so honestly.
4. When comparing projects, present information in a structured table format.
5. Be concise but thorough because consultants are busy people.
6. If the user's question is ambiguous, ask a clarifying question before answering.
"""

APP_TITLE = "Internal RFP Analyst"
APP_SUBTITLE = "AI-powered knowledge agent for fintech consulting"
APP_ICON = "??"
SAMPLE_QUESTIONS = [
    "What tech stack did we use for the last banking audit?",
    "Which projects used Azure services?",
    "Compare the healthcare and insurance projects",
    "What was the budget for the supply chain analytics platform?",
    "List all projects with their timelines",
    "Tell me about projects involving machine learning",
    "What compliance frameworks did we follow in pharma projects?",
    "Which project had the largest team?",
]
