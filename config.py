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


def _get_int_setting(name: str, default: int) -> int:
    raw_value = os.getenv(name, "") or _get_env_file_value(name)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def get_api_keys() -> tuple[str, str]:
    """Resolve API keys dynamically so Streamlit reruns pick up .env changes."""
    groq_api_key = (
        _get_streamlit_secret("GROQ_API_KEY")
        or os.getenv("GROQ_API_KEY", "")
        or _get_env_file_value("GROQ_API_KEY")
    )
    google_api_key = (
        _get_streamlit_secret("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_API_KEY", "")
        or _get_env_file_value("GOOGLE_API_KEY")
    )
    return groq_api_key.strip(), google_api_key.strip()


GROQ_API_KEY, GOOGLE_API_KEY = get_api_keys()

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"

LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

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
