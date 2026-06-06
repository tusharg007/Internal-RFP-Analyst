"""
Central Configuration for the Internal RFP Analyst Agent.
All paths, model settings, and pipeline parameters are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
ASSETS_DIR = BASE_DIR / "assets"

# ─── API Keys ─────────────────────────────────────────────────────────────────
# Supports both local .env and Streamlit Cloud secrets
try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ─── LLM Provider Selection ──────────────────────────────────────────────────
# Priority: Groq (fastest free inference) → Gemini (fallback)
# Groq: 30 RPM, 6000 RPD free tier — https://console.groq.com
# Gemini: 15 RPM free tier — https://aistudio.google.com/apikey

GROQ_MODEL = "llama-3.3-70b-versatile"    # Best quality on Groq free tier
GEMINI_MODEL = "gemini-2.0-flash"          # Fallback

LLM_TEMPERATURE = 0.3                      # Low temp for factual retrieval
LLM_MAX_TOKENS = 2048

# ─── Embeddings — Local (no API calls, no rate limits) ────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ─── RAG Pipeline ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 512         # tokens per chunk
CHUNK_OVERLAP = 50       # overlap for context continuity
COLLECTION_NAME = "rfp_kb_v2"
RETRIEVAL_K = 6          # number of chunks to retrieve per query

# ─── Agent ────────────────────────────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """You are the **Internal RFP Analyst**, an AI-powered knowledge agent 
for a global fintech consulting firm. Your job is to help internal teams quickly find 
information from past proposals, project outlines, RFP responses, and case studies.

RULES:
1. Always ground your answers in the retrieved documents. Never fabricate information.
2. Cite your sources clearly using [Source: <document name>, Page <number>] format.
3. If you cannot find relevant information, say so honestly.
4. When comparing projects, present information in a structured table format.
5. Be concise but thorough — consultants are busy people.
6. If the user's question is ambiguous, ask a clarifying question before answering.
"""

# ─── UI ───────────────────────────────────────────────────────────────────────
APP_TITLE = "🔍 Internal RFP Analyst"
APP_SUBTITLE = "AI-Powered Knowledge Agent for Fintech Consulting"
APP_ICON = "🔍"
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
