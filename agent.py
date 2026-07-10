"""RAG Engine - feature-flagged simple and agentic execution paths."""

from config import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)
from rfp_analyst.agent.runtime import prepare_query_payload, run_query, stream_query_response
from rfp_analyst.exceptions import LLMProviderNotConfiguredError


def _get_provider_name():
    """Return which LLM provider is active."""
    if GROQ_API_KEY:
        return f"Groq ({GROQ_MODEL})"
    if GOOGLE_API_KEY:
        return f"Gemini ({GEMINI_MODEL})"
    return "Not configured"


def is_llm_provider_configured() -> bool:
    """Return whether any supported LLM provider is configured."""
    return bool(GROQ_API_KEY or GOOGLE_API_KEY)


def get_llm():
    """Get LLM with automatic provider selection. Groq preferred (faster)."""
    if GROQ_API_KEY:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    if GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )
    raise LLMProviderNotConfiguredError(
        "No LLM provider is configured. Add GROQ_API_KEY or GOOGLE_API_KEY in your .env file locally, "
        "or in Streamlit secrets on deployment."
    )


def create_agent():
    """Create the LLM instance."""
    return get_llm()


def prepare_query(user_query: str, chat_history: list = None):
    """Prepare a feature-flagged query payload and visible tool trace."""
    payload = prepare_query_payload(user_query, chat_history)
    return payload, payload["reasoning_trace"]


def query_agent_stream(llm, prompt):
    """Stream either the legacy simple path or the new agentic path."""
    yield from stream_query_response(llm, prompt)


def query_agent(llm, user_query: str, thread_id: str = "default", chat_history: list = None):
    """Non-streaming query wrapper."""
    return run_query(llm, user_query, chat_history)


if __name__ == "__main__":
    print(f"Active provider: {_get_provider_name()}")
    llm = create_agent()
    result = query_agent(llm, "List all projects with their timelines")
    print(result["answer"][:500])
