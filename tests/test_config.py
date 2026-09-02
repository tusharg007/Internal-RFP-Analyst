import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


def load_fresh_config():
    sys.modules.pop("config", None)
    return importlib.import_module("config")


def load_fresh_agent():
    sys.modules.pop("agent", None)
    return importlib.import_module("agent")


def test_config_default_paths_and_values(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets={}))

    config = load_fresh_config()

    assert config.BASE_DIR == Path(config.__file__).resolve().parent
    assert config.DATA_DIR == config.BASE_DIR / "data" / "documents"
    assert config.VECTORSTORE_DIR == config.BASE_DIR / "vectorstore"
    assert config.ASSETS_DIR == config.BASE_DIR / "assets"
    assert config.GROQ_MODEL == "openai/gpt-oss-120b"
    assert config.GEMINI_MODEL == "gemini-2.0-flash"
    assert config.LLM_TEMPERATURE == 0.3
    assert config.LLM_MAX_TOKENS == 2048
    assert config.CHUNK_SIZE == 512
    assert config.CHUNK_OVERLAP == 50
    assert config.COLLECTION_NAME == "rfp_kb_v2"
    assert config.RETRIEVAL_K == 6
    assert config.MAX_UPLOAD_FILE_SIZE_BYTES == 25 * 1024 * 1024
    assert config.MAX_UPLOAD_PAGE_COUNT == 250
    assert config.AGENT_MODE == "agentic"
    assert config.SAMPLE_QUESTIONS


def test_config_prefers_streamlit_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-groq")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google")
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        SimpleNamespace(
            secrets={
                "GROQ_API_KEY": "secret-groq",
                "GOOGLE_API_KEY": "secret-google",
            }
        ),
    )

    config = load_fresh_config()

    assert config.GROQ_API_KEY == "secret-groq"
    assert config.GOOGLE_API_KEY == "secret-google"


def test_config_falls_back_to_env_when_streamlit_secrets_fail(monkeypatch):
    class BrokenSecrets:
        def get(self, *_args, **_kwargs):
            raise RuntimeError("secrets unavailable")

    monkeypatch.setenv("GROQ_API_KEY", "env-groq")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google")
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        SimpleNamespace(secrets=BrokenSecrets()),
    )

    config = load_fresh_config()

    assert config.GROQ_API_KEY == "env-groq"
    assert config.GOOGLE_API_KEY == "env-google"


def test_no_llm_provider_state(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets={}))
    load_fresh_config()
    agent = load_fresh_agent()

    assert agent._get_provider_name() == "Not configured"
    assert agent.is_llm_provider_configured() is False
