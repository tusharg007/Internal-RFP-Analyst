import pytest

import agent
from rfp_analyst.exceptions import LLMProviderNotConfiguredError
from rfp_analyst.ui import get_chat_avatar


def test_no_llm_provider_state(monkeypatch):
    monkeypatch.setattr(agent, "get_api_keys", lambda: ("", ""))

    assert agent._get_provider_name() == "Not configured"
    with pytest.raises(LLMProviderNotConfiguredError):
        agent.get_llm()


def test_avatar_helper_uses_valid_emojis():
    assert get_chat_avatar("user") == "👤"
    assert get_chat_avatar("assistant") == "🤖"
    assert get_chat_avatar("anything-else") not in {"User", "AI"}

def test_llm_auth_error_format_hides_provider_json_by_default():
    provider_error = RuntimeError(
        'Error code: 401 - {"error":{"type":"invalid_api_key","message":"bad key"}}'
    )

    message = agent.format_llm_error(provider_error)

    assert message == agent.LLM_AUTH_ERROR_MESSAGE
    assert "invalid_api_key" not in message
    assert "bad key" not in message


def test_llm_auth_error_format_redacts_details_even_in_debug_mode():
    provider_error = RuntimeError(
        'Error code: 401 - {"error":{"type":"invalid_api_key","message":"bad key"}}'
    )

    message = agent.format_llm_error(provider_error, debug=True)

    assert agent.LLM_AUTH_ERROR_MESSAGE in message
    assert "Error type: RuntimeError" in message
    assert "invalid_api_key" not in message
    assert "bad key" not in message


def test_llm_token_budget_error_format_hides_provider_json_by_default():
    provider_error = RuntimeError(
        'Error code: 413 - {"error":{"type":"rate_limit_exceeded","message":"request too large"}}'
    )

    message = agent.format_llm_error(provider_error)

    assert message == agent.LLM_TOKEN_BUDGET_ERROR_MESSAGE
    assert "rate_limit_exceeded" not in message
    assert "request too large" not in message


def test_query_agent_stream_returns_friendly_token_budget_error():
    class FakeLLM:
        def stream(self, _messages):
            raise RuntimeError(
                'Error code: 413 - {"error":{"type":"rate_limit_exceeded","message":"request too large"}}'
            )

    output = "".join(agent.query_agent_stream(FakeLLM(), {"prompt": "hello", "response_mode": "llm"}))

    assert output == agent.LLM_TOKEN_BUDGET_ERROR_MESSAGE


def test_duplicate_source_cards_are_grouped_with_one_based_pages():
    payload = {
        "user_query": "what stack?",
        "traces": [
            {
                "tool": "search_knowledge_base",
                "input": {"query": "what stack?"},
                "documents": [
                    {
                        "source": "client_profile.pdf",
                        "page": 1,
                        "score": "0.85",
                        "chunk_id": "def",
                        "document_origin": "upload",
                    },
                    {
                        "source": "client_profile.pdf",
                        "page": 1,
                        "score": "0.95",
                        "chunk_id": "abc",
                        "document_origin": "upload",
                    },
                ],
            }
        ],
    }

    trace = agent._payload_to_reasoning_trace(payload)
    source_cards = [item for item in trace if item.get("tool_response")]

    assert len(source_cards) == 1
    assert source_cards[0]["tool_response"] == "client_profile.pdf (Page 1)"
    assert source_cards[0]["page"] == 1
    assert source_cards[0]["match_count"] == 2
    assert source_cards[0]["score"] == "0.95"
    assert set(source_cards[0]["chunk_ids"]) == {"abc", "def"}


def test_source_none_never_appears_after_sanitization():
    answer = agent.sanitize_answer_text("Uses React [Source: None]")

    assert "[Source: None]" not in answer
    assert "The retrieved evidence does not support this claim." in answer


def test_tool_trace_contains_summaries_not_hidden_reasoning():
    payload = {
        "user_query": "write proposal",
        "traces": [
            {
                "tool": "proposal_writer",
                "input_summary": "proposal request",
                "output_summary": "Draft proposal outline from retrieved evidence.",
            }
        ],
    }

    trace = agent._payload_to_reasoning_trace(payload)

    assert trace[0]["input_summary"] == "proposal request"
    assert "output_summary" in trace[0]
    assert "chain-of-thought" not in str(trace).lower()
