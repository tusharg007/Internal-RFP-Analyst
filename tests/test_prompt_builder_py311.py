from rfp_analyst.agent.graph import prepare_query_payload


class FakeDoc:
    page_content = "Evidence supporting the banking audit answer."
    metadata = {"source_file": "banking_case_study.pdf", "page": 0}


def test_prompt_builder_imports_and_builds_prompt_on_py311():
    payload = prepare_query_payload(
        user_query="What tech stack did we use for the banking audit?",
        chat_history=[{"role": "user", "content": "Earlier context"}],
        vectorstore_stats={
            "status": "ready",
            "total_documents": 1,
            "total_chunks": 1,
            "document_names": ["banking_case_study.pdf"],
        },
        retrieval_fn=lambda _query, _k: [(FakeDoc(), 0.92)],
    )

    assert payload["response_mode"] == "llm"
    assert "-- Planned Tools --" in payload["prompt"]
    assert "-- Recent Conversation --" in payload["prompt"]
    assert "banking_case_study.pdf" in payload["prompt"]
