"""Fresh graph queries versus genuinely unresolved conversational references."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rfp_analyst.agent import graph as workflow
from rfp_analyst.graph.reader import PROJECT_SEEDS, SUBJECT_FACTS
from tests.test_hybrid_retrieval import (
    environment as environment,
    no_providers as no_providers,
    provider,
    ready,
)


AZURE_QUERY = (
    "List all projects in the knowledge base that used Microsoft Azure. "
    "For each matching project, list the other technologies or frameworks used "
    "and the reported outcomes."
)
STANDALONE_QUERIES = [
    "Which projects used Microsoft Azure?",
    "List projects using Microsoft Azure and their outcomes.",
    "Which Azure projects also mention HIPAA and what outcomes did they report?",
    AZURE_QUERY,
    "Which projects used Azure? For those projects, list technologies and outcomes.",
    "List all projects and their timelines.",
    "List projects that used Azure and show their outcomes.",
    "List projects and their outcomes.",
    "Which healthcare projects reported their outcomes?",
]


@pytest.mark.parametrize("query", STANDALONE_QUERIES)
@pytest.mark.parametrize("history", [[], [{"role": "user", "content": "An earlier query"}]])
def test_standalone_project_selection_needs_no_conversation_antecedent(query, history):
    result = workflow.classify_intent(
        {
            "user_query": query,
            "chat_history": history,
            "retrieval_scope": "all",
            "allow_llm_routing": False,
        }
    )
    assert result["intent"] == "search"
    assert result["source_used"] == "kb"
    assert result["response_mode"] == "llm"
    assert result["resolved_query"] == query
    assert result["resolved_entities"] == []
    assert result["traces"][-1]["resolution_status"] == "not_needed"


@pytest.mark.parametrize(
    "query",
    [
        "What about its budget?",
        "What are their budgets?",
        "Does it use Azure?",
        "Which projects used it?",
        "List those projects and their outcomes.",
        "Compare these projects with Azure.",
        "List projects that used it.",
        "List projects that were previously discussed.",
        "Does that project use Azure?",
        "Which projects used Azure and how does that compare to its budget?",
    ],
)
def test_unresolvable_followup_still_clarifies(query):
    result = workflow.classify_intent(
        {
            "user_query": query,
            "chat_history": [],
            "retrieval_scope": "all",
            "allow_llm_routing": False,
        }
    )
    assert result["intent"] == "ambiguous"
    assert result["answer"] == workflow.FOLLOWUP_CLARIFICATION_MESSAGE
    assert result["source_used"] == "direct"


def test_true_possessive_followup_resolves_prior_document_in_scope():
    history = [
        {
            "role": "assistant",
            "reasoning": [
                {
                    "source": "healthcare.pdf",
                    "page": 2,
                    "document_origin": "sample",
                }
            ],
        }
    ]
    state = {
        "user_query": "What about its budget?",
        "chat_history": history,
        "retrieval_scope": "all",
        "allow_llm_routing": False,
    }
    result = workflow.classify_intent(state)
    assert result["intent"] == "search"
    assert result["resolved_entities"][0]["source"] == "healthcare.pdf"
    assert result["traces"][-1]["resolution_status"] == "resolved"
    # A prior source outside the selected scope is not a usable antecedent.
    result = workflow.classify_intent(state | {"retrieval_scope": "upload"})
    assert result["response_mode"] == "clarification"


def test_singular_followup_with_multiple_prior_documents_still_clarifies():
    history = [
        {
            "role": "assistant",
            "reasoning": [
                {"source": "healthcare.pdf", "document_origin": "sample"},
                {"source": "banking.pdf", "document_origin": "sample"},
            ],
        }
    ]
    assert workflow._resolve_conversational_query("What about its budget?", history, "all")[2] == (
        "ambiguous"
    )


def test_standalone_relative_clause_reaches_structured_kb_router(monkeypatch):
    from rfp_analyst.agent.schemas_decisions import RouteDecision
    from tests.test_langgraph_agent import FakeRouterLLM

    llm = FakeRouterLLM(RouteDecision(route="kb"))
    monkeypatch.setattr("rfp_analyst.agent.router._create_router_llm", lambda: llm)
    result = workflow.classify_intent(
        {
            "user_query": AZURE_QUERY,
            "chat_history": [],
            "retrieval_scope": "all",
        }
    )
    assert llm.schema is RouteDecision
    assert AZURE_QUERY in llm.structured_router.prompt
    assert result["intent"] == "search" and result["source_used"] == "kb"


@pytest.mark.parametrize("query", STANDALONE_QUERIES[:5])
def test_fresh_graph_only_query_enters_reader_and_grounded_generation(
    environment, query, monkeypatch
):
    monkeypatch.setenv("RFP_RETRIEVAL_MODE", "graph_only")
    vector = Mock(side_effect=AssertionError("Graph-only must not issue vector retrieval"))
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content=("Healthcare Migration used Microsoft Azure. [Source: healthcare.pdf, Page 2]")
    )
    service = provider(environment, policy="graph_only", strict=True)
    payload = workflow.prepare_query_payload(
        query,
        chat_history=[],
        retrieval_scope="all",
        vectorstore_stats=ready(environment),
        retrieval_fn=vector,
        retrieval_provider=service,
        llm=llm,
        allow_llm_routing=False,
        allow_llm_grading=False,
        allow_web_search=False,
    )
    assert payload["intent"] == "search"
    assert payload["requested_retrieval_mode"] == payload["retrieval_mode"] == "graph_only"
    assert payload["retrieval_scope"] == "all"
    assert payload["resolved_entities"] == []
    assert payload["graph_query_type"] == "project_constraints"
    statements = [statement for statement, _ in environment[1].calls]
    assert PROJECT_SEEDS in statements and SUBJECT_FACTS in statements
    assert payload["graph_provenance"] and payload["graph_paths"]
    assert payload["generation_contexts"] and payload["generation_evidence"]
    assert {item["chunk_id"] for item in payload["graph_provenance"]} <= {
        item["chunk_id"] for item in payload["retrieved_documents"]
    }
    assert payload["grounded"]
    assert payload["response_mode"] == "llm"
    assert payload["generation_kind"] == "llm_kb"
    tools = [trace.get("tool") for trace in payload["traces"]]
    assert "intent_classifier" in tools and "graph_retrieval" in tools
    assert "final_grounding_verifier" in tools
    assert "web_search" not in tools
    llm.invoke.assert_called_once()
    vector.assert_not_called()


def test_compiled_graph_does_not_retain_previous_clarification_state(environment):
    service = provider(environment, policy="graph_only", strict=True)
    common = dict(
        retrieval_provider=service,
        vectorstore_stats=ready(environment),
        retrieval_scope="all",
        allow_llm_routing=False,
        allow_llm_grading=False,
        allow_web_search=False,
    )
    first = workflow.prepare_query_payload("What about its budget?", chat_history=[], **common)
    assert first["response_mode"] == "clarification"
    assert not environment[1].calls
    second = workflow.prepare_query_payload(AZURE_QUERY, chat_history=[], **common)
    assert second["intent"] == "search"
    assert second["retrieval_mode"] == "graph_only"
    assert second["graph_provenance"]
    assert second["answer"] != workflow.FOLLOWUP_CLARIFICATION_MESSAGE


def test_new_standalone_selection_does_not_bind_stale_prior_source():
    history = [
        {
            "role": "assistant",
            "reasoning": [
                {
                    "source": "unrelated.pdf",
                    "document_origin": "upload",
                    "page": 1,
                }
            ],
        }
    ]
    query = "Which projects used Azure? List outcomes of those projects."
    assert workflow._resolve_conversational_query(query, history, "all") == (
        query,
        [],
        "not_needed",
    )
