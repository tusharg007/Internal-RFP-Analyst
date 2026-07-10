from langchain_core.documents import Document

from rfp_analyst.agent.graph import run_agent_graph
from rfp_analyst.agent.state import AgentState
from rfp_analyst.tools.compare_projects import compare_projects
from rfp_analyst.tools.proposal_writer import generate_proposal_outline
from rfp_analyst.tools.rfp_gap_analyzer import extract_rfp_requirements, find_relevant_case_studies
from rfp_analyst.tools.search_kb import search_knowledge_base
from rfp_analyst.tools.source_verifier import verify_answer_grounding


def make_doc(source: str, page: int, content: str):
    return Document(page_content=content, metadata={"source_file": source, "page": page})


def fake_search_fn(query: str, k: int = 6):
    docs = [
        (
            make_doc(
                "Banking_Audit.pdf",
                0,
                "Timeline & Milestones: 16 weeks. Budget Range: $850,000. Technology Stack: Azure SQL, Power BI. Key Outcomes: Improved data quality.",
            ),
            0.93,
        ),
        (
            make_doc(
                "Insurance_Automation.pdf",
                1,
                "Timeline & Milestones: 12 weeks. Budget Range: $650,000. Technology Stack: Azure Functions, UiPath. Key Outcomes: Claims automation.",
            ),
            0.88,
        ),
    ]
    return docs[:k]


def test_search_knowledge_base_with_mocked_retriever():
    result = search_knowledge_base("azure projects", search_fn=fake_search_fn)
    assert len(result["documents"]) == 2
    assert result["sources"][0]["source"] == "Banking_Audit.pdf"
    assert "[Source: Banking_Audit.pdf, Page 1]" in result["context"]


def test_compare_projects_with_mocked_retriever():
    result = compare_projects("compare banking and insurance", search_fn=fake_search_fn)
    assert "| Project | Timeline | Budget | Tech Stack | Outcomes |" in result["comparison_markdown"]
    assert "Banking_Audit.pdf" in result["comparison_markdown"]


def test_extract_rfp_requirements():
    result = extract_rfp_requirements("The solution must support Azure. The vendor should include dashboards.")
    assert len(result["requirements"]) >= 2
    assert result["requirements"][0]["id"].startswith("REQ-")


def test_find_relevant_case_studies_with_mocked_retriever():
    requirements = [{"id": "REQ-01", "text": "Azure migration"}]
    result = find_relevant_case_studies(requirements, search_fn=fake_search_fn)
    assert result["matches"]
    assert result["matches"][0]["source"] == "Banking_Audit.pdf"


def test_generate_proposal_outline():
    case_studies = {
        "matches": [{"source": "Banking_Audit.pdf", "pages": [0], "snippets": ["Azure SQL and Power BI"]}]
    }
    outline = generate_proposal_outline("Write a proposal", case_studies, [{"id": "REQ-01", "text": "Azure"}])
    assert "## Executive Summary" in outline["outline"]
    assert "[Source: Banking_Audit.pdf, Page 1]" in outline["outline"]


def test_verify_answer_grounding_catches_unsupported_claims():
    answer = (
        "The banking project used Azure SQL and Power BI [Source: Banking_Audit.pdf, Page 1]. "
        "It also deployed Kubernetes to 30 regions."
    )
    verification = verify_answer_grounding(answer, [fake_search_fn("x")[0][0]])
    assert not verification["is_grounded"]
    assert any("Kubernetes" in claim for claim in verification["unsupported_claims"])


def test_graph_happy_path():
    state = run_agent_graph(
        AgentState(query="Compare the banking and insurance projects"),
        search_fn=fake_search_fn,
        stats_fn=lambda: {"status": "ready", "total_documents": 2, "total_chunks": 4, "document_names": ["Banking_Audit.pdf", "Insurance_Automation.pdf"]},
    )
    assert state.intent == "compare_projects"
    assert state.prompt
    assert any(step.get("tool") == "compare_projects" for step in state.tool_trace)


def test_no_documents_fallback():
    state = run_agent_graph(
        AgentState(query="What projects used Azure?"),
        search_fn=fake_search_fn,
        stats_fn=lambda: {"status": "not_initialized", "total_documents": 0, "total_chunks": 0, "document_names": []},
    )
    assert "Please ingest documents first" in state.final_answer
