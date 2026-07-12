import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "config",
        "document_generator",
        "agent",
        "rag_engine",
        "rfp_analyst.exceptions",
        "rfp_analyst.health",
        "rfp_analyst.uploads",
        "rfp_analyst.ui.helpers",
        "rfp_analyst.ingestion.loaders",
        "rfp_analyst.ingestion.chunking",
        "rfp_analyst.ingestion.registry",
        "rfp_analyst.ingestion.pipeline",
        "rfp_analyst.retrieval.vector_store",
        "rfp_analyst.schemas",
        "rfp_analyst.tools.search_kb",
        "rfp_analyst.tools.compare_projects",
        "rfp_analyst.tools.rfp_gap_analyzer",
        "rfp_analyst.tools.proposal_writer",
        "rfp_analyst.tools.source_verifier",
        "rfp_analyst.agent.state",
        "rfp_analyst.agent.prompts",
        "rfp_analyst.agent.graph",
        "rfp_analyst.agent.runtime",
    ],
)
def test_module_imports(module_name):
    module = importlib.import_module(module_name)
    assert module is not None
