from types import SimpleNamespace

from evals import run_kb_evals


class FakeDoc:
    def __init__(self, content: str, source: str, page: int = 0):
        self.page_content = content
        self.metadata = {"source_file": source, "page": page}


class FakeLLM:
    def invoke(self, _messages):
        return SimpleNamespace(content="I could not find grounded evidence for that request.")


def test_real_kb_eval_defaults_to_retrieval_only(monkeypatch, tmp_path):
    monkeypatch.setattr(run_kb_evals, "ensure_sample_documents_ready", lambda: None)
    monkeypatch.setattr(run_kb_evals, "_build_llm", lambda: None)

    monkeypatch.setattr(run_kb_evals, "load_golden_cases", lambda: [
        {"question": "banking", "expected_sources": ["01_Banking_Sector_Digital_Audit_2024.pdf"]},
        {"question": "Mars", "expected_sources": [], "expects_no_answer": True},
    ])

    def fake_prepare_query_payload(question: str, **kwargs):
        source = "01_Banking_Sector_Digital_Audit_2024.pdf"
        if "Compare" in question:
            docs = [
                {"source": "02_Healthcare_Data_Migration_to_Azure_Cloud.pdf", "page": 0, "score": "0.91"},
                {"source": "04_Insurance_Claims_Processing_Automation.pdf", "page": 0, "score": "0.88"},
            ]
        elif "Mars" in question:
            docs = []
        else:
            docs = [{"source": source, "page": 0, "score": "0.95"}]
        return {
            "response_mode": "fallback" if "Mars" in question else "llm",
            "traces": [{"tool": "search_knowledge_base", "documents": docs}],
        }

    monkeypatch.setattr(run_kb_evals, "prepare_query_payload", fake_prepare_query_payload)

    payload = run_kb_evals.run_real_kb_eval(tmp_path / "real_kb_results.json")

    assert payload["evaluation_name"] == "Real KB Evaluation"
    assert payload["mode"] == "retrieval_only"
    assert len(payload["cases"]) == 2


def test_real_kb_eval_can_use_llm_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(run_kb_evals, "ensure_sample_documents_ready", lambda: None)
    monkeypatch.setattr(run_kb_evals, "_build_llm", lambda: FakeLLM())
    monkeypatch.setattr(
        run_kb_evals,
        "prepare_query_payload",
        lambda question, **kwargs: {
            "response_mode": "llm",
            "traces": [{"tool": "search_knowledge_base", "documents": [{"source": "01_Banking_Sector_Digital_Audit_2024.pdf", "page": 0, "score": "0.95"}]}],
        },
    )
    monkeypatch.setattr(
        run_kb_evals,
        "run_query",
        lambda llm, question, **kwargs: {
            "answer": "Grounded answer with citations.",
            "payload": {
                "retrieved_documents": [],
                "traces": [],
                "response_mode": "llm",
            },
        },
    )

    payload = run_kb_evals.run_real_kb_eval(tmp_path / "real_kb_results.json", use_llm=True)

    assert payload["mode"] == "llm_answer"
    assert all(case["mode"] == "llm_answer" for case in payload["cases"])
    monkeypatch.setattr(run_kb_evals, "load_golden_cases", lambda: [
        {"question": "banking", "expected_sources": ["01_Banking_Sector_Digital_Audit_2024.pdf"]}
    ])
