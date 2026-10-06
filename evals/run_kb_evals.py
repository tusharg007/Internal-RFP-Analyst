"""Real knowledge-base evaluation against the actual retrieval layer."""

from __future__ import annotations

import json
import gc
import shutil
import tempfile
import time
from pathlib import Path

import yaml
from fpdf import FPDF

from config import COLLECTION_NAME, DATA_DIR, REAL_KB_EVAL_RESULTS_PATH, get_api_keys
from document_generator import generate_all_documents
from rag_engine import _filter_results_for_scope, _normalize_scope, _release_chroma_resources, get_vectorstore_stats, ingest_documents
from rfp_analyst.agent.graph import prepare_query_payload, run_query
from rfp_analyst.retrieval.vector_store import VectorStoreManager

GOLDEN_QUESTIONS_PATH = Path(__file__).with_name("golden_questions.yaml")


def load_golden_cases(path: Path = GOLDEN_QUESTIONS_PATH) -> list[dict]:
    """Load the shared real-KB golden set."""
    with path.open("r", encoding="utf-8") as handle:
        return [case for case in yaml.safe_load_all(handle) if case]


def _ensure_eval_upload(uploads_dir: Path) -> None:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target = uploads_dir / "eval_target_rfp.pdf"
    if target.exists():
        return
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(
        0,
        7,
        "Client RFP Requirements\nThe solution must support Azure migration, executive dashboards, HIPAA controls, and a phased delivery plan. The proposal should include relevant case studies and measurable outcomes.",
    )
    pdf.output(str(target))


def ensure_sample_documents_ready(persist_dir: Path, uploads_dir: Path) -> dict:
    """Generate bundled PDFs and ingest them into an isolated Chroma directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not list(DATA_DIR.glob("*.pdf")):
        generate_all_documents()
    _ensure_eval_upload(uploads_dir)
    ingest_documents(persist_dir=persist_dir, uploads_dir=uploads_dir)
    return get_vectorstore_stats(persist_dir=persist_dir, uploads_dir=uploads_dir)


def _prepare_eval_vectorstore(persist_dir: Path, uploads_dir: Path) -> dict:
    """Support both the current helper and older test monkeypatches."""
    try:
        stats = ensure_sample_documents_ready(persist_dir, uploads_dir)
    except TypeError:
        stats = ensure_sample_documents_ready()
    return stats or get_vectorstore_stats(persist_dir=persist_dir, uploads_dir=uploads_dir)


def _build_llm() -> object | None:
    groq_api_key, google_api_key = get_api_keys()
    if not (groq_api_key or google_api_key):
        return None

    import agent

    return agent.get_llm()


def _evaluate_no_answer_behavior(payload: dict, answer_text: str, expects_no_answer: bool) -> bool:
    if not expects_no_answer:
        return True

    lowered = (answer_text or "").lower()
    return payload.get("response_mode") == "fallback" or any(
        marker in lowered
        for marker in (
            "could not find",
            "no grounded evidence",
            "cannot find",
            "not ready",
            "do not have",
        )
    )


def _run_case(case: dict, *, vectorstore_stats: dict, retrieval_fn, llm=None) -> dict:
    question = case["question"]
    expected_sources = case.get("expected_sources", [])
    expects_no_answer = bool(case.get("expects_no_answer", False))
    retrieval_scope = case.get("retrieval_scope", "all")
    chat_history = case.get("chat_history", [])
    start = time.perf_counter()
    result = None
    if llm is None:
        payload = prepare_query_payload(
            question,
            chat_history=chat_history,
            retrieval_scope=retrieval_scope,
            vectorstore_stats=vectorstore_stats,
            retrieval_fn=retrieval_fn,
            retrieval_mode="vector_only",
            allow_web_search=False,
            allow_llm_routing=False,
            allow_llm_grading=False,
            allow_llm_rewriting=False,
        )
    else:
        result = run_query(
            llm,
            question,
            chat_history=chat_history,
            vectorstore_stats=vectorstore_stats,
            retrieval_fn=retrieval_fn,
            retrieval_scope=retrieval_scope,
            retrieval_mode="vector_only",
        )
        payload = result["payload"]
    retrieval_latency = time.perf_counter() - start

    traces = payload.get("traces", [])
    retrieved_sources = list(dict.fromkeys(item.get("source") for item in payload.get("retrieved_documents", [])))
    retrieved_origins = sorted({item.get("document_origin") for item in payload.get("retrieved_documents", []) if item.get("document_origin")})
    expected_hits = [source for source in expected_sources if source in retrieved_sources]
    citation_coverage = (
        round(len(expected_hits) / len(expected_sources), 2) if expected_sources else 1.0
    )

    answer_text = ""
    answer_latency = None
    answer_mode = "retrieval_only"
    if llm is not None:
        # Source/tool checks and the answer belong to this same graph execution.
        answer_latency = retrieval_latency
        answer_text = result.get("answer", "")
        answer_mode = "llm_answer"

    no_answer_ok = _evaluate_no_answer_behavior(payload, answer_text, expects_no_answer)
    executed_tools = [step.get("tool") for step in traces]
    tools_ok = all(tool in executed_tools for tool in case.get("expected_tools", []))
    origins_ok = all(origin in retrieved_origins for origin in case.get("expected_origins", []))
    passed = citation_coverage >= 0.5 and no_answer_ok and tools_ok and origins_ok
    if expects_no_answer:
        passed = no_answer_ok and tools_ok and origins_ok

    return {
        "question": question,
        "expected_sources": expected_sources,
        "retrieved_sources": retrieved_sources,
        "retrieved_origins": retrieved_origins,
        "executed_tools": executed_tools,
        "citation_coverage": citation_coverage,
        "no_answer_behavior_ok": no_answer_ok,
        "retrieval_latency_seconds": retrieval_latency,
        "answer_latency_seconds": answer_latency,
        "mode": answer_mode,
        "response_mode": payload.get("response_mode"),
        "answer_preview": answer_text[:280] if answer_text else "",
        "passed": passed,
    }


def run_real_kb_eval(
    output_path: Path = REAL_KB_EVAL_RESULTS_PATH,
    use_llm: bool | None = None,
) -> dict:
    """Run retrieval-backed evaluation against the real sample document set."""
    temp_dir = Path(tempfile.mkdtemp(prefix="kb_eval_vectorstore_"))
    try:
        persist_dir = temp_dir / "vectorstore"
        uploads_dir = temp_dir / "uploads"
        vectorstore_stats = _prepare_eval_vectorstore(persist_dir, uploads_dir)

        def retrieval_fn(query: str, k: int, scope: str = "all"):
            vectorstore = None
            normalized_scope = _normalize_scope(scope)
            try:
                manager = VectorStoreManager(persist_dir=persist_dir, collection_name=COLLECTION_NAME)
                vectorstore = manager.load(create_if_missing=False)
                kwargs = {"k": k}
                if normalized_scope in {"sample", "upload"}:
                    kwargs["filter"] = {"document_origin": normalized_scope}
                try:
                    raw_results = vectorstore.similarity_search_with_relevance_scores(query, **kwargs)
                except TypeError:
                    raw_results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
                return _filter_results_for_scope(raw_results, normalized_scope)
            finally:
                _release_chroma_resources(vectorstore)
                del vectorstore
                gc.collect()

        llm = _build_llm() if use_llm is True else None
        if use_llm is True and llm is None:
            raise RuntimeError("LLM mode requested but no GROQ_API_KEY or GOOGLE_API_KEY is configured.")

        total_start = time.perf_counter()
        cases = load_golden_cases()
        case_results = [
            _run_case(case, vectorstore_stats=vectorstore_stats, retrieval_fn=retrieval_fn, llm=llm)
            for case in cases
        ]
        total_latency = time.perf_counter() - total_start
    finally:
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)

    passed = sum(1 for case in case_results if case["passed"])
    payload = {
        "evaluation_name": "Real KB Evaluation",
        "evaluation_type": "real_kb_eval",
        "mode": "llm_answer" if llm is not None else "retrieval_only",
        "score": f"{passed}/{len(case_results)}",
        "pass_rate": round(passed / len(case_results), 2),
        "latency": total_latency,
        "latency_unit": "seconds",
        "notes": "Runs against generated sample PDFs and the actual Chroma retrieval layer. Retrieval-only mode disables live router/grader/rewriter/web calls. Optional LLM-answer mode measures one full graph execution.",
        "cases": case_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


if __name__ == "__main__":
    summary = run_real_kb_eval()
    print(json.dumps(summary, indent=2))
