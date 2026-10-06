# Groq GPT-OSS matched capture

Date: 2026-10-06. **The single-case smoke passed. The subsequent benchmark stopped on HTTP 429 after 1/48 completed executions; this is not a valid retrieval-mode comparison.**

## Experimental controls

All 18 observed application LLM invocations used `ChatGroq`, provider `groq`, model `openai/gpt-oss-120b`. Successful responses also reported that model ID. Gemini calls: **0**. RAGAS judge calls: **0**.

Evaluation-only process-local pinning masked the Google key through the existing configuration resolver, verified the four existing application factories selected Groq GPT-OSS, and installed a before-transport guard against any other provider/model. Environment overrides were restored afterward. Existing factories, provider/fallback implementation, model constants, prompts, retrieval, grading, grounding, RAGAS configuration and frozen expectations were not edited. Existing SDK retry behavior was not changed; the evaluation stopped when a provider error reached the application boundary.

## semantic_modernization smoke

Requested/effective retrieval mode: `vector_only`. Public frozen sample scope; no external web search was enabled, consistent with the existing matched benchmark policy.

Actual application LLM sequence:

1. `route_question`: completed; intent `search`, source `kb`.
2. `grade_kb_evidence`: valid raw/parsed `weak`.
3. `grade_web_evidence`: valid `weak` in the existing web-disabled fallback path.
4. `rewrite_query`: completed; the existing bounded retry ran.
5. `grade_kb_evidence`: valid raw/parsed `good`.
6. `generate_from_kb`: completed; generation kind `llm_kb`.

The final generated answer covered modernization priorities with banking-document page-2/page-3 citations. First deterministic grounding flagged one table-header line (`| Modernization Priority | Supporting Citation |`). The existing bounded repair removed it; final verification reported **9 claims, 0 unsupported, grounded=true**. The resulting markdown table lacks that header; this is recorded, not repaired through application changes in this task. Smoke latency: **11.913 seconds**. No smoke provider errors occurred.

All smoke gates passed, including actual routing/grading/generation calls, final grounded answer, vector-only effective mode, and provider identity. Corpus parity and namespace fingerprints were checked again before starting the full benchmark.

## Partial benchmark

The unchanged frozen order started `semantic_modernization` in `vector_only`, then `graph_only`; `hybrid` was not reached.

| Mode | Completed / planned | Result |
| --- | ---: | --- |
| vector_only | 1 / 16 | Case completed; retrieval/generation evidence recall and provenance coverage 1.0; route/tool/mode/grounding checks passed. Latency 8.405 seconds. |
| graph_only | 0 / 16 | First case aborted on provider error before a final case result. |
| hybrid | 0 / 16 | Not started. |

Exact error: case `semantic_modernization`, requested mode `graph_only`, stage **`grade_web_evidence`**, provider `groq`, model `openai/gpt-oss-120b`, error `RateLimitError`, **HTTP 429**. The machine-readable failure explicitly has `retrieval_quality_failure=false`. Partial data was preserved; no subsequent execution, Gemini fallback or judge was invoked. The aborted graph case is not assigned a retrieval-quality score.

The one completed vector case is descriptive only: it cannot establish vector/graph/hybrid superiority or a paired aggregate. Full capture: **1/48**, `capture_valid=false`.

## Corpus and source integrity

- Chroma: frozen `evals/ragas_workspace/20261005-public-pilot/vectorstore`, collection `rfp_kb_v2`, **11 documents / 54 chunks**, including synthetic public `eval_target_rfp.pdf`.
- Chroma digest: `641e9d4dad22e33140475ac40b0245dde1edb3ed6e2acc0c5fa8e75919d9d39f`.
- Neo4j evaluation corpus: `rfp-eval-frozen-641e9d4dad22`; active version `d99fc9a7b295f7c9c14cae3de266e0fa97af8dc9dc806ce164fc5a5482d8ce55`.
- Exact document/chunk manifest parity passed before smoke, after smoke/before capture, after capture, and at final verification. Both evaluation and live `internal-rfp` node/relationship fingerprints remained unchanged. No publication or ingestion ran.
- Protected application source/configuration, frozen question files and benchmark lock had identical byte hashes before and after.

## Artifacts and checks

Directory: `evals/results/matched-groq-20261006/`.

- `smoke.json`: exact final answer, actual generation contexts, grades, provenance, full grounding/repair traces and smoke checks.
- `application_calls.json`: provider/model/stage/status/latency for every LLM call, raw grader values, parsed node updates and HTTP error; no prompts or credentials in this audit.
- `capture.json`, `capture-vector_only.json`: preserved partial matched capture.
- `environment_before_smoke.json`, `environment_after_smoke.json`, `environment_after_run.json`: machine-checkable manifest parity.
- `corpus_fingerprints.json`, `protected_files.json`: unchanged corpus/source checks.
- `run_summary.json`: consolidated completion, per-mode deterministic results, provider failure and integrity metadata.

Evaluation-only additions: `evals/groq_matched_capture.py` and mocked `tests/test_groq_capture.py`. Focused runner/compatibility/parity tests: **45 passed**; Ruff and compilation passed. No new full-suite regression claim is made. The live runner genuinely executed and returned exit code **1**, indicating the incomplete/provider-failed capture.

Stop after the 429. Quota-safe completion of the remaining benchmark is still required before interpreting retrieval-mode comparisons.
