# semantic_modernization: Gemini grader diagnostic

Date: 2026-10-06. Scope: one unchanged frozen public case, `vector_only`, no RAGAS judge and no 48-execution benchmark.

## Finding

**Evidence sufficiency: sufficient. Missing expected concepts: none.**

The first KB grading attempt received four banking-document chunks with relevance scores 0.701, 0.625, 0.590 and 0.578 (pages 2, 2, 3 and 1 respectively). All passed the unchanged 0.50 pre-filter. Two other raw candidates were filtered out. The actual model prompt exactly matched the captured question and grader evidence; its length was 2,261 characters, well below the existing budget.

The decisive page-2 objectives chunk was `c43db1e84652a64b0f448d6ff9edf32eb112a6d5251476827c1d27685f6111cc`, from `01_Banking_Sector_Digital_Audit_2024.pdf`:

| Expected concept | Verbatim evidence actually supplied to the grader |
| --- | --- |
| Digital maturity | “Benchmark digital maturity against industry peers” |
| Data governance | “Evaluate data quality, lineage, and governance across 14 business units.” |
| Peer benchmarking | “Benchmark digital maturity against industry peers” |
| Prioritized modernization roadmap | “Deliver a prioritized transformation roadmap with ROI projections.” |
| ROI projections | “ROI projections” |

The executive-summary chunk also explicitly mentions digital transformation maturity and data governance. The reference's modernization roadmap is supported by the document's prioritized transformation roadmap; no external facts were needed.

## Actual grader output

- Provider/model: `google_genai` / `gemini-3.8-flash`, grading temperature 0.0.
- Raw output: `{"grade": "weak"}`.
- Parsed grade and application `kb_grade`: both `weak`.
- Finish reason: `STOP`; no parser error or prompt-budget rejection was observed.
- Rationale/reason: **not returned**. The unchanged prompt/schema requests only a grade; the model's internal reason cannot be established from this response.
- Prompt SHA-256: `b7c357b90735f072cdc27a660f90a11b35d97afb7d1f413feb3a3a0650d6b8a1`.
- Evidence SHA-256: `abfe0f2e23cb9663c2edabb86bbb98062e56d3c8b4f1700d082be956f1bd534c`.

**Root-cause classification for the completed grading attempt: PROVIDER/GRADER COMPATIBILITY, not RETRIEVAL EVIDENCE FAILURE.** This identifies an evidence-sufficiency/grade disagreement for this case, not a proven general Gemini defect or an explanation of its internal reasoning.

## Retry limitation: external failure

Original and first retrieval query: “Summarize the modernization priorities identified by the banking digital audit.”

The application proceeded from weak KB grading to its existing web-grade/rewrite path. It then stopped on **HTTP 429 / GoogleRateLimitError at `rewrite_query`, provider `google_genai`, model `gemini-3.8-flash`**. The rewritten query was not produced, and the second retrieval/KB grading attempt did not execute.

Consequently, the second grade, second prompt/evidence hashes and identical-versus-different evidence comparison are **unavailable**, not assumed. The rate limit is a provider execution failure, not a retrieval-quality failure. No further external calls were started after it.

## Integrity and artifacts

- Frozen Chroma: 11 documents / 54 chunks, including synthetic public `eval_target_rfp.pdf`; digest `641e9d4dad22e33140475ac40b0245dde1edb3ed6e2acc0c5fa8e75919d9d39f`.
- Matching Neo4j namespace: `rfp-eval-frozen-641e9d4dad22`; exact manifest parity passed before and after the run. Graph retrieval was not invoked in this vector-only diagnostic.
- Frozen Chroma, evaluation graph, live `internal-rfp`, question-set/lock files and evaluation expectations remained unchanged. No ingestion, RAGAS judge or full benchmark ran.
- Complete exact candidate chunks, grader-selected chunks, metadata, scores, model response and concept witnesses: `evals/results/semantic-grader-diagnostic-20261006-rerun/grading_trace.json`.
- Machine-readable diagnosis and external failure: `evals/results/semantic-grader-diagnostic-20261006-rerun/diagnostic.json`; manifest comparisons: `environment_before.json` and `environment_after.json` in that directory.
- An earlier capture stopped after the first raw grade because its diagnostic writer encountered a Windows permission error. It remains preserved in `evals/results/semantic-grader-diagnostic-20261006/`. Only the diagnostic recorder was hardened; no application logic was changed. The rerun recorded no writer errors.
- Diagnostic/runner tests: **40 passed**. Ruff and compile checks passed. These are focused checks, not a new full-suite regression claim.

No grader adjustment or retrieval change was made. Stop here: retry evidence comparison remains blocked by the external rate limit.
