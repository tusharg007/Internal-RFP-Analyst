# Gemini-only matched public capture attempt — 2026-10-06

**Outcome: stopped on the one-case provider smoke test. The full 48-execution
benchmark was not started. No deterministic retrieval-quality comparison is
valid from this attempt.**

## Provider selection

The existing application already supports Google through
`ChatGoogleGenerativeAI`. `get_api_keys()` resolves Streamlit secrets, process
environment and `.env`; each existing factory prioritizes Groq over Google.
There is no existing preferred-provider flag.

For this evaluation process only, `GROQ_API_KEY` was masked with whitespace.
Unlike an empty value, whitespace prevents the resolver from falling through
to the `.env` Groq key and is then stripped to an empty key. Actual resolved
keys were checked before execution: Groq absent, Google present. Streamlit
secret precedence was also checked; a defeating Groq secret would stop setup.
No real key was printed, exported or written. All overrides are restored when
the evaluation process exits; `.env` and application defaults were not edited.

The generation, router, grader and rewriter factories all selected:

- Class: `ChatGoogleGenerativeAI`.
- Existing configured model: **`gemini-2.0-flash`**.
- Grading/routing temperature: existing `0.0`.
- Generation/rewriting temperature: existing `0.3`.

Provider selection is recorded in
[`provider_selection.json`](../evals/results/matched-gemini-20261006/provider_selection.json).
Grounding verification and bounded repair are existing deterministic operations;
they do not have a separate LLM provider. They would operate on the selected
application provider's generated answer if generation were reached.

## Corpus verification

- Frozen Chroma index:
  `evals/ragas_workspace/20261005-public-pilot/vectorstore`.
- Collection: `rfp_kb_v2`.
- Documents/chunks: **11 / 54**, including public synthetic
  `eval_target_rfp.pdf` with its original `upload` origin.
- Indexed digest:
  `641e9d4dad22e33140475ac40b0245dde1edb3ed6e2acc0c5fa8e75919d9d39f`.
- Neo4j evaluation corpus: **`rfp-eval-frozen-641e9d4dad22`**.
- Graph version:
  `d99fc9a7b295f7c9c14cae3de266e0fa97af8dc9dc806ce164fc5a5482d8ce55`.

Exact document/chunk property and provenance-edge parity passed both before
and after the smoke attempt. The namespace was reused **without publication
or ingestion writes**. Frozen Chroma's logical digest remained unchanged.
The complete evaluation graph's nodes and relationships were additionally
fingerprinted before and after the smoke; both hashes remained identical.

The live `internal-rfp` corpus also remained identical: same active version,
365 nodes and 569 relationships, with the same complete node/relationship
fingerprints as the previous matched-environment verification.

The exact manifest and comparison are in
[`environment.json`](../evals/results/matched-gemini-20261006/environment.json).
Actual before/after namespace fingerprints are in
[`corpus_fingerprints.json`](../evals/results/matched-gemini-20261006/corpus_fingerprints.json).
No private corpus content was supplied to an application model.

## One-case smoke result

The smoke used the **first existing frozen case**, not a case selected after
looking at model outputs:

| Field | Observed result |
|---|---|
| Case | `semantic_modernization` |
| Query | Summarize the modernization priorities identified by the banking digital audit. |
| Mode | `vector_only` |
| Failure node | **`route_question`** |
| Actual provider | **`google_genai`** |
| Actual model | **`gemini-2.0-flash`** |
| Error | **`GoogleModelNotFoundError`** |
| HTTP status | **404** |
| Pipeline latency | 1.234837 seconds |
| Grading | Not reached |
| Generation | Not reached |
| Grounding verification | Not reached |
| Groq calls | **0** |
| RAGAS judge calls | **0** |

The configured model could not be found/accessed by this Google API call.
The error class/status does not independently establish whether the underlying
cause is retirement, endpoint support or account/model availability. No
replacement model was guessed, configured or attempted after the failure.

The evaluation-scoped stop guard prevented the existing router fallback from
hiding this provider error. Google wraps numeric API errors in classified
LangChain exceptions; status reporting now follows their bounded exception
cause chain, without reading or exporting exception text. This reporting-only
change does not alter application fallback, router or grader policies.

The failure is explicitly marked `retrieval_quality_failure: false`, with no
quality score assigned. The completed-stage callback audit confirms one Google
application invocation and no Groq invocation. Smoke outputs and safe model
metadata are preserved in
[`smoke.json`](../evals/results/matched-gemini-20261006/smoke.json) and
[`application_calls.json`](../evals/results/matched-gemini-20261006/application_calls.json).

## Full benchmark and target-case status

The smoke did not pass, so the benchmark gate correctly did not open.

| Mode | Planned | Completed | Deterministic aggregates |
|---|---:|---:|---|
| vector_only | 16 | 0 | Not run; unavailable, not zero scores |
| graph_only | 16 | 0 | Not run; unavailable, not zero scores |
| hybrid | 16 | 0 | Not run; unavailable, not zero scores |
| Total | **48** | **0** | **No valid comparison** |

All 16 benchmark cases are unexecuted in all three modes. This includes the
banking fact case (`factual_banking_duration`), comparison cases, uploaded
target matching (`matching_uploaded_controls`) and clinical matching
(`matching_clinical_requirements`). The failed smoke is kept outside the
benchmark and is not counted as a completed case or retrieval-quality miss.

[`run_summary.json`](../evals/results/matched-gemini-20261006/run_summary.json)
contains provider/model metadata, corpus manifests, the exact external failure,
completion counters and every frozen case's per-mode `not_run` status with
null benchmark/aggregate values. Previous Groq reports remain preserved.

## Changes and verification

Only operational evaluation support and reporting were added/updated:

- `evals/gemini_matched_capture.py`: process-local selection of existing Google
  factories, one-case smoke gate, safe provider-call audit, existing matched
  benchmark delegation, read-only integrity checks and artifact summary.
- `evals/provider_guard.py`: bounded extraction of numeric HTTP status through
  SDK exception wrappers, for accurate Google failure reports.
- `tests/test_gemini_capture.py`: 10 offline tests for configuration precedence,
  restoration, wrapped error statuses, redacted audit and blocking unexpected
  Groq before transport.
- This document and a fresh `evals/results/matched-gemini-20261006` directory.

No retrieval, routing, prompts, grader logic, grounding policy, graph ingestion,
frozen cases, corpus data, application configuration or expectations were
changed. No judge was instantiated. The new launcher delegates all benchmark
execution to the existing matched coordinator; it is not a new application
provider abstraction or an alternative scoring implementation.

Executed checks:

| Command | Result |
|---|---|
| Existing generation/router/grader/rewriter factory inspection | All four selected Google and `gemini-2.0-flash`; no invocation during inspection |
| `python -m pytest tests/test_gemini_capture.py tests/test_matched_environment.py -q` | **29 passed** |
| `python -m evals.gemini_matched_capture --output-dir evals/results/matched-gemini-20261006` | Exit **1**, intentional provider stop; no full benchmark started |
| `python -m pytest -q` | **534 passed, 5 skipped**, 105.61 seconds |
| `python -m ruff check .` | Passed |
| Compile checks for the launcher, error guard and tests | Passed |
| Saved-artifact summary generation | Completed locally; no model/retrieval calls |

The saved-artifact summary helper was added after the full-suite execution;
it was then compile/lint checked and executed against the actual saved run.
No smoke or application invocation was repeated after the HTTP 404.

## Remaining blocker

An accessible Google application model must be configured before another
smoke/capture attempt. The current task's stop-on-provider-error instruction
was honored: no model migration or further provider calls were attempted.
Once a model change is separately approved, use a **new output directory**:

```powershell
.venv\Scripts\python.exe -m evals.gemini_matched_capture --output-dir evals/results/matched-gemini-next-run
```

That command again gates the unchanged 48-execution capture on one successful
public smoke, refuses existing output directories, performs no ingestion, and
does not enable a semantic judge. Per-mode results become interpretable only
after 48 executions complete without provider/pipeline errors and final corpus
parity/integrity checks pass.
