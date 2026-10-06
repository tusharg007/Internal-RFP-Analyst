# Static evidence-grader compatibility probe

Date: 2026-10-06. **Result: incomplete because of provider quota/backoff; no model compatibility conclusion was established.**

The probe read the exact saved first-attempt evidence from `evals/results/semantic-grader-diagnostic-20261006-rerun/grading_trace.json`. It reconstructed the existing grader prompt and verified its SHA-256 against the saved prompt hash before any external call. No retrieval, routing, rewrite, answer generation, grounding, graph/corpus access, RAGAS judge or full benchmark was executed.

## Availability and outcomes

A fresh `models.list()` call using the configured Google API key confirmed all four exact model names below with `generateContent` support. Listing availability does not guarantee generation quota.

| Exact listed model ID | Static requests | Provider/HTTP result | Raw structured response / parsed grade | Classification |
| --- | ---: | --- | --- | --- |
| `models/gemini-3.1-pro-preview` | 1 | HTTP 429, `GoogleRateLimitError`, `RESOURCE_EXHAUSTED`; 0.602 seconds | None: provider rejected the request | unavailable/provider-error |
| `models/gemini-3.7-flash` | 0 | Not called: quota-safety stop | Not obtained | unavailable/provider-error — not evaluated |
| `models/gemini-3.6-flash` | 0 | Not called: quota-safety stop | Not obtained | unavailable/provider-error — not evaluated |
| `models/gemini-3.8-flash` (control) | 0 | Not called: quota-safety stop | Not obtained | unavailable/provider-error — not evaluated |

The Pro error recorded per-minute and per-day free-tier token/request quota violations for quota model `gemini-3.1-pro`, and an explicit `RetryInfo.retryDelay` of **5,856 seconds (97 minutes 36 seconds)**. The probe made no retry and conservatively stopped remaining calls rather than assume switching models would safely bypass this guidance. This is not evidence that the three Flash models lack availability or have the same exhausted quota. [Google's rate-limit guidance](https://ai.google.dev/gemini-api/docs/rate-limits) and [troubleshooting guidance](https://ai.google.dev/gemini-api/docs/troubleshooting) distinguish quota exhaustion from response quality and advise backoff for transient failures.

## Input/contract identity

- Prompt SHA-256: `b7c357b90735f072cdc27a660f90a11b35d97afb7d1f413feb3a3a0650d6b8a1`.
- Evidence SHA-256: `abfe0f2e23cb9663c2edabb86bbb98062e56d3c8b4f1700d082be956f1bd534c`.
- Every model record contains those identical input hashes. **Only one request actually ran**, so this is not a completed four-model comparison.
- Existing `EvidenceGrade` schema: `good` / `weak`; existing `json_mode`, temperature 0.0 and maximum output tokens 2,048. Raw-response capture changes only the local parser wrapper, not the HTTP prompt/schema.
- Automatic transport retries were disabled for this probe; each available model could receive at most one grader request.
- `good` would mean compatibility with this single sufficient-input grading contract, not proof of answer quality. `strong` is not a valid value under this schema.

## Preservation and verification

`GEMINI_MODEL` remains `gemini-3.8-flash`. Saved trace bytes and hashes of application source/configuration, frozen question files and benchmark lock matched before and after. No saved evidence, expectations, corpus or application logic was edited.

Machine-readable report: `evals/results/static-grader-compatibility-20261006.json`, including fresh model metadata, exact input, hashes, safe quota/backoff fields, latency, null responses/grades and explicit not-called statuses. Earlier diagnostic results remain separate; the prior control's `weak` response is not substituted for a new probe result.

Operational implementation: `evals/static_grader_probe.py`; mocked tests: `tests/test_static_grader_probe.py`. **14 tests passed; Ruff and compilation passed.** No live calls are made by those tests. No full-suite result is claimed for this task.

Stop here. The remaining three tests need a later quota-safe run; no model change or 48-execution benchmark was started.
