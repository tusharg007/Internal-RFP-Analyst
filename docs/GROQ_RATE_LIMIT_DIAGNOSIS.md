# Groq rate-limit diagnosis: historical dimension unknown

Date: 2026-10-06. **No pacing or rerun is justified by the preserved evidence yet.**

## What the historical artifacts establish

Inspected `evals/results/matched-groq-20261006/run_summary.json`, `application_calls.json`, `capture.json` and associated reports. They establish:

- HTTP status: **429**.
- SDK error class: **RateLimitError**.
- Provider/model: **groq / openai/gpt-oss-120b**.
- Case/mode/stage: **semantic_modernization / graph_only / grade_web_evidence**.
- Completed full-capture executions: **1/48**; invalid for a paired retrieval comparison.

The original recorder discarded the SDK response body and HTTP headers. The API's body-level error type/code and every requested header below are **not recorded**, not inferred absent from the server:

| Requested field | Preserved value |
| --- | --- |
| response body / Groq body-level error type | unknown |
| retry-after | unknown |
| x-ratelimit-limit-requests | unknown |
| x-ratelimit-remaining-requests | unknown |
| x-ratelimit-reset-requests | unknown |
| x-ratelimit-limit-tokens | unknown |
| x-ratelimit-remaining-tokens | unknown |
| x-ratelimit-reset-tokens | unknown |

**Classification: unknown.** It is not possible to establish RPM, TPM, RPD, TPD, separate input/output token limits or another provider limit from these artifacts. Call counts, slow successful calls, SDK retry latency and public plan defaults do not prove the failed organization-specific limit.

[Groq's current documentation](https://console.groq.com/docs/rate-limits) specifies that request limit/remaining/reset headers refer to **RPD**, while token limit/remaining/reset headers refer to **TPM**. A short request-reset duration must not be mislabeled RPM. Public default limits are not evidence of this API key's actual exhausted quota.

## Evaluation-only changes

- `evals/groq_rate_limits.py`: offline inspection and safe quota telemetry; no provider calls or waits. Classifies only explicit provider quota clauses, leaving absent/conflicting dimensions unknown.
- `evals/provider_guard.py`: preserves Groq 429 telemetry before its existing fail-fast abort, even if the later audit error callback is skipped.
- `evals/groq_matched_capture.py`: copies that telemetry into the application-call audit.
- `tests/test_groq_rate_limits.py`: mocked SDK responses covering explicit dimensions, missing/conflicting data, header semantics, secret rejection, wrapper handling and preservation before abort.

Future 429 telemetry retains only the seven whitelisted rate-limit headers, SDK status, safe body-level type/code and a matched quota clause. It excludes organization IDs, credentials, arbitrary exception messages, prompts, request headers and document text. This cannot retroactively recover the missing historical response.

No application-level logic, retry/fallback, prompt, model, retrieval, grading, grounding, RAGAS configuration, corpus or frozen expectation changed. No pacing implementation was added because the required short-window diagnosis is unproven. No sleeps or retry loops were introduced. Neither new smoke case nor a 48-execution attempt ran. Additional Groq/Gemini/judge calls: **0**.

## Verification and next prerequisite

The offline command genuinely executed:

```powershell
.\.venv\Scripts\python.exe -m evals.groq_rate_limits --source-dir evals/results/matched-groq-20261006 --output evals/results/groq-rate-limit-diagnosis-20261006.json
```

Machine-readable result: `evals/results/groq-rate-limit-diagnosis-20261006.json`. Original report byte hashes matched before/after; the partial results were not rewritten or interpreted as quality scores. Previously verified 11-document/54-chunk parity and unchanged Neo4j evaluation/live fingerprints are recorded as **historical** checks, not a new live-corpus verification.

Focused tests: **71 passed**. Ruff and compilation passed. No full-suite regression result is claimed for this telemetry-only task.

Before proceeding, obtain the original 429 body/headers or provider quota-usage evidence identifying the exhausted window. If it proves RPM/TPM, implement evaluation-only proactive pacing and the consecutive two-case gate before another unchanged 48-run capture. If it proves RPD/TPD exhaustion, do not add sleeps/retries: wait for the provider quota reset or use a higher-quota Groq organization/project/plan. Until then, stop rather than guess.
