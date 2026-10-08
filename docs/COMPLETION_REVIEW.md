# Completion review — 2026-10-08

Branch: `feature/graphrag-neo4j`. This closes a bounded set of reproducibility,
presentation, dependency and evaluation-safety gaps. It is **not a claim that all
production-readiness gates are closed**.

## Gaps addressed

| Gap | Smallest supported correction | Regression coverage |
| --- | --- | --- |
| Graph snapshot loading and normal vector loading disagreed on Chroma settings, causing startup failure for the same persistent directory | Align vector client telemetry settings with the existing snapshot/evaluation loader; no retrieval/ranking changes | Real temporary Chroma collection loaded through both paths; original digest preserved |
| Windows Git CRLF checkout falsely appeared to change the frozen question bytes | Normalize only CRLF to the original LF representation in frozen text hashes; retain parsed hashes and source witnesses | LF/CRLF equality, changed content rejected; no YAML/lock edits |
| Trace cards interpolated unescaped query/source text into HTML | HTML-escape variable text | Script/image-tag injection checks |
| No concise explanation of effective retrieval and provenance | Add a bounded, whitelisted operational explanation panel, preserving unknown/missing claim links | Mode/fallback, witness IDs, modality, privacy, bounds and legacy adapter coverage |
| Public screenshots risked reading private root uploads or modifying a corpus | Explicit frozen-public-index guard, process-local demo paths and disabled document mutations | Wrong digest rejected before app imports; read-only Streamlit smoke |
| Google judge client constructed successfully but could not prepare structured requests | Include `instructor[google-genai]` in optional evaluation dependencies (provides `jsonref`) | Mocked transport-boundary test prepares a real structured request, without network |
| Judge HTTP errors could trigger subsequent metrics/cases | Evaluation-only fail-stop/latch, numeric HTTP status, checkpoint and unrun work | Quota stop across metrics/cases; comparison does not generate remaining modes |
| No fresh, auditable visual walkthrough | Capture actual public UI and retain source trace; add public post-hoc judge and integrity-receipt tools | Exact capture/reference validation; parity mismatch rejection |

No application provider/fallback logic, routing, prompts, relevance thresholds,
grounding policy, graph extraction/ingestion, Cypher templates, retrieval ranking,
frozen expectations or corpus contents were changed. The vector-store change is
connection-settings compatibility, not a retrieval-quality adjustment. RAGAS
dependency changes apply only to optional evaluation installation.

## Commands actually executed

Python commands used the existing virtual environment. The workspace's reused
pytest temp directory was inaccessible on one attempt (13 fixture setup errors,
75 tests passed). Subsequent runs used a **new** isolated `--basetemp` and disabled
the inaccessible cache; no tests were skipped to hide those setup errors.

| Command | Result |
| --- | --- |
| `python -m pytest --basetemp=.capture_runtime/pytest-final-2 -p no:cacheprovider -q` | **641 passed, 5 skipped, 0 failed** |
| Focused RAGAS/public-probe/explanation tests with a fresh temp directory | 67 passed before the final additional parity/comparison tests |
| Python `py_compile` source sweep | Passed; excludes ignored runtime SQLite/index directories |
| `python -m ruff check .` | Passed |
| `python -m pip check` | No broken requirements |
| `python -m evals.run_evals` | 3/3 deterministic mock smoke; not semantic accuracy |
| `python -m evals.run_kb_evals` | 9/9, retrieval-only; real public PDFs/Chroma in a separate temporary evaluation index |
| `python -m evals.matched_environment --output-dir .capture_runtime/parity-before` | Exact parity passed; existing graph namespace reused without writes |
| `python -m evals.matched_environment --output-dir .capture_runtime/parity-after` | Exact parity passed; protected namespace fingerprints unchanged |
| `python tools/verify_walkthrough.py --before … --after … --output …` | Public receipt generated; checks before/after immutability, not only counts |
| `python tools/capture_walkthrough.py --submit` against the guarded local demo | Four actual public UI screenshots; real Groq generation/grounding, vector fallback |
| Strict `HybridRetrievalProvider.retrieve(...)` public Azure read, no LLM | Graph projection timeout; zero paths/chunks; **not a graph success** |
| `python tools/judge_public_smoke.py --input … --output … --allow-judge` | First: optional-dependency configuration failure. After correction: Google HTTP 503 on Faithfulness; later metrics unrun |
| `git diff --check` | Passed |

The five local skips are opt-in real Neo4j integration tests, not database-level
RBAC verification. The repository's CI includes a separate disposable Neo4j job
and optional evaluation-contract job; hosted results must be checked on the
pushed commit independently of these local results.

## Actual public execution

Question: `Which projects used Microsoft Azure?`

```text
fresh public UI
→ KB route / all-document scope
→ requested graph_only
→ bounded Neo4j read unavailable
→ existing vector_only fallback / original Chroma evidence
→ KB grade good
→ Groq openai/gpt-oss-120b generation
→ grounding_verifier
→ final_grounding_verifier
→ source-cited answer
```

Both grounding checks reported five checked claims and zero unsupported claims.
That is the existing deterministic verifier's observation, **not formal semantic
entailment**. The visible response distinguishes proposed healthcare work from
reported outcomes. Source citations and traces are retained in
[capture.json](assets/walkthrough/capture.json). The screenshot explanation makes
the vector fallback explicit; graph results were not fabricated for presentation.

## Public corpus and protected live corpus

The frozen corpus remains 11 documents / 54 chunks, with public synthetic
`eval_target_rfp.pdf`. Neo4j namespace: `rfp-eval-frozen-641e9d4dad22`.
Version: `d99fc9a7b295f7c9c14cae3de266e0fa97af8dc9dc806ce164fc5a5482d8ce55`.
The original manifest's document/chunk properties and edges matched before and
after. Complete protected `internal-rfp` namespace fingerprints were unchanged.
Neither existing corpus was republished, deleted or reingested.

[Machine-checkable receipt](assets/walkthrough/corpus-integrity.json).
The receipt intentionally omits private records, API keys, Aura addresses and
machine-specific index paths.

## RAGAS: real attempt, incomplete measurement

The user confirmed free-tier/no-billing projects. Only the existing configured
Google judge was used; there was no paid upgrade, application-model change or
automatic judge-provider substitution. Free-tier quota is still provider-managed.

The saved matched `semantic_modernization` vector-only output was judged post-hoc.
Its exact original contexts, final answer and independent frozen reference were
preserved. No retrieval or generation was repeated. Source capture revision and
package metadata are recorded separately from current judge versions; the older
capture is not represented as a current-commit benchmark.

Google `models.list()` confirmed the configured `gemini-3.8-flash` ID. Following
the structured-output dependency fix, the Faithfulness request returned HTTP 503
(`InstructorRetryException`, numeric provider status preserved). No aggressive
retry was performed. Answer Relevancy, Context Precision, Context Recall and
Answer Correctness were marked `not_run`; all semantic scores remain null.

[Initial dependency result](assets/walkthrough/ragas-smoke.json) ·
[Post-fix provider result](assets/walkthrough/ragas-smoke-after-dependency-fix.json).

This attempt is not a baseline and cannot establish regression thresholds or a
retrieval winner. The unchanged full 48-execution benchmark was **not launched**
in this task. Its known provider-quota/reset gate and graph-runtime reliability
gate remain prerequisites; partial historical captures are not republished as
quality metrics.

## README quantitative claims and their evidence

| Claim | Repository support |
| --- | --- |
| Package `0.1.0`; Python 3.11+ | `pyproject.toml` |
| 641 passed / 5 skipped / zero failed | Executed full-suite command above; 28 added regression cases relative to the prior 613-pass suite |
| Offline 3/3 | Executed `evals.run_evals`; deterministic mock cases in that module |
| Real-KB 9/9 | Executed `evals.run_kb_evals`; original nine cases in `golden_questions.yaml` |
| Public 11 documents / 54 chunks | `assets/walkthrough/corpus-integrity.json`; original frozen manifest digest |
| 16 frozen cases / 48 planned executions | `evals/retrieval_questions.yaml`, existing lock and three-mode protocol; **planned, not completed** |
| Five checked claims / zero unsupported in each UI check | Actual `assets/walkthrough/capture.json` and grounding screenshot |
| One bounded query retry / one repair (where described) | `MAX_QUERY_RETRIES` in config and existing graph/verifier implementation |
| Gemini HTTP 503; four later metrics unrun | Actual post-fix RAGAS report; no semantic score claimed |

Model IDs are configuration identifiers, not measured quality claims. Source
page numbers in answers identify original evidence; they are not invented
benchmark results. No superiority, semantic pass rate or latency improvement is
asserted.

## Remaining release gates—explicitly not closed

1. **Cloud graph projection reliability.** Exact manifest parity succeeded, but
   this query missed the unchanged bounded runtime read deadline. Diagnose query
   round-trip latency, service/network conditions and supported bounds separately
   before declaring live graph-only success. No retrieval redesign was made here.
2. **Complete semantic baseline and ablation.** The judge returned HTTP 503;
   application quota has previously blocked capture. Wait for provider recovery
   and an explicitly confirmed quota reset. Collect complete matched runs before
   calibration or winner claims; keep public/private corpora separated.
3. **Neo4j server-enforced least privilege.** READ sessions and fixed templates
   are not proof of database-level RBAC. Independently verify denied writes for
   deployment reader credentials. Do not publish credentials as evidence.
4. **Production authentication, ACLs, retention and hard erasure.** Retrieval
   scope/corpus namespaces are not tenant isolation. Historical snapshots require
   an explicit retention/erasure design; that is not a safe ad-hoc cleanup.
5. **Claim-level lineage and semantic assurance.** Recorded page/chunk/path
   witnesses do not structurally bind every final claim. The current verifier is
   heuristic; it does not prove semantic entailment.

These are meaningful operating/architecture decisions or external-service gates,
not gaps that can honestly be closed by changing README language or adjusting
frozen expectations. They remain visible rather than being hidden behind a
successful fallback screenshot.
