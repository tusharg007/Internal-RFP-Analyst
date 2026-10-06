# GraphRAG retrieval evaluation

## Scope and integrity

This is an exploratory, closed-corpus ablation of `vector_only`, `graph_only`,
and `hybrid`, not a demonstration designed to make GraphRAG win.

The independently authored extension, `evals/retrieval_questions.yaml`, contains
16 cases: two each for semantic single-hop, exact factual document, relationship,
multi-hop, cross-document comparison, requirement matching, unsupported/no-answer,
and ambiguous queries. The original nine cases in `evals/golden_questions.yaml`
are unchanged and remain part of the existing regression evaluations. Their web
fallback tests are not repurposed as closed-KB retrieval comparisons.

Each new case includes an expected optimal mode and a rationale written before
scores. These are hypotheses, not labels declaring the eventual winner. Prose,
outcome modality, and fields outside the current graph ontology are deliberately
represented alongside graph-friendly connected questions. Multi-hop cases require
two source witnesses for a Project → Technology ← Project join, not merely several
filters on a single project.

`evals/retrieval_benchmark.lock.json` was created before any execution. It freezes
case-file bytes, canonical case expectations, the original golden-file hash, the
actual indexed corpus fingerprint, and source/page/anchor/chunk witnesses. The
runner refuses to overwrite a changed freeze. It also checks cases, corpus, and
implementation version during execution. No cases were changed after scoring.

Public corpus: ten generated case-study/proposal PDFs plus the public
`eval_target_rfp.pdf` fixture, 54 indexed chunks across 11 documents. No private
uploads, credentials, or internal prompts are exported. Source pages are
one-based in the dataset and zero-based in application metadata; the scorer
converts explicitly.

Frozen case hash: `ba097a4fed2d707445f0362eb8a26220e41ac1cf571edfb3fde1aee302b56752`.

Frozen corpus hash: `641e9d4dad22e33140475ac40b0245dde1edb3ed6e2acc0c5fa8e75919d9d39f`.

## Execution protocol

The matrix executes the actual application once for every case and mode: 48
executions on precisely the same cases and index. Modes rotate their execution
order by case index. There is no answer cache and no second retrieval for semantic
judging. Tavily is disabled equally across modes to avoid external evidence
confounding a closed-corpus comparison. The ordinary application behavior is
unchanged. Graph-only uses strict no-vector fallback; hybrid fallbacks are recorded
as failure to exercise the requested backend, not as hybrid quality gains.

Generation uses the existing `openai/gpt-oss-120b`, temperature 0.3; control
temperature is 0.0. FastEmbed uses `BAAI/bge-small-en-v1.5`. Every report records
package versions, source-tree hash, retrieval budgets, model settings, prompt hash,
actual outputs, exact generation contexts, and execution events. Provider timing,
shared embedding initialization, and one repetition make latency exploratory.

## Metrics and interpretation

- Retrieval success: all independently verified source/page/anchor clauses appear
  in the actual accepted retrieval records. A filename match alone is insufficient.
  This is strict gold-witness coverage, not proof that no alternative passage
  could answer the question. Semantic context recall is a separate measure.
- Generation evidence recall: those same clauses survive into the exact original
  text supplied to generation; this is separate from retrieval success.
- Provenance coverage: supplied chunk identities match the frozen source, page,
  origin, and optional evidence ID. Empty evidence has no provenance score.
- Answer grounding: the existing final verifier result for evidence-generated
  answers, not an invented semantic entailment check or claim-level provenance graph.
- Route/tool correctness: existing deterministic expectations. Exact numeric probes
  exclude citations and prevent substring matches such as 9 versus 19.
- Optimal-auto-route correctness: independent pre-score hypothesis versus the
  current auto planner. This is not the same as compliance with a forced ablation
  mode; the experiment itself does not run an auto-routed deployment.
- Expected behavior: answer, abstain, or clarify. Unsupported and ambiguous cases
  do not receive perfect semantic scores for empty contexts.
- Graph path correctness: complete provenance witnesses must survive generation.
  No observed graph paths means this check is not evaluated, not passed.
- RAGAS: faithfulness, answer relevancy, context precision, context recall, and
  answer correctness where a reference exists. Missing, inapplicable, timed-out,
  or failed scores remain distinguishable and are never replaced with zero or one.
- Latency: wall-clock pipeline time, including grading, generation, and repair,
  excluding judge time and deliberate between-case pacing.

Class aggregates retain evaluated/total counts. Averages over partially scored
cases are not comparable to fully scored means. No semantic thresholds or combined
weighted score are introduced. Deterministic "better" categories require all
requested backends to operate and Pareto dominance over jointly available quality
dimensions. Semantic comparisons require complete paired scores for that metric.
Trade-offs and missing-backend cases are inconclusive. Ties do not prove equivalent
retrievers, and clarification bypasses do not test retrieval accuracy.

## Availability boundary

At experiment start Neo4j was disabled, no endpoint/read account was configured,
and neither Docker nor Java was available locally. Real Neo4j graph quality cannot
be measured in this environment yet. An in-memory graph or mock is deliberately
not substituted for production graph execution. Graph-only declines and hybrid
vector fallbacks test safe degradation, not GraphRAG performance.

The judge provider/model are not configured in the persistent environment.
Generation capture and semantic scoring are separate operations; the captured
matrix must not be represented as a completed semantic baseline.

The graph-ingestion dry run validated this exact corpus without writes:
11 documents, 54 chunks, 80 entities, 171 assertions, 327 persistence rows.
Validated extraction version:
`35d5004db07fd0441bbec0cc95490b492a3dc315bbd7d8721a050cb6855dad7f`.
This is an extraction/schema check, not a Neo4j retrieval benchmark.

## Results

The initial 48 executions completed without pipeline exceptions. The original
report is retained in `evals/results/retrieval_comparison-pre-fix.json`. Neo4j was
not exercised. Fourteen three-way comparisons were inconclusive; the two
clarification cases bypassed retrieval in every mode, adding no retrieval benefit.
Neither vector, graph, nor hybrid superiority was established.

The initial vector run retrieved all required evidence clauses in 6 of 12
answerable cases. Graph-only retrieved none; these are declines with disabled or
unsupported graph plans, not measured graph recall. Hybrid retrieved the same
six complete evidence sets through vector fallback, not graph evidence. The
grounding verifier passed 7/8 vector evidence-generated outputs and 8/8 hybrid
fallback outputs. Citation identity coverage averaged 0.625 and 0.75 respectively
over those eight outputs. These stochastic fallback-answer differences do not
demonstrate any graph benefit and are not semantic correctness scores.

### Objectively diagnosed implementation bug

`relationship_regulatory` was incorrectly routed to clarification in all modes.
The phrase "alongside it" triggered the generic conversation-reference check,
even though GDPR was named locally in the same self-contained question. This
prevented retrieval altogether, independently of Neo4j availability.

The resolver now recognizes narrowly bounded local relationship references
(alongside/associated with/related to/linked to) when a known technology or
framework precedes the sole deictic reference. It still requires history for
"Does it use Azure?", multiple unresolved references, unnamed "it", and ordinary
vague follow-ups. Nine positive/negative regression tests cover the change.
No query wording, reference answer, evidence anchor, expected mode, or dataset
hash was changed. The entire matrix is rerun with the same immutable lock;
implementation hashes distinguish the before/after reports.

### Final unchanged-set rerun

**Status: INCOMPLETE.** All 48 executions were attempted. Five pipeline executions
raised `RateLimitError` (three vector, two hybrid); those are retained rather than
silently retried or dropped. Fourteen cases still cannot support a valid three-way
conclusion because Neo4j was unavailable. The two clarification cases correctly
bypassed retrieval in every mode.

| Case | Class | Expected optimal | Vector supplied-evidence coverage | Graph coverage | Hybrid coverage |
|---|---|---|---:|---:|---:|
| semantic_modernization | semantic single-hop | vector_only | 1.00 | declined | 1.00 fallback |
| semantic_healthcare_security | semantic single-hop | vector_only | 1.00 | declined | 1.00 fallback |
| factual_banking_duration | exact factual | graph_only | 0.00 | declined | 0.00 fallback |
| factual_insurance_team | exact factual | vector_only | 1.00 | declined | 1.00 fallback |
| relationship_rpa | relationship | graph_only | 1.00 | declined | 1.00 fallback |
| relationship_regulatory | relationship | hybrid | 1.00 | declined | 1.00 fallback |
| multihop_shared_retail_pharma | multi-hop | hybrid | 1.00 | declined | 1.00 fallback |
| multihop_shared_banking_healthcare | multi-hop | hybrid | 0.50 | declined | 0.50 fallback |
| comparison_delivery_budget | cross-document | hybrid | 0.00 | declined | 0.00 fallback |
| comparison_outcome_modality | cross-document | vector_only | 1.00 | declined | 1.00 fallback |
| matching_uploaded_controls | requirement matching | hybrid | unknown: API error | declined | 0.00 fallback |
| matching_clinical_requirements | requirement matching | hybrid | unknown: API error | declined | unknown: API error |
| unsupported_private_contact | no-answer | vector_only | unknown: API error | declined | unknown: API error |
| unsupported_certification | no-answer | vector_only | N/A | declined | N/A fallback |
| ambiguous_missing_referent | ambiguous | vector_only default/bypass | N/A | N/A | N/A |
| ambiguous_missing_action | ambiguous | vector_only default/bypass | N/A | N/A | N/A |

Here coverage means independently declared evidence clauses in the **actual
generation contexts**, not an answer correctness score. Full accepted-retrieval
coverage is recorded separately in JSON. Graph declines are not assigned measured
graph accuracy. Hybrid's original-text evidence came from vector fallback.

| Measurement | vector_only | graph_only | hybrid |
|---|---:|---:|---:|
| Pipeline completed / attempted | 13/16 | 16/16 | 14/16 |
| Answerable cases with complete retrieved gold witnesses / all 12 answerable cases | 7/12 | 0/12 declines, invalid backend | 7/12 via vector fallback |
| Answerable cases with unknown retrieval state due to API exception | 2 | 0 | 1 |
| Valid requested-mode full executions, including clarification bypass | 13 | 2 bypasses only | 2 bypasses only |
| Supplied chunk provenance identity coverage on evidence-generated answers | 1.00, 9 outputs | N/A | 1.00, 9 outputs |
| Final deterministic grounding verifier passed | 8/9 | N/A | 8/9 |
| Citation identity coverage mean over those 9 generated outputs | 0.778 | N/A | 0.667 |
| Intent route correctness on completed executions | 12/13 | 14/16 | 12/14 |
| Expected tool correctness on completed executions | 12/13 | 12/16 | 12/14 |
| Pipeline median seconds, all attempted cases | 8.619 | 6.443 | 11.928 |
| Pipeline total seconds | 149.016 | 109.730 | 210.753 |

The automatic retrieval planner agrees with 15/16 pre-score optimal-mode
hypotheses. This does not establish optimality: the ablation forces its requested
mode. The remaining mismatch is `comparison_delivery_budget`, planned as a
graph-only field lookup rather than a hybrid cross-document comparison.

The repaired relationship query now reaches retrieval and supplies its declared
evidence in vector and hybrid-fallback runs. That establishes the orchestration
bug fix, **not** a graph improvement. The incomplete two-project witness set and
cross-document/requirement misses remain visible. Questions about an existing
proposal also expose a keyword-refinement limitation: mentioning "proposal" can
select proposal-writing intent rather than ordinary lookup.

### Explicit conclusions by requested category

- **Vector better:** not established against a functioning Neo4j backend.
- **Graph better:** not established; Neo4j did not participate.
- **Hybrid better:** not established; observed hybrid evidence is vector fallback.
- **GraphRAG adds no retrieval benefit:** the two ambiguous cases, where all modes
  correctly bypass retrieval and clarify. This is not a retrieval accuracy tie.
- **Inconclusive:** the other 14 cases. No universal-superiority claim is made.

Raw timing includes distinct response paths and rate-limit behavior; the graph
median must not be advertised as a graph speed advantage. There is one repetition,
no significance test, and no warmed/repeated latency estimate.

### RAGAS attempt and provider diagnosis

The saved, unchanged final outputs were passed to the actual RAGAS metrics with
the existing Groq key and `openai/gpt-oss-120b`, selected only in the scoring
process. Persistent `.env` credentials/settings were not edited. Judge temperature
was 0, maximum output tokens 4096, metric timeout 90 seconds, retries 0, pacing
2 seconds, and explicit case budget 48. The judge is the same model as generation;
even successful future scores would require consideration of correlated judge bias.

**No usable semantic score was obtained.** A sanitized nested-exception diagnostic
confirmed HTTP 429, `rate_limit_exceeded`, specifically **TPD (tokens per day)**.
A tiny provider request succeeded, which confirms that an available endpoint/key
does not imply sufficient quota for full-context judge requests. Pacing does not
restore an exhausted daily allowance. Do not infer answer correctness from these
errors or calibrate thresholds from this run.

| Each of the five metrics | Successful scores | Judge-error samples | Pipeline-error samples | Inapplicable samples |
|---|---:|---:|---:|---:|
| vector_only | 0/16 | 9 | 3 | 4 |
| graph_only | 0/16 | 0 | 0 | 16 |
| hybrid | 0/16 | 9 | 2 | 5 |

Faithfulness, answer relevancy, context precision, context recall, and answer
correctness all retain null means with those denominators. Ninety metric entries
failed during judging with `InstructorRetryException`; another 25 error entries
represent the five failed pipelines across five metrics, not 25 additional judge
calls. There is no fabricated score, success threshold, or partial-score ranking.
The real-KB regression overlapped the first portion of judging; this additional
shared-quota/latency confound is recorded in JSON. This semantic attempt is
diagnostic only and cannot establish a calibrated baseline.

### Scorer correction, not dataset manipulation

An API exception does not export the intermediate retrieval state. The first
scorer revision incorrectly treated absent records on failed executions as zero
retrieval coverage. This was corrected to **unknown**, with a regression test.
All five failures still count in attempted/pipeline-failure totals; the conservative
7/12 complete-witness counts above still include all answerable cases. Conditional
class means expose evaluated/total denominators. No answers, contexts, raw semantic
scores, timing, references, or frozen expectations were changed.

`retrieval_comparison-before-scorer-correction.json` preserves the original judged
report; `scoring_revision` in the final JSON records the reason and implementation
version. `retrieval_comparison-capture.json` preserves original post-fix outputs,
and the judge record hashes that exact capture. The pre-fix report is separately
retained. No incomplete run is promoted to a release-quality baseline.

Machine-readable checkpoints remain per mode under `evals/results/`; the final
combined output is `evals/results/retrieval_comparison.json`.

## Reproduction and completion prerequisites

```powershell
.venv\Scripts\python.exe -m evals.retrieval_benchmark --index evals/ragas_workspace/20261005-public-pilot/vectorstore --collection rfp_kb_v2 --freeze-only
.venv\Scripts\python.exe -m evals.retrieval_benchmark --index evals/ragas_workspace/20261005-public-pilot/vectorstore --collection rfp_kb_v2 --capture-only --case-interval-seconds 5
```

For a valid live graph comparison, configure read-only Neo4j runtime credentials
and use separate migration/ingestion accounts to publish exactly this indexed
corpus into an isolated graph corpus. Follow `docs/NEO4J_FOUNDATION.md` and
`docs/GRAPH_INGESTION.md`; do not rebuild or overwrite a private production corpus.
Set `RFP_GRAPH_CORPUS_ID=rfp-public-evaluation-20261005` in the evaluation process
and use that same `--corpus-id` for the graph rebuild. Enable `NEO4J_ENABLED` only
after configuring the endpoint and role-specific credentials. The corpus ID is a
non-secret isolation boundary, not a substitute for read-only server privileges.
Then configure the existing judge settings and explicitly allow paid semantic
calls:

```powershell
# Environment values must be configured by the operator; never commit secrets.
# RAGAS_MAX_CASES must explicitly permit 48 executions.
$env:RFP_GRAPH_CORPUS_ID='rfp-public-evaluation-20261005'
.venv\Scripts\python.exe -m evals.retrieval_benchmark --index evals/ragas_workspace/20261005-public-pilot/vectorstore --collection rfp_kb_v2 --allow-judge
```

Also ensure adequate generation **and** judge daily token allowance, or configure
a supported independent judge/provider with sufficient quota. Run regression
jobs separately from paid judging; do not share their rate budget during a valid
semantic baseline trial. Do not select cases or discard failures to fit quota.

To judge an already captured matrix without repeating retrieval or generation,
preserve the capture in a separate file and use:

```powershell
.venv\Scripts\python.exe -m evals.judge_retrieval_capture --input evals/results/retrieval_comparison-capture.json --output evals/results/retrieval_comparison.json --allow-judge
```

The post-hoc adapter retains a hash of the complete original capture and separate
judge implementation/time/provider metadata. Questions and references must still
match the lock. Exact original contexts, responses, and pipeline latencies are
preserved; unit tests reject any second pipeline execution.

Retain earlier results for before/after implementation comparisons. Rerun the same
lock after fixes. Do not re-freeze a convenient new case set after observing scores.
Any objectively necessary dataset correction requires a documented bug, old/new
hashes, and an entirely new separately named dataset version.

## Verification

| Check | Result |
|---|---|
| Initial benchmark + RAGAS adapter tests | 64 passed |
| Final complete pytest suite | 413 passed, 5 opt-in Neo4j tests skipped; 60.92 s |
| Targeted source compilation | All 84 Python source files compiled, including app.py |
| Changed-file Ruff | Passed |
| Full-repository Ruff | Eight pre-existing unused import/variable findings; no new findings |
| Existing offline evaluation | 3/3 smoke; mock corpus, not real semantic accuracy |
| Real KB evaluation | Before fix: 9/9 at 46.46 s; after fix: 9/9 at 47.13 s, both retrieval-only |
| Frozen source checks | All page/anchor witnesses validated before execution |
| Graph ingestion dry run | Validated 327 rows; published false |
| Provider dependency check | No broken requirements |

Executed commands used `.venv\Scripts\python.exe`: `-m pytest -q`,
`-m ruff check` on the changed files and separately on the repository,
`-m evals.run_evals`, `-m evals.run_kb_evals`, `-m pip check`, the freeze-only
command, two full capture matrices, and the post-hoc judging command above.
Compilation used `py_compile.compile(..., doraise=True)` on all source/test/eval
Python files and the four root application/config files, excluding runtime SQLite
directories. The graph dry run used `-m rfp_analyst.graph rebuild --dry-run` with
the same public index and isolated corpus ID `rfp-public-evaluation-20261005`.

No Chroma retrieval implementation, graph query templates, ontology, generator,
production configuration, existing golden expectations, or grounding verifier was
changed. The one production behavior fix is the independently diagnosed local
conversation-reference safeguard described above.

## What remains before a valid comparative baseline

1. Provide an isolated, functioning Neo4j instance and publish the frozen indexed
   corpus with the correct corpus ID; enforce read-only credentials for queries.
2. Restore sufficient generation/judge daily allowance or configure a supported
   independently provisioned judge. Keep other evaluation jobs out of that budget.
3. Rerun the identical locked cases. Preserve this incomplete measurement rather
   than editing questions, lowering support criteria, changing references, or
   dropping failure cases.
4. Diagnose live graph witness misses, query-shape coverage, and text-fusion budgets
   on connected cases before changing ranking or schema. The disabled server here
   cannot establish whether multi-hop graph traversal helps or fails.
5. Collect repeated complete paired trials before drawing a population-level
   conclusion, comparing latency robustly, or setting release thresholds.

This report does not claim the requested live GraphRAG/RAGAS comparison is finished.
It provides the frozen benchmark, real attempted runs, a verified orchestration
fix, explicit environmental blockers, and an auditable path to completion.
