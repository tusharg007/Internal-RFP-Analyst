# Targeted GraphRAG workflow fixes — 2026-10-06

## Status and evidence boundary

The generic workflow fixes are implemented. Local regression checks pass, including tests using the actual bundled PDF structure, the real extraction/retrieval/tool/generation boundary, an injected graph driver, and controlled model responses. Those tests are not live Aura/model benchmarks.

**The four live end-to-end acceptance criteria are not established.** The configured structured-output provider returned HTTP 429 (`RateLimitError`) on an independent synthetic prompt. In addition, the immutable benchmark snapshot and currently published Aura corpus differ. No semantic judge ran, no graph corpus was republished, and no private upload was sent in the post-fix model runs.

### Corpus separation

| Cohort | Documents/chunks | Indexed digest | Interpretation |
|---|---:|---|---|
| Current local index and live Aura `internal-rfp` | 13 / 102 | `e8ab4d01bd8dfc1a2cc6e88b7ddecbe209123e803508431867875cc213785619` | Pre-fix diagnostic cohort; also used for post-fix sample-only retrieval diagnostics with every LLM call disabled. It does not contain `eval_target_rfp.pdf`. |
| Immutable public benchmark snapshot | 11 / 54 | `641e9d4dad22e33140475ac40b0245dde1edb3ed6e2acc0c5fa8e75919d9d39f` | Contains the synthetic target RFP. Used for post-fix model/capture runs. Aura rejects this digest as `unsynchronized_snapshot`. |

The user identified the existing live diagnostic captures during this task, and their metadata was inspected:

- `evals/ragas_results/vector-smoke.json`
- `evals/ragas_results/graph-smoke.json`
- `evals/ragas_results/hybrid-smoke.json`

Each contains 16 completed pipeline executions against the root 13/102 corpus with digest `e8ab4d...`; graph/hybrid used live `internal-rfp`. The graph banking row explicitly records `timeout` then `unsupported_plan`. The comparison row records insufficient output and missing tool execution. These captures are useful integration diagnostics, **not a valid frozen 16-case ablation**: the locked gold corpus differs. The uploaded-control run retrieved unrelated uploads, not the synthetic target RFP; its failure cannot establish an answerable-case graph retrieval defect. No missing frozen witness was used to justify retrieval changes. Neither these reports, the earlier checked-in comparison nor the frozen manifest was overwritten.

The user's final scope instruction was **public sample data only; do not modify either corpus**. Post-fix model runs used only the frozen public/synthetic snapshot. Live sample-only retrieval diagnostics had every model call disabled. Results from different corpora are not treated as a paired before/after experiment.

Unchanged file hashes:

- `evals/retrieval_questions.yaml`: `7846cea3aa3eccc3b281d44dd1a3eecff51d65b08265f045ff1e34a896bc6689`
- `evals/retrieval_benchmark.lock.json`: `a76610f63e0047e13033cbd45cb7b732c75ff17e36f316b6b557bc07536c0eb7`
- `evals/golden_questions.yaml`: `13eb016110e872fa2ddf6f8e47a6c451b1fbb0969ecd570fc1125d3290f6ee2f`

## Root causes, reproduced before the fixes

All four original questions were executed individually in vector-only, graph-only, and hybrid modes against the live 13/102 diagnostic cohort before edits. Route, planned/executed tools, retrieval events, source evidence, plans, retries and grounding outcomes were retained locally. A root-cause table was shared before editing.

| Defect | Reproduced behavior / root cause | Generic correction |
|---|---|---|
| Banking duration | Initial typed timeline plan was supported. A graph timeout was followed by a semantic rewrite that dropped the parser-supported named-field form; graph-only then declined as `unsupported_plan`. Hybrid retrieved the field. | One existing bounded transport retry retains the same supported query/plan and clears only a transient provider circuit. No vector fallback was added to strict graph-only and no timeout was increased. |
| Comparison sequencing | Intent and tool planner selected `compare_projects`, but the initial grader required a complete final comparison before allowing that tool to retrieve remaining fields. All three paths abstained without executing the tool. | Explicit tool-input grading precedes deterministic tools; strict answer-quality grading still runs on the actual final evidence before generation. Tests prove that a weak final grade blocks generation in both vector and graph modes. |
| Comparison retrieval | The original query became a healthcare/insurance industry constraint, rather than two named projects plus timeline/budget fields. It did not retrieve all four scalar witnesses. | Parse explicit named-field comparisons; resolve shortened ordered title words only when unique; retrieve both typed fields with original witness chunks. Auto mode remains hybrid. |
| Comparison extraction | The section parser stopped at capitalized `Phase 2`. Real PDF chunking also separated `Total Duration` from `Timeline & Milestones`. | Stop only at known section headings and recognize exact `Total Duration` / `Estimated project cost` lines when headings are absent. Missing-field lookups are bounded and restricted to selected sources. |
| Uploaded target | The live failure is expected because the synthetic target is absent and unrelated uploads cannot answer it. Independently, the workflow required final-answer sufficiency before candidate tools, and the parser could misread “delivered evidence” as a delivered-outcome filter. These are corpus-independent defects, tested with a valid synthetic target. | Tool-input grading permits relevant target requirements, not unrelated uploads. A delivered-outcome constraint now requires an explicit projects/case-studies-delivered construction. Missing corpus membership remains a real limitation, not an invented graph fact. |
| Inline clinical requirements | Every `rfp_analysis` intent was forced into uploaded-target scope, even when the user supplied requirements inline and selected sample scope. Outcome witness pages were not selected by the constraint path. | Distinguish explicit inline RFP constraints from uploaded/target requests; extract inline requirements from the user query without fabricating document provenance; retain requested outcome witnesses as original evidence. |

Additional root causes were reproduced during the focused PDF regressions/read-only live diagnostic, before their corrections:

- Short filenames need not contain full project names. The comparison tool now considers bounded first-page title lines too and still requires uniqueness.
- Derived `Compare fit for: ...` requirement text could be misparsed as two named projects. Named-field comparison syntax now requires an explicit timeline/budget/outcome comparison construction.
- The final reader head check conflated an elapsed deadline with a changed corpus version. These now produce separate `timeout` and `snapshot_changed` reasons. Version-integrity guards and deadline bounds remain unchanged.

### Secondary hybrid invariant

No primary fallback evidence-loss defect was established. Unsupported graph planning returns the same vector-supplier evidence, without extra ranking/truncation. A focused identity-preservation regression protects that behavior. Primary hybrid fallback, vector thresholds, fusion weights and vector-optimal case expectations were not retuned.

## Before / after verification

| Target | Pre-fix live diagnostic | Post-fix verified boundary | Remaining acceptance gap |
|---|---|---|---|
| Banking duration | Graph timeout → unsupported rewritten plan → insufficient. Hybrid answered with a graph path. | Same-plan transport retry regression passes. Real public-PDF unit test retrieves the original 16-week witness. A genuine live, LLM-disabled sample diagnostic returned graph-only, 1 path, 2 chunks and validated source provenance after a retry. | Final live generated/grounded answer is not verified; provider quota and frozen-corpus synchronization block the model benchmark. |
| Delivery/budget comparison | All modes insufficient; planned comparison tool never executed. | Real PDF regression extracts 18 weeks/$1.2–1.5M and 16 weeks/$650–800K with four distinct page witnesses. A controlled four-citation answer passes the unchanged verifier. Live LLM-disabled graph-only diagnostic executed comparison and returned 2 paths / 6 chunks for the two projects. | Live LLM completion of all four claims remains unverified. |
| Uploaded controls | All modes insufficient; target/candidate tools blocked. Expected synthetic target absent in current live cohort. | Public-PDF/injected-driver workflow processes the synthetic upload, executes requirement/candidate tools, retains sample/upload provenance and preserves healthcare proposal/not-delivered wording in a grounded controlled answer. | Must use a graph corpus synchronized with the frozen snapshot and a working generation/grader provider. Secondary lookups of non-domain target prose may correctly decline in strict graph-only; no relationships are fabricated for it. |
| Clinical requirements | Inline request incorrectly searched uploads; no graph candidate. | Public-PDF regression retains FDA framework and original achieved 75%/85% outcome pages, executes candidate tools and passes grounding for controlled cited claims. Live LLM-disabled graph-only diagnostic returned Pharma only, 1 path / 3 chunks and executed the RFP tools. | End-to-end live hybrid generation remains unverified. |

The live diagnostics have `generation_kind=not_generated`, not an answer-quality pass. Their corpus is not the immutable benchmark corpus. The generated public target rerun completed all 12 case/mode executions but abstained because of grader/provider failures; graph-backed modes also recorded the digest mismatch. This is not a successful before/after quality comparison.

## Files changed in this task

Existing unrelated/earlier upgrade edits in the dirty worktree were preserved.

| File | Task change |
|---|---|
| `src/rfp_analyst/retrieval/decisions.py` | Named-field comparison plans, unique shortened-title matching, precise delivered-language detection, requested scalar witness fields. |
| `src/rfp_analyst/retrieval/hybrid.py` | Original outcome witness retention and explicit transient-circuit reset for the bounded workflow retry. |
| `src/rfp_analyst/graph/reader.py` | Distinguish final-head deadline expiration from real version change. |
| `src/rfp_analyst/agent/graph.py` | Tool-input versus answer grading, same-query transport retry, inline/upload scoping, retention of comparison-tool evidence. |
| `src/rfp_analyst/agent/grader.py` | Stage-specific tool-input grading prompt; final-answer prompt remains unchanged. |
| `src/rfp_analyst/agent/prompts.py` | Generic separate-per-field numeric citation guidance and modality preservation. |
| `src/rfp_analyst/tools/compare_projects.py` | Section/chunk-boundary extraction, bounded source-locked supplementation, per-field cited derived notes. |
| `tests/test_target_workflows.py` | 24 new parametrized regression cases for transport, planning, tool sequencing, provenance, real PDFs and final-grade non-bypass. |
| `docs/GRAPHRAG_TARGET_FIXES.md` | This report. |

New capture files use a separate result prefix; no earlier comparison report was replaced. Local diagnosis artifacts under ignored `evals/ragas_workspace/` are not public exports and are not intended for commit.

## Commands and results

Commands use `.venv\Scripts\python.exe`.

```powershell
python -m pytest tests/test_target_workflows.py tests/test_hybrid_retrieval.py tests/test_langgraph_agent.py tests/test_conversation_routing.py tests/test_production_readiness.py tests/test_retrieval_benchmark.py -q
python -m pytest -q
python -m ruff check .
python -m compileall -q config.py agent.py app.py rag_engine.py src tests
$taskEvalFiles = @(Get-ChildItem -LiteralPath evals -File -Filter '*.py' | ForEach-Object { $_.FullName })
python -m compileall -q $taskEvalFiles
python -m evals.run_evals
python -m evals.run_kb_evals
python -m evals.retrieval_benchmark --index evals/ragas_workspace/20261005-public-pilot/vectorstore --collection rfp_kb_v2 --capture-only --case-interval-seconds 5 --output evals/results/retrieval_comparison_target_fixes_final_20261006.json
```

- Focused suite, final code: **247 passed**.
- Full regression, final code: **505 passed, 5 skipped**, exit 0, 128.51 seconds (prior stated baseline: 481 passed, 5 skipped). The five opt-in integration skips are not represented as live Neo4j test passes.
- Ruff: passed.
- Compile checks: passed for application/package/tests and top-level evaluation modules. An earlier recursive compile command emitted ACL warnings for protected runtime index folders; the source-only commands above avoid traversing those non-source folders.
- Offline smoke evaluation: **3/3**; deterministic mock smoke test, not live RAG quality.
- Real-KB evaluation: **9/9**, retrieval-only; actual public PDF ingestion/Chroma, not live answer grading.
- No RAGAS judge: all semantic judge scores remain unrun/null.

One earlier benchmark attempt was deliberately interrupted before a source fix to avoid mixing implementations. Its checkpoints are not a final 16-case result. The separate `final` capture command completed all **48 executions** and returned **exit 1 / `status=incomplete`**. This is a genuinely executed command, not a successful quality verification.

## Frozen comparison / protected cases

Machine-readable final capture and mode reports:

- `evals/results/retrieval_comparison_target_fixes_final_20261006.json`
- `evals/results/retrieval_comparison_target_fixes_final_20261006-vector_only.json`
- `evals/results/retrieval_comparison_target_fixes_final_20261006-graph_only.json`
- `evals/results/retrieval_comparison_target_fixes_final_20261006-hybrid.json`

| Mode | Completed pipeline executions | Pipeline exceptions | Valid requested backend/bypass | Expected answer/abstain/clarify behavior | Generated-answer grounding evaluated |
|---|---:|---:|---:|---:|---:|
| vector_only | 16/16 | 0 | 16/16 | 4/16 | 0 |
| graph_only | 16/16 | 0 | 2/16 | 4/16 | 0 |
| hybrid | 16/16 | 0 | 2/16 | 4/16 | 0 |

The two valid graph/hybrid entries are **clarification bypasses**, not working graph retrieval evaluations. The four behavior agreements per mode are the two no-answer and two clarification cases; no answerable case produced a KB answer in this final capture. Every target was `insufficient` in every mode. Grader failure logs accompanied the model runs, and the independent synthetic provider probe established HTTP 429; the capture schema does not attach a provider HTTP status to every individual rejected grade. Graph-backed cases additionally retained `unsynchronized_snapshot` or intentional unsupported-plan reasons. Availability failures are not interpreted as retrieval quality changes.

There were 76 failed deterministic check entries across the reports (not 76 failed cases). Fourteen case comparisons are inconclusive. The two remaining cases bypassed retrieval for clarification, so the harness's `no_benefit` label for those cases says nothing about vector-versus-graph accuracy. No mode superiority is established. Semantic judge scores are all unrun/null. The manifest matched before, during and after execution; no case expectations were changed.

This mismatched run is a recorded availability/abstention diagnostic, **not a valid quality ablation**. It cannot establish target acceptance or a live generated-answer non-regression and is not justification for any retrieval changes.

The frozen question text, expected mode/intent/tool/reference/evidence fields and manifest were not edited. Existing routing/retrieval tests, including relationship and shared-technology paths, remain passing. Live non-regression of generated answers cannot be established while the model provider is rejecting grading calls and the graph cohort does not match the frozen cohort. Unsupported graph-only behavior on vector-optimal cases remains acceptable and has not been converted into graph support for scoring purposes.

An additional read-only diagnostic on the current live cohort verified these retrieval boundaries with all LLM calls disabled:

| Case | Requested / effective | Graph paths | Result boundary |
|---|---|---:|---|
| `semantic_modernization` | vector / vector | 0 | 4 chunks, expected banking source, no graph use. |
| `semantic_healthcare_security` | vector / vector | 0 | 5 chunks, healthcare source retained, no graph use. |
| `factual_insurance_team` | vector / vector | 0 | 2 insurance chunks, no graph use. |
| `comparison_outcome_modality` | vector / vector | 0 | Retail/manufacturing evidence and comparison tool retained, no graph use. |
| `relationship_rpa` | graph / graph | 1 | 2 chunks, zero vector calls. |
| `multihop_shared_banking_healthcare` | hybrid / hybrid | 5 | 6 chunks. |
| `multihop_shared_retail_pharma` | hybrid / hybrid on one explicit rerun | 3 | 6 chunks; first diagnostic fell back after Aura availability failure. |
| `relationship_regulatory` | hybrid / vector fallback | 0 | Availability failure initially and timeout on one explicit rerun; 5 Pharma vector chunks preserved. Live graph-path retention is not confirmed by this post-fix diagnostic. |
| `comparison_delivery_budget` | hybrid / hybrid | 2 | 6 chunks; comparison tool executed. |
| `matching_clinical_requirements` | hybrid / hybrid | 1 | 5 chunks; required candidate tool executed. |

Banking's final-code live diagnostic specifically recorded `project_fields/timeout` followed by `project_fields/success`, one retry and **zero vector searches**. This confirms that the transient retry no longer converts that query into an unsupported plan, but remains retrieval-only evidence, not a generated-answer pass.

## Remaining limits and required next step

1. Restore configured grader/generation provider capacity (HTTP 429 was observed independently). Do not turn provider errors into positive evidence grades.
2. A meaningful frozen graph ablation requires a graph snapshot matching the frozen public index. The user explicitly instructed that neither existing corpus be modified, and both were left unchanged. Any separately populated evaluation graph requires a later, separately authorized task; do not replace `internal-rfp` or re-freeze cases to avoid the mismatch.
3. The safety review denied post-fix model generation from the live corpus because it may contain private uploads, and the user chose public sample data only. No private-corpus post-fix model run was performed. Public synthetic generation and sample-only live retrieval with all LLM calls disabled were used instead.
4. Re-run targets, then protected successful cases, then all frozen modes after resolving the above. Keep provider/corpus metadata with the results.
5. Aura's existing bounded read deadline can still expire. The deadline was not increased; the one bounded typed transport retry is tested, not a guarantee of cloud availability. Genuine snapshot changes still invalidate evidence.

No changes were made to Neo4j ingestion, schema, security policy, unrestricted Cypher handling, frozen data, final citation/grounding verifier, repair limit, prompt token limit, graph traversal/result bounds, RAGAS judging, or primary Chroma retrieval/fallback algorithms.
