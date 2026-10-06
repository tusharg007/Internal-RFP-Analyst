# GraphRAG / RAGAS production-readiness audit

Audit date: 2026-10-05. Reviewed the working tree, not only committed HEAD.
Initial inspection and the baseline below preceded all changes in this audit.
Verdict: **not ready for production sign-off**. The application remains a bounded,
single-user demonstration; passing unit tests alone does not establish production
Neo4j authorization or semantic answer quality.

## Ranked findings (before fixes)

No Critical defect established in the inspected implementation.

| ID | Severity | Evidence / impact | Planned disposition |
| --- | --- | --- | --- |
| H1 | High | `tools/source_verifier.py:verify_answer_grounding` adds the cited filename even when its page is absent. A reproduced claim citing `Banking_Digital_Audit.pdf, Page 999` passed with only page 1 retrieved. Follow-up inspection found publication was also possible after failed final verification. | Reject all unknown citation identities and withhold answers failing the final gate after bounded repair. Keep the verifier explicitly heuristic. |
| H2 | High | `agent/graph.py:generate_from_kb` rebuilds a prompt from full `retrieval_context` and unbounded `specialized_notes`; `generate_from_web` also has no budget guard. The budgeted `synthesize_prompt` result is not the prompt sent. Token limits, capture and verification can disagree. | Budget the actual submitted prompt; retain whole evidence blocks, verify only supplied documents, and report the actual budget. |
| H3 | High | `agent/grader.py:_grade_with_llm` returns `good` after provider/parsing failures whenever evidence is nonempty. Semantic rejection silently becomes approval, particularly unsafe for web evidence. Factory errors also occur outside its exception boundary. | Fail weak on configured-provider errors; preserve the explicit offline deterministic fallback. |
| H4 | High | `graph/extraction.py:SectionExtractor` emits `uses_technology=Microsoft Azure` for `Technology Stack / Cloud: We do not use Azure.` Exact quotes and model confidence do not establish a positive relationship. Structured extraction has the same lexical-only acceptance gap. | Conservatively reject negated/conditional relationship spans at the shared validation boundary; do not infer semantic entailment from quote matching. |
| H5 | High | `tools/web_search.py` logs complete SDK exceptions; `agent.py:format_llm_error(debug=True)`, generic UI handlers and ingestion errors expose raw exception text. SDK errors can contain queries, credentials or document content. | Expose safe error categories only, including debug mode; sanitize generic UI/ingestion failures and logs. |
| M1 | Medium | `router.py:_refine_kb_intent` treats any occurrence of `proposal` as a writing request. Factual questions about a proposal and matching queries execute the wrong tool. | Use action-oriented writing/refinement rules and regression tests; do not change frozen evaluation cases. |
| M2 | Medium | `evals/retrieval_benchmark.py:summarize` labels valid tradeoffs or deterministic quality failures as incomplete execution; semantic N/A can also make a valid run incomplete. | Separate execution completeness from quality findings and inconclusive comparisons. |
| M3 | Medium | Full repository Ruff fails with eight unused imports/variables; CI does not run lint and all real Neo4j tests are opt-in/skipped. | Remove safe lint defects, enable lint and a separate disposable Neo4j integration CI job. |
| M4 | Medium | Historical graph snapshots retain deleted document facts; active-version filtering correctly hides them but no hard-erasure/retention operation exists. | Deferred: require an explicit retention/erasure policy before private production ingestion. Do not delete audit history silently. |
| M5 | Medium | READ sessions and reader-role settings are not server-side RBAC. Community integration tests borrow an admin driver. No actual least-privilege deployment has been demonstrated. | Deployment gate: distinct server-enforced reader identity and negative write-privilege test required. No unsafe probe writes or fabricated RBAC claim. |
| M6 | Medium | RAGAS may use the same model as generation; only a small one-repetition dataset exists, semantic baselines are uncalibrated, and the last live run had quota errors. | Require independent judge sensitivity checks, repeated matched runs and operator-approved tolerances before quality sign-off. No invented scores/thresholds. |
| M7 | Medium | Shared process logging levels are changed/restored by `RagasJudge`; overlapping judges can restore verbose logging while another is active. | Make redaction lifecycle reference-counted and test overlapping instances. |
| M8 | Medium | Found during verification: real-KB retrieval-only evaluation invokes live control/web factories. Optional answer evaluation executes twice, checking one run's sources against another run's answer. | Explicitly disable live calls in retrieval-only; check one complete execution in answer mode. |
| M9 | Medium | Follow-up lifecycle inspection: currentness failures retain an owned reader driver; judge wrapping can fail after creating a client but before assigning ownership. | Close failed owned readers; establish judge ownership before wrapping. Preserve borrowed-driver ownership. |

## Controls reviewed / remaining boundaries

- Neo4j user questions use finite source-reviewed Cypher templates, parameterized
  values and no generic query API or dangerous chain. Server transactions, seed,
  fact, record, path, chunk and context limits are bounded. No arbitrary traversal.
- Persistence revalidates Pydantic models, unique composite identities and provenance;
  publication locks the corpus head and atomically commits a complete snapshot.
  Lost-acknowledgement retries of the same version are idempotent. Driver/session
  ownership is explicit and guarded by instance locks; no hidden transaction retries.
- Canonicalization uses a controlled ontology and keeps cloud products distinct.
  Malformed extraction quarantines the entire snapshot. Span validation is not
  semantic entailment; unknown coverage and compliance attestations remain unknown.
- Graph retrieval validates active-version/content hashes and source spans, hydrates
  original text, deduplicates vector/graph chunks, retains complete multi-hop witnesses
  under budgets and rechecks graph/index currentness before and after generation.
  Disabled/unavailable Neo4j falls back explicitly; strict evaluation records it.
- The cyclic agent keeps bounded rewriting, existing tools, source labels, web fallback,
  deterministic backend and one repair pass. Web generation is prompt-grounded, not
  checked by the PDF citation verifier: do not equate it with Private KB verification.
- RAGAS adapts captured generation evidence without another retrieval. References are
  evaluation-only, not passed to generation. Missing references have explicit metric
  applicability; per-case scores and denominators are retained. Context-precision
  variants and judge models must not be compared as interchangeable measurements.
  Derived tool notes are separately hashed, not masqueraded as original evidence.
- The question/corpus freeze, source anchors and comparison cohorts prohibit editing
  questions to make GraphRAG win. Failed metrics remain null; there is no claim-level
  provenance graph or guarantee of semantic entailment from the lexical verifier.
- Chroma clients may share internal process state; do not stop a shared Chroma system
  from a graph helper. Concurrent distributed Chroma publication/Neo4j rebuild needs
  a quiescent immutable index, as documented. Large-corpus staging, multi-tenant ACLs,
  automatic reconciliation workers and erasure policy remain outside this patch.

## Initial commands actually executed

| Command | Result |
| --- | --- |
| `.venv/Scripts/python.exe -m pytest -q` | 413 passed, 5 Neo4j integration tests skipped, 62.45 s. |
| `.venv/Scripts/python.exe -m ruff check .` | Failed: 8 unused-import/variable defects. |
| `.venv/Scripts/python.exe -m evals.run_evals` | Passed, 3/3 mock offline smoke; not semantic quality. |
| `.venv/Scripts/python.exe -m pip check` | Passed, no broken installed requirements. |
| Read-only Python reproductions of H1 and H4 | Invalid page accepted; negated Azure emitted as a positive relationship. |

## Post-fix verification

### Remediation

H1-H5 and safe Medium findings M1-M3/M7-M9 are addressed. M4-M6 remain operational
release gates, not hidden successes.

- All citation identities must exist in supplied evidence, including headings,
  short claims and mixed valid/invalid citations. Final failure after one repair
  withholds the answer and sets insufficient-evidence status; attempted generation
  context remains recorded for audit.
- Actual vector/web generation prompts are bounded. Private chunks stay whole;
  omitted chunks cannot support verification. Oversized derived notes are omitted.
  Clipped web and reduced KB context are regraded and captured exactly. The graph
  path keeps its whole-witness/currentness guards. Preparation-only APIs retain
  legacy planning prompts. Token budgets use the existing character estimate,
  not a claimed provider-exact tokenizer.
- Grader errors, including initialization/parsing failures, fail weak. Oversized
  grader prompts do not invoke providers. Explicit offline grading remains.
  Router and rewriter construction failures also degrade.
- Shared extraction validation rejects negated/conditional object relationships
  for deterministic and structured extraction; ambiguous spans quarantine the
  snapshot. Versions are now rfp-sections-v2 and structured-v2:<model>. Rebuild
  enabled graphs. This conservative guard can reject valid mixed sentences and
  is not universal semantic entailment.
- SDK/ingestion/UI error handlers no longer expose raw exception text, even in
  debug mode. Benchmark setup errors also generate sanitized JSON.
- Action-oriented intent refinement distinguishes proposal writing from factual
  proposal questions and recognizes case-study matching.
- Completion is separate from quality failures/tradeoffs. Deterministic comparison
  uses the same available dimensions for all modes; metric N/A is not an error.
- Judge log suppression is reference-counted; owned clients/readers close on
  initialization/currentness failure.
- Retrieval-only KB evaluation disables live controls and forces vector retrieval.
  Optional answer evaluation checks the one run actually producing the answer.
  This is an evaluation-recipe correction, not a dataset edit; older network-
  dependent retrieval-only latencies are not directly comparable.
- CI adds lint, optional RAGAS contracts without live calls and disposable Neo4j
  schema/transaction/retrieval tests. Hosted jobs have NOT been executed locally.

### Commands and results

Final full-suite rerun: **443 passed, 5 skipped in 65.12 seconds**.
Thirty new parametrized regression cases were added (28 in production-readiness
tests and two benchmark completion tests). Existing security tests that explicitly
expected raw errors were updated to require redaction; their lifecycle/retry
assertions remain. No runtime failures remain in the executed pytest suite.

All Python commands used the repository .venv/Scripts/python.exe.

| Command actually executed | Result |
| --- | --- |
| python -m pytest -q | Passed, 443 tests; 5 opt-in real Neo4j tests skipped. 65.12 s. |
| python -m ruff check . | Passed; all initial lint defects removed. |
| Source py_compile sweep over src/tests/evals and five entry points | Passed, 86 files; runtime index folders excluded. |
| python -m pip check | Passed; no broken installed requirements. |
| python -m evals.run_evals | Passed, 3/3 mock smoke, not semantic accuracy. |
| python -m evals.run_kb_evals | Final rerun passed, 9/9 retrieval-only, 8.719 s case latency (earlier isolated run: 8.956 s); actual Chroma/public PDFs. The initial network-dependent attempt experienced grader failures and was not accepted as a quality baseline. |
| python -m evals.run_ragas --mode vector/graph/hybrid --allow-judge --output evals/results/production-audit-ragas-<mode>.json (three commands) | Attempted; each setup failed with ValueError. Judge provider/model are unconfigured. Sanitized setup-error JSON saved; no semantic scores. |
| python -m evals.compare_retrieval_modes --allow-judge --output evals/results/production-audit-ragas-comparison.json | Attempted; same configuration blocker, saved setup-error JSON. Not a successful comparison. |
| python -m evals.retrieval_benchmark --index evals/ragas_workspace/20261005-public-pilot/vectorstore --freeze-only --output evals/results/production-audit-freeze.json | Initial sandbox access denied; approved isolated-public-index retry passed. Verified original 16-case signature dbab6df4cae79b001bb2d54fae9adad7109216c601fa404a5f16ecb32096173b. |
| python -m rfp_analyst.graph rebuild --dry-run --persist-dir evals/ragas_workspace/20261005-public-pilot/vectorstore --corpus-id internal-rfp | Passed with approved public-index access: 11 documents, 54 chunks, 80 entities, 171 assertions, 327 rows; no publication. Snapshot 19b5b483a3ffe35104375e4eea6cf5f698bd3c2aa83399ac1e0c784476db66f7. |
| python -m pytest tests/test_graph_store_integration.py -q | Five skipped, NOT a real-server pass. Neo4j disabled/unconfigured and Docker unavailable. |
| git diff --check | Passed; Windows CRLF warnings only. |
| CI YAML parse | Passed; not a hosted-job execution. |

Local interpreter is Python 3.12; CI's Python 3.11 remains a hosted verification
gate. Streamlit smoke tests are in pytest. Frozen questions/references, historical
comparison captures, private PDFs, graph data and .env credentials were not edited.

### Files changed in this audit

This list excludes pre-existing changes from earlier implementation stages.

- Runtime: agent.py, app.py, rag_engine.py; agent/{graph,grader,router,
  query_rewriter,runtime}.py; tools/{source_verifier,web_search}.py.
- Graph: graph/{extraction,reader}.py.
- Evaluation: evals/{run_kb_evals,ragas_judge,retrieval_benchmark}.py.
- Engineering/docs: document_generator.py, .github/workflows/ci.yml, README,
  this audit and testing/RAGAS guides.
- Tests: new test_production_readiness.py; updates to test_retrieval_benchmark,
  test_ragas_evaluation, test_kb_evals, test_agent_and_ui, test_streamlit_app_smoke;
  unused import removal in test_runtime_hardening.
- Artifacts: four sanitized production-audit-ragas-*.json setup reports. Existing
  smoke/real-KB result files were regenerated by their evaluation commands.

### Release gates still open

1. Run real Neo4j integration CI and demonstrate server-enforced read-only runtime
   credentials, including denied write privileges. READ routing is not RBAC.
2. Configure RAGAS judge/provider/model and adequate quota. Rerun the unchanged
   frozen cohort against a synchronized graph, repeat, assess independent judge
   sensitivity and calibrate approved tolerances. No scores/thresholds invented.
3. Establish private-document retention/hard erasure, quiescent index/worker
   operations, resource and ACL policy. Historical snapshots are not active facts,
   but retaining them is not equivalent to deleting private data.
4. Review semantic failure cases manually: lexical/numeric checks and quote gates
   do not establish formal entailment or claim-level provenance. Neither unit
   tests nor smoke evals justify a production-assurance claim.
