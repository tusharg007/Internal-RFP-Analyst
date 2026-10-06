# RAGAS integration verification — 2026-10-05

## Preserved baseline

Before this integration: 338 passing tests, five opt-in Neo4j skips, offline smoke
3/3 and real-KB retrieval evaluation 9/9. Existing harnesses, golden questions,
deterministic metrics, vector defaults, graph templates, grounding/repair logic and
normal CI judge-free execution were preserved.

## Commands and results

Commands used `.venv/Scripts/python.exe` in this Windows workspace.

| Command | Result |
| --- | --- |
| `python -m pytest -q` | 390 passed, 5 opt-in Neo4j tests skipped; zero failures |
| `python -m pytest tests/test_ragas_evaluation.py -q` | 52 offline tests included in full suite; no live judge calls |
| `python -m evals.run_evals` | 3/3, pass rate 1.0; unchanged deterministic smoke |
| `python -m evals.run_kb_evals` | 9/9, pass rate 1.0, retrieval-only, 47.61 seconds |
| `python -m compileall -q src tests` and compile every top-level `evals/*.py` | Passed |
| `python -m py_compile app.py agent.py rag_engine.py config.py` | Passed |
| Ruff on new/changed evaluation modules, capture helper, graph/config/web-search and touched tests | Passed |
| `python -m ruff check .` | Eight pre-existing unused import/local-variable findings; no new findings |
| `python -m pip check` | No broken requirements |
| CLI help for RAGAS runner, comparison and baseline calibration | Passed without provider calls |

Recursive `compileall evals` also encountered unreadable runtime SQLite directories
created by elevated benchmark processes. These are ignored data workspaces, not
Python source failures. Explicit evaluation-module compilation avoids traversing them.
The pre-existing Ruff findings are in `document_generator.py` (2),
`evals/run_kb_evals.py` (1), legacy `agent/runtime.py` (4), and
`tests/test_runtime_hardening.py` (1); unrelated files were not silently cleaned up.

## Actual semantic pilot, not a fabricated quality baseline

A real public-sample vector-only pilot used the configured Groq key and
`openai/gpt-oss-120b` as the existing generator and explicitly selected judge.
Selection was process-local; `.env` and real credentials were not edited.

```powershell
$env:RAGAS_JUDGE_PROVIDER='groq'
$env:RAGAS_JUDGE_MODEL='openai/gpt-oss-120b'
python -m evals.run_ragas --mode vector --index evals/ragas_workspace/20261005-public-pilot/vectorstore --allow-judge --output evals/ragas_results/vector-pilot-1.json
```

Sample preparation indexed 11 public fixture PDFs into 54 chunks in an isolated
persistent evaluation workspace. No production index or private upload was used.
Initial attempts exposed memory pressure from concurrent embedding jobs and an
evaluation-only Chroma settings conflict; the latter was fixed and unit-tested.
The completed retry executed all nine cases once, without pipeline exceptions.

| Metric | Successful scores / all cases | Successful-score mean | Judge errors | NA |
| --- | --- | --- | --- | --- |
| Faithfulness | 1 / 9 | 1.0 | 2 | 6 |
| Answer Relevancy | 1 / 9 | 0.856118 | 2 | 6 |
| Context Precision | 0 / 9 | Not measured | 3 | 6 |
| Context Recall | 0 / 9 | Not measured | 2 | 7 |
| Answer Correctness | 0 / 9 | Not measured | 2 | 7 |

Groq returned HTTP 429 token-per-minute quota errors (observed limit: 8,000 TPM).
Failures were preserved as `InstructorRetryException`, not zero/perfect scores or
omitted cases. Direct, deterministic catalog and insufficient-evidence paths were
inapplicable. Cross-corpus analysis abstained on an expected answerable task:
tool/origin/no-answer checks each reported 8/9, not a universal end-to-end pass.

This is a **partial measurement only**. It cannot support release thresholds,
comparisons, or a claim that the whole system scored 1.0. The Git-ignored JSON
artifact retains actual answers, contexts, per-case failures and exact pilot
configuration/version metadata. It predates the subsequently added pacing/token-budget
and deterministic-failure aggregation improvements, as its metadata records.

No successful two-pilot baseline exists. Calibration rejects failed/partial pilots;
no invented thresholds or fake baseline JSON were committed. Collect at least two
complete matching pilots with an appropriate provider quota/pacing configuration,
then choose an approved tolerance using the documented calibration CLI.

## Environment-dependent work

- Persist `RAGAS_JUDGE_PROVIDER` / `RAGAS_JUDGE_MODEL`: these remain blank locally.
  Existing provider keys can be reused without duplication.
- Configure pacing and sufficient metric timeout for the provider quota. Pacing is
  not token accounting; shared generation calls/retries can still cause 429s. Avoid
  concurrent embedding benchmarks on memory-limited hosts.
- Supply read-only Neo4j and publish the graph for the same frozen benchmark index
  before graph-only/hybrid quality comparison. Live graph comparison and opt-in
  Neo4j integration tests were not run here.
- Google async construction is tested without network; live Google judging is
  unmeasured. Reference-based metrics have API/unit coverage but their live scores
  were blocked by Groq quota in this pilot.
- Freeze the local embedding cache/weight revision for stronger reproducibility;
  model name and package versions are recorded now.

## Integration files

New: `evals/ragas_adapter.py`, `ragas_judge.py`, `ragas_pipeline.py`, `ragas_reports.py`,
`run_ragas.py`, `compare_retrieval_modes.py`, `ragas_baselines.py`,
`ragas_annotations.yaml`, `src/rfp_analyst/agent/generation_evidence.py`,
`tests/test_ragas_evaluation.py`, `requirements-eval.txt`, and RAGAS documentation.

Updated: generation fields/nodes in `agent/graph.py`; per-run web-disable flag in
`tools/web_search.py` (default enabled); optional settings in `config.py` and
`.env.example`; evaluation extras in `pyproject.toml`; artifact ignores;
graph-generation assertions in `tests/test_hybrid_retrieval.py`; README and testing
documentation. Existing uncommitted Neo4j/hybrid work was preserved. No commits,
pushes, graph rebuilds or production data migrations were performed in this task.
