# Semantic evaluation of actual pipeline outputs

Audit update (2026-10-05): benchmark completion now distinguishes execution/judge
failures from quality findings and valid tradeoffs. Matched not-applicable metrics
do not imply failure; unexecuted or errored metrics remain incomplete. Pareto
comparisons use the same available dimensions across all three modes.

Final KB verification failure after bounded repair withholds the answer. The
attempt's supplied contexts remain captured, but final generation_kind is
insufficient; semantic answer metrics are N/A and deterministic no-answer checks
apply. This is an honest rejected attempt, not a scored successful answer.

Judge logging suppression is reference-counted across overlapping lifetimes,
and failed initialization closes owned clients. Live judge configuration, matched
repeated baselines and deployment verification remain required: see the
[production-readiness audit](PRODUCTION_READINESS_AUDIT.md).

RAGAS is an optional, costly evaluation layer, not a replacement for the offline
smoke evaluation, real-KB golden evaluation, deterministic tools, or final citation
verifier. Normal application startup and pytest do not require RAGAS or judge keys.
No tutorial answers, mock contexts or notebook examples are production evaluation inputs.

## Install and configure

Prefer a separate evaluation environment with the application dependencies installed:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -r requirements-eval.txt
```

The optional stack pins RAGAS 0.4.3 and its compatible LangChain community 0.3.31.
RAGAS 0.4.3 imports a legacy VertexAI class removed in community 0.4; an unconstrained
install can fail at import. Core/orchestration packages do not need replacement.
The Groq/Google adapters use public Instructor provider-specific async factories:
RAGAS 0.4.3's generic Groq factory assumes an incompatible client interface.
Do not monkeypatch third-party modules or use deprecated Google SDKs to bypass this.

Configure `.env` using the existing environment/secrets resolution:

```dotenv
RAGAS_JUDGE_PROVIDER=groq
RAGAS_JUDGE_MODEL=<your chosen supported judge model>
RAGAS_JUDGE_API_KEY=
RAGAS_METRIC_TIMEOUT_SECONDS=90
RAGAS_JUDGE_MAX_RETRIES=1
RAGAS_MAX_CASES=30
RAGAS_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RAGAS_JUDGE_MAX_TOKENS=4096
RAGAS_JUDGE_MIN_CALL_INTERVAL_SECONDS=0
```

Providers supported here are `groq` and `google`. An empty override key uses the
existing `GROQ_API_KEY` / `GOOGLE_API_KEY`. There is no default judge provider/model.
Generation continues using `agent.get_llm()` and the existing generation settings.
Judge credentials never enter reports or reprs. Temperature is zero; the embedding
implementation is local FastEmbed, not an implicit paid embedding API. Its model
name and FastEmbed package version are recorded; upstream embedding-weight revision
is not automatically resolved, so freeze the downloaded model cache for strict runs.

## What is measured

Each case executes the complete application graph **once**. Generation nodes record:

- `generation_contexts`: verbatim source blocks in prompt order, after fusion and
  graph witness/budget selection. If blocks cannot exactly reconstruct the actual
  supplied context, the exact supplied string is retained as one context.
- `generation_evidence`: source/page/chunk metadata for that boundary.
- `generation_prompt_hash`: SHA-256 of the actual invoked prompt, not the older
  synthesized prompt that some generation paths do not use.
- `generation_kind` and a hash of auxiliary derived tool notes.

The adapter exports `user_input`, `retrieved_contexts`, the **final repaired**
`response`, and `reference` when provided. It cannot retrieve again and rejects
missing boundary captures instead of reconstructing stronger lineage after the fact.
Discarding stale graph evidence also clears capture fields. Generation fields are
additive to QueryState/payload; existing UI and retrieval defaults remain unchanged.

Specialized tool notes can contain derived analysis and are not promoted to original
source evidence. They remain in the existing generation prompt, with a separate hash.
This makes an unsupported derived claim visible to Faithfulness. Deterministic catalog
outputs, direct chat, clarification, abstention and ungenerated answers do not receive
synthetic perfect semantic scores; they are marked not applicable and checked by
deterministic metrics. Failures and applicability counts remain in the denominator.

| Metric | RAGAS collection | Inputs / applicability |
| --- | --- | --- |
| Faithfulness | `Faithfulness` | Question, final answer, captured evidence |
| Answer Relevancy | `AnswerRelevancy` | Question, final answer; local embeddings; strictness 3 |
| Context Precision | `ContextPrecisionWithReference` or `ContextPrecisionWithoutReference` | Uses an independent reference when present, otherwise final answer; variants recorded separately |
| Context Recall | `ContextRecall` | Requires an independent reference; missing reference is NA |
| Answer Correctness | `AnswerCorrectness` | Only with reference; RAGAS weights 0.75 factuality / 0.25 similarity |

Modern `.ascore()` APIs are used. See the official
[metric documentation](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
and [RAGAS 0.4.3 release source](https://github.com/vibrantlabsai/ragas/tree/v0.4.3/src/ragas/metrics/collections).
Scores are heuristic semantic judgments, not formal entailment/provenance proofs.
Cosine-based relevancy can be negative; negative values are retained, not clamped
into perfect/zero scores. Nonfinite or out-of-domain judge values are recorded as errors.
Precision without a reference is answer-conditioned and cannot establish independent
retrieval completeness. Recall cannot be fabricated using the generated answer as its
own reference. Successful score means must always be read with coverage/error counts.

Tool correctness, route correctness, effective retrieval-mode correctness (including
secondary retrieval fallbacks), source/origin expectations, no-answer behavior,
citation identity coverage and the existing grounding result are deterministic. No
judge is asked to validate those exact facts. Unknown route expectations remain NA.
Existing golden definitions and deterministic harness implementations are preserved.

## Datasets and corpus

`evals/golden_questions.yaml` remains the same nine-case golden set.
`evals/ragas_annotations.yaml` adds intent expectations and two source-checked
references for the public banking PDF and the public evaluation RFP upload. Those
references come from the repository's real PDF generation definitions, not fabricated
pipeline outputs. Other cases have no reference until independently reviewed answers
are added. `--no-annotations` retains the original cases alone; `--annotations PATH`
accepts a matching sidecar. Sidecars cannot change original questions/source/tool checks.

For graph-heavy extensions, add reviewed conjunctive industry/technology/compliance,
shared-technology and multi-capability cases to a new frozen YAML set. Retain the
original golden regression set. Explicitly define references and expected tool/intent
checks; do not manufacture references from the current model answer. The current
nine-case set is a regression baseline, not sufficient evidence of multi-hop superiority.

Prepare a persistent public sample workspace once, or use an existing index:

```powershell
python -m evals.run_ragas --mode vector --prepare-samples evals/ragas_workspace/pilot --capture-only --output evals/ragas_results/capture.json
```

This runs real generation and therefore still requires the existing generation key
and can incur generation cost. It skips judge calls. Sample preparation refuses an
existing workspace, uses the existing public PDFs/upload fixture, and never rebuilds
between modes. Existing indexed corpora are only read, with currentness checks before
and after execution. Do not edit the corpus concurrently.

Publish Neo4j for **that same index** using the existing graph rebuild CLI (see
[graph ingestion](GRAPH_INGESTION.md)); configure the read-only runtime identity and
matching corpus ID. Evaluation does not write, extract or silently synchronize Neo4j.

## Paid semantic evaluation

```powershell
python -m evals.run_ragas --mode vector --index evals/ragas_workspace/pilot/vectorstore --allow-judge --output evals/ragas_results/vector-pilot-1.json
python -m evals.run_ragas --mode graph --index evals/ragas_workspace/pilot/vectorstore --allow-judge --output evals/ragas_results/graph-pilot-1.json
python -m evals.run_ragas --mode hybrid --index evals/ragas_workspace/pilot/vectorstore --allow-judge --output evals/ragas_results/hybrid-pilot-1.json
python -m evals.compare_retrieval_modes --index evals/ragas_workspace/pilot/vectorstore --allow-judge --output evals/ragas_results/comparison.json
```

`vector`, `graph`, `hybrid` map to `vector_only`, `graph_only`, `hybrid`.
Full comparison shares the question set, references, corpus manifest, generator/judge
configuration and software versions. It checks these before combining results.
Graph-only uses the existing strict evaluation mode and cannot covertly fall back to
vector retrieval. Production hybrid fallback remains unchanged but is marked as a
retrieval-mode correctness failure. All primary and secondary provider events are
considered. Live web search is disabled per invocation for KB retrieval ablation;
normal application web fallback remains enabled. The original real-KB harness still
covers web fallback/rewrite behavior.

`--allow-judge` explicitly authorizes sending captured answers and document evidence
to the selected provider. Do not use confidential/private corpora without the required
data-processing approval. Artifacts contain questions, answers and actual evidence;
`evals/ragas_results/` and sample workspaces are Git-ignored. Treat custom output paths
as equally sensitive. RAGAS telemetry is disabled and no judge cache is enabled.

Cases/metrics run sequentially. Case budget includes all three mode executions in a
comparison. Each metric has a timeout and bounded Instructor retries; transport retries
are disabled for Groq. Actual billable tokens/cost are not estimated as if measured.
Generation/router/grader may make multiple calls under existing bounded retry rules.
Set `RAGAS_JUDGE_MIN_CALL_INTERVAL_SECONDS` for provider quotas; it paces each judge
LLM request (including within multi-call metrics) and waits before the first call.
This is request pacing, not token accounting: generation calls and Instructor retries
also consume the same provider quota. Increase metric timeout if pacing makes a
multi-context metric exceed the default. Provider failures remain explicit, not scores.

JSON artifacts contain per-case scores/status/variant, aggregates, scored/NA/error
counts, judge provider/model, generator model, effective/requested mode, captured
provenance, latencies, failures, UTC timestamp, package versions, Git revision and
source-tree fingerprint (including uncommitted code). Completed cases checkpoint
atomically. SDK exception types, not messages containing secrets/prompts, are persisted.
Setup failure exits 2; recorded execution/judge failures exit 1. Semantic pass/fail is
only meaningful after calibration; no arbitrary score cutoff is built into the runner.

## Establish a baseline before regression gates

Run at least two genuine judge pilots per mode on the same frozen cohort, with no
judge/pipeline failures and stable applicability. Review variability and choose an
**explicit absolute tolerance**; there is no default. Calibrate each mode separately:

```powershell
# Replace <approved-tolerance> with the reviewed allowance; it is not a preset threshold.
python -m evals.ragas_baselines calibrate evals/ragas_results/vector-pilot-1.json evals/ragas_results/vector-pilot-2.json --tolerance <approved-tolerance> --output evals/ragas_results/vector-baseline.json
python -m evals.ragas_baselines check evals/ragas_results/vector-current.json --baseline evals/ragas_results/vector-baseline.json --output evals/ragas_results/vector-regression.json
```

For each measured metric, the aggregate floor is the minimum observed pilot mean
minus the approved allowance. Required scored case IDs and metric variants are retained:
abstaining on a previously scored case cannot improve the result by dropping it.
Deterministic baseline floors have no LLM-score tolerance. Metrics with no applicable
cases remain uncalibrated, not assigned zero/perfect values. Changed corpus, question
set/references, judge/model/package versions or generation recipe require recalibration.
Code revisions can change during regression testing and remain recorded for audit.

Two pilots only establish an initial lower-envelope guardrail, not statistical confidence.
Expand repeats and reviewed cases before using scores as release criteria. Keep exact
deterministic assertions in pytest regardless of measured semantic floors.

## CI and verification

```powershell
python -m pytest -q
python -m evals.run_evals
python -m evals.run_kb_evals
python -m pytest tests/test_ragas_evaluation.py -q
```

Normal CI retains offline pytest/smoke without optional judge dependencies or secrets.
Adapter tests inject scorers and pipeline outputs solely to test contracts; those doubles
are never production benchmark data. Optional installed-provider contract tests construct
clients but make no network calls. Run semantic pilots in a manually approved trusted
environment, never in fork PRs; retain sensitive artifacts privately. Only use calibrated,
cohort-matched baseline checks as release gates. Live Neo4j and judge checks are not normal
pytest tests.

See [the verification record](RAGAS_VERIFICATION.md) for actual local results and any
unmeasured semantic baselines. Capture-only success is not a RAGAS quality score.
