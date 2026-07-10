# Evaluation Suite

This project includes a deterministic evaluation suite in `evals/` so we can validate retrieval, tool orchestration, grounding, and fallback behavior without depending on paid model calls.

## What it covers

- Direct fact lookup
- Project comparison
- Budget extraction
- Timeline extraction
- Tech stack search
- Compliance framework search
- Ambiguous question handling
- No-answer / insufficient evidence behavior
- Proposal outline generation
- Multi-document synthesis

## Files

- `evals/golden_questions.yaml`: the golden question set
- `evals/run_evals.py`: deterministic runner that uses mocked retrieval and non-API answer synthesis
- `evals/metrics.py`: aggregate metric calculations
- `evals/results.json`: generated output after an eval run

## Metrics

- `retrieval_hit_rate`: how often expected source documents are retrieved
- `citation_coverage`: how often answers that should cite sources actually include citations
- `grounded_answer_score`: fraction of answers that pass grounding verification
- `average_latency`: average per-question runtime in milliseconds
- `tool_call_count`: average number of tool steps used per question
- `failure_rate`: fraction of eval cases that fail expectations

## UI snapshot

If `evals/results.json` exists, the Streamlit sidebar shows an `Evaluation Snapshot` section with the latest metric summary and pass count.

## Notes

The eval runner is intentionally deterministic. It uses a small in-memory corpus and mocked retrieval behavior so the suite can run in CI or on local machines without external API dependencies.
