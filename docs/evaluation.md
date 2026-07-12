# Evaluation Modes

This project now separates two very different kinds of evaluation output.

## Offline Smoke Evaluation

The offline smoke evaluation is intentionally lightweight and deterministic.

- Script: `evals/run_evals.py`
- Data source: a small mock corpus in Python
- Answer generation: deterministic mock synthesis
- Purpose: catch obvious packaging, formatting, and snapshot-regression issues quickly

This evaluation is **not** evidence of real retrieval quality. A perfect score here only means the mock harness still behaves as expected.

## Real KB Evaluation

The real knowledge-base evaluation runs against the actual sample PDF corpus, the generated evaluation upload fixture, and the Chroma-backed retrieval stack.

- Script: `evals/run_kb_evals.py`
- Data source: generated sample PDFs in `data/documents` plus a temporary evaluation upload fixture
- Retrieval path: real ingestion into an isolated temporary vectorstore plus the actual graph and retrieval layer used by the app
- Metrics recorded per case:
  - retrieved sources
  - citation coverage against expected source files
  - no-answer behavior
  - retrieval latency
  - optional LLM-answer latency when an API key is configured

By default, the real KB evaluation runs in **retrieval-only mode**. This keeps evaluation available even when no `GROQ_API_KEY` or `GOOGLE_API_KEY` is configured.

If either API key is present, the script can also run in **LLM-answer mode** and record answer previews plus answer latency.

## Running the Evaluations

### Offline smoke evaluation

```bash
python -m evals.run_evals
```

This writes `evals/offline_smoke_results.json`.

### Real KB evaluation

```bash
python -m evals.run_kb_evals
```

This script will:

1. ensure the sample PDFs exist
2. ingest them into an isolated temporary Chroma index
3. run golden questions through the actual retrieval layer
4. persist `evals/real_kb_results.json`

The isolated vectorstore and temporary upload directory keep evaluation from depending on local uploaded files or replacing/locking the app's local `vectorstore/` directory on Windows.

## Streamlit UI

The sidebar shows these snapshots separately:

- `Offline Smoke Evaluation`
- `Real KB Evaluation`

If no real KB evaluation has been run yet, the app explicitly shows:

`No real KB evaluation run found`

That distinction is important because only the real KB evaluation exercises the actual retrieval stack.
