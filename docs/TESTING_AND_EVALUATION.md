# Testing and Evaluation

## Test Layers

The repository uses several layers of validation:

- unit tests for tool logic, configuration, and helper behavior
- integration-style tests for graph routing, scoped retrieval, and ingestion behavior
- runtime-hardening tests for error handling and health behavior
- Streamlit smoke tests using `streamlit.testing.v1.AppTest`
- evaluation runners for deterministic smoke validation and real KB validation

## Core Validation Commands

```powershell
python -m py_compile app.py agent.py rag_engine.py config.py document_generator.py
python -m compileall -f src tests evals
python -m pytest -q
python -m evals.run_evals
python -m evals.run_kb_evals
```

Latest verified local pytest count in this repository pass: `119 passed`.

## Offline Smoke Evaluation

Script:

```powershell
python -m evals.run_evals
```

Characteristics:

- deterministic mock corpus
- deterministic answer synthesis
- no external API calls
- fast regression signal for packaging and evaluation plumbing

What it proves:

- the smoke harness runs
- golden mock cases still behave as expected

What it does not prove:

- real retrieval quality
- case-study ranking quality
- live LLM answer quality

## Real Knowledge-Base Evaluation

Script:

```powershell
python -m evals.run_kb_evals
```

Characteristics:

- generates or reuses the sample PDF corpus
- creates a temporary evaluation upload fixture
- ingests into an isolated temporary vectorstore
- exercises the real retrieval and graph path
- defaults to retrieval-only mode
- can run in LLM-answer mode only when explicitly enabled and a provider key exists

Important recorded fields:

- `retrieved_sources`
- `retrieved_origins`
- `executed_tools`
- `citation_coverage`
- `no_answer_behavior_ok`
- `retrieval_latency_seconds`
- `answer_latency_seconds`
- `response_mode`

## Manual Validation Matrix

Manual checks are still useful for:

- readability of final answers
- usefulness of inferred gaps
- relevance of ranked case studies
- quality of grouped source traces
- provider-specific answer style differences
- behavior under large prompts and low-evidence scenarios

Useful manual prompts:

- `Compare the healthcare cloud migration and insurance automation projects.`
- `Which documents were used for the previous answer?`
- `Treat uploaded documents as target requirements and numbered PDFs as internal case studies. Return technical requirements, gaps, three case studies and a proposal outline.`
- `What is the CEO's private phone number?`

## Evaluation Dataset Notes

- `evals/golden_questions.yaml` is the real KB golden set
- `evals/run_evals.py` contains the deterministic smoke cases
- `document_generator.py` creates the synthetic internal sample corpus

## Adding a Golden Question

To extend the real KB evaluation:

1. open `evals/golden_questions.yaml`
2. add a new case with a question
3. add expected source files and any expected origins or tools
4. rerun `python -m evals.run_kb_evals`

Keep the expected sources tied to files that actually exist in the generated sample corpus or evaluation upload fixture.

## How to Interpret Failures

- `py_compile` or `compileall` failures usually indicate syntax or import breakage
- `pytest` failures indicate contract regressions in runtime, tools, or UI behavior
- offline smoke failures indicate broken evaluation plumbing or stale deterministic expectations
- real KB evaluation failures indicate retrieval, routing, or evaluation-definition drift

## What the Evaluations Do and Do Not Prove

They do prove:

- the indexed sample corpus is loadable
- retrieval and graph execution work end to end
- citations and no-answer behavior can be checked automatically
- the offline harness and real KB harness both run reproducibly

They do not prove:

- universal answer correctness
- production-scale robustness
- perfect grounding
- performance across a large proprietary document corpus
