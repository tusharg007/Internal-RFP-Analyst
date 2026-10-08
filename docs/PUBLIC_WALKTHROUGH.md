# Actual public-data walkthrough

Captured and reviewed on 2026-10-08. The screenshots show the real Streamlit app,
not an injected answer, fake provider, edited DOM or image-generated UI.

## Sequence and interpretation

| Step | Artifact | Observed behavior |
| --- | --- | --- |
| Ready | [01-public-ready.png](assets/walkthrough/01-public-ready.png) | Frozen synthetic public corpus; 11 documents, 54 chunks. |
| Question/answer | [02-azure-answer.png](assets/walkthrough/02-azure-answer.png) | Actual Azure relationship question and source-cited Groq answer. |
| Explainability | [03-retrieval-explanation.png](assets/walkthrough/03-retrieval-explanation.png) | Requested graph_only; effective vector_only; graph unavailable within existing deadline. |
| Grounding | [04-grounding-check.png](assets/walkthrough/04-grounding-check.png) | Two grounding checks, each reporting five checked claims and zero unsupported claims. |

The captured response is **not a live GraphRAG success**. Neo4j manifest inspection
worked, but runtime graph projection hit its bounded deadline. The untouched
application fallback kept the user-facing answer available through Chroma. A
separate strict read-only graph check returned `timeout`, zero paths and zero
supporting chunks. No deadline, Cypher template, grader or grounding policy was
relaxed for presentation.

The explanation panel displays only recorded operational metadata: intent, scope,
requested/effective mode, graph query/version/fallback, bounded path relationships
and modality, source/chunk/evidence IDs, grades, retry count and grounding status.
Unknown values remain unknown. `claim_to_chunk_links: not_recorded` preserves the
known lineage boundary. Graph paths do not establish semantic entailment for
every generated claim. Variable trace-card text is HTML-escaped.

## Reproduce the UI capture

Use the **existing approved frozen public index** and already-published matching
evaluation graph. The index itself is intentionally not committed. A fresh PDF
regeneration may have different byte hashes and is not silently treated as the
same frozen corpus. Obtain/verify the original fixture before following this exact
capture recipe; do not refreeze questions or republish a live corpus to bypass it.

```powershell
$env:RFP_PUBLIC_DEMO_INDEX = "PATH_TO_EXISTING_FROZEN_CHROMA_INDEX"
$env:RFP_RETRIEVAL_MODE = "graph_only"
python -m streamlit run tools/public_demo_app.py --server.port 8512 --server.address 127.0.0.1 --server.headless true
```

The wrapper validates the exact logical indexed digest before importing the app.
It binds process-local data/upload directories, selects the isolated evaluation
graph namespace, and disables document generation, uploads and ingestion. It
does not mock the app, rewrite `.env`, publish graph snapshots or alter the private
index. This is a trusted local demo safeguard, **not a production ACL or tenant boundary**.

In another terminal, optionally install browser-capture tooling separately:

```powershell
python -m pip install playwright
# Uses an existing Microsoft Edge installation; no synthetic screenshots.
python tools/capture_walkthrough.py --url http://127.0.0.1:8512 --output docs/assets/walkthrough --submit
```

`--submit` makes a real application request and consumes the configured provider's
quota. Browser tooling is not an application dependency. The script refuses a
non-local URL or an unavailable public app and checkpoints an incomplete capture
on failure. Use a new output directory when preserving an earlier run.

## Corpus protection

- Chroma collection: `rfp_kb_v2`, 11 documents / 54 chunks, including public
  `eval_target_rfp.pdf` with origin `upload`.
- Indexed digest: `641e9d4dad22e33140475ac40b0245dde1edb3ed6e2acc0c5fa8e75919d9d39f`.
- Evaluation graph: `rfp-eval-frozen-641e9d4dad22`.
- Graph version: `d99fc9a7b295f7c9c14cae3de266e0fa97af8dc9dc806ce164fc5a5482d8ce55`.
- Exact before/after document and chunk manifests match. Equal counts alone are
  not accepted. Existing namespace was reused without writes.
- Complete protected `internal-rfp` namespace fingerprints matched before/after;
  it was not republished, pruned or deleted.

The [machine-checkable integrity receipt](assets/walkthrough/corpus-integrity.json)
contains public document names, digests, exact parity results and only hashes of
the protected namespace—not private records, credentials or local paths.

```powershell
python -m evals.matched_environment --output-dir PATH_TO_NEW_BEFORE_REPORT
# Capture; do not use --publish or --capture on the parity-only verification.
python -m evals.matched_environment --output-dir PATH_TO_NEW_AFTER_REPORT
python tools/verify_walkthrough.py --before PATH_TO_NEW_BEFORE_REPORT/environment.json --after PATH_TO_NEW_AFTER_REPORT/environment.json --output PATH_TO_NEW_RECEIPT.json
```

The current parity CLI uses its existing frozen-index convention; do not point it
at a private index. [Public environment protocol](FROZEN_PUBLIC_EVALUATION.md).

## Real RAGAS attempt, not a manufactured score

The opt-in post-hoc probe reads an immutable **original matched capture**. It
selects the real saved `semantic_modernization` vector-only execution and preserves
its exact generation contexts, answer and independent reference. No second search,
answer generation, graph publication or application fallback call occurs.

```powershell
python -m pip install -r requirements-eval.txt
# Configure judge settings in environment/secret manager, never in source.
python tools/judge_public_smoke.py --input PATH_TO_ORIGINAL_MATCHED_CAPTURE.json --output PATH_TO_NEW_JUDGE_REPORT.json --allow-judge
```

This is a one-case compatibility smoke, not a 48-run ablation. Provider/model and
source-capture versions are recorded separately. Availability was confirmed through
Google `models.list()`. The operator confirmed free-tier/no-billing projects; no
paid upgrade or provider switch was performed. An API key alone cannot prove billing.

The first attempt exposed missing `jsonref` in the optional Google structured-output
dependencies. After the dependency correction, Google `gemini-3.8-flash` returned
HTTP 503 during Faithfulness. The remaining metrics were unrun and all scores are
null. The result is [preserved here](assets/walkthrough/ragas-smoke-after-dependency-fix.json).
The [initial dependency diagnostic](assets/walkthrough/ragas-smoke.json) is retained
separately. There is no usable semantic baseline, threshold or graph superiority claim.

## What the walkthrough does not validate

It does not prove cloud graph-read reliability, full-corpus recall, Neo4j
database-level RBAC, production multi-tenancy, formal claim-level provenance,
semantic entailment, or a winning retrieval mode. Those gates remain explicit in
the [completion review](COMPLETION_REVIEW.md).
