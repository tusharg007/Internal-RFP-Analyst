# Corpus-matched public evaluation environment

Verified on 2026-10-06. Environment preparation succeeded; the application capture
did **not** complete. There are no valid vector/graph/hybrid quality conclusions
from this run, and no RAGAS judge was invoked.

## Frozen Chroma manifest

- Existing index: `evals/ragas_workspace/20261005-public-pilot/vectorstore`.
- Existing collection: `rfp_kb_v2`.
- Documents: **11**; chunks: **54**.
- Ten public sample documents contribute 53 chunks. The remaining chunk belongs
  to the existing synthetic `eval_target_rfp.pdf`, whose origin is deliberately
  `upload` so that uploaded-target evaluation retains its original semantics.
  No private uploads were used or transmitted to an application model.
- Logical indexed-corpus SHA-256:
  `641e9d4dad22e33140475ac40b0245dde1edb3ed6e2acc0c5fa8e75919d9d39f`.
  This is the existing digest of sorted complete indexed records, including text
  and provenance; it is not a physical SQLite-file checksum.
- Freeze time in the unchanged lock: `2026-10-05T01:25:18.360602+00:00`.
- Legacy embedding version was not recorded in the original index. It was not
  guessed or recreated; graph publication used the original indexed text and
  metadata, without PDF regeneration, embedding, or Chroma upserts.

| Document | Origin | Chunks |
|---|---|---:|
| 01_Banking_Sector_Digital_Audit_2024.pdf | sample | 6 |
| 02_Healthcare_Data_Migration_to_Azure_Cloud.pdf | sample | 6 |
| 03_Retail_Supply_Chain_Analytics_Platform.pdf | sample | 5 |
| 04_Insurance_Claims_Processing_Automation.pdf | sample | 5 |
| 05_Telecom_Network_Optimization_with_AI.pdf | sample | 5 |
| 06_Government_Tax_Filing_Modernization.pdf | sample | 5 |
| 07_Pharma_Clinical_Trial_Data_Platform.pdf | sample | 5 |
| 08_Energy_Sector_ESG_Reporting_Dashboard.pdf | sample | 6 |
| 09_Financial_Services_Anti-Fraud_Detection_System.pdf | sample | 5 |
| 10_Manufacturing_IoT_Predictive_Maintenance_Platform.pdf | sample | 5 |
| eval_target_rfp.pdf | upload (public synthetic) | 1 |

Every document's original file hash, deterministic document ID, and every
chunk's ID, evidence ID, text hash, original chunk index, zero-based page,
origin and span are recorded in the machine-readable
[`environment.json`](../evals/results/matched-public-20261006/environment.json).
File hashes are preserved ingestion metadata, not newly computed PDF hashes.

Unchanged frozen file checksums:

| File | SHA-256 |
|---|---|
| retrieval_questions.yaml | `7846cea3aa3eccc3b281d44dd1a3eecff51d65b08265f045ff1e34a896bc6689` |
| retrieval_benchmark.lock.json | `a76610f63e0047e13033cbd45cb7b732c75ff17e36f316b6b557bc07536c0eb7` |
| golden_questions.yaml | `13eb016110e872fa2ddf6f8e47a6c451b1fbb0969ecd570fc1125d3290f6ee2f` |

## Isolated Neo4j publication and parity

The existing composite constraints, MERGE/MATCH statements, and corpus-head
publication transaction are scoped by corpus ID and version. They support an
isolated namespace in the existing Aura database; a disposable database was
therefore unnecessary for this trusted-operator evaluation.

- New evaluation corpus: **`rfp-eval-frozen-641e9d4dad22`**.
- Graph version:
  `d99fc9a7b295f7c9c14cae3de266e0fa97af8dc9dc806ce164fc5a5482d8ce55`.
- Existing deterministic extractor: `rfp-sections-v2`.
- Existing canonicalization: `curated-aliases-v1`.
- Existing atomic `repository.publish(snapshot, expected_version=None)` published
  the new namespace. No global schema initialization, pruning, deletion or live
  corpus republishing was performed. Writer credentials were resolved through
  the existing separate ingestion-role configuration.
- Capture subsequently **reused** the evaluation namespace without writes;
  the current environment report correctly labels that verification invocation
  `reused_without_writes`.
- **Exact parity passed before and after capture**: 11 documents, 54 chunks,
  all persisted document/chunk properties, text hashes, complete provenance
  identities, the indexed digest, active version and exact chunk/document edges.
  Equal counts alone were not accepted as parity. Cross-namespace edges are
  rejected by the verification.

| Record set | Expected SHA-256 | Actual SHA-256 |
|---|---|---|
| Documents | `8902b322be74133c00c36f41a76174536c72931d9f0fc78ae11fccea0ec0aabd` | identical |
| Chunks | `de085fb8476ebc09fd0ff3a26e074ca07979310924c504e39255d32ebadca988` | identical |

These record-set hashes cover graph-shaped provenance properties. They differ
from the logical indexed-corpus digest because the indexed records contain
original text and chunk indices, while graph anchors contain derived evidence
IDs and text hashes. Each representation was compared like-for-like.

### Protected existing corpora

`internal-rfp` was **not deleted, replaced or republished**. Before publication,
after publication and after the stopped capture, its complete namespace node
and relationship fingerprints were identical:

- Nodes: 365; relationships: 569 (includes corpus head and snapshot).
- Active version:
  `eddc57727f1f70adb54c3d4d98d6896e4245daaa6cc896f8bfbafc63b88fdfdc`.
- Node SHA-256:
  `e610c788846bc9aa625c133104cbf7948f34bbe14cb6499d1313a27fd3df781c`.
- Relationship SHA-256:
  `eb39381b92932696a5eea8bbcffba022d9974d8dd503b3043c7870e16730d6d5`.

Private node properties were hashed locally, not exported in reports or sent to
an LLM. The root Chroma corpus was not selected, rebuilt or written. The frozen
public index was read-only at the application/API level and its logical digest
was checked again after publication and capture. Chroma client housekeeping is
not a byte-for-byte filesystem immutability guarantee.

## Capture outcome: provider failure, not retrieval failure

[`capture.json`](../evals/results/matched-public-20261006/capture.json) preserves
the run configuration, unchanged freeze, empty completed-mode reports, and the
aborted execution. No previous reports were overwritten.

| Field | Observed value |
|---|---|
| First case | `semantic_modernization` |
| Requested mode | `vector_only` |
| Exact application node | `grade_kb_evidence` |
| Provider | `groq` |
| Model | `openai/gpt-oss-120b` |
| Error | `RateLimitError`, HTTP **429** |
| Status | `stopped_provider_failure` |
| Attempted / completed | **1 attempted, 0 completed / 48 required** |
| Valid paired capture | **false** |
| RAGAS judge calls | **0** |

The coordinator stopped without starting another case/mode. The failure is
explicitly marked `retrieval_quality_failure: false`; no zero-quality score was
assigned to the aborted execution. This run does not establish live graph
retrieval success, since it stopped during the first vector execution before
the graph execution began.

An evaluation-scoped LangChain callback observes provider failures before
ordinary application exception fallbacks can hide them. It raises a dedicated
evaluation abort, checkpoints the partial report, and restores its context.
Application routing, grading, rewriting, generation and grounding policies
remain unchanged. Existing SDK retries inside one application invocation are
unchanged; there are no coordinator retries after a provider failure.

## Configuration and reproduction

Only the evaluation process sets `RFP_GRAPH_CORPUS_ID` to the isolated namespace,
verifies actual resolved configuration, and restores the previous environment.
Each frozen case is explicitly forced through the original `vector_only`,
`graph_only`, and `hybrid` runner modes in the existing rotated order. `.env`,
application defaults and the frozen case set were not edited.

After resolving the Groq quota/rate limit, start a **new** capture directory to
preserve this partial run:

```powershell
.venv\Scripts\python.exe -m evals.matched_environment --capture --output-dir evals/results/matched-public-next-run
```

Omit `--publish` to verify/reuse the existing evaluation namespace without
writes. The initial setup used:

```powershell
.venv\Scripts\python.exe -m evals.matched_environment --publish
```

The default index/collection/namespace are the exact matched public pair listed
above. The command rejects `internal-rfp`, rejects a changed freeze, verifies
parity before any application model call, preserves an existing capture file,
and never initializes a semantic judge. The first diagnostic attempt failed
with a Neo4j driver `TypeError` before any publication; the evaluation-only
transaction-call usage was corrected, and publication/parity then succeeded.

A valid comparison capture requires all **48 completed executions**, preserved
frozen cases, exact corpus parity, no application/provider/backend failures,
and an unchanged protected corpus. Unsupported deterministic graph plans are
reported limitations, not transport failures or proof of vector superiority.
Passing environment setup or unit tests is not a quality comparison result.

## Added files and verification

Only evaluation infrastructure, its tests and this report were added:

- `evals/matched_environment.py`: bounded read-only parity inspection, isolated
  publication through existing ingestion, process-local configuration and
  capture-only coordinator.
- `evals/provider_guard.py`: evaluation-only provider-error observation and
  clean stop, with redacted failure metadata.
- `tests/test_matched_environment.py`: 19 tests covering exact parity,
  equal-count mismatches, duplicate/missing/cross-namespace relationships,
  protected-corpus fingerprints, namespace rejection, callback context/thread
  propagation, partial checkpoints and 48-execution validity.
- `docs/FROZEN_PUBLIC_EVALUATION.md` and the new public run reports.

Executed results:

| Command | Result |
|---|---|
| `python -m evals.matched_environment --publish` | Successful isolated publication; parity true; live corpus unchanged (after the corrected diagnostic attempt) |
| `python -m evals.matched_environment --capture` | Exit 1, intentional clean stop on HTTP 429; partial report preserved |
| `python -m pytest tests/test_matched_environment.py -q` | 19 passed |
| `python -m pytest -q` | **524 passed, 5 skipped**, 70.66 seconds |
| `python -m ruff check .` | Passed |
| `python -m compileall -q evals/provider_guard.py evals/matched_environment.py tests/test_matched_environment.py` | Passed |
| `python -m evals.run_evals` | Offline smoke **3/3**; not real retrieval/answer-quality evidence |

No retrieval implementation, routing, prompts, grounding policy, ingestion
implementation, frozen questions, lock or application configuration was changed
in this task. Namespace isolation is an application/data-model safeguard, not
a separate tenant authorization boundary; a separately secured database would
still be appropriate for untrusted operators or stronger tenancy requirements.
