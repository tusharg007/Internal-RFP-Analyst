# RFP domain graph ingestion

Implemented on 2026-10-05 against the approved GraphRAG/RAGAS upgrade plan.
This phase adds ingestion only. No query routing, graph retrieval, RAGAS,
Chroma embedding/search changes, prompt changes or final-answer lineage claims.
This is the historical ingestion-phase report. The later opt-in retrieval layer
is documented in [GRAPH_RETRIEVAL.md](GRAPH_RETRIEVAL.md); vector-only remains the default.
This is the historical ingestion-phase report. The later opt-in retrieval layer
is documented in [GRAPH_RETRIEVAL.md](GRAPH_RETRIEVAL.md); vector-only remains the default.

## Corpus inspection and extraction boundary

Inspected all 30 public sample PDF pages, their generator, PDF loading/chunking,
the live ingestion facade, project catalog/comparison, requirement/gap/case-study
tools and proposal generation before implementing extraction. The samples have
cover-page titles/document types/industries, page-2 technologies and timelines,
and page-3 budgets/outcomes. Clients are anonymized; there is no reliable shared
Client identity. The evaluation upload is a target RFP, not a delivered project.

The public samples validated with the current 512/50 chunk configuration produce:

| Record | Count |
| --- | ---: |
| Documents / indexed chunks | 10 / 53 |
| Projects | 10 |
| Domain entities, including projects | 79 |
| Assertions | 169 |
| Industry / total duration / budget assertions | 10 / 10 / 10 |
| Outcome / technology / framework assertions | 40 / 79 / 20 |
| Total bounded ingestion rows, including identity supports | 321 |

Counts are this fixture's observations, not corpus-completeness guarantees.
Extraction is deliberately conservative and coverage is recorded as
`conservative_partial`. Unknown products, free-form uploads and unsupported
sections remain vector-searchable but are not forced into graph facts. This is
not generic entity extraction or a semantic correctness benchmark.

## Allowed schema

Keep `RFPDocument` and `RFPChunk` foundation anchors. Domain entities use one
fixed `RFPDomainEntity` label with a validated, indexed `kind` enum:
`Project`, `Requirement`, `Technology`, `Industry`, `ComplianceFramework`.
This implements the plan's logical node types without dynamic LLM-supplied labels.
Project identity is document-scoped; a shared title does not merge two projects.
Requirement identity includes document, page and verbatim requirement text.
Canonical technologies/industries/frameworks share identities within a corpus
version, across documents. Entity IDs are deterministic SHA-256 values.

Fixed physical relationships:

```text
Project/Requirement -HAS_ASSERTION-> RFPAssertion -OBJECT-> canonical domain entity
                               \-SUPPORTED_BY {start,end,span_scope}-> RFPChunk
Project/Requirement -SUPPORTED_BY {start,end,span_scope}-> identity-witness chunk
Project/Requirement -IN_DOCUMENT-> RFPDocument
RFPChunk -IN_DOCUMENT-> RFPDocument
```

Allowed assertion predicates are `uses_technology`, `in_industry`,
`mentions_framework`, `requires_technology`, `requires_framework`, `timeline`,
`budget`, `outcome`. Scalar assertions do not have an object edge. Framework
mentions have `proposed_control` modality; they do not establish compliance.
Outcomes preserve explicit `achieved`, `projected`, `estimated` or `unknown`
wording. Duration units and budget currency are retained without conversions.
No Client/Capability/Outcome/RFP ontology expansion or inferred SATISFIES edges.

## Provenance and validation

`IndexedChunk` requires existing metadata: `chunk_id`, `source_file`,
`document_origin`, `file_hash`, zero-based `page`, `chunk_index`, and content.
Missing or conflicting metadata fails the rebuild rather than assigning defaults.
The original Chroma chunk ID is retained unchanged; it remains path-dependent.
A separate portable evidence alias includes document identity, page, chunk index
and raw UTF-8 content SHA-256. Document identity includes filename, origin and
PDF byte hash. Changing source bytes, origin, page, split or legacy mapping
changes the graph snapshot version. Rendering a citation still uses page + 1.

Legacy chunks have no original page offsets. New supports therefore explicitly
use **chunk-local** half-open offsets (`span_scope=chunk`), not fabricated page
coordinates. Every quote is checked against the original indexed text. Project
titles and requirements have their own identity supports. Object entities are
supported through validated assertions; standalone unsupported entities fail.

Extraction uses frozen, extra-forbidden Pydantic models. The default worker uses
deterministic parsing of the actual known sections and curated alias registries.
Microsoft Azure / Azure / MS Azure share one canonical entity. Azure SQL and
other specific products remain distinct; product use does not imply use of every
provider product. Registry edits require a normalization-version update.

`StructuredLLMExtractor` is available through dependency injection, not enabled
automatically by an API key. Inject a temperature-zero model and an explicit
model/version identifier. Its output must satisfy the same schema, exact quotes,
offsets, allowed entities, subject/predicate rules and outcome modality checks.
It cannot return labels, relationships or Cypher. Unknown entities, confidence
below 0.85, extra fields, invalid JSON, fabricated quotes and provider failures
reject the **whole** snapshot with `ExtractionRejected.issues`. The CLI emits a
sanitized quarantine report (chunk IDs and reason codes); it does not write a
persistent content-bearing quarantine archive. Confidence is an extraction
gate, not a calibrated probability or proof of entailment.

## Transaction and lifecycle design

`IngestionRepository` isolates the service from driver details;
`Neo4jIngestionRepository` extends the existing lazy, injected-driver lifecycle.
`NullIngestionRepository` explicitly reports disabled/unconfigured states.
No global connection state, automatic schema writes or application startup probe.

Initialize the domain migration explicitly with the admin identity. It adds four
unique constraints (entity, assertion, snapshot and corpus head) and two domain
indexes, alongside the existing foundation constraints/indexes. Writes verify
constraint type, label and composite properties, not just constraint names.

A rebuild validates and freezes the complete bounded corpus before data writes.
Inside ONE explicit Neo4j data transaction it locks the corpus head, checks the
expected prior version, MERGEs immutable anchors/entities/assertions/supports,
checks persisted values and relationship counts, records the snapshot manifest,
then publishes its version. Any failure rolls everything back. Competing stale
publications are rejected. Retrying an identical committed snapshot after an
uncertain acknowledgement is safe and does not duplicate facts. There are no
hidden automatic driver retries; retry/reconcile explicitly after an outage.
[Neo4j explicit transaction behavior](https://neo4j.com/docs/python-manual/current/transactions/).

Snapshot manifests include source-content digest, extraction/normalization/format
versions, counts and server `ingested_at`. Exact historical chunker/embedding
versions were not stored in the legacy index, so they are explicitly marked
unspecified rather than inferred from today's config.

The entire rebuild must fit `NEO4J_MAX_BATCH_SIZE` (default 500, validated maximum
1000) counting documents, chunks, entities, assertions and identity supports.
Index export is separately bounded to 1000 chunks. Oversized corpora fail before
data writes; **large-corpus multi-transaction staging/worker scaling is deferred**.
This intentional small-corpus implementation uses atomic publication rather than
claiming a production-scale background queue or distributed Chroma/Neo4j commit.

## Synchronization, deletion and recovery

After successful live Chroma publication, `rag_engine.ingest_documents` records
`graph_sync/pending.json` only when the graph flag is enabled. It contains a
content digest and existing chunk IDs, not PDF text, prompts, source paths,
credentials or filenames. File replacement is atomic. Recording failure logs a
redacted warning and cannot roll back a successful vector ingestion. Failed
vector publication never queues graph synchronization. The directory is ignored
by Git. There is no automatic extraction daemon in the Streamlit process.

The pending file is a last-writer-wins reconciliation hint, not an authoritative
distributed queue. The rebuild operation always exports the actual current index
whether that hint exists or not. It does not consume/acknowledge the hint yet;
successful graph publication is established by the repository result/head and
the matching Neo4j snapshot manifest. Schedule/retry the worker externally if
automatic synchronization is required.

Rebuilding after document removal publishes only the remaining documents.
An empty indexed collection explicitly publishes an empty active snapshot.
Re-ingesting identical bytes can reuse an existing immutable snapshot; changed
bytes produce a new document/version. Historic snapshots are retained as audit
history, **not active facts**. Hard erasure/retention-policy cleanup is deferred.
Future graph retrieval MUST filter by active version and verify its indexed
digest against the current corpus before using any facts, including after a
deletion or synchronization failure. No such graph query path is enabled here.

There is no portable atomic snapshot API spanning concurrent Chroma writers and
Neo4j publication. Export compares two bounded captures to detect ordinary
changes; this is not a guarantee against ABA changes. **Quiesce PDF/vector
ingestion during rebuild**, or use a captured compatible immutable index copy.
An incompatible/missing/inaccessible index is an error, never a new collection.
An extraction/schema/integrity error preserves the old graph head. Neo4j outages
return non-success status; the existing application continues using Chroma.

## Operations

Install the existing optional graph dependency and configure the distinct
admin/worker identities described in `NEO4J_FOUNDATION.md`.

```powershell
python -m pip install -r requirements-graph.txt
# Set NEO4J_ENABLED=true, verified URI/database and admin/worker credentials in .env.
# Migration is explicit; the worker cannot initialize schema with its reader role.
python -m rfp_analyst.graph init-schema

# Validate the EXISTING collection without graph writes (also works with flag off):
python -m rfp_analyst.graph rebuild --dry-run --corpus-id internal-rfp

# Publish the bounded validated corpus; no embedding or similarity search:
python -m rfp_analyst.graph rebuild --corpus-id internal-rfp
# Optional: --persist-dir <existing-index> --collection rfp_kb_v2
```

The CLI uses `get_collection(..., embedding_function=None)` and `Collection.get`
with documents/metadata; never get-or-create, upsert, reset or embedding. The
Chroma client may manage its own internal metadata, so dry-run means no GRAPH
writes, not a promise of filesystem-read-only Chroma access.
[Chroma Collection API](https://docs.trychroma.com/reference/python/collection).
Exit codes: 0 validated/published, 1 unavailable/configuration/index/integrity
failure, 2 quarantined extraction. Disabled graph support does not pretend to
persist anything. The dry-run flag is only valid for rebuild.

Service composition for an explicitly approved LLM ingestion job:

```python
from rfp_analyst.graph.extraction import StructuredLLMExtractor
from rfp_analyst.graph.ingestion import read_indexed_corpus, rebuild_graph
from rfp_analyst.graph.repository import create_ingestion_repository

inputs = read_indexed_corpus(existing_collection)  # caller owns collection lifecycle
extractor = StructuredLLMExtractor(temperature_zero_llm, model_version="provider/model/revision")
with create_ingestion_repository(role="writer") as repository:
    result = rebuild_graph(repository, "internal-rfp", inputs, extractor=extractor)
    # Check result['published']; model output never becomes a query.
```

Review provider privacy/retention before sending private chunks to an LLM.
The default CLI is deterministic, local and requires no LLM credentials.
Enforce Neo4j privileges server-side; application role guards are not RBAC.
Avoid granting schema/admin privileges to the ingestion or runtime account.

## Verification and files

New implementation: `graph/extraction.py`, `graph/ingestion.py`,
`graph/repository.py`, `graph/__main__.py`. Modified: foundation `graph/models.py`
(explicit coordinate scope), `rag_engine.py` (non-blocking manifest hook),
`.gitignore`, `.env.example`, `config.py` comments and foundation documentation.
The existing foundation dependency/settings/store/schema files are preserved.

Tests: 44 added cases in `tests/test_graph_ingestion.py` cover repeat/retry, duplicate and
cross-document entities, product/provider separation, malformed and valid LLM
outputs, service failures, complete rollback, stale publication, deletion/empty
corpus/re-ingestion, metadata and version integrity, domain schema, source export,
concurrent export changes, disabled CLI, pending-job failures and generated public
PDFs. An isolated real Chroma collection test verifies export/rebuild and the
dry-run CLI without changing documents, metadata or existing vectors. Two
additional live-server cases extend `test_graph_store_integration.py`.
Docker support remains in `compose.neo4j-test.yml`; no credentials are committed.

Recorded checks on 2026-10-05:

| Command | Result |
| --- | --- |
| `python -m pytest -q` | Final rerun: 254 passed, 4 opt-in Neo4j tests skipped; 41.37 seconds. No existing pytest failures. |
| `python -m compileall -q src config.py rag_engine.py tests evals` | Passed. |
| `python -m ruff check src/rfp_analyst/graph rag_engine.py tests/test_graph_ingestion.py tests/test_graph_store_integration.py` | Passed. |
| `python -m ruff check .` | Same 10 pre-existing failures in untouched files; no new lint failures. |
| `python -m evals.run_evals` | 3/3 offline smoke; mock corpus only. |
| `python -m evals.run_kb_evals` | 9/9, retrieval-only; 58.13 seconds measured query latency. No graph retrieval or live-answer benchmark. |
| `git diff --check` / `python -m rfp_analyst.graph --help` | Passed. |

The earlier design baseline observed 8/9 on the real KB evaluation; helper LLM
and web decisions can vary between runs even in its retrieval-only mode. This
9/9 run does not demonstrate an ingestion-caused quality improvement: retrieval
and routing have not been changed, and golden questions were not edited.

The existing `langchain-community` deprecation warning remains. Full-repo lint
failures are unused imports/variables in `agent.py`, `document_generator.py`,
`evals/run_kb_evals.py`, legacy `agent/runtime.py`, and `test_runtime_hardening.py`.
Docker is unavailable here; actual Neo4j template execution remains unverified
until the four opt-in integration tests are run against the disposable service.
The local live `vectorstore` is access-denied, so it was not modified/rebuilt.
Public-PDF extraction, injected indexed-corpus tests and an isolated actual Chroma
collection supplied safe validation without reading private uploaded PDF contents.

Stop at ingestion: do not turn on GraphRAG routing, alter vector behavior, import
RAGAS, replace existing tools/verifiers, or assert formal final-claim provenance.
