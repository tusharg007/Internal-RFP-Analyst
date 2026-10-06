# Optional Neo4j persistence foundation

This phase adds a standalone persistence interface, not GraphRAG. It does not
change Chroma search, LangGraph routing, tools, prompts, app readiness, or the
existing evaluation harness. The foundation alone has no extraction, graph retrieval or RAGAS.
The application does not create a graph connection on startup, even if configured.

The additive ingestion phase is documented in [GRAPH_INGESTION.md](GRAPH_INGESTION.md).
Use its separate ingestion repository/CLI for the domain migration and snapshot
publication; the original foundation interface remains compatible.
The subsequent optional retrieval implementation is documented in
[GRAPH_RETRIEVAL.md](GRAPH_RETRIEVAL.md). This foundation report describes its
original phase boundary, not the full set of later features.
The subsequent optional retrieval implementation is documented in
[GRAPH_RETRIEVAL.md](GRAPH_RETRIEVAL.md). This foundation report describes its
original phase boundary, not the full set of later features.

## Installation and configuration

Install the existing application as usual. To use graph persistence, additionally run:

```powershell
python -m pip install -r requirements-graph.txt
# Alternatively, install the optional package extra:
python -m pip install -e ".[graph]"
```

The foundation uses `neo4j>=6.3.1,<7`, tested locally with 6.3.1. The official
driver supports the project's Python 3.11 minimum. No deprecated `neo4j-driver`,
`py2neo` or `langchain_community.graphs` wrapper is used.
[Official driver package](https://pypi.org/project/neo4j/).

The current `langchain-neo4j` package is the appropriate maintained integration
if a future LangChain wrapper is needed. It is not installed in this phase:
a persistence-only boundary benefits from a direct driver and does not need
chat history, vector replacement, arbitrary queries or LLM-generated Cypher.
[LangChain Neo4j integration](https://pypi.org/project/langchain-neo4j/).

Set `NEO4J_ENABLED=true` and the URI/database in `.env` or deployment settings.
Keep it false for existing vector-only operation. This is an infrastructure flag,
not a GraphRAG routing flag. Remote URIs must use verified TLS (`neo4j+s://` or
`bolt+s://`); unencrypted URIs are accepted only for loopback development.
Passwords must not be embedded in the URI. `.env.example` contains no credentials.

| Role | Username/password settings | Allowed repository operations |
| --- | --- | --- |
| Reader (default) | `NEO4J_USERNAME`, `NEO4J_PASSWORD` | Connectivity/database health, close. |
| Writer | `NEO4J_INGEST_USERNAME`, `NEO4J_INGEST_PASSWORD` | Health and provenance ingestion. |
| Admin | `NEO4J_ADMIN_USERNAME`, `NEO4J_ADMIN_PASSWORD` | Health, explicit schema initialization and ingestion. |

There is no credential fallback from reader to writer/admin or vice versa.
Writer/admin settings are resolved only when that role is explicitly requested.
Keep those identities out of the user-question runtime process in production.
Enforce the roles in Neo4j too; application role checks do not grant or prove
database authorization. Community Docker tests use a disposable admin identity
and do not validate production RBAC.

Connection/acquisition/transaction timeouts, pool size and ingestion batch size
are bounded configurable settings. The default driver has no automatic managed
transaction retry loop. Timeouts are per connection/transaction, not a promised
end-to-end ingestion deadline. Invalid enabled settings degrade to an explicit
`invalid_configuration` store when loaded through the factory.

## Interface and lifecycle

Import from `rfp_analyst.graph`:

```python
from rfp_analyst.graph import GraphStore, create_graph_store

# Resolve environment at the composition boundary; no connection on construction.
with create_graph_store() as store:
    status = store.health()
    if not status.available:
        # Continue using the existing application; never treat this as graph evidence.
        print(status.status)
```

`GraphStore` is a small protocol. The implementation owns lazily created drivers
and closes them on context exit/`close()`. Close is idempotent. Injected drivers
are borrowed by default; pass `owns_driver=True` only to transfer ownership.
Alternatively inject `driver_factory(settings)` for tests or a composition root.
There is no global connection cache. Each call creates and closes its own session
and explicit transaction; a per-instance lock coordinates creation/use/close.

Health verifies connectivity and executes a fixed read probe against the configured
database, not just the server's home database. It does not certify schema, semantic
quality, tenant isolation or reader privileges. A failed owned driver is closed;
a later explicit call can create a fresh driver. Borrowed drivers remain caller-owned.
Driver exception contents are not included in health results or repository logs.

Disabled, missing-config, missing-driver, unavailable and closed states are explicit.
They return non-success write results instead of throwing infrastructure failures
into the application. Callers must check `result.ok`; a disabled write is not
a successful persistence operation. Validation, permission and integrity failures
are programming/data errors and deliberately raise safe exceptions.

## Explicit schema initialization

Use a migration/admin process, never automatic initialization in Streamlit:

```python
from rfp_analyst.graph import create_graph_store

with create_graph_store(role="admin") as store:
    result = store.initialize_schema()
    if not result.ok:
        raise RuntimeError(f"Schema initialization did not complete: {result.status}")
```

The schema is limited to `RFPDocument` and `RFPChunk` provenance anchors:

- Composite document uniqueness: corpus ID, corpus version, document ID.
- Composite portable evidence uniqueness: corpus ID, corpus version, evidence ID.
- Composite legacy chunk uniqueness: corpus ID, corpus version, existing chunk ID.
- Document origin/hash indexes and a chunk document/page index.
- Parameterized `IN_DOCUMENT` relationships, created by provenance ingestion.

All DDL uses fixed statements and `IF NOT EXISTS`. Uniqueness constraints supply
their own backing indexes. Initialization validates actual constraint/index
definitions, not just object names. Ingestion refuses to proceed if required
uniqueness constraints are missing or incompatible. Neo4j schemas are not initialized
through LLM text. [Neo4j constraint documentation](https://neo4j.com/docs/cypher-manual/current/schema/constraints/create-constraints/).

DDL statements run separately from data transactions. Initialization can partially
complete if a later statement fails; correct permissions/conflicts and rerun.
Existing objects are not deleted or destructively rewritten. Domain entities and
assertions will require a separately authorized extraction/schema phase.

## Idempotent caller-supplied provenance

`GraphDocument` and `GraphChunk` are immutable validated Pydantic models. The
foundation accepts explicit metadata; it does not read PDFs, create portable IDs,
extract facts, publish an active corpus snapshot or alter existing chunk IDs.

Required shared fields: corpus/version/document ID, source filename, source origin
(`sample` or `upload`) and file SHA-256. Documents also carry kind, revision and
access partition. Chunks carry portable evidence ID, existing Chroma chunk ID,
zero-based page index, text SHA-256 and half-open source span offsets. Full document
text, absolute filesystem paths and prompts are not stored by this foundation.

```python
from rfp_analyst.graph import GraphDocument, GraphChunk, create_graph_store

# validated_documents and validated_chunks are supplied by a future ingestion caller.
with create_graph_store(role="writer") as store:
    result = store.ingest_provenance(validated_documents, validated_chunks)
    if not result.ok:
        # Persist/retry the upstream job later; do not claim the graph was updated.
        print(result.status)
```

Each bounded call writes in one data transaction. A repeated identical identity
does not create duplicate nodes/relationships or overwrite metadata. Identical
duplicates inside a batch are collapsed. Changed metadata for a frozen identity,
conflicting portable/legacy mappings or chunks without a matching document
hash/name/origin cause rollback. A chunk may refer to a document supplied in the
same call or a previously persisted matching document. Versions/corpora are
independent namespaces. Larger ingestions must be orchestrated as bounded calls;
there is no cross-batch atomicity or publication/outbox in this phase.

An infrastructure error during commit can leave its outcome unknown to the caller.
An unavailable result is not a guarantee that the server wrote nothing; repeat
the identical batch safely rather than inventing success or changing its IDs.

Processed counts count unique accepted inputs, not newly created nodes. The schema
and repository preserve structural provenance; they cannot prove span/quote accuracy
without original source text, semantic entailment, ACL correctness or final
claim-to-evidence lineage. Caller authorization is required; access partitions
are metadata, not an authentication system.

## Tests

Unit tests use an in-memory transactional fake, dependency/connection failures,
credential/config isolation and existing Streamlit startup checks:

```powershell
python -m pytest tests/test_graph_store.py -q
```

Optional real Neo4j tests use `compose.neo4j-test.yml` on loopback ports 17474/17687.
No host data volume or application `.env` is used. Set a disposable password in
your shell, then run:

```powershell
# Set NEO4J_TEST_PASSWORD in your shell without committing it.
docker compose -f compose.neo4j-test.yml up -d --wait
$env:RFP_ANALYST_NEO4J_INTEGRATION = "1"
python -m pytest tests/test_graph_store_integration.py -v
Remove-Item Env:\RFP_ANALYST_NEO4J_INTEGRATION
docker compose -f compose.neo4j-test.yml down
```

Tests are skipped unless explicitly enabled. When enabled, missing dependency,
credentials or an unhealthy service fail rather than silently skip. Data cleanup
deletes only each fixture's generated `test-neo4j-foundation-<uuid>` namespace;
schema is left in the disposable database. Never repoint this fixture at a real
corpus. The Docker tag tracks the 5.26 Community line; pin an approved patch/digest
for repeatable CI/deployment. [Neo4j Docker documentation](https://neo4j.com/docs/operations-manual/current/docker/introduction/).

## Scope boundary

No graph traversal/query API, domain extraction, vector/graph fusion, LangGraph
integration, RAGAS, snapshot publication, tenancy or production RBAC provisioning
has been introduced. None is implied by enabling the infrastructure flag.
The existing Chroma-only application remains the active answering system.

## Implementation verification: 2026-10-05

### Files changed in this foundation phase

Modified: `config.py`, `.env.example`, `pyproject.toml`, `requirements.txt`.

Added:

- `requirements-graph.txt`: optional current official driver dependency.
- `src/rfp_analyst/graph/__init__.py`, `models.py`, `settings.py`, `schema.py`,
  `store.py`: interface, validated provenance, instance-local settings, fixed
  constraints/indexes and lazy persistence/lifecycle implementation.
- `tests/test_graph_store.py`: 65 unit/startup cases, including parameterized cases.
- `tests/test_graph_store_integration.py`: two opt-in real-server tests.
- `compose.neo4j-test.yml`: isolated loopback Docker test service.
- `docs/NEO4J_FOUNDATION.md`: setup, usage, safety boundaries and this report.

The pre-existing upgrade plan was read but not edited. No application, agent,
Chroma retrieval, existing test, evaluation dataset or CI workflow code was changed.

### Commands and final results

The commands used the existing Windows `.venv` (Python 3.12.14); `python` below
means `.\\.venv\\Scripts\\python.exe`. Dependency installation initially hit the
sandbox's network restriction, then succeeded with approved network access.
Installed Neo4j 6.3.1, Ruff 0.16.10 and the driver's `pytz` dependency without
upgrading the existing LangChain/Chroma stack.

| Command/check | Final result |
| --- | --- |
| `python -m pip install -r requirements-graph.txt "ruff>=0.15,<1"` | Passed after approved network rerun. |
| `python -m py_compile app.py agent.py rag_engine.py config.py document_generator.py` | Passed. |
| `python -m compileall -q -f src tests evals` | Passed; verbose compileall also ran successfully. |
| `python -m pytest tests/test_graph_store.py tests/test_graph_store_integration.py -q` | **65 passed, 2 skipped**, 5.10 s; one existing deprecation warning. |
| `python -m pytest -q` | **210 passed, 2 skipped**, 38.04 s; all 145 existing cases retained. One existing `langchain-community` deprecation warning. |
| `python -m evals.run_evals` | **3/3**, pass rate 1.0, exit 0; mock smoke evaluation, not semantic quality. |
| `python -m ruff check config.py src/rfp_analyst/graph tests/test_graph_store.py tests/test_graph_store_integration.py` | Passed. |
| `python -m ruff format --check src/rfp_analyst/graph tests/test_graph_store.py tests/test_graph_store_integration.py` | Passed, seven files already formatted. Formatting was applied only to new Python files. |
| `python -m ruff check . --output-format concise` | Exit 1: ten pre-existing findings in untouched files; no new findings. |
| `python -m pip check` | No broken requirements found. |
| Real-driver unavailable-local-endpoint probe | Returned `unavailable` in approximately 3.009 s; explicit close returned `closed`. No real credentials used. |
| Compose YAML and optional dependency metadata checks | Parsed successfully; loopback bindings and matching dependency specifications verified. |
| `git diff --check` | Passed; only normal LF/CRLF working-copy notices. |

Existing full-repository Ruff findings: two unused imports in `agent.py`, two in
`document_generator.py`, one unused local in `evals/run_kb_evals.py`, four unused
imports in `src/rfp_analyst/agent/runtime.py`, and one unused import in
`tests/test_runtime_hardening.py`. They were not fixed or suppressed because they
are outside this phase. Changed/new Python files pass the configured lint rules.

### Known limitations / not performed

Docker is not installed on this host. The two real-server tests were collected
and deliberately skipped, not represented as passing. The provided service/tests
still need an authorized Docker run to verify live Cypher/schema behavior.
Mocks cover transactional semantics, not real-server concurrency or production
authorization. Python 3.11 CI was not run locally.

Schema initialization is explicit and admin-only; no production roles/ACLs,
backup system or Neo4j service were provisioned. Ingestion accepts only supplied
document/chunk metadata; extraction, active-snapshot publication, cross-store
synchronization and retrieval are intentionally absent. No live/private KB
evaluation, RAGAS, commit, push or deployment was performed in this phase.
