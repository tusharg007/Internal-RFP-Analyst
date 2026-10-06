# Optional graph retrieval and hybrid GraphRAG

Implemented on 2026-10-05, building on the approved upgrade plan and existing
[graph ingestion](GRAPH_INGESTION.md). Chroma remains the original text and
semantic evidence store. Neo4j adds controlled relationship selection, not a
replacement vector database or a source of uncited generated facts.

## Enablement and rollback

The default is unchanged: `RFP_RETRIEVAL_MODE=vector_only`.
No graph driver, connection, corpus export or graph lookup is created for that
mode. Existing Streamlit calls and source labels remain compatible.

For optional graph retrieval:

1. Install `requirements-graph.txt` or the existing `.[graph]` extra.
2. Deploy Neo4j with verified TLS for remote connections. Provision separate
   **server-enforced read-only**, ingestion and migration identities. Never
   give the application writer/admin credentials in production.
3. Set `NEO4J_ENABLED=true`, URI/database and the reader
   `NEO4J_USERNAME`/`NEO4J_PASSWORD`. Other settings are in `.env.example`.
4. Initialize the domain schema and rebuild the existing indexed corpus using
   the separate admin/ingestion CLI. Use the same corpus ID for both operations:

   ```powershell
   python -m rfp_analyst.graph init-schema
   python -m rfp_analyst.graph rebuild --corpus-id internal-rfp
   ```

5. Set `RFP_GRAPH_CORPUS_ID=internal-rfp` and choose
   `RFP_RETRIEVAL_MODE=auto`, `graph_only` or `hybrid`. Restart the application
   after deployment configuration changes.

Reader, ingestion and migration credentials are not interchangeable. The CLI
does not run automatically inside a user question. Graph publication must
match the **entire current indexed corpus manifest**, including uploads, not
merely the selected query scope. Rebuild after vector ingestion/deletion changes.

Rollback is only `RFP_RETRIEVAL_MODE=vector_only` and an application restart.
No Chroma migration, embedding change or Neo4j deletion is necessary.
`NEO4J_ENABLED=false` additionally disables infrastructure use. A selected
graph mode with disabled/missing graph settings degrades explicitly to vectors.

## Decisions and supported queries

`RetrievalDecision` is an immutable, extra-forbidden Pydantic model containing
the selected mode, a typed `GraphPlan`, and a reason. It cannot contain Cypher,
arbitrary labels, procedures or relationship names. Decisions are deterministic
and local: the existing LLM KB/direct router still decides whether retrieval is
needed; the new provider then decides **how** to retrieve.

| Query characteristics under `auto` | Mode / graph plan |
| --- | --- |
| Summaries, passages, resume questions, "what does this RFP say…" | `vector_only`; unchanged adaptive retrieval. |
| "Which projects used Azure?" | `graph_only` / `project_constraints`. |
| "Which healthcare projects used Azure and had compliance requirements?" | `hybrid` / explicit technology, industry and framework intersection. |
| "What technologies are shared by Project A and Project B?" | `hybrid` / `shared_technologies`, with both projects' supporting chunks. |
| "What is the timeline for Healthcare Migration?" | `graph_only` / `project_fields`; also budget and outcome fields. |
| "Which previous projects match target RFP requirements?" | `hybrid` / `requirement_candidates`; source-backed requirement constraints and case witnesses. |
| "Which projects delivered fraud reduction for banking?" | `hybrid`; only source assertions marked `achieved` can meet the delivery keyword filter. |
| Unsupported/over-sized plans or ambiguous project references | Vector fallback; strict graph-only declines instead. |

Multiple recognized uploaded target RFPs are treated as ambiguous, not silently
merged into one conjunctive target; production falls back to the existing
vector flow, and strict graph-only declines.

The parser is intentionally conservative, not a universal natural-language
query compiler. Project references resolve against bounded source-backed
titles/filenames, and must resolve uniquely. Technology/framework aliases use
the ingestion registry. An Azure-provider query admits curated Azure products
as **candidates**, without merging distinct products or claiming every product
was used. Industry names follow the actual public corpus sections.

The existing minimum ontology remains Project, Requirement, Technology,
Industry and ComplianceFramework plus Document/Chunk/assertion provenance.
There is **no fabricated Capability or SATISFIES graph**. Arbitrary capabilities
and multi-clause contractual requirements need original-text analysis; a
requirement candidate is not proof that every target capability is fulfilled.
Framework mentions do not prove compliance or certification. Projected or
estimated outcomes are not promoted to achieved results. These limitations
are also included in the graph generation instructions.

Explicit mode overrides are supported by `prepare_query_payload`, `run_query`,
and the root `agent.py` adapters. For a controlled ablation, inject a fresh
provider per query:

```python
from rfp_analyst.agent.graph import prepare_query_payload
from rfp_analyst.retrieval.hybrid import create_retrieval_provider

provider = create_retrieval_provider(policy="graph_only")
provider.strict = True
try:
    payload = prepare_query_payload(
        question,
        retrieval_provider=provider,
        retrieval_fn=existing_vector_callable,
        vectorstore_stats=existing_stats,
        llm=existing_llm,
    )
finally:
    provider.close()
```

Production `graph_only` may fall back to vectors on failure/no results; the
requested/effective mode and reason are distinct in the trace. Strict
graph-only never issues a vector search, including secondary RFP tools.
It controls retrieval, **not** the existing web/direct routes: disable/mock
Tavily and freeze those routes when running retrieval ablations. No RAGAS
or new semantic evaluation harness is introduced in this phase.

## Retrieval, fusion and generation

```text
Existing KB/direct router
  -> existing tool planner
  -> query-characteristic retrieval decision
       vector_only -> unchanged Chroma retrieval
       graph_only  -> bounded Neo4j projection -> validated original chunks
       hybrid      -> same graph witnesses + existing vector candidates
  -> deduplicate / reciprocal-rank fusion
  -> existing KB evidence grader
  -> existing deterministic tools (same per-run provider for secondary searches)
  -> currentness check + complete-witness prompt selection + final-context grading
  -> existing KB generator -> citation/page verifier -> bounded repair
```

The Neo4j reader executes fixed header, project/requirement seed, subject-fact
and head-check templates. Explicit graph assertions and canonical object IDs
are returned, with support relationships. Constraint intersections, shared
technology joins and requirement/case joins run in Python over that bounded
graph projection. This is **not** free-form generated Cypher or unbounded
server-side traversal. The initial projection can scan eligible subjects; it
deliberately falls back when the bounded corpus/result limits are exceeded.

The provider reads the existing Chroma collection without embedding generation
or writes, using the ingestion export's stable two-pass manifest capture. It
checks the active Neo4j snapshot against that exact digest. Every hydrated
witness validates corpus/version, document ID, source filename/origin/hash,
page, existing chunk ID, portable evidence ID, text hash, identity spans,
assertion IDs, allowed predicates/canonical objects and exact support offsets.
Text is the original indexed chunk, not a graph summary. Internal page indices
remain zero-based; final citations still use `[Source: filename.pdf, Page N]`.

Hybrid candidates are de-duplicated by unchanged Chroma chunk identity, with
source/content agreement required. Equal-weight reciprocal rank fusion uses
`1/(60 + rank)` for each channel. It does not substitute graph rank for cosine
similarity. The existing 0.50 vector prefilter remains; graph witnesses have
`score=None`, a separate fusion rank, and no fabricated relevance score.

Complete witness groups are reserved before optional vector evidence. A shared
technology needs both project identities and both assertion supports. If a
group cannot fit, it is dropped whole, not sliced into an incomplete path.
`k` is a vector target, not permission to cut mandatory graph witnesses; a
bounded graph result can contain more than `k` chunks.

Before prompt construction, after grading and after generation, the provider
rechecks active graph version and current index digest. Changed/unavailable state invalidates **all**
graph evidence, prompts, generated answers and tool notes. Production reruns original vector retrieval and
tools; strict graph-only declines. One budgeted prompt is actually passed to
the LLM, containing only selected original chunks and conservative relationship
interpretation. Unverified tool prose is not injected into this new prompt.
The selected context is graded again, and the existing post-generation
grounding/one-repair control still executes. Existing vector-only prompt and
deterministic catalog behavior are not rewritten.

## Security, bounds and lifecycle

- Only immutable reviewed query strings are executable. There is no generic
  `run_cypher` endpoint/tool and no `GraphCypherQAChain`.
- Every runtime session uses READ access and explicit transactions against the
  configured database. Server-side reader privileges remain mandatory:
  application role guards and READ routing are not database RBAC.
- Labels and relationships are literal source constants. All external values
  are parameters or locally resolved IDs, never query interpolation. No writes,
  CALL/APOC, arbitrary procedures, UNION or variable-length paths are used.
- Per projection: at most 100 subject seeds per type, 30 assertions per subject
  and 500 returned seed/fact records. Sentinel records detect over-limit
  results and reject the projection rather than asserting completeness.
- Query bounds: at most five technology groups, five framework names, five
  industry aliases, two project references, and three allowed scalar fields.
  Excess constraints are declined, not silently truncated.
- At most two business relationship hops; literal physical Cypher patterns are
  fixed-length (up to three relationships including source document support).
- Per response: 20 paths, 30 distinct assertions, 20 original chunks and 16,000
  content characters, further constrained by the actual prompt-token estimate.
- Connection/acquisition timeouts are capped at one second, transactions at
  two seconds. A three-second projection deadline is checked throughout result
  consumption. These driver/server controls are **not** a hard interrupt for
  every possible OS/network stall or the entire workflow, which also has LLM,
  web and tool calls. No hidden driver retry loop is enabled.
- Logs contain template enum, counts and elapsed time, not query values, source
  content, URI/passwords or raw driver exceptions. User-visible trace fields
  may contain filenames/IDs; sanitize those separately before external sharing.
- Providers are per-run dependencies, not mutable global connection state.
  Automatically composed providers close in `finally`; injected providers and
  borrowed drivers remain caller-owned. Payload exports remove the provider.
- Disabled/missing configuration bypasses graph index reads. A per-run service
  outage circuit prevents repeated connection attempts across secondary tools
  and rewritten queries. Invalid/stale graph evidence is never used for answers.

Neo4j documentation: [explicit transactions and timeouts](https://neo4j.com/docs/python-manual/current/transactions/),
[MATCH patterns](https://neo4j.com/docs/cypher-manual/current/clauses/match/).

The application still has no authenticated multi-tenant ACL boundary. The
existing `all`/`sample`/`upload` scope and `internal` partition are enforced
retrieval filters, **not** user authorization. Do not deploy it as a shared
private-data service without separately implementing authentication/ACLs.

## State and trace lineage

Additive output fields:

- `retrieval_mode`, `requested_retrieval_mode`, `retrieval_decision`;
- `graph_query_type`, `graph_version`, `graph_fallback_reason`;
- `graph_entities`: validated entity IDs/kinds;
- `graph_paths`: subject/object/assertion IDs, typed relationships with modality,
  support evidence/chunk IDs, hop count and candidate interpretation;
- `graph_provenance`: original chunk/portable evidence/document IDs,
  filename/page/origin/file hash and associated assertion IDs.

`graph_retrieval` and `graph_integrity_check` events supplement the existing
health, planner, retrieval, tool, budget and grounding traces. Corpus coverage
is reported as `conservative_partial`; the separate `partial` flag indicates
path truncation, not exhaustive extraction. Final prompt selection removes
dropped paths and their assertion references from exported provenance.
The Streamlit UI still displays Private KB, Web Search and Direct; graph
witnesses are explicitly marked rather than shown with artificial cosine scores.

This strengthens **graph assertion -> original chunk** lineage, not formal
**final claim -> evidence** provenance. The final answer's citations and the
existing heuristic verifier are not a semantic entailment proof or a claim
provenance graph. No missing claim links are reconstructed after the fact.

## Files and verification

Added:

- `src/rfp_analyst/graph/reader.py`;
- `src/rfp_analyst/retrieval/decisions.py` and `hybrid.py`;
- `tests/test_hybrid_retrieval.py` and this document.

Modified for this retrieval phase: `config.py`, `.env.example`, root `agent.py`,
`agent/graph.py`, `agent/schemas_decisions.py`, `agent/__init__.py`,
`tools/search_kb.py`, `tests/test_graph_store_integration.py`, and documentation
links. Pre-existing foundation/ingestion work is preserved. No new dependency
upgrade is necessary beyond the existing optional official Neo4j driver.

The new offline tests cover schema/routing, parameter injection, read-only
templates/roles, complete multi-hop witnesses, all four public document kinds,
actual generated PDF structure, scope isolation, original text and quote/ID
integrity, native score/RRF separation, secondary tools, strict ablations,
stale snapshots, budgets, timeouts, connection failure, lifecycle and exact
legacy-vector differential behavior. Existing golden questions are unchanged.

The opt-in real-server test extends the disposable loopback Docker fixture with
all four graph plan types. Run:

```powershell
# Set a disposable NEO4J_TEST_PASSWORD; never use application/private credentials.
docker compose -f compose.neo4j-test.yml up -d --wait
$env:RFP_ANALYST_NEO4J_INTEGRATION = "1"
python -m pytest tests/test_graph_store_integration.py -v
Remove-Item Env:\RFP_ANALYST_NEO4J_INTEGRATION
docker compose -f compose.neo4j-test.yml down
```

This Community fixture validates real Cypher/persistence/hydration, not
production RBAC; its reader wrapper borrows the disposable admin driver.
Provision and independently test a real least-privilege reader before deployment.
Docker is unavailable on this host, so live-server results are not claimed.

### Verification results (2026-10-05)

Recorded commands use `.venv\Scripts\python.exe` (Python 3.12.14). Final full-suite
results include all pre-existing tests and the new retrieval tests. Existing smoke and KB
scores are regression results, **not** GraphRAG quality-improvement claims.

| Check | Result |
| --- | --- |
| `python -m pytest tests/test_hybrid_retrieval.py -q` | 84 passed; 4.70 seconds. |
| `python -m pytest -q` | Final rerun: 338 passed, 5 opt-in Neo4j tests skipped; 43.01 seconds. Zero pytest failures. |
| Offline `python -m evals.run_evals` | 3/3, exit 0. Mock smoke only. |
| `python -m evals.run_kb_evals` | 9/9, retrieval-only; 46.79 seconds measured query latency. |
| Changed/new Python Ruff checks | Passed. |
| Full-repository Ruff | Eight pre-existing unused-import/local findings remain in unrelated files; no new failures. |
| `python -m compileall -q -f src tests evals` | Passed, exit 0. |
| `python -m py_compile app.py agent.py rag_engine.py config.py document_generator.py` | Passed, exit 0. |
| New module/test `ruff format --check` / `git diff --check` | Passed. |

The prior foundation/ingestion baseline was 254 passed and four skipped. The
new real-server retrieval test accounts for the fifth skip. Local production
`vectorstore` access is denied; it was not modified or rebuilt. Validation used
synthetic/injected snapshots, generated public PDFs and the evaluation harness's
isolated real Chroma collection. Live answer quality, Neo4j concurrency/RBAC,
Python 3.11 CI, deployment load and RAGAS have not been evaluated locally.
The existing `langchain-community` deprecation warning remains. Full-repository
lint findings are in `document_generator.py` (two), `evals/run_kb_evals.py` (one),
legacy `agent/runtime.py` (four), and `tests/test_runtime_hardening.py` (one).
The two previous root `agent.py` lint warnings were resolved as explicit
backward-compatible re-exports while updating that adapter.

Unchanged: Chroma embeddings/chunking/search thresholds, existing KB/direct
router, web fallback/rewrite topology, deterministic tools, citation format,
final verifier/repair semantics, default UI behavior, golden data and existing
evaluation harnesses. No graph extraction redesign, RAGAS, database provisioning,
commit, push or deployment is performed in this phase.
