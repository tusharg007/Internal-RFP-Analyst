# Hybrid GraphRAG and RAGAS upgrade plan

Status: design only; no implementation authorized by this review.

Review dates: 2026-10-04 to 2026-10-05. Repository baseline: `a729b48f895f976bb20770187cf37567ad415c8a`. Package version: `0.1.0` in `pyproject.toml`.

## Executive decision

Add Neo4j as a provenance-bearing relationship index alongside ChromaDB, behind an opt-in feature flag. Keep LangGraph orchestration, existing deterministic tools, scope handling, evidence grading, Tavily fallback, bounded rewriting, source verification and bounded repair. Add RAGAS as a separate evaluation process over captured runs, not as a runtime answering dependency or replacement for existing tests.

Do not enable hybrid retrieval by default until a frozen, paired evaluation demonstrates useful improvement on relationship questions without weakening provenance, scope isolation, existing tool behavior or outage handling. A small synthetic corpus does not by itself justify production infrastructure or establish real-user quality.

The first prerequisites are trustworthy context/run capture and reproducible evaluation. The current tests pass, but the real-KB baseline is **8/9**, not the historical 9/9 stated in the README. One RFP-analysis case follows a web branch and misses the expected tools. GraphRAG is not a fix for that orchestration issue by itself.

## 1. Current architecture and repository findings

### Review coverage

Inspected the tracked application, configuration, dependency/build files, CI, all Python files under `src/rfp_analyst/`, all tests, all evaluation code/questions, and the documentation, including:

- `README.md`, `docs/ARCHITECTURE.md`, `docs/AGENTIC_RAG.md`, `docs/TESTING_AND_EVALUATION.md`, `docs/REPRODUCIBILITY.md`, `docs/FILE_MAP.md`, `docs/evaluation.md` and troubleshooting documentation.
- `app.py`, `agent.py`, `rag_engine.py`, `config.py`, `document_generator.py`, `requirements.txt`, `pyproject.toml`, `Makefile`, `.github/workflows/ci.yml`, `.python-version`, `.env.example` and Streamlit configuration.
- The ten public synthetic PDF texts and their generator definitions. Local notebook routing/grading/topology sections were reference material only, not the source of truth for this application.

Private uploaded document contents and `.env` values were not inspected or copied. Screenshots/assets and ignored runtime directories were inventoried, not treated as authoritative architecture. Assertions below about the domain are based on the public corpus and the synthetic evaluation RFP, not private client data.

### Runtime architecture

| Component | Current responsibility and important contract |
| --- | --- |
| Streamlit `app.py` | Chat/history, sample/upload/all scope, uploads, ingestion, readiness and evaluation display. Chat currently requires both provider and vector-store readiness. |
| Root `agent.py` | LLM creation, graph preparation, response/trace adaptation and user-friendly errors. Groq `openai/gpt-oss-120b` is primary; Google is a key-based alternative, not an automatic error failover. |
| `rag_engine.py` | Live ingestion/retrieval facade; scoped Chroma search, statistics and resource cleanup. Successful ingestion publishes a rebuilt temporary store with backup/swap handling. |
| Ingestion package | PDF validation/loading, chunking, metadata and identity; a separate registry/pipeline also exists. It is not the only ingestion entry point. |
| Chroma + FastEmbed | Semantic evidence and raw chunk text, with sample/upload metadata. No existing graph store. |
| `agent/graph.py` | Canonical `QueryState` TypedDict, graph compilation, retrieval orchestration, specialized tools, generation and grounding/repair. `agent/state.py` contains a separate legacy dataclass; do not mistake it for the live graph state. |
| Router/grader/rewriter modules | Structured JSON decisions; keyword/degraded fallbacks; bounded retry count. Helpers may create their own configured LLMs. |
| Deterministic tools | Project catalogs, comparisons, requirement extraction, case-study matching, proposal outlines and page/source verification. |
| Evaluation | Mock smoke evaluation, real sample-KB golden evaluation, pytest and compilation checks. No installed Neo4j or RAGAS integration. |

The active graph is cyclic, despite older architecture documentation describing a linear pipeline:

```text
START -> health_check -> route_question
  direct -> direct_answer -> END
  kb     -> retrieve_kb -> grade_kb_evidence
                            good -> execute_tools -> synthesize_prompt
                                    -> generate_from_kb -> END
                            weak -> search_web -> grade_web_evidence
                                                  good -> generate_from_web -> END
                                                  weak, retry available -> rewrite_query
                                                                          -> retrieve_kb
                                                  weak, retry exhausted -> answer_insufficient -> END
```

There are three conditional edge sets. KB verification and one bounded repair are helpers inside generation, not separately compiled graph nodes. `DeterministicCompiledGraph` intentionally provides a degraded linear path without the full web/rewrite cycle. The compiled graph is cached; injected retrieval/runtime policy should be per-run, not captured in mutable global settings.

### Evidence, tools and generation today

- Retrieved normalized records contain content, source filename, zero-based page, score/raw score, chunk ID and document origin. Ingestion also has file hash/path information, but normalization does not carry all of it into graph state.
- Chunk identity incorporates absolute-path namespace as well as file/content metadata. It is repeatable for the same location/content, not a portable identity across machines or differently located temporary uploads.
- Generic retrieval uses the 0.50 relevance pre-filter. Existing special handling for catalogs, resumes and target RFPs broadens/adjusts retrieval. Preserve those behaviors; do not mistake this for one universal threshold.
- Catalogs use deterministic structured extraction and broad sample coverage. Comparisons and requirement-to-case matching can perform additional retrieval through the injected retrieval function. An upgrade only to the first `retrieve_kb` call would leave those tools vector-only and invalidate ablation comparisons.
- Some specialized-tool document normalization uses score `1.0`; that is not a measured semantic score or calibrated confidence. Comparison-returned evidence also needs explicit capture before it can be used for graph provenance/evaluation.
- Requirement source attachment can default to a high-scoring uploaded page rather than identify an exact supporting span. Existing requirement IDs such as `REQ-01` are local sequence IDs. Neither should be promoted into precise graph provenance without re-extraction/validation.
- Prompts instruct page citations, and the verifier checks retrieved source/page identities plus lexical/numeric support. It is heuristic, can assess uncited lines against available evidence, and skips some scaffolding. A final grounded result is not formal semantic entailment or a claim-to-chunk graph. Repair can remove table rows, so grounded-but-incomplete answers are possible.
- `synthesize_prompt` computes a compact prompt and budget, but `generate_from_kb` currently constructs a different prompt from the full retrieval context plus specialized notes. The recorded budget is therefore not a reliable limit on the actual generation input. New graph context must not amplify this discrepancy.
- Web generation does not run the private-KB verifier. Public search cannot establish private project facts. The existing system does not expose a complete private/public query policy; address that separately and explicitly.
- UI source labels are `Private KB`, `Web Search` and `Direct`. UI source cards currently understand the `search_knowledge_base` trace shape; arbitrary new trace fields will not automatically become visible.

### Production gaps that exist independently of GraphRAG

Scopes are corpus-selection filters, not authenticated tenant authorization. Dependencies largely use open-ended lower bounds and have no complete lock. The optional development dependency group does not install Ruff. Python 3.11 is documented/used by CI, whereas the reviewed local interpreter is 3.12.14. Readiness, ingestion consistency, real generation-context budgets and evaluation determinism require explicit acceptance tests; adding Neo4j or an LLM judge does not resolve them automatically.

## 2. Exact shortcomings GraphRAG should solve

| Current limitation | Corpus-grounded question/use case | Additive graph benefit | What it does not establish |
| --- | --- | --- | --- |
| Top-k similarity can miss a conjunction spread across pages/documents. | Which Azure projects explicitly mention healthcare compliance and phased delivery? | Indexed project/technology/control candidates, followed by source-backed intersection and chunk hydration. | Azure use alone does not establish HIPAA compliance or delivery capability. |
| Separate tool searches do not model a reusable relationship path. | Which projects share technologies with the healthcare migration proposal? | Project-to-technology-to-project paths with both projects' supporting assertions. | Shared technology is not proof of equal expertise or implementation success. |
| Catalog completeness depends on broad retrieval/extraction. | List every project with its timeline and outcome status. | Enumerate eligible project records, preserving field-specific evidence and explicit coverage. | A bounded partial result must not claim to list all projects. |
| Requirement-to-case scoring is lexical and relatively coarse. | Find cases relevant to Azure migration, dashboards and regulatory controls. | Source-backed candidate sets for each requirement dimension and explainable intersections. | A candidate match is not a contractual `SATISFIES` relationship. |
| Timeline/budget/outcome text is dispersed and can be conflated. | Compare delivered reductions with proposed savings. | Typed assertions with units and achieved/projected/estimated/unknown modality. | A proposal's projected outcome is not an achieved result. |
| Page citations lack structural graph witnesses. | Explain why a project was selected through a two-hop relationship. | Return assertion IDs, path witnesses and exact supporting chunk references. | This alone does not bind every generated final sentence to those references. |

Vector retrieval remains better suited to narrative summaries, unknown entities, resumes, arbitrary uploads, fuzzy language and evidence outside the graph ontology. Graph-only performance may be worse there; report that honestly. No presumed global quality uplift, automatic answer correctness, or production-scale benchmark is claimed.

## 3. Minimum domain graph schema

### Corpus justification

The public corpus contains ten anonymized project documents. Cover pages identify titles/industries; page 2 generally contains objectives, technologies and timelines; page 3 generally contains budgets/outcomes. Examples include:

- Banking: Azure/Power BI, a 16-week timeline and stated achieved audit-cycle/data-quality improvements.
- Healthcare: Azure migration with HIPAA/SOC 2 mentions and projected cost reductions.
- Retail: AWS/Snowflake/Tableau and stated achieved forecasting/stockout improvements.
- Government: Azure Government, FedRAMP/WCAG mentions and projected processing improvements.
- Pharma: AWS/Redshift, FDA/GDPR/GCP mentions and claimed achieved outcomes.
- Manufacturing: Azure IoT/ML/Power BI and projected downtime savings.

Document types include proposals, case studies, project outlines and RFP responses. There are no reliable repeated named-client identifiers. The synthetic evaluation upload is a target RFP with requirements, not an additional delivered consulting project. Do not infer project status merely from a document title or section heading.

### Nodes

| Node | Minimum fields | Reason for inclusion |
| --- | --- | --- |
| `Document` | corpus ID, document ID, file hash, safe source name, origin, kind, revision, access partition | Provenance root; `kind=target_rfp` distinguishes a request from project evidence. |
| `Chunk` | portable evidence ID, current legacy chunk ID mapping, document ID, page index, text hash, span coordinates, corpus version | Hydrates exact evidence and checks graph/vector consistency. |
| `Project` | scoped project ID, title, aliases, identity/review status | Joins related facts without conflating unnamed clients/projects. Initially document-scoped. |
| `Technology` | canonical ID/name, controlled aliases | Azure/AWS/Power BI/etc. matching and two-hop project discovery. |
| `Industry` | canonical ID/name, explicit source wording | Sector filters; do not infer solely from a filename. |
| `ComplianceFramework` | canonical ID/name/version if explicit | HIPAA, SOC 2, FedRAMP, FDA 21 CFR Part 11, GDPR, etc.; records mentions, not certification truth. |
| `Requirement` | document-scoped ID, verbatim text/span, kind, normalized constraints, modality | Target-RFP matching. Use `kind=compliance` where appropriate rather than a duplicate class. |
| `Assertion` | assertion ID, subject, predicate, object or scalar value, unit, modality, extraction version/method, review status | Makes every queryable domain fact individually supportable, including timelines/budgets/outcomes. |

`Client`, a broad `Capability` ontology, separate `Outcome`, separate `RFP` and separate `ComplianceRequirement` nodes are deferred. Anonymous client descriptions should be retained as cited text, not resolved into fabricated client identities. Initially outcomes are typed assertions; RFP is a document kind; compliance requirements are requirements referencing a framework. Add entities later only when real questions and annotation demonstrate value.

### Relationships and assertion semantics

Use a small reified model:

```text
Project or Requirement -> HAS_ASSERTION -> Assertion
Assertion -> OBJECT -> Technology / Industry / ComplianceFramework
Assertion -> SUPPORTED_BY -> Chunk -> IN_DOCUMENT -> Document
```

Scalar assertions have a predicate/value instead of an entity object. Initial predicates: `uses_technology`, `in_industry`, `mentions_framework`, `requires_technology`, `requires_framework`, `timeline`, `budget`, `outcome`. Requirement/document membership is an explicit structural link. Each support link records the actual source span; a domain fact without valid support is not eligible for publication or retrieval.

Examples of temporal/modality distinctions:

- `outcome: audit cycle reduced 35%; achieved` versus `outcome: cost reduction 40%; projected`.
- `mentions_framework: HIPAA; proposed_control` versus a source's explicit certification claim. Preserve the claim's status; neither constitutes an independent legal/compliance attestation.
- Keep duration units (`weeks`, `months`) and budget currency/range. Do not silently equate a month to four weeks or normalize unspecified currency.

Project identity begins with one project record per recognized project document. Merge across documents only with an explicit identity rule or reviewed mapping, never because two documents describe a generic bank. Technology aliases are curated, versioned and unambiguous; broad `Azure` is not automatically equivalent to every Azure product. Unknown/contradictory facts remain unknown/contradictory.

Requirement-to-case matches are query-time candidate assessments with the target requirement evidence and case evidence attached. Do not write inferred `SATISFIES`, `COMPLIANT_WITH`, or transitive capability claims into the factual graph.

### Identity and integrity

Preserve existing Chroma chunk IDs and file/page citation formatting. Add a portable evidence alias based on a frozen document identity, file/content hash, origin and chunk/page/span coordinates, mapped to the existing chunk ID for hydration. Portable aliases require identical frozen source bytes and chunking rules; regenerated PDF timestamps or changed splits produce a new version, not false stability.

Metadata page indices remain zero-based; rendered citations use page index + 1. Enforce unique composite keys for corpus/version/node ID, required provenance fields, valid references and allowed predicate/object types. Evidence from different origins/revisions must not be silently merged. Validate no dangling support links or facts whose sources have been revoked.

## 4. Graph ingestion architecture

### Publication flow

```text
Existing PDF validation/loading/chunking -> existing Chroma publication succeeds
  -> persist corpus manifest + graph synchronization job
  -> isolated ingestion worker extracts/validates assertions
  -> Neo4j staging snapshot -> integrity checks -> publish graph version
```

Hook the live `rag_engine.ingest_documents` facade after successful vector publication, not just the separate ingestion pipeline. Graph failure must not roll back an otherwise successful Chroma ingestion. There is no atomic transaction across a Chroma directory swap and Neo4j; use manifests, idempotent jobs and reconciliation rather than claiming distributed atomicity.

The manifest records exact source hashes, current chunk IDs/portable aliases, text hashes, chunker/embedding/extractor versions and corpus version. A persisted pending job plus reconciliation against the active store handles a crash between vector publication and job recording. Graph queries must reject mismatched versions and degrade to vector-only until synchronization completes. Maintain a consistent captured snapshot for each query, including text hydration; fail/retry safely if the active corpus changes mid-run.

### Extraction and validation

1. Start with deterministic parsing of known sample sections and explicit, controlled aliases. Extract per chunk/page span before applying corpus-level joins.
2. Classify recognized project versus target RFP documents from validated content/metadata. Unrecognized documents and resumes remain vector-searchable; do not force them into `Project` records.
3. Extract requirements from their own source spans. Do not import approximate tool source attachment or sequential requirement IDs as exact lineage.
4. An optional later LLM extractor returns schema-validated entities/assertions only, never Cypher. It runs in the ingestion worker, not inside user-question retrieval.
5. Validate quoted supporting spans against normalized source text with an offset mapping back to original text. Facts spanning chunks must list all required supports. Reject fabricated quotes, missing pages, malformed values and unsupported entity merges.
6. Record extraction uncertainty separately from source wording and semantic truth. Route ambiguous identity/modality or unsupported extraction to review; unpublished candidates are not evidence.
7. Writer-owned fixed statements perform idempotent upserts in bounded batches. Migrations/constraints use a separate administrative identity.
8. Compare manifest counts/hashes, support integrity and extraction fixture results before publishing the snapshot. Retire deleted/replaced/revoked evidence; reconcile orphans and retain approved snapshots for rollback.

Do not scrape public web results into the private graph or persist user-query-generated inferences as factual relationships. A graph job processes validated corpus documents only.

Operational metrics: synchronization lag, active vector/graph versions, extracted/accepted/rejected assertions, unresolved identities, orphan count, batch failures and extraction cost. Alert on stale publication or revoked evidence remaining queryable. Protect manifests and graph backups as sensitive data even when full PDF text is stored elsewhere.

## 5. Graph retrieval architecture

Introduce a typed retrieval provider alongside the legacy callable contract. Resolve normalized entities from controlled aliases and a restricted structured plan, then execute one of a finite set of reviewed templates:

| Template | Inputs | Returned evidence |
| --- | --- | --- |
| `projects_by_technology` | canonical technology IDs | Project/technology assertions and supports. |
| `projects_by_industry_and_framework` | industry/framework IDs | Both explicit property witnesses per matching project. |
| `projects_sharing_technology` | project ID, optional technology constraint | Two project/technology assertions for each path. |
| `project_fields` | eligible project IDs, allowed fields | Timeline/budget/outcome assertions with modality and completeness metadata. |
| `requirement_case_candidates` | target document/requirement IDs, controlled constraints | Requirement supports plus relevant project supports; no proof of satisfaction. |

The planner may select a template enum and validated parameters; it cannot supply query text, labels, relationship syntax, procedures or arbitrary field names. Unknown aliases/plans return `unsupported_plan`/`ambiguous_entity`, not unconstrained exploration. Clarify ambiguous identities where necessary; hybrid can still use semantic retrieval for unsupported language.

Use the official Neo4j driver with sessions bound to an explicit database and server transaction timeouts. Do not expose `GraphCypherQAChain`, a generic `run_cypher` tool or text-to-Cypher to the user/LLM. Driver read access is not itself an authorization boundary. [Neo4j transaction documentation](https://neo4j.com/docs/python-manual/current/transactions/).

Initial configurable upper bounds, subject to load-test validation:

| Bound | Initial ceiling |
| --- | --- |
| Canonical entity seeds | 5 |
| Business relationship hops | 2 |
| Domain-path physical edges | 8 for two reified project/technology hops; separately fixed provenance joins |
| Returned paths / records / assertions | 20 / 50 / 30 |
| Hydrated chunks | 20 before the generation-context budget |
| Graph connection / transaction / total retrieval budget | 1 s / 2 s / 3 s |
| Graph transient retry | At most one, within the same total budget |

A `LIMIT` alone does not bound traversal work. Use indexed scoped seeds, fixed-length patterns, bounded fan-out at each expansion, constrained property filters, per-query cardinality limits and server timeouts. Disable unbounded variable-length patterns and unrestricted full-graph scans. Account for driver retries and queue/pool waiting in the total wall-clock deadline.

Catalog listing uses a separate bounded, stable pagination template with an initial 100-document ceiling and explicit eligible/returned counts. If the ceiling, scope, extraction gaps or budget prevents complete coverage, mark `complete=false` and state the limitation. Never claim all projects from a partial traversal.

Hydrate original text by existing chunk ID/portable mapping from Chroma or a validated immutable corpus snapshot; check document hash, page and version before passing it to grading/generation. Graph-only means no semantic similarity search, not that original evidence text is unnecessary. A broken support mapping invalidates the graph fact.

## 6. Vector/graph/hybrid routing rules

Keep two separate decisions: current `kb` versus `direct` routing, followed by private-KB retrieval strategy. Preserve RFP `compare`, `proposal` and `rfp_analysis` intent refinement and source-recall/clarification behavior.

| Question class | Proposed production strategy when feature enabled |
| --- | --- |
| Greeting, simple chat, previous-source recall, clarification | Existing direct/clarification path; no graph call. |
| Narrative/fuzzy document question, resume, unknown ontology, arbitrary uploaded PDF | `vector_only`. |
| Exact supported entity/property lookup or complete recognized project catalog | `graph_only` when snapshot/coverage is valid; otherwise production vector fallback with degradation recorded. |
| Multi-constraint, shared-technology, cross-document RFP matching | `hybrid`: semantic evidence plus supported template paths. |
| Unsupported graph-only plan in a forced ablation | Explicit unsupported/no-evidence result, not hidden vector fallback. |

Default config remains `vector_only`; a feature flag enables an `auto` policy choosing the above strategies. Forced evaluation modes bypass auto strategy selection but not scope/security constraints. Capture requested and effective modes separately. Mode is not a new answer-source label: Neo4j evidence hydrated from private documents is still `Private KB`.

Inject the same provider into initial retrieval and secondary searches inside comparisons/case matching. Preserve the old `retrieval_fn(query, k, scope)` and two-argument compatibility path. Do not encode mode in the query string or mutate global graph compilation state. Capture per-call mode, arguments and evidence to detect tools secretly issuing vector searches during graph-only runs.

Do not force web fallback just because graph evidence is absent. Graph unavailable/unsupported first invokes the existing vector evidence path in production. Only the existing KB-grading decision should then enter the public fallback/rewrite cycle. A separate explicit public-search policy must prevent private-only gaps or sensitive queries from being turned into public search; its rollout is independent and tested, not a silent change to the default legacy branch.

## 7. Evidence fusion and final grounding

### Typed evidence contract

Add an `EvidenceBundle` concept with vector candidates, graph assertions/paths, hydrated source chunks and selection diagnostics. Each evidence item includes corpus/version, source/document ID, page index, origin, portable evidence ID and legacy chunk ID. Graph items also include assertion IDs, template/path witnesses and extraction method/modality. Keep native vector relevance, graph ranking and extraction uncertainty in separate fields.

For opt-in hybrid retrieval:

1. Apply existing vector scope and relevance/adaptive pre-filter behavior to vector candidates only.
2. Validate graph facts, permissions, version and support independently. Never force graph evidence through the 0.50 cosine threshold or set an artificial semantic score of 1.0.
3. Rank the validated vector and graph evidence lists separately. Initially use equal-weight reciprocal rank fusion with rank constant 60; tune only on the development split. Fused rank is ordering, not probability or semantic score.
4. Deduplicate by exact evidence identity/mapping, not just source/page. Preserve multiple supports and distinguish source retrieval route from document origin.
5. Reserve all mandatory witnesses of an accepted multi-hop explanation. If they cannot fit, remove the claim/path or explicitly limit the answer; do not retain an unsupported half-path.
6. Apply document diversity and a measured context/token budget. Preserve target-RFP evidence and field-specific source spans. Keep conflicting values/dates/modalities visible rather than selecting whichever scores highest.
7. Grade the fused original-source context using the existing KB grader. Pass original source chunks and supported notes to deterministic tools/generation, then run existing verification and at most one repair.

Keep the existing vector-only code path unchanged initially. An optional evidence-provider integration must collect additional tool evidence without altering tool scoring/output contracts. Do not disguise generated graph summaries as raw PDF text merely to make the verifier pass.

### Generation context and lineage boundary

Before opt-in generation, make one context assembler authoritative for the prompt actually sent to the model, its budget and its captured evidence list. Record raw candidates separately from selected/truncated generation contexts and context order. Differential tests must show vector-only behavior is preserved before sharing the new assembler by default.

Add structural graph-provenance validation before generation and before final graph-derived facts are emitted: every selected fact/path must reference authorized, current, hydrated supporting evidence. Retain the existing verifier as final grounding control; structural validation does not replace semantic/lexical checks.

For deterministic catalog rows or explicit graph-fact renderings, a future additive sidecar can record final item/claim IDs and their already-established assertion/evidence IDs at generation time. Free-form answer claims with only page citations remain page-level or unknown linkage. Do not manufacture claim-to-chunk links after generation from lexical similarity. Retrieval path provenance and final claim provenance are different guarantees.

Capture pre-repair and post-repair answers and removed/qualified items. Evaluate answer coverage as well as groundedness so an empty table or repeated disclaimer is not classified as a successful answer.

## 8. Security model

### Database and query boundaries

- Runtime reader, ingestion writer and migration administrator are separate identities. Reader credentials cannot write data, modify schema, load external resources or execute unsafe procedures.
- Enforce least privilege in Neo4j itself; an application enum/template allowlist is a second boundary, not a substitute. Verify inherited `PUBLIC` privileges and procedure/loading permissions for the chosen deployment. The built-in reader role alone should not be assumed to meet a custom deployment's complete policy. [Neo4j roles and privileges](https://neo4j.com/docs/operations-manual/current/authentication-authorization/built-in-roles/).
- Production acceptance requires a deployment/edition that demonstrably enforces the required permissions. A local development service with an administrator account is not evidence of read-only security.
- Only fixed, reviewed statements execute in the query process. All values are parameters; no interpolation, dynamic labels, LLM-returned syntax, arbitrary Cypher endpoint, `APOC`, `LOAD CSV`, or unrestricted procedure calls.
- Test attempted writes/schema changes using the actual reader principal, injection-like values, oversized parameter lists and forbidden template IDs. Assert rejection and no state change.
- Enforce authorization and active version before expansion and on every supporting witness, not only final result rendering. Revoked or cross-scope supports invalidate the result.

### Application/data boundaries

Sample/upload/all scope remains supported, but is not an ACL. Initial deployment can retain a single authorized corpus/instance. Before multi-user/multi-tenant hosting, authenticated identities must produce authoritative allowed document/corpus sets applied to both Chroma and Neo4j, hydration, caches, traces and exports. Do not market metadata filters as tenant isolation.

Use TLS with certificate verification for remote Neo4j, private networking where supported, secret-manager-to-environment injection and rotation. Reader credentials alone enter the application process; writer/admin credentials remain in worker/deployment jobs. Keep `_resolve_key`-style env/config patterns, redaction and non-secret `.env.example` placeholders.

Treat PDFs, entities, graph records and prompts as untrusted data. Source instructions cannot choose tools, query templates, credentials or public-search policy. Bound extraction and generation budgets. Default trace exports are identifiers, hashes, decisions and aggregate scores, not source text, prompts, connection strings or private queries. Evaluation source text is permitted only in explicitly approved synthetic/private evaluation storage.

The existing application may send a weak-KB query to Tavily. A cloud rollout must explicitly approve public query disclosure and prevent public results from answering private-fact questions. Keep web evidence separate from private graph assertions and labels.

## 9. Failure/fallback and observability

| Failure | Production response | Evaluation response |
| --- | --- | --- |
| Graph feature off, missing package/credentials | Existing vector pipeline; optional graph imports must not break startup. | Graph modes `unavailable`, not fabricated scores. |
| Neo4j down/timeout/circuit open | Bounded vector fallback; trace requested/effective mode and reason. | Forced graph-only does not silently use vectors; record failure and denominator. |
| Graph snapshot stale, support/hash mismatch, revoked witness | Discard affected graph evidence; use valid vector path. | Integrity failure; no unsupported graph facts admitted. |
| Valid graph query has no result | Hybrid retains vector evidence; graph-only production can degrade according to policy. | Empty graph result is an observed retrieval outcome. |
| Vector store unavailable | Preserve current readiness/error behavior initially. | Do not claim graph-only independence unless a separately tested immutable hydration store exists. |
| Both KB strategies weak | Existing permitted Tavily/grading/rewrite/insufficient branch and retry bound. | Public search disabled or fixture-pinned for retrieval ablations. |
| Graph ingestion fails after vector publication | Chroma remains usable; graph marked unsynchronized; retry/reconcile job. | Extraction/publication failure is explicit. |
| RAGAS package/provider/quota/judge fails | No impact on answering. | Failed metric with reason, not 0, 1 or silently dropped row. |

A proposed graph circuit breaker opens after three consecutive infrastructure failures and probes after 30 seconds. Tune these defaults under load. Circuit state must not bypass authorization or treat invalid provenance as successful graph evidence. Avoid a separate unbounded graph retry loop on top of LangGraph's one query rewrite.

Expose additive trace/state fields: run ID, requested/effective retrieval mode, template ID, corpus/graph versions, assertion/path/evidence IDs, support validation, per-stage latency, fallback reason, selected generation contexts, pre/post-repair coverage and retry count. Preserve current trace tool names where contracts/tests rely on them. Sanitize UI summaries; keep detailed records in access-controlled run artifacts rather than relying on shortened UI traces.

Metrics/SLOs: graph p50/p95/p99 latency, timeout/error/degradation rate, synchronization lag, context-token use, no-answer rate, graph witness coverage, pre/post-repair answer completeness and user-visible latency by mode. Initial hard graph deadline is three seconds; total application latency targets require a load-tested provider budget, not a promise from this single-machine baseline.

## 10. RAGAS integration design

### Evaluation runs, not live nodes

Add a separate command that reads captured, versioned run artifacts and writes metric records/aggregates. No RAGAS import in normal UI/agent startup, no judge calls inside generation, no judge-driven runtime rewriting and no reference answers fed into retrieval or generation. Keep current evaluators available unchanged.

Each captured sample includes question ID, sanitized user input/history, mode requested/effective, scope, actual tool route, exact selected contexts in model order, source IDs, answer before/after repair, optional human reference, corpus/question hashes, model/embedding/judge versions, latency, retries and applicability/failure status. Separate retrieval-only runs from runs that actually generated answers.

Use an isolated, pinned evaluation dependency set compatible with the chosen RAGAS release. Test an adapter against that exact release rather than assuming old examples still work. Current documentation uses metric classes under `ragas.metrics.collections` and asynchronous scoring; do not mix them blindly with legacy APIs. A future implementation must record the installed version and tested signatures. This review neither installs RAGAS nor invents a tested version pin.

### Metric contracts

| Requested metric | Inputs and interpretation | Applicability |
| --- | --- | --- |
| Faithfulness | Question, actual response and actual model-visible contexts; assesses response support. | Evidence-grounded generated answers. Score pre/post repair separately. |
| Answer Relevancy | Question/response with explicit judge and embedding configuration; relevance does not prove truth. | Answer-bearing cases; evaluate direct chat separately if desired. |
| Context Precision | Ordered generation contexts and reference for the reference-based variant. | Reference-annotated answerable cases. |
| Context Recall | Reference answer and retrieved/selected contexts; evaluates required reference support. | Reference-annotated cases, not arbitrary no-answer rows. |
| Answer Correctness | Question, final answer, reference, judge and embeddings; combines factual comparison and semantic similarity. | Only cases with reviewed reference answers. |

Use documented current implementations/adapters: [Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/), [Answer Relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/), [Context Precision](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/), [Context Recall](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/) and [Answer Correctness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_correctness/).

For no-reference cases, a separately named reference-free context-utilization metric may be reported, but not mislabeled as the same reference-based precision score. Do not substitute a different factual-correctness metric for Answer Correctness without stating the change. Pin metric prompts, similarity weights, embedding configuration and versions in results.

Configure a judge LLM explicitly, separately from the answer LLM. Configure evaluation embeddings explicitly using a tested adapter; FastEmbed is an option only if the selected RAGAS contracts support the adapter. No implicit OpenAI key/provider/model selection. Provider/model selection, timeouts, maximum concurrency, retry count (initially at most two attempts), token limits and a monetary/token budget belong to evaluation configuration. Keep provider SDKs in optional evaluation dependencies. Never evaluate private client content with an external judge without authorization and an approved retention policy.

Judge scores are noisy and not assurance proofs. Calibrate on human-annotated examples, optionally use an independent judge model, repeat a pilot three times, and report paired differences/confidence intervals and judge failure counts. A same-family judge may introduce shared-model bias. No live RAGAS quality baseline is available from this review.

### Missing data and score denominators

No-answer/clarification/direct cases lacking evidence receive an explicit metric applicability reason, not automatic perfect scores. An answerable case that fails to answer still fails deterministic task coverage; do not make a mode look better by silently excluding its failed cases. Record semantic metric success, not-applicable and error counts separately from all-question task success.

Store both raw retrieved evidence and the exact selected generation context. RAGAS answer faithfulness uses the latter; retrieval-level recall/precision can be a separate, clearly named experiment. Do not put unsupported graph paraphrases into the context merely to make the judge consider a claim supported.

## 11. Evaluation datasets and acceptance metrics

### Frozen datasets

Preserve `evals/golden_questions.yaml` as the legacy regression set. Add a versioned graph benchmark with initially about 30 reviewed questions, explicit development/held-out separation, fixed source PDF bytes, fixed synthetic upload fixtures and a corpus manifest. The identical question IDs, bytes, references and scopes must be used for A/B/C comparisons:

- A: `vector_only`.
- B: `graph_only`, no vector similarity seeds or secondary vector tool searches.
- C: `hybrid` with fixed fusion parameters.

Do not regenerate PDF bytes per mode. Freeze chunking/embedding revision, extraction/alias rules, corpus version, question/reference hashes, model parameters and context budget. Portable evidence IDs, not machine-local absolute-path IDs, are the benchmark reference keys. Keep all legacy source expectations intact; stricter new checks have separate names/results.

### Question categories and annotation

| Category | Representative question | Required annotation |
| --- | --- | --- |
| Existing retrieval/tool regression | Banking outcomes; retail comparison; target RFP analysis. | Expected sources/origins/tools, target versus supporting cases. |
| Conjunction | Projects mentioning Azure and HIPAA controls. | Explicit witnesses for each constraint; no inferred certification. |
| Two-hop | Projects sharing technologies with the healthcare project. | Accepted technology-specific paths and both source supports. |
| Catalog completeness | All project timelines. | Ten eligible sample projects, correct durations/source pages; missing fields explicit. |
| Modality/conflicts | Achieved versus projected reductions. | Units, temporal/modality distinctions and acceptable wording. |
| Negative/no-answer | Unsupported client identity or unmentioned requirement. | Must abstain/qualify; cannot invent an entity/path. |
| Non-graph private documents | Synthetic resume or unfamiliar upload. | Vector evidence where appropriate; graph-only limitations measured. |
| Security/failure | Scope crossing, stale support, timeout, injection-like input. | Rejection/degradation result and absence of unauthorized evidence. |

References must be human-reviewed, include acceptable alternatives and field-specific supporting spans, and distinguish source claims from independently established facts. Do not generate held-out references using the same model being evaluated. Annotation conflicts require adjudication; incomplete graph extraction is not permission to relax a known reference.

### Deterministic layers

Measure independently:

1. Expected-source precision/recall, required origin/scope correctness and source-page/chunk coverage. Existing `citation_coverage` mostly measures expected-source retrieval hits; retain it but do not call it final citation provenance.
2. Actual retrieval-mode correctness, requested versus effective modes, all secondary retrieval calls and forbidden-mode use.
3. Route correctness and tool correctness: selected tools, validated inputs, output values and source supports, not only a tool-name presence check.
4. Citation identity validity; graph-fact support coverage; final item/claim links where genuinely recorded. Unknown final-claim linkage stays unknown. Report structural coverage separately from heuristic grounding.
5. Graph path correctness against accepted path sets and every mandatory witness. No path score can excuse an unsupported edge.
6. No-answer behavior, catalog/task completeness and preservation of units/modality. A catalog must cover all eligible projects or explicitly declare partial coverage.
7. Latency per stage and p95 under repeated/load runs, failures/timeouts, recovery and graph-outage overhead.
8. Bounded retry/tool-call counts, prompt/context budget adherence and answer coverage lost during repair.

Mandatory gates: zero unauthorized/cross-scope evidence, zero accepted orphan graph facts, zero write capability through the query boundary, and complete support for every accepted graph-derived fact. Require 100% eligible coverage when an answer claims a complete catalog. Semantic score thresholds and performance targets must be established on a pilot; do not fabricate them from the current deterministic smoke score.

### Fair comparisons and agent evaluation

Run a retrieval ablation with public web search disabled, fixed routing policy, the same tool/provider interfaces and identical generation/judge settings. Grading must assess each mode's actual evidence with the same policy; do not replay a vector-mode grade onto different graph evidence. Capture/cache immutable external outputs where appropriate and report randomness.

Separately run end-to-end Agentic RAG tests with pinned Tavily fixtures and the real bounded cycle, plus authorized live-provider trials when needed. Otherwise a web answer can mask failed private retrieval and contaminate a GraphRAG comparison. Forced evaluation modes never silently fall back to another mode; production degradation is tested as a separate experiment.

Report per-category and paired per-question results, task failure denominators, semantic applicability, confidence intervals, graph extraction coverage and cost. Graph-only can legitimately lose on narrative/resume queries. Hybrid must show an improvement on the relationship subset without material regression on the legacy set, excessive latency, or weaker source guarantees before promotion.

## 12. CI strategy

Keep current CI commands and Python 3.11 coverage. Add, rather than substitute, the following layers:

| Layer | Trigger | Dependencies/data | Blocking behavior |
| --- | --- | --- | --- |
| Existing compile/pytest/smoke | Every PR/push | Existing fixtures; no new service requirement. | Remains mandatory. |
| Graph disabled/adapter unit tests | Every PR | Fake driver, fixed extraction/path fixtures. | Mandatory; startup works without Neo4j/RAGAS packages. |
| RAGAS artifact/schema tests | Every PR | Fake deterministic judge/embeddings, no external calls. | Mandatory; validates integration, not semantic quality. |
| Neo4j integration | Dedicated PR job when graph files change | Disposable pinned Neo4j service and synthetic manifest. | Mandatory once feature ships; verify templates, versioning, hydration and outage behavior. |
| Production privilege tests | Protected deployment/staging job | Actual least-privilege database identities and supported permission model. | Required deployment gate; an admin-backed local service cannot substitute. |
| Frozen KB/mode comparison | Trusted integration job | Pinned/prewarmed embedding model, frozen PDFs/index inputs. | Deterministic scope/path/provenance regressions block. |
| Live RAGAS comparison | Manual/protected scheduled job | Approved judge credentials, synthetic artifacts, pinned metric environment. | Pilot reporting first; calibrated quality gates only after stable baseline. |

Fork PRs must not receive secrets or run arbitrary contributor code with privileged credentials. Provision/cache embedding models in a clearly network-enabled setup step; genuinely offline tests must not depend on opportunistic Hugging Face downloads or configured local grader keys. No passing CI semantic-quality claim from fake judges.

Use optional dependency groups/isolated constraints for the driver and evaluator, recording exact resolved versions and model revisions in artifacts. Do not upgrade all LangChain dependencies just to add RAGAS. Preserve compatibility tests on Python 3.11 and add the supported local Python version deliberately.

Retain sanitized result manifests and aggregate reports with explicit access control and initial 14-day retention; private evidence artifacts need separate approval. Version pinning, redaction and serialization failures are tested. Update documentation's historical test counts only when a new recorded baseline exists, not as part of this design-only task.

## 13. Migration phases and gates

| Phase | Deliverable | Exit gate |
| --- | --- | --- |
| 0: contracts/baseline | Capture actual context and pre/post-repair output; portable mappings; deterministic external-call controls; reproduce cross-corpus failure. | Legacy tests unchanged; failing live baseline explained; vector-only differential behavior documented; no default graph behavior change. |
| 1: ingestion shadow | Minimal schema, deterministic extraction, manifests, separate writer/admin and offline fixtures. | All published assertions have exact support; repeat ingestion is idempotent; deletion/version tests pass. |
| 2: bounded retrieval | Read-only templates, limits, hydration/version/ACL validation, graph-only test provider. | Real reader cannot write; paths/witnesses correct; timeout/outage tests pass. |
| 3: hybrid shadow | Typed fusion and consistent secondary tool retrieval; no user-visible default switch. | No graph pseudo-score misuse; mandatory supports fit; context budgets enforced; source labels/contracts preserved. |
| 4: evaluation pilot | Frozen A/B/C runs, human references, isolated RAGAS adapter and calibrated judge. | Paired relationship gains demonstrated; legacy/task completeness preserved; costs/uncertainty reported. |
| 5: canary | Opt-in small authorized corpus, dashboards, restore drills and public-search policy. | Latency/error limits validated; support coverage intact; independent reader permissions/rollback proven. |
| 6: controlled promotion | Expand approved use cases and later ontology only where justified. | Operator approval with recorded evidence; vector-only remains supported and reversible. |

Cloud deployment initially needs a persistent Chroma volume/snapshot strategy and a separately managed Neo4j service, not a simultaneous vector-store migration. Define operator ownership for graph backups, manifests, credential rotation, failed ingestion/reconciliation and paid evaluation budgets. HA/multi-tenant changes are separate projects with explicit authorization and tests.

## 14. Expected new and modified files

These are future implementation targets, not files created by this review. Final names can be consolidated, but responsibilities must stay separated.

| New files/directories | Responsibility |
| --- | --- |
| `src/rfp_analyst/graph/schema.py` | Typed entity/assertion/provenance schemas and predicate validation. |
| `src/rfp_analyst/graph/client.py` | Driver lifecycle, read-only execution, deadlines/circuit handling. |
| `src/rfp_analyst/graph/templates.py` | Reviewed finite query-template registry and parameter validation. |
| `src/rfp_analyst/graph/extraction.py` | Deterministic corpus extraction and span/modality validation. |
| `src/rfp_analyst/graph/ingestion.py`, `manifest.py` | Writer-only publication, idempotent jobs, snapshot/reconciliation. |
| `src/rfp_analyst/retrieval/evidence.py`, `hybrid.py`, `routing.py` | Additive evidence protocol, provider/fusion and strategy policy. |
| `src/rfp_analyst/tools/graph_provenance.py` | Structural graph witness validation; does not replace source verifier. |
| `evals/run_retrieval_comparison.py`, `run_ragas_evals.py`, `ragas_adapter.py` | Mode runs, capture and isolated semantic scoring. |
| `evals/datasets/graphrag_questions.yaml`, references and corpus manifest | Frozen questions, reviewed answers/spans/paths and hashes. |
| Graph/RAGAS tests and extraction/security fixtures | Disabled-mode, extraction, scope, path, role, failure, semantic-adapter and UI compatibility tests. |
| Versioned graph migration artifacts; optional evaluation constraints | Schema/index migrations and tested isolated dependency pins. |

| Existing files to change later | Bounded change |
| --- | --- |
| `config.py`, `.env.example` | Optional graph/evaluation settings; secrets remain env-driven. |
| `pyproject.toml` / dependency constraints | Optional `graph` and `eval` extras; no mandatory graph/judge for legacy users. |
| `rag_engine.py` and ingestion helpers | Post-publication manifest/jobs; existing vector success/cleanup preserved. |
| `retrieval/vector_store.py` | Scoped/version-checked by-ID hydration; not replacement of semantic retrieval. |
| `agent/graph.py` | Additive state/provider capture and opt-in retrieval; keep current conditional topology/retry semantics. |
| `tools/search_kb.py`, comparison/case-matching retrieval seams | Optional evidence provider, consistent secondary retrieval and evidence capture; preserve algorithms/public outputs. |
| `agent/prompts.py` | Explicit supported graph facts/modality instructions where enabled; do not remove existing prompts. |
| Root `agent.py`, `app.py`, UI helpers | Additive diagnostics/health and source cards; keep source labels, return types and streaming behavior. |
| `evals/metrics.py`, `run_kb_evals.py` | Add exact mode/context/tool/provenance metrics and reproducible controls; keep legacy summary metrics/commands. |
| Existing tests, `.github/workflows/ci.yml` | Regression coverage and separate optional jobs; preserve existing mandatory checks. |
| README and architecture/testing/reproducibility docs | Reconcile linear/cyclic and metric claims with implemented, tested behavior. |

Proposed settings: `GRAPHRAG_ENABLED=false`, `KB_RETRIEVAL_MODE=vector_only`, `NEO4J_URI`, `NEO4J_DATABASE`, reader username/password, graph limits/timeouts and snapshot policy. Writer credentials belong only in worker settings. Evaluator settings include an explicit judge provider/model/key reference, embeddings, concurrency, timeout and budget. Secret values never appear in checked-in examples or run reports. Configuration parsing must be lazy enough that unused optional settings cannot break vector-only operation.

## 15. Backward compatibility and things NOT to change

The following are explicit constraints, not modernization opportunities:

- Do not replace Chroma/FastEmbed with Neo4j vectors or Pinecone in this upgrade. Do not re-embed the corpus merely to add graph relationships.
- Do not replace the LangGraph cyclic workflow, existing router/intent refinements, grading, Tavily integration, one-retry rewriting or degraded graph fallback.
- Do not remove sample/upload/all filtering, upload limits, atomic vector publication/backup handling, Windows resource cleanup or readiness checks.
- Do not change existing chunk IDs, zero-based page metadata, rendered page citation format or `Private KB`/`Web Search`/`Direct` labels.
- Do not silently move `QueryState` into the legacy dataclass module, change public agent/retrieval callable signatures, or introduce eager optional imports/circular dependencies.
- Do not replace deterministic catalogs, comparisons, requirement/gap tools or proposal output contracts with free-form graph/LLM answers.
- Do not remove the final verifier/one bounded repair or describe its heuristic checks as formal entailment.
- Do not turn every extracted requirement into a proven case-study satisfaction/compliance relationship.
- Do not add unrestricted Cypher, user-question writes, dynamic procedures, query-generated graph facts or silent cross-origin/tenant joins.
- Do not replace pytest, mock smoke metrics or legacy real-KB questions with RAGAS scores. Do not rewrite reference expectations just to make new modes pass.
- Do not make paid judges, Neo4j credentials or network access mandatory for existing local/CI users.
- Do not include credentials/private PDF text/prompts in evaluation exports or assume external scoring is approved for private data.
- Do not claim final claim-level lineage where the runtime only records retrieved chunks and page citations. Unknown links remain unknown.

Use additive optional state/output fields with defaults. Preserve existing UI/trace adapters and fixture expectations. Driver connections should initialize only when enabled; RAGAS should initialize only in its evaluator process. Tests must cover import/startup with neither optional package installed and graph-down behavior with ordinary vector queries.

## 16. Rollback plan

1. Disable GraphRAG and force `vector_only` through an operator-controlled setting. Restart/refresh affected runtime dependencies safely; do not mutate a cached compiled graph mid-query.
2. Confirm existing health, scoped retrieval, tools, source labels, generation and verification operate normally without Neo4j access.
3. Stop graph publication workers or pause their queue while preserving pending jobs/manifests for inspection. Do not delete the active Chroma store or private source files.
4. Restore a validated prior graph snapshot only if its corpus manifest matches the intended vector/source snapshot. Otherwise leave graph disabled. Restoring one database does not restore cross-store consistency.
5. Restore matched vector/source/manifest snapshots if ingestion data itself regressed, using existing safe publication/backup handling and an operator-approved restore procedure.
6. Roll back optional app changes to a recorded release while retaining provenance/evaluation artifacts for diagnosis. Graph namespaces/schema migrations remain separately versioned; no destructive automatic downgrade.
7. Disable semantic-evaluation jobs independently. Their failure or rollback must have no effect on user answering.

Rehearse graph disconnect, stale manifest, bad extraction release and credential revocation rollbacks before promotion. Preserve snapshot retention/access policies; deleting graph data is not the default rollback mechanism.

## 17. Risks and mitigations

| Risk | Mitigation / acceptance evidence |
| --- | --- |
| Small corpus overfitting or graph cost without useful gains | Held-out relationship questions, honest graph-only limitations, canary/no-go decision based on paired results. |
| Hallucinated extraction or anonymous-client conflation | Deterministic-first extraction, exact spans, document-scoped identities, reviewed merges and rejected unsupported assertions. |
| Proposed outcomes/compliance become achievements | Typed assertion modality/units, reference annotations and dedicated negative tests. |
| Graph/vector drift, deleted data or broken portable IDs | Versioned manifests, source hashes, reconciliation, revocation tests and fail-closed support validation. |
| Unbounded query work or credentials capable of writes | Fixed templates, bounded indexed traversal, deadlines and real database privilege tests. |
| Cross-scope/tenant leakage | Authorization before traversal/hydration, witness validation and dedicated multi-user prerequisites. |
| Graph rank mistaken for semantic relevance | Separate score fields; vector-only threshold preserved; provider contract tests. |
| Multi-hop answer lacks a supporting edge | Mandatory witness packing, structural validation and explicit partial/no-answer behavior. |
| Existing prompt budget does not cover actual generation | Capture/assemble the real prompt first, differential legacy tests and measured token/context checks. |
| Repair produces grounded but useless answers | Pre/post-repair task coverage, catalog completeness and semantic/reference evaluation. |
| Tool secondary retrieval invalidates mode comparison | One injected provider for all searches; assert no vector calls in forced graph-only runs. |
| RAGAS/LangChain API or dependency drift | Isolated pinned extras/constraints and adapter signature tests, not blanket dependency upgrades. |
| Judge bias, variance, cost or disclosure | Human calibration, explicit judge/embedding config, repeated pilot, quotas and approved synthetic datasets. |
| Existing retrieval-only evaluation uses live branches | Explicit external-call controls; separate retrieval, end-to-end and semantic experiments. |
| Documentation/score overclaims | Record exact baseline/modes/failures; separate retrieval hits, grounding heuristics and formal lineage. |

## 18. Executed baseline report

Only this plan document is an intentional tracked-file change. Existing test/evaluation commands were run without code fixes, dependency installation, private corpus ingestion or schema/database implementation. Tests/evaluators generate their normal ignored artifacts.

### Environment and commands

Windows / PowerShell, repository commit stated above, local `.venv` Python **3.12.14**. CI targets Python 3.11, so this run is not a fresh CI-host reproduction. Selected installed versions: LangGraph 1.2.11, LangChain 1.3.18, ChromaDB 1.5.9, FastEmbed 0.8.0, Pydantic 2.13.5 and pytest 9.1.1. Neo4j and RAGAS packages were not installed. Secret values were not examined or exported.

Commands below used the existing venv; `pytest` is the `make test` equivalent on this Windows environment:

```powershell
& .\.venv\Scripts\python.exe --version
& .\.venv\Scripts\python.exe -m py_compile app.py agent.py rag_engine.py config.py document_generator.py
& .\.venv\Scripts\python.exe -m compileall -f src tests evals
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m evals.run_evals
& .\.venv\Scripts\python.exe -m evals.run_kb_evals
& .\.venv\Scripts\python.exe -m ruff check .
```

| Check | Observed result | Interpretation |
| --- | --- | --- |
| Critical-entry compilation | Exit 0 | Listed root entry points compile. |
| Source/test/eval compilation | Exit 0 | Tree compiles. |
| Pytest | **145 passed, 1 warning, 11.69 s**, exit 0 | All existing tests pass in this environment. Warning concerns `langchain-community` deprecation. |
| Offline smoke evaluation | **3/3**, pass rate 1.0, approximately 0.0000125 s, exit 0 | Mock corpus/deterministic synthesis; not real retrieval or semantic answer quality. |
| Real-KB golden evaluation | **8/9**, pass rate 0.89, 119.216 s case-execution latency, exit 0 | One case fails despite process success. Default mode is `retrieval_only`; source/tool/no-answer checks, not generated-answer correctness. |
| Ruff | Exit 1: `No module named ruff` | Optional lint command unavailable in the existing environment. Not a lint pass and no installation performed. |
| Neo4j / RAGAS evaluation | Not run / not available | Design-only scope; no new services, packages or semantic scores. |

The real-KB command initially failed because the sandbox blocked a network socket (`WinError 10013`) while FastEmbed resolved its model through Hugging Face. An approved rerun completed. Therefore this result is not evidence of a fully air-gapped KB evaluation. Model setup/ingestion time is outside the reported aggregate case latency.

### Real-KB details

The evaluator built an isolated temporary Chroma store from ten public sample PDFs (30 pages) and one synthetic uploaded target RFP (one page): **11 documents, 54 chunks**. It did not reindex the application's private uploads/store.

| Golden case | Result | Case latency, seconds |
| --- | --- | ---: |
| Sample retrieval | Pass | 2.577 |
| Upload retrieval | Pass | 2.112 |
| Cross-corpus RFP analysis | **Fail** | 8.462 |
| Previous-source recall | Pass | 0.004 |
| Unsupported query | Pass | 11.113 |
| Comparison | Pass | 3.888 |
| Web fallback | Pass | 32.286 |
| Bounded query rewrite | Pass | 35.780 |
| Project catalog | Pass | 22.987 |

For the failing cross-corpus case, expected-source hit coverage was 1.0, but returned origins contained only `upload` (`eval_target_rfp.pdf`), not the required sample case-study evidence. The recorded path included `target_context_retrieval`, KB evidence checking, `web_search`, `web_evidence_grader` and final response, rather than the required requirement extraction, case matching and proposal tools. This establishes a branch/tool-path failure. The result does not include the detailed grading explanation, so the exact reason for the grader decisions is not established by the saved summary.

The evaluator supplies no generation LLM in its default mode, but routing/grading/rewriting helpers can independently instantiate configured providers. Thus `retrieval_only` does **not** mean that all graph decisions/public-search branches are guaranteed offline. Its JSON does not capture generated-answer quality, exact model-visible contexts, or all final claim links. Live-provider settings can affect this baseline even when answer generation is disabled.

The legacy pass rule admits expected-source coverage of at least 0.5 and checks tool-name/origin/no-answer expectations. It is not sufficient as a GraphRAG completeness/provenance gate. The evaluator also exits successfully when individual cases fail; future CI must explicitly assert the desired case results rather than rely solely on exit status.

Reproducibility anchors: golden-question file SHA-256 `13EB016110E872FA2DDF6F8E47A6C451B1FBB0969ECD570FC1125D3290F6EE2F` and the repository commit above. Current local PDF hashes were inspected; a future frozen-corpus manifest must persist them with chunking/model configuration and the synthetic upload bytes. Existing machine-path IDs and regenerated PDF metadata must not become cross-machine benchmark keys.

### Baseline conclusions and stop condition

- No existing test failure was observed; one real-KB evaluation case failed and lint was unavailable.
- No full generated-answer/RAGAS baseline, Neo4j load benchmark, production permission test or fresh Python 3.11 CI run is claimed.
- Resolve/reproduce the orchestration/evaluation determinism issue before interpreting A/B/C results. GraphRAG must not receive credit for changes in web routing, source expectations or test fixtures.
- This task ends with the plan and baseline report. No application code, dependencies, graph schema, credentials, evaluation questions or CI workflow were changed; no commit/push or deployment was performed.
