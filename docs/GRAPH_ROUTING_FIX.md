# Standalone GraphRAG query routing regression

Verification date: 2026-10-06. Scope: conversational classification only.
No graph ingestion, graph retrieval algorithm, schema, timeout policy, frozen
benchmark, or RAGAS changes were made.

## Root cause and reproduction

With `NEO4J_ENABLED=true`, `RFP_RETRIEVAL_MODE=graph_only`, scope `all`, and
`chat_history=[]`, this exact question reproduced the reported failure:

> List all projects in the knowledge base that used Microsoft Azure. For each
> matching project, list the other technologies or frameworks used and the
> reported outcomes.

`VAGUE_REFERENCE_PATTERN` matched the word **that**, although it introduces a
relative clause selecting projects in the current question. The conversational
resolver searched for a prior source, found none, and reported `ambiguous`.
`classify_intent` then set `source_used=direct` and
`response_mode=clarification` **without invoking the structured KB router**.
The conditional edge selected `direct_answer`, which returned the follow-up
clarification. This was a deterministic pre-router error, not a Neo4j or
LLM-generated intent classification error.

Observed pre-fix workflow:

```text
START -> health_check -> route_question/classify_intent
      -> unresolved "that" -> ambiguous/direct -> direct_answer -> END
```

The live provider's `reader.fetch` call count was **zero**. Requested mode was
`graph_only`; the state's untouched initial effective-mode field was
`vector_only`. No vector retrieval actually ran either.

## Minimal change

In `src/rfp_analyst/agent/graph.py`:

- Exclude relative `that` attached to an explicit project/document noun,
  optionally qualified by `in/from/within the knowledge base`, when followed
  by a supported factual relation verb such as `used`, `mention`, or `support`.
- Resolve later plural references locally only when the question already
  explicitly selects a project/document set. Recognized graph constraints,
  explicit list/search instructions, or an all-project request establish that
  local set. No prior source is injected into a new standalone selection.
- Recognize possessive/plural pronouns (`its`, `they`, `them`, `their`) as
  possible conversation references rather than silently overlooking them.
  Unresolved pronouns still clarify; singular references with multiple possible
  prior documents still clarify; out-of-scope antecedents still do not resolve.
- Preserve the existing narrow locally named technology/framework relationship
  exception and all existing clarification, generation, and grounding gates.

This is not a universal coreference parser. Leading `those projects`, `Does
it use Azure?`, `Which projects used it?`, and `What about its budget?` still
need a usable conversational antecedent. An Azure mention alone cannot resolve
an earlier `it`. `List projects that used it` also remains unresolved.

## Scope and Streamlit reruns

`app.process_query` passes `st.session_state.messages` and the selected scope
to the existing root adapter. A UI history containing only the current user
message has no prior source and is covered by regression tests.
`prepare_query_payload` constructs a new state for each invocation, including
fresh traces, answer, resolved entities, retry count, and graph metadata.
The cached compiled graph has no checkpointer retaining a previous answer.
The UI caches its LLM client, not a graph query state. No state-retention defect
was found, and neither `app.py` nor `agent.py` was changed for this fix.

Tests verify that a standalone query does not inherit an unrelated prior
source, and that a new invocation after clarification still enters graph
retrieval. All-document scope remains `all` through classification and graph
retrieval. No sample/upload scope broadening was introduced.

## Actual post-fix live execution

The same exact Azure question was executed with empty conversation history and
scope `all`, using the configured Aura reader and application generation LLM.
Router/grader calls were live, not mocked. Web fallback was disabled for this
verification; a rejecting vector supplier proved no vector retrieval occurred
in the successful graph-only run. No ingestion or graph writes were executed.

First, a connection-health-warmed run succeeded with template-level observation:
the header, project seeds, subject fact reads, and corpus-currentness reads
all returned. Both initial and final grounding checks passed (four checked
claims, zero unsupported). Then a separate successful run used the normal
per-query provider factory, **without a connection warm-up**. Observation only
wrapped the reader; its queries, bounds, configuration, and results were not
substituted. The owned provider was closed by the existing `finally` block.

Observed normal-composition run:

| Field | Actual result |
| --- | --- |
| KB health | Ready; 13 indexed documents, 102 chunks |
| Input intent / resolution | `search` / `not_needed` |
| Requested / effective retrieval | `graph_only` / `graph_only` |
| Reader | Real `Neo4jGraphReader` |
| Live projection calls | 1, returned in 2.930 seconds |
| Vector search calls | 0 |
| Fallback reason | Empty |
| Validated graph paths | 8 |
| Provenance records / original generation chunks | 16 / 16 |
| Generation kind | `llm_kb` |
| Initial grounding | 6 checked claims; 4 unsupported |
| Existing bounded repair | Executed once |
| Final grounding | 2 checked claims; 0 unsupported; `grounded=true` |

Active version observed:
`eddc57727f1f70adb54c3d4d98d6896e4245daaa6cc896f8bfbafc63b88fdfdc`.

The actual successful tool/node path was:

```text
health_check
 -> route_question/classify_intent: search, KB, resolution not_needed
 -> retrieve_kb: plan_tools -> execute_retrieval
 -> Neo4j graph_retrieval: graph_only, validated original chunk provenance
 -> grade_kb_evidence: good
 -> execute_tools -> synthesize_prompt / prompt_budget
 -> generate_from_kb: graph_integrity_check, final-context grading/currentness
 -> final_response (live LLM)
 -> grounding_verifier -> one bounded answer_repair -> final_grounding_verifier
 -> END
```

No follow-up clarification, web search, query rewrite, or vector search was
needed in this successful run. Retrieved chunks were hydrated from the existing
index and retained in the actual generation capture. This is a genuine live
graph retrieval/generation verification, not just an asserted routing decision.
Grounding remains the existing citation/lexical/numeric heuristic, not formal
semantic entailment or proof of exhaustive answer completeness.

## Regression tests and commands

Added `tests/test_conversation_routing.py`: **38 test cases** covering:

- Fresh plural project, relationship, multi-hop, relative-clause, and later
  `those projects` selections, including empty and user-only histories.
- A structured KB-router invocation after the conversational gate.
- Prior-context possessive follow-up resolution, unresolved follow-ups,
  out-of-scope antecedents, and multiple possible singular antecedents.
- All-document scope and requested/effective graph-only mode.
- Real workflow execution against the synthetic driver projection: project
  seeds/fact reads, original supporting chunks, provenance, generation capture,
  final grounding, and absence of vector/web retrieval.
- New invocation after clarification and no binding to stale prior sources.

All commands used `.venv\Scripts\python.exe`:

| Command | Final result |
| --- | --- |
| `-m pytest tests/test_conversation_routing.py tests/test_langgraph_agent.py tests/test_hybrid_retrieval.py tests/test_retrieval_benchmark.py -q` | 195 passed; 43.03 seconds |
| `-m pytest -q` | 481 passed, 5 skipped; 98.37 seconds |
| `-m ruff check .` | All checks passed |
| `-m py_compile src/rfp_analyst/agent/graph.py tests/test_conversation_routing.py` | Passed |
| `-m evals.run_evals` | 3/3; offline smoke, not live quality proof |
| `-m evals.run_kb_evals` | 9/9; isolated real Chroma retrieval-only evaluation |

The five skipped tests are the existing opt-in disposable Neo4j integration
tests; they are not replaced by or conflated with the separate live Aura check.
Existing evaluation result files are refreshed by their normal commands.

Frozen benchmark files were not edited. Recorded SHA-256 fingerprints:

- `evals/retrieval_questions.yaml`:
  `7846cea3aa3eccc3b281d44dd1a3eecff51d65b08265f045ff1e34a896bc6689`
- `evals/retrieval_benchmark.lock.json`:
  `a76610f63e0047e13033cbd45cb7b732c75ff17e36f316b6b557bc07536c0eb7`

## Remaining limitations

- **Aura latency is intermittent.** Earlier live attempts reached the graph
  reader but timed out before returning evidence; a normal-composition probe
  also attempted the existing vector fallback after a failed read. Successful
  normal-composition projection latency was 2.930 seconds, close to the
  unchanged three-second projection deadline. This fix does not guarantee
  cold connection reliability or alter the security/timeouts/fallback design.
- `graph_only` production policy still permits the existing degraded vector
  fallback on graph failures; strict graph-only is still a separate option.
  A requested graph mode alone is never evidence of effective graph retrieval.
- Generation can still require the existing repair pass, which may remove
  unsupported claims and reduce answer completeness. This routing task does
  not weaken the final verifier or claim comprehensive factual coverage.
- A separate parser boundary was noticed: `outcomes of those projects` can
  become a `project_fields` plan with a nonliteral project reference. The
  local follow-up gate now accepts that standalone reference, but resolving
  that particular retrieval-plan wording is outside this conversational fix.
  Neither the retrieval planner nor the frozen benchmark was changed to hide
  this limitation.

Task-specific files changed: the agent graph's conversational helper logic;
the new focused test module; and this report. Existing unrelated worktree
changes, credentials, dependencies, ingestion, RAGAS, and benchmark data were
preserved. No commit, push, or deployment was performed.
