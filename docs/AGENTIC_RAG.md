# Agentic RAG

## Why This System Is Agentic

This repository does not use a single retrieve-and-generate chain as its primary execution model. The canonical runtime classifies intent, carries structured state across graph nodes, executes deterministic tools conditionally, separates target and case-study corpora, compacts evidence before model generation, and performs post-generation verification with an optional bounded repair pass.

That combination of routing, tool execution, scoped retrieval, and verification is what makes the system agentic in practice.

## Actual Tools

The runtime uses the following deterministic tools:

- `search_knowledge_base`
- `extract_rfp_requirements`
- `find_relevant_case_studies`
- `compare_projects`
- `generate_proposal_outline`
- `verify_answer_grounding`

These tools operate on retrieved evidence and produce structured outputs that later nodes consume.

## Graph State

Key graph state values:

- user request and resolved query
- chat history and resolved conversational entities
- retrieval scope and retrieval function
- vectorstore health/stats
- planned tools
- retrieved documents
- tool outputs
- prompt budget metadata
- answer mode, answer text, and verification results
- visible traces

## Intents and Conditional Execution

Supported intents:

- `search`
- `compare`
- `proposal`
- `rfp_analysis`
- `previous_sources`
- `ambiguous`

Conditional behavior:

- `previous_sources` returns direct source recall without retrieval
- `ambiguous` returns a clarification prompt
- `rfp_analysis` forces upload-scoped target retrieval and sample-scoped case-study retrieval
- `compare` executes the comparison tool
- `proposal` and `rfp_analysis` execute requirements, case studies, and proposal helpers

## Target and Sample Corpus Separation

Target/sample separation is enforced by:

- separate directories for generated samples and uploaded documents
- `document_origin` metadata
- scoped Chroma filters
- separate retrieval stages in `rfp_analysis`
- prompt compaction that keeps uploaded target evidence and sample case-study evidence in distinct sections

This prevents internal sample documents from being mistaken for uploaded target requirements.

## Structured Tool Outputs

The runtime keeps full internal tool outputs for verification, but the prompt builder uses compact representations:

- requirement summaries and inferred gaps
- ranked case-study summaries with fit scores and citations
- compact project comparison rows
- six-section proposal-outline bullets

That allows the graph to stay grounded without sending raw nested objects to the model.

## Verification and Repair

The answer is generated first, then checked by `verify_answer_grounding`. If unsupported claims or vague citations remain:

1. `grounding_verifier` records the first verification result
2. `answer_repair` removes or qualifies unsupported claims once
3. `final_grounding_verifier` verifies the repaired answer

The repair pass is intentionally bounded and never loops indefinitely.

## Agent Traces

The UI exposes safe trace summaries only:

- tool name
- short input summary
- short output summary
- selected sources
- prompt-budget data
- grounding status

No private chain-of-thought is emitted.

## Implementation Mapping

| Capability | Implementation |
| --- | --- |
| Intent classification | `src/rfp_analyst/agent/graph.py::classify_intent` and `_classify_query` |
| Tool planning | `src/rfp_analyst/agent/graph.py::plan_tools` |
| Scoped retrieval | `src/rfp_analyst/agent/graph.py::execute_retrieval`, `rag_engine.py::similarity_search`, `src/rfp_analyst/retrieval/vector_store.py::VectorStoreManager.similarity_search` |
| Requirement extraction | `src/rfp_analyst/tools/rfp_gap_analyzer.py::extract_rfp_requirements` |
| Case-study search | `src/rfp_analyst/tools/rfp_gap_analyzer.py::find_relevant_case_studies` |
| Comparison | `src/rfp_analyst/tools/compare_projects.py::compare_projects` |
| Proposal generation | `src/rfp_analyst/tools/proposal_writer.py::generate_proposal_outline` |
| Prompt budgeting | `src/rfp_analyst/agent/graph.py::synthesize_prompt`, `_estimate_tokens`, `_build_compact_prompt_sections`, `compact_tool_outputs_for_prompt` |
| Grounding | `src/rfp_analyst/tools/source_verifier.py::verify_answer_grounding` and `src/rfp_analyst/agent/graph.py::_verify_generated_answer` |
| Previous-source recall | `src/rfp_analyst/agent/graph.py::_previous_answer_sources` and `_format_previous_sources` |

## How This Differs from Basic RAG

Basic RAG usually:

- retrieves one mixed context set
- prompts once
- returns the answer

This system instead:

- routes by intent
- resolves conversational references
- separates target and internal corpora
- executes deterministic tools
- scores case studies explicitly
- compacts prompts to fit model limits
- verifies claims after generation
- performs one bounded repair pass when needed
