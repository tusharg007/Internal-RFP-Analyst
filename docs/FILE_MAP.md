# File Map

This map covers the meaningful tracked project files and explains how they fit into the live application. Generated PDFs, local vectorstores, caches, upload artifacts, and `*.egg-info` metadata are intentionally excluded.

## Root Application Files

| Path | Purpose | Important functions/classes | Inputs | Outputs | Dependencies | Live path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `app.py` | Streamlit UI entry point | `build_health_snapshot`, `handle_uploaded_files`, `run_manual_ingestion`, `process_query`, `render_eval_snapshot` | user interaction, session state, uploaded files | Streamlit views, warnings, chat responses | `agent.py`, `rag_engine.py`, `src/rfp_analyst/*` | Yes | Primary application entry point |
| `agent.py` | Provider selection and UI-facing wrappers | `get_llm`, `prepare_query`, `query_agent_stream`, `query_agent`, `format_llm_error` | provider keys, payloads, user query | LLM client, stream output, reasoning traces | `config.py`, graph runtime | Yes | Compatibility adapter over canonical graph runtime |
| `rag_engine.py` | Ingestion/retrieval facade | `ingest_documents`, `similarity_search`, `get_vectorstore_stats`, `get_retriever` | sample dir, uploads dir, persist dir, query | ingestion report, results, stats | ingestion modules, retrieval manager | Yes | Core KB management module |
| `config.py` | Global configuration | `get_api_keys`, `_get_int_setting` | env vars, `.env`, Streamlit secrets | constants and resolved keys | `dotenv`, `streamlit` | Yes | Source of truth for paths, limits, budgets |
| `document_generator.py` | Synthetic sample PDF generator | `generate_all_documents`, `PDFDocument` | static document definitions | generated sample PDFs | `fpdf2`, `config.py` | Yes | Used by UI and real KB eval setup |
| `Makefile` | Convenience commands | `install`, `test`, `lint`, `run`, `generate-docs` | shell target | local command execution | Python tooling | No | Helpful locally, not part of runtime |
| `pyproject.toml` | Packaging and tool config | project metadata, pytest/ruff config | package metadata | editable install config | setuptools | Indirect | Required for `pip install -e .` |
| `requirements.txt` | Runtime dependencies | package list | pip install | installed dependencies | pip | Indirect | Used by local setup and CI |
| `.env.example` | Example local configuration | environment variable template | developer edits | local `.env` seed | `config.py` | Indirect | Safe template only |
| `.gitignore` | Repository hygiene | ignore rules | git status/add | ignored runtime files | git | Indirect | Protects secrets and local artifacts |
| `.python-version` | Python version hint | `3.11` | pyenv/asdf style tools | version hint | local tooling | Indirect | Matches project requirement |
| `.streamlit/config.toml` | Streamlit theme/server config | theme and upload-size settings | Streamlit startup | UI theme / upload limit | Streamlit | Yes | Affects app appearance and file-upload cap |

## Agent Modules

| Path | Purpose | Important functions/classes | Inputs | Outputs | Dependencies | Live path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `src/rfp_analyst/agent/graph.py` | Canonical orchestration runtime | `compile_query_graph`, `prepare_query_payload`, `execute_retrieval`, `execute_specialized_tool`, `synthesize_prompt`, `run_query`, `stream_query_response` | query, scope, chat history, stats, retrieval fn | prompt, answer, traces, verification state | LangGraph, tools, `rag_engine.py` | Yes | Main source of agentic behavior |
| `src/rfp_analyst/agent/runtime.py` | Compatibility runtime helpers | `prepare_simple_query`, `prepare_agentic_query`, deprecated wrappers | user query, chat history | compatibility payloads | graph runtime, prompts | Partial | Supports simple/agentic compatibility modes |
| `src/rfp_analyst/agent/state.py` | Legacy-compatible state container | `AgentState` | query, trace state | dataclass instances | stdlib dataclasses | Partial | Used by compatibility graph wrapper |
| `src/rfp_analyst/agent/prompts.py` | Prompt helpers for compatibility mode | `build_simple_prompt`, `build_agentic_prompt`, `classify_query_intent` | user query, tool outputs, stats | prompt text | `config.py` | Partial | Graph runtime is canonical; this remains for compatibility |
| `src/rfp_analyst/agent/__init__.py` | Package exports | graph/runtime exports | imports | module API | local agent modules | Indirect | Packaging convenience |

## Ingestion Modules

| Path | Purpose | Important functions/classes | Inputs | Outputs | Dependencies | Live path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `src/rfp_analyst/ingestion/loaders.py` | PDF validation and loading | `sanitize_filename`, `validate_pdf`, `load_pdf_sources`, `sha256_file` | PDF paths | `LoadedSource[]` | PyMuPDF, config, exceptions | Yes | Enforces size and page-count limits |
| `src/rfp_analyst/ingestion/chunking.py` | Deterministic chunk generation | `build_chunk_id`, `chunk_loaded_sources` | `LoadedSource[]` | LangChain `Document[]` chunks | text splitter, schemas | Yes | Adds deterministic metadata |
| `src/rfp_analyst/ingestion/pipeline.py` | Shared ingestion helpers | project-specific utilities | internal module inputs | helper outputs | local ingestion modules | Indirect | Support code, not the main facade |
| `src/rfp_analyst/ingestion/registry.py` | Duplicate-prevention registry helpers | registry functions | file/chunk metadata | registry state | local ingestion logic | Indirect | Supports idempotent ingestion behavior |
| `src/rfp_analyst/ingestion/__init__.py` | Package marker | package exports | imports | module API | local modules | Indirect | Packaging convenience |

## Retrieval Modules

| Path | Purpose | Important functions/classes | Inputs | Outputs | Dependencies | Live path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `src/rfp_analyst/retrieval/vector_store.py` | Chroma manager and dedupe logic | `VectorStoreManager`, `deduplicate_documents_by_chunk_id`, `get_embeddings` | chunks, query, scope | Chroma collections, search results | Chroma, FastEmbed | Yes | Core persistence and scoped search |
| `src/rfp_analyst/retrieval/__init__.py` | Package marker | package exports | imports | module API | local module | Indirect | Packaging convenience |

## Tools

| Path | Purpose | Important functions/classes | Inputs | Outputs | Dependencies | Live path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `src/rfp_analyst/tools/search_kb.py` | Normalize scoped search results | `search_knowledge_base` | query, search fn, k | documents, sources, context | retrieval manager or injected search fn | Yes | Tool-level retrieval normalization |
| `src/rfp_analyst/tools/rfp_gap_analyzer.py` | Requirement extraction, inferred gaps, and case-study scoring | `extract_rfp_requirements`, `find_relevant_case_studies` | target evidence text, requirements, search fn | requirement list, gaps, ranked matches | search tool, config | Yes | Core cross-corpus analysis helper |
| `src/rfp_analyst/tools/compare_projects.py` | Structured project comparison | `compare_projects` | query, search fn | comparison rows and markdown | search tool | Yes | Used for compare intent and RFP fit comparison |
| `src/rfp_analyst/tools/proposal_writer.py` | Six-section proposal outline generation | `generate_proposal_outline` | user query, case studies, requirements | proposal outline dict | deterministic structured inputs | Yes | Used for proposal and RFP analysis |
| `src/rfp_analyst/tools/source_verifier.py` | Post-generation claim verification | `verify_answer_grounding` | answer text, supporting docs | grounding result dict | regex/token logic | Yes | Supports repair flow |
| `src/rfp_analyst/tools/__init__.py` | Package marker | package exports | imports | module API | local modules | Indirect | Packaging convenience |

## UI Helpers and App Support

| Path | Purpose | Important functions/classes | Inputs | Outputs | Dependencies | Live path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `src/rfp_analyst/ui/helpers.py` | UI helper functions | `get_chat_avatar`, `format_latency_display` | role, metrics | avatar / latency text | none | Yes | Used by Streamlit UI and tests |
| `src/rfp_analyst/ui/__init__.py` | UI helper exports | exports helper functions | imports | module API | helpers | Yes | Supports package-style imports |
| `src/rfp_analyst/health.py` | Central health snapshot builder | `get_app_health`, `get_provider_status` | stats, keys, paths | health dict | config, vector store manager | Yes | Used by UI and tests |
| `src/rfp_analyst/uploads.py` | Upload validation and persistence | `sanitize_uploaded_filename`, `validate_uploaded_pdf`, `persist_uploaded_pdf` | uploaded file objects | saved paths | config, exceptions | Yes | Used directly by `app.py` |
| `src/rfp_analyst/exceptions.py` | Custom exception types | exception classes | raised error conditions | typed exceptions | stdlib | Yes | Shared error boundary definitions |
| `src/rfp_analyst/schemas.py` | Shared data structures | `LoadedSource` and related models | loader/chunker data | typed containers | stdlib/dataclasses | Yes | Cross-module ingestion schema |
| `src/rfp_analyst/__init__.py` | Package marker | package version/module exports | imports | package API | local modules | Indirect | Editable install target |

## Evaluation

| Path | Purpose | Important functions/classes | Inputs | Outputs | Dependencies | Live path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `evals/run_evals.py` | Deterministic smoke evaluation | `run_offline_smoke_eval`, `synthesize_offline_answer` | mock golden cases | JSON snapshot | config | Yes | No live retrieval or LLM calls |
| `evals/run_kb_evals.py` | Real KB evaluation runner | `run_real_kb_eval`, `_run_case`, `load_golden_cases` | generated corpus, temporary upload, graph runtime | JSON snapshot | document generator, rag engine, graph runtime | Yes | Uses isolated temp vectorstore/uploads |
| `evals/golden_questions.yaml` | Real KB golden cases | YAML documents | evaluation runner | case definitions | YAML parser | Yes | Source of real evaluation cases |
| `evals/metrics.py` | Evaluation metrics helpers | metrics utilities | eval payloads | aggregated metrics | stdlib | Indirect | Supporting evaluation code |
| `evals/__init__.py` | Package marker | package exports | imports | module API | local eval modules | Indirect | Enables `python -m evals.*` |
| `src/rfp_analyst/evals.py` | Evaluation snapshot loading for UI | `load_eval_snapshot`, `format_latency` | result paths | UI-ready snapshot dict | json | Yes | Used in Streamlit sidebar |
| `docs/evaluation.md` | Evaluation overview document | documentation | human reader | documentation | repository docs | No | Supplemental documentation |

## Tests

| Path | Purpose | Important coverage | Inputs | Outputs | Dependencies | Live path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tests/test_config.py` | Config resolution behavior | env/secrets precedence, default values | monkeypatched env/secrets | assertions | config module | No | Regression coverage |
| `tests/test_document_generator.py` | Sample PDF generation | generator output expectations | filesystem | assertions | document generator | No | Functional check |
| `tests/test_document_scope.py` | corpus separation and retrieval scope behavior | upload/sample separation, dedupe, scope routing | mocks and temp data | assertions | rag engine, graph | No | Key cross-corpus guardrails |
| `tests/test_agentic_tools.py` | deterministic tool behavior | search, compare, requirements, scoring, outline, verifier | fake docs/search fn | assertions | tools, graph wrapper | No | Tool-level regression coverage |
| `tests/test_agent_and_ui.py` | provider and UI helper behavior | safe errors, grouped traces, latency formatting | fake errors/payloads | assertions | agent.py, UI helpers | No | UI-facing safety coverage |
| `tests/test_langgraph_agent.py` | graph orchestration behavior | intent routing, scopes, prompt budget, repair flow | fake retrieval/LLM | assertions | graph runtime | No | Canonical agent regression suite |
| `tests/test_ingestion_pipeline.py` | ingestion behavior | chunking, dedupe, errors | temp files and mocks | assertions | ingestion/rag engine | No | Ingestion regression coverage |
| `tests/test_health_and_uploads.py` | health and upload logic | readiness and upload validation | temp files, fake uploads | assertions | health/uploads | No | Safety coverage |
| `tests/test_runtime_hardening.py` | runtime hardening | helper correctness, app health assumptions | monkeypatching | assertions | runtime modules | No | Runtime safety checks |
| `tests/test_streamlit_app_smoke.py` | Streamlit smoke test | app launch without unhandled exception | `AppTest` | assertions | app.py | No | High-level UI smoke coverage |
| `tests/test_kb_evals.py` | real KB eval runner behavior | retrieval-only and optional LLM paths | monkeypatches | assertions | eval runner | No | Evaluation reproducibility coverage |
| `tests/test_evals.py` | snapshot/eval helper behavior | smoke eval labeling and snapshot loading | temp files | assertions | eval modules | No | Evaluation utility coverage |
| `tests/test_prompt_builder_py311.py` | Python 3.11 import safety | graph import syntax safety | import machinery | assertions | graph runtime | No | Compile-time regression |
| `tests/test_imports.py` | import surface checks | package imports | import runtime | assertions | package modules | No | Packaging sanity check |
| `tests/conftest.py` | shared pytest fixtures | test configuration | pytest runtime | fixtures | pytest | No | Test support |
| `tests/__init__.py` | package marker | package importability | imports | package marker | stdlib | No | Test package support |

## Configuration and Packaging Files

| Path | Purpose | Important fields | Inputs | Outputs | Dependencies | Live path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `.env.example` | Safe configuration template | provider keys and optional overrides | developer edits | local `.env` seed | config module | Indirect | Never contains secrets |
| `.gitignore` | Git hygiene | secret/runtime ignore rules | git | ignored files | git | Indirect | Prevents accidental commits |
| `pyproject.toml` | Packaging metadata | name, version, Python requirement, setuptools config, pytest/ruff config | build tools | editable package install | setuptools | Indirect | Needed for `pip install -e .` |
| `requirements.txt` | Runtime dependencies | LangChain, LangGraph, Chroma, Streamlit, PDF, dotenv, YAML | pip | installed deps | pip | Indirect | Used by setup and CI |
| `Makefile` | local helper tasks | install/test/lint/run/generate-docs | shell | command execution | local tooling | No | Convenience only |
| `.streamlit/config.toml` | Streamlit runtime config | theme colors, `maxUploadSize` | Streamlit | UI/server settings | Streamlit | Yes | Used at app startup |

## Compatibility and Deprecated Surfaces

- `src/rfp_analyst/agent/runtime.py` contains compatibility wrappers around the canonical graph runtime.
- `src/rfp_analyst/agent/prompts.py` still provides prompt helpers for compatibility/simple mode, but the live Streamlit execution path goes through `agent.py` into `src/rfp_analyst/agent/graph.py`.
- `run_agent_graph` in the graph module exists as a backward-compatible wrapper for older tests and helper paths.

## Documentation and Automation

| Path | Purpose | Important contents | Inputs | Outputs | Dependencies | Live path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `README.md` | Main project guide | setup, configuration, workflow overview, deployment guidance | repository state | human-readable project entry point | docs, codebase | No | Source-of-truth overview for new users |
| `docs/ARCHITECTURE.md` | Deep architecture reference | component design, ingestion flow, graph nodes, Mermaid diagrams | repository state | human-readable architecture details | codebase | No | Detailed technical narrative |
| `docs/AGENTIC_RAG.md` | Agentic behavior reference | tools, state, intents, implementation mapping | graph runtime and tools | human-readable explanation | codebase | No | Focused on agentic execution model |
| `docs/REPRODUCIBILITY.md` | Reproducible setup guide | environment setup, ingestion, reset and validation commands | repository state | repeatable local workflow | codebase | No | Local reproduction and reset guide |
| `docs/TESTING_AND_EVALUATION.md` | Test and eval guide | validation commands, evaluation layers, interpretation notes | tests and eval runners | human-readable validation guide | tests, evals | No | Distinguishes smoke checks from real KB evals |
| `docs/FILE_MAP.md` | Repository map | file-by-file explanation of meaningful tracked files | repository tree | human-readable inventory | codebase | No | Keep aligned with actual tracked files |
| `docs/TROUBLESHOOTING.md` | Operational troubleshooting reference | common failures and fixes | repository behavior | human-readable fix guide | codebase | No | Documents real observed project issues |
| `.github/workflows/ci.yml` | GitHub Actions CI workflow | install, compile, pytest, offline smoke eval | pushes and pull requests | CI job status | GitHub Actions, requirements | No | Does not run live paid LLM calls |
