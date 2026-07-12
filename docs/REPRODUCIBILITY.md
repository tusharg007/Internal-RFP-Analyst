# Reproducibility

## Supported Python Version

- Python `3.11`
- The repository pins package behavior around `py311` in `pyproject.toml`
- `.python-version` contains `3.11`

## Fresh Environment Setup

### Windows PowerShell

```powershell
git clone https://github.com/tusharg007/Internal-RFP-Analyst.git
cd Internal-RFP-Analyst
git checkout agentic-rag-v2
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
Copy-Item .env.example .env
```

### POSIX / macOS / Linux

```bash
git clone https://github.com/tusharg007/Internal-RFP-Analyst.git
cd Internal-RFP-Analyst
git checkout agentic-rag-v2
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
```

## Environment Configuration

Populate `.env` with one provider key if chat generation is needed:

```text
GROQ_API_KEY=
GOOGLE_API_KEY=
```

Most local validation, ingestion, and evaluation paths run without an LLM key.

## Sample Data Generation

Generate the synthetic corpus:

```powershell
python document_generator.py
```

Or use the `Generate Sample PDFs` button in the app.

## Clean Ingestion

Run explicit ingestion from the UI with `Ingest Documents`, or from Python:

```powershell
python -c "from rag_engine import ingest_documents; ingest_documents()"
```

This loads sample and uploaded PDFs, assigns metadata, chunks content, deduplicates files and chunks, and persists the Chroma index.

## Repeated-Ingestion Idempotency Check

The implementation is designed so unchanged corpora do not produce duplicate chunks. A simple reproducibility check is:

```powershell
python -c "from rag_engine import ingest_documents, get_vectorstore_stats; ingest_documents(); print(get_vectorstore_stats())"
python -c "from rag_engine import ingest_documents, get_vectorstore_stats; ingest_documents(); print(get_vectorstore_stats())"
```

Chunk counts should remain stable.

## Local Application Startup

```powershell
python -m streamlit run app.py
```

Streamlit typically serves the app at `http://localhost:8501`.

## Validation Commands

```powershell
python -m py_compile app.py agent.py rag_engine.py config.py document_generator.py
python -m compileall -f src tests evals
python -m pytest -q
python -m evals.run_evals
python -m evals.run_kb_evals
```

## Windows Troubleshooting

- If PowerShell activation is blocked, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.
- If the vectorstore cannot be replaced because files are locked, stop the running app and retry ingestion.
- If ONNX or embedding setup runs out of memory, close other heavy applications and rerun the ingestion or evaluation command in a fresh shell.

## Clean Reset Commands

Use these only when you want to clear local runtime artifacts, not source files:

```powershell
Remove-Item -Recurse -Force .pytest_cache, .pytest_tmp, vectorstore -ErrorAction SilentlyContinue
Remove-Item -Force evals\offline_smoke_results.json, evals\real_kb_results.json -ErrorAction SilentlyContinue
```

These commands are for local reproduction resets only and are intentionally not run by CI.

## Deployment Reproduction

To reproduce a Streamlit Community Cloud deployment locally:

1. install dependencies from `requirements.txt`
2. install the project editable with `python -m pip install -e .`
3. set provider keys via environment variables or Streamlit secrets
4. generate or upload PDFs
5. ingest documents
6. run `python -m streamlit run app.py`
