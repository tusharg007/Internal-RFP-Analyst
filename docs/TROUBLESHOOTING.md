# Troubleshooting

## Wrong Python Interpreter

Symptoms:

- import errors for installed packages
- `python` points to a different environment
- tests pass in one shell and fail in another

Fix:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

Expected interpreter: `F:\Internal-RFP-Analyst\.venv\Scripts\python.exe`

## Broken Virtual Environment

Symptoms:

- missing packages after activation
- corrupted environment after upgrades

Fix:

```powershell
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Missing Dependencies

Symptoms:

- `ModuleNotFoundError`
- app starts but fails importing LangChain/Chroma/Streamlit modules

Fix:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Invalid API Key

Symptoms:

- provider returns `401`
- provider reports `invalid_api_key`

Behavior:

- the UI should show a friendly authentication message instead of raw provider JSON unless debug mode is enabled

Fix:

- replace the key in `.env` or Streamlit Secrets
- restart or rerun the app

## No Provider Configured

Symptoms:

- sidebar shows `Not configured`
- chat input is disabled or returns the missing-provider warning

Fix:

- add `GROQ_API_KEY` or `GOOGLE_API_KEY`
- keep using upload and ingestion features even without a provider

## Blank Embedded VS Code Browser

Symptoms:

- the embedded browser panel fails to render Streamlit correctly

Fix:

- open the app in a normal browser at the Streamlit URL
- refresh the Streamlit process if the embedded panel held a stale session

## Knowledge Base Not Ready

Symptoms:

- health panel shows not ready
- chat warns that the knowledge base is not ready

Fix:

1. generate sample PDFs or upload PDFs
2. click `Ingest Documents`
3. wait for chunk count to become nonzero

## Duplicate Chroma IDs

Symptoms:

- ingestion fails with duplicate ID errors

Fix:

- ensure uploads are stored only in `data/uploads`
- ensure sample documents stay in `data/documents`
- rerun ingestion after the duplicate file/chunk fix already present in the repository

The current implementation deduplicates files by hash and chunks by `chunk_id` before Chroma upsert.

## Windows Vectorstore File Locks

Symptoms:

- access denied while replacing `vectorstore`
- ingestion reports the vectorstore is locked

Fix:

- stop the running Streamlit app
- rerun ingestion
- if needed, delete the local `vectorstore/` directory manually and ingest again

The app clears cached vectorstore-related objects before ingestion, but open file handles from another process can still block replacement.

## ONNX Memory Allocation Errors

Symptoms:

- embedding or evaluation commands fail with ONNX or allocation errors

Fix:

- close other memory-intensive programs
- rerun in a fresh shell
- reduce parallel local load while ingesting large PDFs

## Request-Too-Large / 413 Token Failures

Symptoms:

- provider reports request too large, context length, or token-limit style failures

Behavior:

- the app compacts the prompt automatically
- if still too large, it shows a friendly token-budget message

Fix:

- reduce document scope
- shorten the request
- use a provider/model with a larger context window if available

## Uploaded Documents Pending Indexing

Symptoms:

- upload succeeded but chat warns that uploads are pending indexing

Fix:

- click `Ingest Documents`
- wait for ingestion to finish successfully

This prevents stale sample-only answers from being treated as if they describe uploaded files.

## Relevance Threshold Producing No Matches

Symptoms:

- scope has indexed documents but the app returns insufficient evidence

Fix:

- ask a more specific question
- switch scope
- confirm the relevant document was actually indexed

For `rfp_analysis`, the runtime also has a bounded near-threshold target fallback when uploads exist but evidence is only slightly below threshold.

## Streamlit Rerun Behavior

Symptoms:

- a button click appears to restart the script
- chat or ingestion state seems to reset unexpectedly

Explanation:

- Streamlit reruns the script on interaction by design

Fix:

- rely on `st.session_state` values already used by the app
- avoid assuming a local variable survives across clicks

The app already uses session-state guards for ingestion, pending uploads, and chat history.
