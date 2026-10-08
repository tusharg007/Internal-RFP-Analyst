"""Run the real app against the verified frozen PUBLIC corpus, without writes.

Usage: set RFP_PUBLIC_DEMO_INDEX to the existing frozen index, then
streamlit run tools/public_demo_app.py --server.port 8512
No LLM, graph, retrieval, grading or grounding component is mocked.
"""

import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evals.matched_environment import CORPUS_ID, PUBLIC_HASH  # noqa: E402
from rfp_analyst.retrieval.hybrid import load_indexed_snapshot  # noqa: E402


def validate_public_index(index: Path):
    """Fail before provider calls if this is not the exact approved public corpus."""
    snapshot = load_indexed_snapshot(index)
    if snapshot.fingerprint != PUBLIC_HASH:
        raise ValueError("Public demo requires the exact frozen public index; private/changed corpora refused")
    return snapshot


def main():
    import config

    raw_index = os.environ.get("RFP_PUBLIC_DEMO_INDEX", "")
    if not raw_index:
        raise ValueError("Set RFP_PUBLIC_DEMO_INDEX to the verified frozen public Chroma index")
    index = Path(raw_index).resolve()
    validate_public_index(index)
    # Bind existing dependency defaults BEFORE importing rag_engine/app. These
    # process-local directories are empty: root private PDFs are never discovered.
    workspace = ROOT / ".capture_runtime" / "public_demo"
    config.DATA_DIR = config.SAMPLE_DOCS_DIR = workspace / "documents"
    config.UPLOADS_DIR = workspace / "uploads"
    config.VECTORSTORE_DIR = index
    config.PUBLIC_DEMO_READ_ONLY = True
    os.environ["RFP_GRAPH_CORPUS_ID"] = CORPUS_ID
    runpy.run_path(str(ROOT / "app.py"), run_name="__main__")


if __name__ == "__main__":
    main()
