"""Run: python -m rfp_analyst.graph {init-schema,rebuild} --help."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingestion import ExtractionRejected, build_snapshot, read_indexed_corpus, rebuild_graph
from .repository import create_ingestion_repository


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Opt-in graph ingestion only; never changes vector retrieval"
    )
    parser.add_argument("operation", choices=["init-schema", "rebuild"])
    parser.add_argument("--corpus-id", default="internal-rfp")
    parser.add_argument("--persist-dir", type=Path)
    parser.add_argument("--collection")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate deterministic extraction without graph writes",
    )
    args = parser.parse_args(argv)
    try:
        with create_ingestion_repository(
            role="admin" if args.operation == "init-schema" else "writer"
        ) as repository:
            if args.operation == "init-schema":
                if args.dry_run:
                    parser.error("--dry-run is only valid for rebuild")
                result = {"status": repository.initialize_schema().status}
            else:
                if not args.dry_run:
                    health = repository.health()
                    if not health.available:
                        print(json.dumps({"status": health.status, "published": False}))
                        return 1
                from config import COLLECTION_NAME, VECTORSTORE_DIR
                from chromadb import PersistentClient
                from chromadb.config import Settings

                directory = args.persist_dir or VECTORSTORE_DIR
                if not (directory / "chroma.sqlite3").is_file():
                    raise ValueError("An existing indexed corpus is required")
                # An existing collection is mandatory. No get_or_create, embeddings,
                # reset, upsert or PDF re-ingestion. Chroma's client may manage its
                # own metadata; run against a quiescent compatible index.
                client = PersistentClient(
                    path=str(directory), settings=Settings(anonymized_telemetry=False)
                )
                collection = client.get_collection(
                    args.collection or COLLECTION_NAME, embedding_function=None
                )
                inputs = read_indexed_corpus(collection)
                if args.dry_run:
                    snapshot = build_snapshot(args.corpus_id, inputs)
                    result = {
                        "status": "validated",
                        "published": False,
                        "version": snapshot.version,
                        "documents": len(snapshot.documents),
                        "chunks": len(snapshot.chunks),
                        "entities": len(snapshot.entities),
                        "assertions": len(snapshot.assertions),
                        "rows": snapshot.row_count,
                    }
                else:
                    result = rebuild_graph(repository, args.corpus_id, inputs)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] in {"ready", "validated"} else 1
    except ExtractionRejected as error:
        print(
            json.dumps(
                {
                    "status": "quarantined",
                    "published": False,
                    "issues": [issue.__dict__ for issue in error.issues],
                }
            )
        )
        return 2
    except Exception:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "published": False,
                    "reason": "Check index availability, schema, bounds and configuration; details redacted",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
