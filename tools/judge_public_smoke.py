"""Opt-in, one-case RAGAS probe of an immutable matched PUBLIC capture.

No retrieval, generation, corpus writes or full benchmark. This diagnostic is
not a matched 48-run comparison or a calibrated semantic baseline.
"""

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evals.matched_environment import PUBLIC_HASH  # noqa: E402
from evals.ragas_adapter import AdaptedExecution, EvaluationSample, fingerprint  # noqa: E402
from evals.ragas_reports import version_metadata, write_report  # noqa: E402
from evals.retrieval_benchmark import LOCK, QUESTIONS  # noqa: E402
from evals.run_ragas import load_cases  # noqa: E402


def select_public_capture(capture, cases, lock):
    """Refuse mismatched corpora, missing boundary capture and altered references."""
    if (capture.get("freeze") != lock
            or lock["signature"]["corpus_hash"] != PUBLIC_HASH
            or fingerprint(cases) != lock["signature"]["question_set_hash"]):
        raise ValueError("Requires the original matched frozen public capture")
    report = next(r for r in capture["reports"] if r["retrieval_mode"] == "vector_only")
    row = next(r for r in report["cases"] if r["id"] == "semantic_modernization")
    case = next(c for c in cases if c["id"] == row["id"])
    sample = EvaluationSample.model_validate(row["sample"])
    if (report["corpus_hash"] != PUBLIC_HASH or row["status"] != "completed"
            or row["generation_kind"] != "llm_kb" or not sample.retrieved_contexts
            or not row["provenance"].get("generation_prompt_hash")
            or sample.user_input != case["question"] or sample.reference != case["reference"]):
        raise ValueError("Incomplete or altered original generation capture")
    return report, row, AdaptedExecution(sample, row["generation_kind"],
                                       row["deterministic"], row["provenance"])


async def run(args):
    from evals.ragas_judge import JudgeSettings, RagasJudge

    if not args.allow_judge:
        raise ValueError("Explicit --allow-judge consent required; billing is operator-managed")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Do not overwrite the immutable capture")
    capture = json.loads(args.input.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    report, row, execution = select_public_capture(capture, load_cases(QUESTIONS, None), lock)
    settings = replace(JudgeSettings.from_environment(), max_retries=0,
                       min_call_interval_seconds=20, max_cases=1)
    result = {
        "kind": "public_posthoc_ragas_smoke", "comparison_valid": False,
        "baseline_status": "not_established", "status": "pending",
        "capture_sha256": fingerprint(capture), "capture_versions": report["versions"],
        "judge_versions": version_metadata(), "judge": settings.public_metadata(),
        "case_id": row["id"], "retrieval_mode": "vector_only",
        "corpus_hash": PUBLIC_HASH, "generation": report["generation"],
        "sample": execution.sample.model_dump(), "provenance": execution.provenance,
        "deterministic": execution.deterministic, "scores": {},
        "policy": "exact saved contexts/answer/reference; no retrieval or generation; stop on provider error",
    }
    write_report(args.output, result)
    judge = None
    start = time.perf_counter()
    try:
        if settings.provider == "google":
            from google import genai

            with genai.Client(api_key=settings.api_key) as client:
                available = {m.name.removeprefix("models/") for m in client.models.list()}
            if settings.model.removeprefix("models/") not in available:
                raise ValueError("Configured judge model is not available to this API project")
            result["model_availability"] = "confirmed_by_models_list"
        judge = await RagasJudge.create(settings)
        result["scores"] = await judge.score(execution)
        errors = [v for v in result["scores"].values() if v["status"] == "error"]
        result["status"] = "incomplete_due_to_judge_error" if errors else "completed"
    except Exception as exc:
        from evals.provider_guard import provider_status_code

        result.update({"status": "setup_or_provider_error", "error_type": type(exc).__name__,
                       "http_status": provider_status_code(exc)})
    finally:
        result["latency_seconds"] = round(time.perf_counter() - start, 6)
        write_report(args.output, result)
        if judge:
            await judge.close()
    print(json.dumps({k: result[k] for k in ("status", "case_id", "judge", "scores")}))
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-judge", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
