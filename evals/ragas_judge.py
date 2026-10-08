"""Optional modern RAGAS metrics, isolated from app imports and fast pytest."""

from __future__ import annotations

import asyncio
import math
import os
import time
import logging
from threading import RLock
from dataclasses import dataclass, field

from evals.ragas_adapter import METRICS, AdaptedExecution, applicability, metric_score_range

_LOG_LOCK = RLock()
_LOG_USERS = 0
_LOG_LEVELS = {}


def _acquire_redacted_logging():
    global _LOG_USERS
    with _LOG_LOCK:
        if not _LOG_USERS:
            for name in ("instructor", "instructor.v2.retry", "ragas"):
                logger = logging.getLogger(name)
                _LOG_LEVELS[name] = logger.level
                logger.setLevel(logging.CRITICAL)
        _LOG_USERS += 1


def _release_redacted_logging():
    global _LOG_USERS
    with _LOG_LOCK:
        _LOG_USERS -= 1
        if not _LOG_USERS:
            for name, level in _LOG_LEVELS.items():
                logging.getLogger(name).setLevel(level)
            _LOG_LEVELS.clear()


@dataclass(frozen=True)
class JudgeSettings:
    provider: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 90.0
    max_retries: int = 1
    max_cases: int = 30
    embedding_model: str = ""
    max_tokens: int = 4096
    min_call_interval_seconds: float = 0.0

    def __post_init__(self):
        if self.provider not in {"groq", "google"} or not self.model or not self.api_key:
            raise ValueError("Configure RAGAS judge provider, model and credentials explicitly")
        if (
            not 1 <= self.timeout_seconds <= 300
            or not 0 <= self.max_retries <= 3
            or not 1 <= self.max_cases <= 1000
        ):
            raise ValueError("Invalid RAGAS timeout/retry/case budget")
        if not self.embedding_model:
            raise ValueError("Configure an embedding model")
        if not 256 <= self.max_tokens <= 32768 or not 0 <= self.min_call_interval_seconds <= 300:
            raise ValueError("Invalid judge token/pacing budget")

    @classmethod
    def from_environment(cls):
        from config import get_ragas_settings

        return cls(**get_ragas_settings())

    def public_metadata(self):
        return {
            key: getattr(self, key)
            for key in (
                "provider",
                "model",
                "timeout_seconds",
                "max_retries",
                "max_cases",
                "embedding_model",
                "max_tokens",
                "min_call_interval_seconds",
            )
        } | {
            "temperature": 0,
            "answer_relevancy_strictness": 3,
            "answer_correctness_weights": [0.75, 0.25],
            "telemetry": "disabled",
        }


def metric_arguments(metric: str, sample) -> tuple[str, dict]:
    data = sample.model_dump(exclude_none=True)
    fields = {
        "faithfulness": ("faithfulness", ("user_input", "response", "retrieved_contexts")),
        "answer_relevancy": ("answer_relevancy", ("user_input", "response")),
        "context_recall": ("context_recall", ("user_input", "retrieved_contexts", "reference")),
        "answer_correctness": ("answer_correctness", ("user_input", "response", "reference")),
    }
    if metric == "context_precision":
        variant = (
            "context_precision_with_reference"
            if sample.reference
            else "context_precision_without_reference"
        )
        keys = ("user_input", "retrieved_contexts", "reference" if sample.reference else "response")
    else:
        variant, keys = fields[metric]
    return variant, {key: data[key] for key in keys}


async def score_execution(
    execution: AdaptedExecution, registry: dict, *, timeout: float = 90
) -> dict:
    """Injectable metric registry: tests use fake scorers, never external providers."""
    scores = {}
    provider_failure = None
    for name in METRICS:
        reason = applicability(execution.generation_kind, execution.sample, name)
        if reason:
            scores[name] = {
                "status": "not_applicable",
                "score": None,
                "reason": reason,
                "latency_seconds": 0,
            }
            continue
        if provider_failure:
            scores[name] = {
                "status": "not_run", "score": None, "reason": "provider_error_stop",
                "http_status": provider_failure, "latency_seconds": 0,
            }
            continue
        variant, arguments = metric_arguments(name, execution.sample)
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(registry[variant].ascore(**arguments), timeout=timeout)
            value = float(result.value)
            lower, upper = metric_score_range(name)
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError("Nonfinite/out-of-range judge score")
            scores[name] = {"status": "scored", "score": value, "variant": variant}
        except Exception as exc:
            from evals.provider_guard import provider_status_code

            status = provider_status_code(exc)
            # SDK exceptions may contain prompts, URLs or credentials; persist types only.
            scores[name] = {
                "status": "error",
                "score": None,
                "variant": variant,
                "error_type": type(exc).__name__,
            }
            if status is not None and status >= 400:
                scores[name]["http_status"] = status
                provider_failure = status
        scores[name]["latency_seconds"] = round(time.perf_counter() - start, 6)
    return scores


class RagasJudge:
    def __init__(self, settings: JudgeSettings):
        self.settings = settings
        self.client = None
        self._logging_acquired = False
        self.registry = {}
        self.provider_failure = None

    @classmethod
    async def create(cls, settings: JudgeSettings):
        instance = cls(settings)
        try:
            instance._initialize()
        except BaseException:
            await instance.close()
            raise
        return instance

    def _initialize(self):
        settings = self.settings
        # No tracking of questions/contexts through third-party telemetry.
        os.environ["RAGAS_DO_NOT_TRACK"] = "true"
        from ragas.embeddings.base import BaseRagasEmbedding
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextRecall,
            Faithfulness,
            ContextPrecisionWithReference,
            ContextPrecisionWithoutReference,
        )
        from fastembed import TextEmbedding

        # Instructor otherwise logs complete provider errors, potentially including
        # failed structured outputs/private content. Restore logger levels on close.
        _acquire_redacted_logging()
        self._logging_acquired = True
        self.client, llm = build_judge_llm(
            settings, client_created=lambda client: setattr(self, "client", client)
        )

        class LocalEmbeddings(BaseRagasEmbedding):
            def __init__(self):
                super().__init__()
                self.model = TextEmbedding(model_name=settings.embedding_model)

            def embed_text(self, text, **kwargs):
                return next(iter(self.model.embed([text]))).tolist()

            async def aembed_text(self, text, **kwargs):
                return await asyncio.to_thread(self.embed_text, text, **kwargs)

        embeddings = LocalEmbeddings()
        self.registry = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_precision_with_reference": ContextPrecisionWithReference(llm=llm),
            "context_precision_without_reference": ContextPrecisionWithoutReference(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
        }

    async def score(self, execution):
        if self.provider_failure:
            return {
                name: {"status": "not_run", "score": None, "reason": "provider_error_stop",
                       "http_status": self.provider_failure, "latency_seconds": 0}
                for name in METRICS
            }
        scores = await score_execution(
            execution, self.registry, timeout=self.settings.timeout_seconds
        )
        self.provider_failure = next(
            (value["http_status"] for value in scores.values()
             if value.get("status") == "error" and value.get("http_status")), None
        )
        return scores

    async def close(self):
        try:
            if self.client is None:
                return
            if self.settings.provider == "groq":
                await self.client.close()
            else:
                await self.client.aio.aclose()
                self.client.close()
        finally:
            if self._logging_acquired:
                _release_redacted_logging()
                self._logging_acquired = False
            self.client = None


def build_judge_llm(settings: JudgeSettings, *, client_created=None):
    """Use public provider-specific adapters (RAGAS 0.4.3 generic Groq adapter is incompatible)."""
    import instructor
    from ragas.llms import InstructorLLM

    if settings.provider == "groq":
        from groq import AsyncGroq

        client = AsyncGroq(
            api_key=settings.api_key, timeout=settings.timeout_seconds, max_retries=0
        )
        if client_created:
            client_created(client)
        patched = instructor.from_groq(client, mode=instructor.Mode.JSON)
    else:
        from google import genai

        client = genai.Client(
            api_key=settings.api_key, http_options={"timeout": int(settings.timeout_seconds * 1000)}
        )
        if client_created:
            client_created(client)
        patched = instructor.from_genai(client, use_async=True)

    class PacedInstructorLLM(InstructorLLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.last_call = time.monotonic()
            self.call_lock = asyncio.Lock()

        async def agenerate(self, prompt, response_model):
            async with self.call_lock:
                if self.last_call is not None:
                    remaining = settings.min_call_interval_seconds - (
                        time.monotonic() - self.last_call
                    )
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                self.last_call = time.monotonic()
                return await super().agenerate(prompt, response_model)

    # Instructor's integer is total attempts, whereas configuration counts retries.
    return client, PacedInstructorLLM(
        client=patched,
        model=settings.model,
        provider=settings.provider,
        temperature=0,
        max_tokens=settings.max_tokens,
        max_retries=settings.max_retries + 1,
    )
