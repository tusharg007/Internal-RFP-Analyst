"""Capture-only provider fail-fast control, inactive outside the evaluation context.

Application fallbacks catch Exception. This deliberately separate BaseException
abort reaches the evaluation coordinator without changing those production policies.
Never retain callback inputs, prompts, exception messages, or credentials.
Groq 429 telemetry retains whitelisted headers and a quota-only body excerpt.
"""

from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook


class ProviderAbort(BaseException):
    def __init__(self, failure):
        self.failure = failure
        super().__init__("Evaluation stopped on application provider failure")


def provider_status_code(error):
    """Read numeric status through bounded SDK wrapper causes, never error text.

Google's LangChain adapter classifies ClientError and chains the original API
error, whose numeric code is not copied to the outer exception.
"""
    seen = set()
    for _ in range(8):
        if error is None or id(error) in seen:
            break
        seen.add(id(error))
        for value in (
            getattr(error, "status_code", None),
            getattr(getattr(error, "response", None), "status_code", None),
            getattr(error, "code", None),
        ):
            if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
                return int(value)
        error = error.__cause__ or error.__context__
    return None


class ProviderGuard(BaseCallbackHandler):
    raise_error = True

    def __init__(self):
        self.calls = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        metadata = metadata or {}
        params = kwargs.get("invocation_params", {})
        identity = (serialized or {}).get("id", [])
        classname = identity[-1] if identity else "unknown"
        provider = metadata.get("ls_provider") or {
            "ChatGroq": "groq",
            "ChatGoogleGenerativeAI": "google_genai",
        }.get(classname, classname)
        self.calls[run_id] = {
            "stage": metadata.get("langgraph_node", "unknown"),
            "provider": provider,
            "model": metadata.get("ls_model_name") or params.get("model_name")
            or params.get("model") or "unknown",
        }

    def on_llm_end(self, response, *, run_id, **kwargs):
        self.calls.pop(run_id, None)

    def on_llm_error(self, error, *, run_id, **kwargs):
        if isinstance(error, ProviderAbort):
            raise error
        status = provider_status_code(error)
        failure = {
            **self.calls.pop(run_id, {"stage": "unknown", "provider": "unknown", "model": "unknown"}),
            "error_type": type(error).__name__,
            "http_status": status,
            "category": "rate_limit" if status == 429 else "provider_failure",
            "retrieval_quality_failure": False,
        }
        if failure["provider"] == "groq" and status == 429:
            from evals.groq_rate_limits import extract_groq_limit

            failure["rate_limit"] = extract_groq_limit(error)
        raise ProviderAbort(failure)


_guard = ContextVar("rfp_evaluation_provider_guard", default=None)
register_configure_hook(_guard, inheritable=True)


@contextmanager
def stop_on_provider_failure():
    token = _guard.set(ProviderGuard())
    try:
        yield
    finally:
        _guard.reset(token)
