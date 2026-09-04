"""Classify a provider failure into a typed ModelError the loop can recover from."""

from __future__ import annotations

import enum
import logging
import re
from typing import Any, Optional

log = logging.getLogger(__name__)


class Recovery(enum.Enum):
    """What the loop should do next. Transient retries live in the provider."""

    COMPACT = "compact"
    STOP = "stop"


class ModelError(Exception):
    """A classified provider failure and the loop response it calls for.

    Raised at the model-call boundary, so the loop reads a typed recovery instead
    of classifying a raw exception. An error from our own code stays a plain
    exception and remains visible rather than being reported as a provider fault.
    """

    def __init__(
        self, recovery: Recovery, reason: str, message: str, hint: str = ""
    ) -> None:
        super().__init__(message)
        self.recovery = recovery
        #: Short slug for logs and tests: ``"context_overflow"``, ``"auth"``.
        self.reason = reason
        #: One line for the user. Already stripped of library noise.
        self.message = message
        #: A verified action the user can take, when one is available.
        self.hint = hint


# Matched against the lower-cased message, first hit wins (specific before general); duck-typed, no provider library imported.
_CONTEXT_OVERFLOW = re.compile(
    r"context[ _-]?(?:length|window|size)|too many tokens|maximum context|"
    r"reduce the length|prompt is too long|"
    r"exceeds?(?: the)?(?: available)? (?:model|context)|"
    r"exceed[ _-]?context[ _-]?size|n[ _-]?ctx",
)
_AUTH = re.compile(
    r"invalid[ _-]?api[ _-]?key|incorrect api key|unauthorized|authentication|"
    r"no api key|api key not found|permission denied|forbidden",
)
_BILLING = re.compile(
    r"insufficient[ _-](credit|quota|balance|fund)|billing|payment required|"
    r"credit balance|exceeded your current quota",
)
_MODEL_MISSING = re.compile(
    r"model[ _-]?not[ _-]?found|does not exist|unknown model|no such model|"
    r"invalid model|model_not_found",
)
_CONTENT_POLICY = re.compile(
    r"content[ _-]?(policy|filter)|safety|responsible ?ai|blocked by|flagged",
)
_TOOL_UNSUPPORTED = re.compile(
    r"tools? (is |are )?not supported|function calling|does not support tools",
)
_RATE_LIMIT = re.compile(r"rate[ _-]?limit|too many requests|quota|429")
_TRANSPORT = re.compile(
    r"timed? ?out|timeout|connection|network|unreachable|reset by peer|"
    r"temporarily unavailable|overloaded|502|503|529",
)


def _status_of(exc: Any) -> Optional[int]:
    """An HTTP status, wherever this provider decided to put it."""
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _clean(exc: Any) -> str:
    """Return a normalized, redacted provider error message."""
    from lamssi_agents.providers.errors import clean_model_error
    return clean_model_error(exc)


def classify(exc: BaseException) -> ModelError:
    """Classify a provider exception for loop recovery or termination."""
    message = _clean(exc)
    lowered = message.lower()
    status = _status_of(exc)
    haystack = f"{type(exc).__name__.lower()} {lowered}"

    # Some providers report context overflow as a generic HTTP 400.
    if _CONTEXT_OVERFLOW.search(haystack):
        return ModelError(
            Recovery.COMPACT, "context_overflow",
            "The conversation no longer fits in the model's context window.",
            "It will be compacted and retried. If this repeats, lower "
            "history_budget_tokens or max_tool_result_chars.",
        )
    if status == 413 or "payload too large" in lowered or "request too large" in lowered:
        return ModelError(
            Recovery.COMPACT, "payload_too_large",
            "The request was larger than the provider accepts.",
            "It will be compacted and retried.",
        )

    if _AUTH.search(haystack) or status in (401, 403):
        return ModelError(
            Recovery.STOP, "auth",
            f"The provider rejected the credentials: {message}",
            "Check the API key in your environment (ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY, ...) or the one passed to Agent.",
        )
    if _BILLING.search(haystack) or status == 402:
        return ModelError(
            Recovery.STOP, "billing",
            f"The account cannot pay for this request: {message}",
            "Retrying will not help - top up the account or switch model.",
        )
    if _MODEL_MISSING.search(haystack) or status == 404:
        return ModelError(
            Recovery.STOP, "model_not_found",
            f"The provider does not have that model: {message}",
            "Check the spelling, or that the model is loaded if it is a local server.",
        )
    if _CONTENT_POLICY.search(haystack):
        return ModelError(
            Recovery.STOP, "content_policy",
            f"The provider's safety filter refused this request: {message}",
            "The same prompt will be refused the same way - rephrase the request.",
        )
    if _TOOL_UNSUPPORTED.search(haystack):
        return ModelError(
            Recovery.STOP, "tools_unsupported",
            f"This model does not accept tool definitions: {message}",
            "Use a model with tool-calling support, or build the agent with no tools.",
        )

    # Transient errors only reach the loop after the provider exhausted its own retries, so they stop, not retry again.
    if _RATE_LIMIT.search(haystack) or status == 429:
        return ModelError(
            Recovery.STOP, "rate_limit",
            "The provider is rate-limiting this key, and automatic retries were exhausted.",
            "Wait and try again, or switch the model or key.",
        )
    if status in (500, 502, 503, 504, 529) or _TRANSPORT.search(haystack):
        reason = "overloaded" if status in (503, 529) else "transport"
        return ModelError(
            Recovery.STOP, reason,
            f"The provider is temporarily unavailable, and retries were exhausted: {message}",
            "Try again shortly.",
        )

    return ModelError(
        Recovery.STOP, "unknown", f"Provider error: {message}",
        "This was not recognised as recoverable, so it was not retried.",
    )


__all__ = ["Recovery", "ModelError", "classify"]
