"""Extract LiteLLM token-usage data into :class:`Usage` objects."""

from __future__ import annotations

from typing import Any

from lamssi_agents.providers.models import Usage


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read *key* from a mapping or attribute-bearing object."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _int_field(obj: Any, key: str, default: int = 0) -> int:
    """Like :func:`_field`, coerced to a non-negative-safe int; never raises."""
    try:
        return int(_field(obj, key, default) or 0)
    except (TypeError, ValueError):
        return default


def extract_usage(raw: Any) -> Usage:
    """Convert a LiteLLM usage object (or dict) into a :class:`Usage`."""
    if raw is None:
        return Usage()

    prompt = _int_field(raw, "prompt_tokens")
    completion = _int_field(raw, "completion_tokens")
    total = _int_field(raw, "total_tokens") or (prompt + completion)

    # Optional nested details (OpenAI/Anthropic extras).
    cached = 0
    reasoning = 0
    details = _field(raw, "prompt_tokens_details")
    if details is not None:
        cached = _int_field(details, "cached_tokens")

    # Anthropic reports cache stats at usage top-level: cache_read_input_tokens (fallback) / cache_creation_input_tokens (write).
    cache_write = _int_field(raw, "cache_creation_input_tokens")
    if not cached:
        cached = _int_field(raw, "cache_read_input_tokens")
    comp_details = _field(raw, "completion_tokens_details")
    if comp_details is not None:
        reasoning = _int_field(comp_details, "reasoning_tokens")

    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
    )
