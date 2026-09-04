"""Human-facing cleanup for model transport failures."""

from __future__ import annotations

from lamssi_agents.redaction import redact


def clean_model_error(error: object) -> str:
    """Render and redact a transport error without leaking LiteLLM internals."""
    try:
        text = str(error)
    except Exception:
        return type(error).__name__
    return redact(text.replace("litellm.", "").strip())


__all__ = ["clean_model_error"]
