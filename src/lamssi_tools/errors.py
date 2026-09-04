"""The tool layer's typed exception tree and the failed-tool-result builder."""

from __future__ import annotations

from typing import Any, Dict


class LamssiError(Exception):
    """Base class for every expected failure raised by this library.

    Catch at a subsystem boundary to handle anticipated failures without
    swallowing bugs or control-flow signals.
    """


def err(
    message: str,
    *,
    hint: str | None = None,
    retriable: bool | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a failed tool-result dict.

    A tool failure is a dict carrying an ``"error"`` key; ``is_error`` is
    decided by its presence. ``retriable`` and ``hint`` are included only when
    given (so this reproduces the many hand-written variants exactly), and
    *extra* carries any other keys a specific tool reports, e.g. ``available``.
    """
    result: Dict[str, Any] = {"error": message}
    if retriable is not None:
        result["retriable"] = retriable
    if hint is not None:
        result["hint"] = hint
    result.update(extra)
    return result


__all__ = ["LamssiError", "err"]
