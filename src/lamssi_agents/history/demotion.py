"""Replace old tool results with compact stubs."""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any, Dict, List, Tuple

from lamssi_agents.history.tokens import rough_tokens
from lamssi_agents.providers import Message

log = logging.getLogger(__name__)

#: How many of the most recent tool results keep their full body.
DEFAULT_KEEP_RESULTS = 6

#: Minimum body length eligible for demotion.
MIN_DEMOTABLE_CHARS = 400

#: Short result fields preserved in demoted stubs.
_KEEP_KEYS = ("path", "status", "root", "count", "artifact", "error", "retriable")
_KEEP_VALUE_CHARS = 120


def demote_tool_results(
    history: List[Message],
    *,
    keep_results: int = DEFAULT_KEEP_RESULTS,
    min_chars: int = MIN_DEMOTABLE_CHARS,
) -> Tuple[List[Message], int]:
    """Stub older tool results and return ``(history, characters saved)``."""
    if keep_results < 0:
        keep_results = 0

    positions = [i for i, m in enumerate(history) if m.role == "tool"]
    if len(positions) <= keep_results:
        return history, 0

    demotable = set(positions[: len(positions) - keep_results] if keep_results else positions)

    out: List[Message] = []
    saved = 0
    demoted = 0
    for i, message in enumerate(history):
        if i not in demotable:
            out.append(message)
            continue

        body = message.content or ""
        if len(body) <= min_chars or _already_demoted(body):
            out.append(message)
            continue

        stub = _stub_for(message.name or "", body)
        saved += len(body) - len(stub)
        demoted += 1
        out.append(_replace_content(message, stub))

    if not demoted:
        return history, 0

    log.info(
        "Demoted %d tool result(s), saving ~%d chars (~%d tokens); "
        "the %d most recent kept in full",
        demoted, saved, rough_tokens(saved), keep_results,
    )
    return out, saved


def _already_demoted(body: str) -> bool:
    """Return whether *body* is already a demotion stub."""
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, dict) and "elided" in parsed


def _stub_for(tool_name: str, body: str) -> str:
    """Build a compact result stub while retaining short identifiers."""
    kept: Dict[str, Any] = {}
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        for key in _KEEP_KEYS:
            if key not in parsed:
                continue
            value = parsed[key]
            if isinstance(value, (int, float, bool)) or value is None:
                kept[key] = value
            elif isinstance(value, str) and len(value) <= _KEEP_VALUE_CHARS:
                kept[key] = value

    kept["elided"] = f"{len(body):,} chars"
    kept["note"] = (
        f"You already ran this {tool_name or 'tool'} call and acted on the result; "
        "the body was dropped to save context. Work from what you concluded. "
        "Re-run it only if you need a detail you did not record."
    )
    return json.dumps(kept, ensure_ascii=False, separators=(",", ":"))


def _replace_content(message: Message, content: str) -> Message:
    """Copy *message* with replacement content."""
    return dataclasses.replace(message, content=content)


def largest_tool_results(
    history: List[Message], limit: int = 5
) -> List[Tuple[str, int]]:
    """``(tool name, chars)`` for the heaviest results, largest first."""
    weights = [
        (m.name or "?", len(m.content or ""))
        for m in history
        if m.role == "tool"
    ]
    weights.sort(key=lambda pair: pair[1], reverse=True)
    return weights[:limit]


__all__ = [
    "demote_tool_results",
    "largest_tool_results",
    "DEFAULT_KEEP_RESULTS",
    "MIN_DEMOTABLE_CHARS",
]
