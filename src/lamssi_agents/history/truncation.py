"""Clips a single oversize tool result once, as it lands in history: a safety cap on pathological output."""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol

from lamssi_agents.redaction import redact

#: Used when neither the caller nor the tool names one. Deliberately small: about
#: 3% of the default history budget, so many results fit before compaction is needed.
DEFAULT_MAX_TOOL_RESULT_CHARS = 8_000

#: Tools returning a ``{"content": ...}`` envelope, cut inside ``content`` so the
#: JSON survives and the metadata keys stay readable.
_ENVELOPE_TOOLS = frozenset({"read_file"})


class Truncator(Protocol):
    """Clip one oversize tool result as it lands in history.

    The text is already redacted. Return it at or under the resolved cap, marking
    what was removed, or unchanged when it already fits.
    """

    def __call__(
        self,
        result_str: str,
        *,
        tool_name: str = "",
        definition: Any = None,
        max_chars: Optional[int] = None,
        default_max_chars: Optional[int] = None,
    ) -> str: ...


def clip_result(
    result_str: str,
    *,
    tool_name: str = "",
    definition: Any = None,
    max_chars: Optional[int] = None,
    default_max_chars: Optional[int] = None,
) -> str:
    """The default truncator: clip already-redacted text to the resolved cap.

    Cap precedence: ``max_chars``, then the tool's declared ``truncation``, then
    ``default_max_chars``. A content-envelope tool keeps a true prefix; otherwise
    the tool's ``truncation_side`` ("middle", "head", or "tail") decides which end
    survives.
    """
    cap = _resolve_cap(definition, max_chars, default_max_chars)
    if len(result_str) <= cap:
        return result_str

    if tool_name in _ENVELOPE_TOOLS:
        out = _truncate_envelope(result_str, cap)
        if out is not None:
            return out
        # Unexpected shape: fall through to the side-based cut.

    removed = len(result_str) - cap
    hint = getattr(definition, "truncation_hint", "") if definition is not None else ""
    suffix = f": {hint}" if hint else ""
    side = getattr(definition, "truncation_side", "middle")

    if side == "tail":
        marker = f"... [truncated to the last {cap:,} chars, {removed:,} removed{suffix}] ..."
        return f"{marker}\n\n{result_str[-cap:]}"
    if side == "head":
        marker = f"... [truncated to the first {cap:,} chars, {removed:,} removed{suffix}] ..."
        return f"{result_str[:cap]}\n\n{marker}"

    half = cap // 2
    marker = f"... [truncated: {removed:,} chars removed{suffix}] ..."
    return f"{result_str[:half]}\n\n{marker}\n\n{result_str[-half:]}"


def truncate_tool_result(
    result_str: str,
    *,
    tool_name: str = "",
    definition: Any = None,
    max_chars: Optional[int] = None,
    default_max_chars: Optional[int] = None,
) -> str:
    """Redact then clip a tool result, for callers that shape a result on their own.

    The dispatch path redacts at its boundary and then calls the agent's truncator,
    so redaction runs whichever truncator is installed; this wrapper keeps the same
    guarantee for direct callers.
    """
    return clip_result(
        redact(result_str),
        tool_name=tool_name,
        definition=definition,
        max_chars=max_chars,
        default_max_chars=default_max_chars,
    )


def _resolve_cap(
    definition: Any,
    max_chars: Optional[int],
    default_max_chars: Optional[int] = None,
) -> int:
    """The cap for one result: explicit arg, then declared, then runtime default, then module default."""
    if max_chars is not None and max_chars > 0:
        return int(max_chars)
    declared = getattr(definition, "truncation", None) if definition is not None else None
    if isinstance(declared, int) and declared > 0:
        return declared
    if default_max_chars is not None and default_max_chars > 0:
        return int(default_max_chars)
    return DEFAULT_MAX_TOOL_RESULT_CHARS


def _write_head(
    envelope: dict, content: str, budget: int, *, set_hint: bool,
) -> tuple[str, str]:
    """Keep a line-bounded prefix and update envelope truncation metadata."""
    head = _clip_at_line(content, budget)

    head_lines = head.count("\n") + (0 if head.endswith("\n") else 1)
    dropped = len(content) - len(head)
    dropped_lines = content[len(head):].count("\n")

    kept = head
    if dropped > 0:
        kept = (
            f"{head}\n\n... [{dropped:,} chars / {dropped_lines:,} more lines follow - "
            f"this is the start of the file, not all of it] ..."
        )

    envelope["content"] = kept
    envelope["truncated_after_local_line"] = head_lines
    envelope["truncated_chars_remaining"] = dropped
    if set_hint:
        envelope["truncation_hint"] = (
            f"You have lines 1-{head_lines} only; {dropped_lines:,} lines follow that "
            f"you have not seen. To continue, re-call read_file with "
            f"start_line={head_lines + 1}."
        )

    out = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return out, kept


def _clip_at_line(text: str, budget: int) -> str:
    """At most *budget* chars from the start of *text*, cut on a line boundary."""
    if budget <= 0 or not text:
        return ""
    if len(text) <= budget:
        return text
    piece = text[:budget]
    nl = piece.rfind("\n")
    return piece[:nl] if nl > 0 else piece


def _truncate_envelope(result_str: str, cap: int) -> Optional[str]:
    """Truncate a JSON content envelope while preserving valid structure."""
    try:
        envelope = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(envelope, dict):
        return None
    content = envelope.get("content")
    if not isinstance(content, str):
        return None

    # Allow for envelope metadata and JSON escaping.
    OVERHEAD = 1_000
    EXPANSION = 1.10
    budget = max(int((cap - OVERHEAD) / EXPANSION), 1)
    if len(content) <= budget:
        return None  # nothing to do; let the caller pass it through

    out, head = _write_head(envelope, content, budget, set_hint=True)

    # Retry with a smaller prefix if serialization exceeded the cap.
    if len(out) > cap:
        excess_factor = len(out) / cap
        tighter_budget = int(len(head) / (excess_factor * 1.05))
        out, head = _write_head(envelope, content, tighter_budget, set_hint=False)
    return out
