# SPDX-License-Identifier: MIT
"""Shared helpers for examples that do not need a live model.

``ScriptedModel`` keeps the examples quick and repeatable. Examples that call a
live model say so at the top and read ``LAMSSI_MODEL``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from lamssi_agents.providers.models import StreamDelta, ToolCall, Usage


def says(text: str) -> List[StreamDelta]:
    """One turn: the model replies with text and stops."""
    return [
        StreamDelta(type="text", text=text),
        StreamDelta(type="done", finish_reason="stop"),
    ]


def calls(name: str, **arguments: Any) -> List[StreamDelta]:
    """One turn: the model calls a tool and waits for the result."""
    return [
        StreamDelta(
            type="tool_call",
            tool_call=ToolCall(id=f"call-{name}", name=name, arguments=arguments),
        ),
        StreamDelta(type="done", finish_reason="tool_calls"),
    ]


class ScriptedModel:
    """Replays the turns it was given, then says "done".

    Satisfies the same ``Model`` protocol as the real thing, which is the
    point worth noticing: the framework never learns whether a model is real.
    The custom seam is small, so a hosted application model is not a big adapter.
    """

    model = name = "scripted"
    is_local = supports_tools = True
    reasoning_effort = None

    def __init__(self, *turns: List[StreamDelta]) -> None:
        self._turns = list(turns)
        self._usage = Usage()
        self.seen: List[Dict[str, Any]] = []

    def stream(self, messages, tools=None, **kw):
        self.seen.append({"messages": len(messages), "tools": len(tools or [])})
        yield from self._turns.pop(0) if self._turns else says("done")

    @property
    def cumulative_usage(self) -> Usage:
        return self._usage


def real_model() -> str:
    """The model an example should use when it needs a live one.

    Set ``LAMSSI_MODEL`` to anything LiteLLM accepts. The default points at a
    local OpenAI-compatible server (LM Studio's port), so the examples cost
    nothing to try.
    """
    return os.environ.get("LAMSSI_MODEL", "local/qwen3")


def heading(title: str) -> None:
    """Plain ASCII on purpose.

    A Windows console defaults to cp1252, and a box-drawing character raises
    UnicodeEncodeError there: which would make every example fail on the first
    line, on the platform most likely to be someone's first try.
    """
    print(f"\n{'-' * 68}\n{title}\n{'-' * 68}")
