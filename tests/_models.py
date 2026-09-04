"""Test helpers: a scripted provider and stream-delta builders."""

from __future__ import annotations

from lamssi_agents.providers.models import StreamDelta, ToolCall, Usage


def calls(*specs, finish: str = "tool_calls") -> list:
    """Build one provider turn announcing the given ``(id, name, arguments)`` calls."""
    deltas = [
        StreamDelta(type="tool_call", tool_call=ToolCall(id=i, name=n, arguments=a))
        for i, n, a in specs
    ]
    deltas.append(StreamDelta(type="done", finish_reason=finish))
    return deltas


def says(text: str) -> list:
    """Build one provider turn that streams *text* and stops."""
    return [
        StreamDelta(type="text", text=text),
        StreamDelta(type="done", finish_reason="stop"),
    ]


class ScriptedModel:
    """A provider that yields one prepared turn of deltas per call."""

    is_local = True
    supports_tools = True
    reasoning_effort = None

    def __init__(self, *turns, name: str = "scripted") -> None:
        self.model = self.name = name
        self._turns = [list(turn) for turn in turns]
        self.calls = 0
        self._usage = Usage()

    def stream(self, messages, tools=None, *, abort_event=None, **kwargs):
        """Yield the next scripted turn, or a filler answer once exhausted."""
        self.calls += 1
        if not self._turns:
            yield from says("script exhausted")
            return
        yield from self._turns.pop(0)

    def check_connectivity(self):
        """Report a healthy scripted connection."""
        return True, "scripted"

    @property
    def cumulative_usage(self) -> Usage:
        """Return the usage this scripted model accrued."""
        return self._usage
