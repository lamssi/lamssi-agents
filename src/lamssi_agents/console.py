"""Optional stdin/stdout helpers for terminal hosts."""

from __future__ import annotations

from typing import Any, Callable


def printer() -> Callable[[Any], None]:
    """A listener that narrates the run: streamed text, tool calls, results."""
    from lamssi_agents.events import AgentEventType

    state = {"streaming": False}

    def on_event(event: Any) -> None:
        kind = event.type
        if kind is AgentEventType.TEXT_DELTA:
            if not state["streaming"]:
                print("  ", end="", flush=True)
                state["streaming"] = True
            print(event.data, end="", flush=True)
        elif kind is AgentEventType.TEXT_DONE:
            if state["streaming"]:
                print()
                state["streaming"] = False
        elif kind is AgentEventType.TOOL_START:
            args = event.metadata.get("arguments") or {}
            shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(f"  -> {event.data}({shown[:100]})")
        elif kind is AgentEventType.TOOL_RESULT:
            print(f"  <- {str(event.data)[:140]}")

    return on_event


__all__ = ["printer"]
