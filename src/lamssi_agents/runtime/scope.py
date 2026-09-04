"""The agent running the current tool call, carried as a capability.

The tool subsystem binds the executing agent's capability context for the
duration of a call (gates and body). Each Agent provides ``RunScope(self)``, so
a tool resolves to the agent whose capabilities are bound while it runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from lamssi_tools.context import current_capability_context


@dataclass(frozen=True, slots=True)
class RunScope:
    """The Agent running the current tool call.

    Reach it with ``ctx.get(RunScope)`` from a context-injected tool, or with
    :func:`active_agent` from a helper that has no ``ctx`` parameter.
    """

    agent: Any


def active_agent() -> Optional[Any]:
    """The Agent running the current tool call, or ``None`` outside one."""
    ctx = current_capability_context()
    scope = ctx.get(RunScope) if ctx is not None else None
    return scope.agent if scope is not None else None


__all__ = ["RunScope", "active_agent"]
