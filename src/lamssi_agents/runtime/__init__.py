"""Config and the run-scope capability for tool execution."""

from __future__ import annotations

from lamssi_agents.runtime.config import AgentConfig
from lamssi_agents.runtime.scope import RunScope, active_agent

__all__ = [
    "AgentConfig",
    "RunScope",
    "active_agent",
]
