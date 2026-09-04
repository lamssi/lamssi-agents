"""The minimal structural interface accepted by :class:`Agent`."""

from __future__ import annotations

import threading
from typing import Iterator, List, Optional, Protocol

from lamssi_agents.providers.models import Message, StreamDelta
from lamssi_tools import ToolDefinition


class Model(Protocol):
    """A custom model adapter.

    Only streaming is required by the agent loop. Adapters may publish optional
    metadata such as ``name``, ``context_window``, and ``cumulative_usage``;
    Lamssi reads those defensively when available.
    """

    model: str

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        abort_event: threading.Event | None = None,
    ) -> Iterator[StreamDelta]: ...


__all__ = ["Model"]
