"""Agent events and the EventBus that fans them out to listeners."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List

log = logging.getLogger(__name__)


class AgentEventType(str, Enum):
    """Everything the agent announces during a turn."""

    USER_MESSAGE = "user_message"      # emitted at the top of chat()
    TEXT_DELTA = "text_delta"
    TEXT_DONE = "text_done"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    TOOL_APPROVAL = "tool_approval"
    TOOL_REJECTED = "tool_rejected"
    USER_INPUT_REQUEST = "user_input_request"
    USER_INPUT_RESPONSE = "user_input_response"
    THINKING = "thinking"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    # Emitted before the provider call; data is the system prompt, metadata
    # carries turn/tool counts and prompt_blocks (per-block size breakdown).
    MESSAGES_SENT = "messages_sent"
    USAGE = "usage"
    ERROR = "error"
    RECOVERING = "recovering"          # a failure the loop is retrying past
    HISTORY_COMPACTING = "history_compacting"  # about to reduce history; the summary pass can take seconds
    HISTORY_COMPACTED = "history_compacted"
    ABORTED = "aborted"
    DONE = "done"


@dataclass
class AgentEvent:
    type: AgentEventType
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


AgentEventCallback = Callable[[AgentEvent], None]


class EventBus:
    """Fan-out to any number of listeners, isolating each one's failures.

    A raising listener is logged and skipped, so a broken UI subscriber cannot
    abort the turn it is reporting on.
    """

    __slots__ = ("_listeners", "_lock")

    def __init__(self) -> None:
        self._listeners: List[AgentEventCallback] = []
        self._lock = threading.RLock()

    def subscribe(self, cb: AgentEventCallback) -> Callable[[], None]:
        """Add a listener; returns a callable that removes it again."""
        with self._lock:
            self._listeners.append(cb)

        def _unsubscribe() -> None:
            self.unsubscribe(cb)

        return _unsubscribe

    def unsubscribe(self, cb: AgentEventCallback) -> None:
        with self._lock:
            try:
                self._listeners.remove(cb)
            except ValueError:
                pass

    def listeners(self) -> List[AgentEventCallback]:
        with self._lock:
            return list(self._listeners)

    def emit(self, event: AgentEvent) -> None:
        for cb in self.listeners():
            try:
                cb(event)
            except Exception as exc:
                log.debug("event listener %r raised: %s", cb, exc, exc_info=True)

    def publish(self, etype: AgentEventType, data: Any = None, **meta: Any) -> None:
        """Convenience: build the event and emit it."""
        self.emit(AgentEvent(type=etype, data=data, metadata=meta))

    def clear(self) -> None:
        with self._lock:
            self._listeners.clear()


class AgentAborted(Exception):
    """Raised when a run is cancelled mid-flight."""


__all__ = [
    "AgentEventType", "AgentEvent", "AgentEventCallback", "EventBus",
    "AgentAborted",
]
