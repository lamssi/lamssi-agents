"""RunControl: one Agent's run authority.

Named services: cancellation, approvals, interaction, events, outputs, and tool
access. Each Agent owns its RunControl and keeps its own conversation and
tool-safety state.

The convenience methods on RunControl delegate to these services.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from lamssi_agents.approval import ApprovalPolicy
from lamssi_agents.events import EventBus

if TYPE_CHECKING:
    from lamssi_agents.interaction import InteractionHandler


class CancellationToken:
    """Cooperative cancellation for one run."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self) -> None:
        """Ask the run to stop at the next cooperative checkpoint."""
        self._event.set()

    def clear(self) -> None:
        """Reset for a new top-level run."""
        self._event.clear()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def event(self) -> threading.Event:
        """The underlying event, for a caller that waits on it."""
        return self._event


class ApprovalService:
    """The tool-approval policy for a run, replaceable in place."""

    __slots__ = ("_policy",)

    def __init__(self, policy: ApprovalPolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> ApprovalPolicy:
        return self._policy

    @policy.setter
    def policy(self, value: ApprovalPolicy) -> None:
        if not isinstance(value, ApprovalPolicy):
            raise TypeError("approval must be an ApprovalPolicy")
        self._policy = value


class InteractionService:
    """The typed application interaction handler for a run."""

    __slots__ = ("handler",)

    def __init__(self, handler: "InteractionHandler | None" = None) -> None:
        self.handler = handler


class RunOutputs:
    """Feature-contributed outputs for the current run."""

    __slots__ = ("_lock", "_values")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._values: dict[str, list[Any]] = {}

    def record(self, name: str, value: Any, *, unique: bool = False) -> None:
        """Append one feature-owned output to the current run."""
        if not name:
            return
        with self._lock:
            values = self._values.setdefault(name, [])
            if not unique or value not in values:
                values.append(value)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def snapshot(self) -> dict[str, tuple[Any, ...]]:
        with self._lock:
            return {name: tuple(values) for name, values in self._values.items()}


class ToolAccess:
    """Which tools are disabled for the run."""

    __slots__ = ("_lock", "_disabled")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._disabled: set[str] = set()

    def disable(self, name: str) -> None:
        with self._lock:
            self._disabled.add(name)

    def enable(self, name: str) -> None:
        with self._lock:
            self._disabled.discard(name)

    def is_disabled(self, name: str) -> bool:
        with self._lock:
            return name in self._disabled

    @property
    def disabled(self) -> frozenset[str]:
        """A thread-safe snapshot of disabled tool names."""
        with self._lock:
            return frozenset(self._disabled)


class RunControl:
    """One Agent's run authority: cancellation, approval, interaction, events,
    outputs, and tool access.
    """

    __slots__ = (
        "cancellation",
        "approvals",
        "interaction",
        "events",
        "outputs",
        "tool_access",
        "_active_runs",
        "_lock",
    )

    def __init__(self, *, approval: ApprovalPolicy | None = None) -> None:
        self.cancellation = CancellationToken()
        self.approvals = ApprovalService(
            approval or ApprovalPolicy.reject_when_required()
        )
        self.interaction = InteractionService()
        self.events = EventBus()
        self.outputs = RunOutputs()
        self.tool_access = ToolAccess()
        self._active_runs = 0
        self._lock = threading.RLock()

    @property
    def is_running(self) -> bool:
        """Whether a run is currently active on this agent."""
        with self._lock:
            return self._active_runs > 0

    def enter_run(self) -> "RunLease":
        """Lease the run; entering resets run-scoped state, exiting closes it."""
        return RunLease(self)

    # Convenience surface delegating to the services above.

    @property
    def approval(self) -> ApprovalPolicy:
        return self.approvals.policy

    @approval.setter
    def approval(self, value: ApprovalPolicy) -> None:
        self.approvals.policy = value

    @property
    def aborted(self) -> threading.Event:
        return self.cancellation.event

    @property
    def is_aborted(self) -> bool:
        return self.cancellation.requested

    def abort(self) -> None:
        self.cancellation.request()

    def record_output(self, name: str, value: Any, *, unique: bool = False) -> None:
        self.outputs.record(name, value, unique=unique)

    def subscribe(self, cb: Any):
        return self.events.subscribe(cb)

    def emit(self, etype: Any, data: Any = None, **meta: Any) -> None:
        self.events.publish(etype, data, **meta)

    def disable_tool(self, name: str) -> None:
        self.tool_access.disable(name)

    def enable_tool(self, name: str) -> None:
        self.tool_access.enable(name)

    @property
    def disabled_tool_names(self) -> frozenset[str]:
        return self.tool_access.disabled

    def __repr__(self) -> str:
        return (
            f"<RunControl approval={self.approvals.policy.name!r} "
            f"aborted={self.is_aborted}>"
        )


class RunLease:
    """One agent's run, entered as a context manager."""

    __slots__ = ("_control",)

    def __init__(self, control: RunControl) -> None:
        self._control = control

    def __enter__(self) -> "RunLease":
        c = self._control
        with c._lock:
            c._active_runs += 1
            c.outputs.clear()
            c.cancellation.clear()
        return self

    def __exit__(self, *exc: Any) -> None:
        with self._control._lock:
            self._control._active_runs -= 1
        return None


__all__ = [
    "RunControl",
    "RunLease",
    "CancellationToken",
    "ApprovalService",
    "InteractionService",
    "RunOutputs",
    "ToolAccess",
]
