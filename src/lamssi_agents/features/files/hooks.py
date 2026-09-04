"""Write notifications owned by the Files feature."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lamssi_agents.agent.control import RunControl

log = logging.getLogger(__name__)

WriteKind = Literal["write", "edit", "delete"]


@dataclass(frozen=True, slots=True)
class WriteEvent:
    """A file change emitted by the :class:`Files` feature."""

    rel_path: str
    content: str
    kind: WriteKind
    control: RunControl
    action: str = ""


@runtime_checkable
class WriteHook(Protocol):
    """React after a matching file change has completed."""

    name: str

    def matches(self, rel_path: str) -> bool:
        """Return whether this hook handles the project-relative path."""
        ...

    def after_write(self, event: WriteEvent) -> Mapping[str, Any] | None:
        """Return fields to merge into the tool result, or ``None``."""
        ...


def as_write_hook(hook: Any) -> WriteHook:
    """Adapt a WriteHook, ``(pattern, fn)`` pair, or bare callable."""
    pattern = None
    if isinstance(hook, tuple):
        pattern, hook = hook
    if hasattr(hook, "after_write") and hasattr(hook, "matches"):
        return hook

    if pattern is None:
        matches = lambda _rel: True  # noqa: E731
    elif callable(pattern):
        matches = pattern
    else:
        matches = lambda rel, value=str(pattern): fnmatch(rel, value)  # noqa: E731

    class CallableHook:
        name = getattr(hook, "__name__", "on_write")

        def matches(self, rel_path: str) -> bool:
            return bool(matches(rel_path))

        def after_write(self, event: WriteEvent) -> Any:
            return hook(event)

    return CallableHook()


class HookChain:
    """Write hooks for one Files runtime, applied in registration order."""

    __slots__ = ("_hooks", "_lock")

    def __init__(self) -> None:
        self._hooks: list[WriteHook] = []
        self._lock = threading.RLock()

    def add(self, hook: WriteHook) -> Callable[[], None]:
        """Register *hook* and return a function that removes it."""
        with self._lock:
            if hook not in self._hooks:
                self._hooks.append(hook)

        def remove() -> None:
            with self._lock:
                try:
                    self._hooks.remove(hook)
                except ValueError:
                    pass

        return remove

    def fire(self, event: WriteEvent) -> dict[str, Any]:
        """Run matching hooks and merge their result fields."""
        with self._lock:
            hooks = list(self._hooks)
        extra: dict[str, Any] = {}
        for hook in hooks:
            try:
                if hook.matches(event.rel_path):
                    result = hook.after_write(event)
                    if result:
                        extra.update(result)
            except Exception as exc:
                log.warning(
                    "write hook %r raised for %s: %s",
                    getattr(hook, "name", hook),
                    event.rel_path,
                    exc,
                )
        return extra

    def __len__(self) -> int:
        with self._lock:
            return len(self._hooks)


def make_event(
    rel_path: str,
    content: str,
    kind: WriteKind,
    control: RunControl | None,
    action: str = "",
) -> WriteEvent | None:
    """Build a write event when the tool is running inside an Agent run."""
    if control is None:
        return None
    return WriteEvent(rel_path, content, kind, control, action)


__all__ = [
    "HookChain",
    "WriteEvent",
    "WriteHook",
    "WriteKind",
    "as_write_hook",
    "make_event",
]
