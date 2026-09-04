"""Typed host capabilities available to tool implementations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import inspect
from typing import Any, Dict, Iterator, List, Mapping, Type, TypeVar

from lamssi_tools.errors import LamssiError

T = TypeVar("T")

_ACTIVE_CAPABILITIES: ContextVar["CapabilityContext | None"] = ContextVar(
    "lamssi_active_capabilities",
    default=None,
)


class CapabilityMissing(LamssiError):
    """A tool required a capability the host did not register: a configuration error, not a tool bug."""

    def __init__(self, capability: type, available: List[type] | None = None) -> None:
        self.capability = capability
        self.available = list(available or [])
        name = getattr(capability, "__name__", repr(capability))
        have = ", ".join(sorted(getattr(c, "__name__", repr(c)) for c in self.available))
        super().__init__(
            f"no {name} capability is registered"
            + (f" (host provides: {have})" if have else " (host provides nothing)")
        )


class CapabilityContext:
    """An immutable-by-convention bag of host capabilities, keyed by protocol.

    A tool body receives one as its first argument when decorated with
    ``@tool(inject_context=True)``. Multiple implementations may be registered
    under one protocol: see :meth:`get_all` for fan-in capabilities.
    """

    __slots__ = ("_caps",)

    def __init__(self, capabilities: Mapping[type, Any] | None = None) -> None:
        self._caps: Dict[type, List[Any]] = {}
        for proto, impl in (capabilities or {}).items():
            self.register(proto, impl)

    def register(self, capability: Type[T], impl: T) -> None:
        """Register an implementation, adapting a function to a one-method protocol."""
        self._caps.setdefault(capability, []).append(
            _adapt_callable(capability, impl)
        )

    def get(self, capability: Type[T]) -> T | None:
        """The registered implementation of *capability*, or ``None``.

        Last registration wins, so a host can override a pack's default.
        """
        impls = self._caps.get(capability)
        return impls[-1] if impls else None

    def get_all(self, capability: Type[T]) -> List[T]:
        """Every implementation of *capability*, in registration order."""
        return list(self._caps.get(capability, ()))

    def require(self, capability: Type[T]) -> T:
        """Like :meth:`get`, but raises :class:`CapabilityMissing` when absent."""
        impl = self.get(capability)
        if impl is None:
            raise CapabilityMissing(capability, list(self._caps))
        return impl

    def has(self, capability: type) -> bool:
        return self.get(capability) is not None

    def __repr__(self) -> str:
        names = sorted(getattr(c, "__name__", repr(c)) for c in self._caps)
        return f"CapabilityContext({', '.join(names) or 'empty'})"


def current_capability_context() -> CapabilityContext | None:
    """Capability context active for the current tool invocation, if any."""
    return _ACTIVE_CAPABILITIES.get()


@contextmanager
def capability_context_active(
    context: CapabilityContext,
) -> Iterator[CapabilityContext]:
    """Temporarily make *context* available to context-injected tools."""
    token = _ACTIVE_CAPABILITIES.set(context)
    try:
        yield context
    finally:
        _ACTIVE_CAPABILITIES.reset(token)


def _adapt_callable(capability: type, value: Any) -> Any:
    """Adapt a bare callable to a capability declaring exactly one method."""
    method = _single_method(capability)
    if method is None or not callable(value) or hasattr(value, method):
        return value
    shim = type(
        f"{capability.__name__}Fn",
        (),
        {
            method: staticmethod(value),
            "__doc__": f"{getattr(value, '__name__', value)!r} as {capability.__name__}.",
            "__repr__": lambda self: (
                f"<{capability.__name__} via "
                f"{getattr(value, '__qualname__', value)!r}>"
            ),
        },
    )
    return shim()


def _single_method(capability: type) -> str | None:
    """Return the only declared method when no public data fields exist."""
    if not inspect.isclass(capability):
        return None
    methods = sorted(
        name
        for name, member in vars(capability).items()
        if not name.startswith("_") and callable(member)
    )
    if len(methods) != 1:
        return None
    fields = {
        name
        for cls in getattr(capability, "__mro__", (capability,))
        for name in getattr(cls, "__annotations__", {})
        if not name.startswith("_")
    }
    return None if fields else methods[0]


__all__ = ["CapabilityContext", "CapabilityMissing"]
