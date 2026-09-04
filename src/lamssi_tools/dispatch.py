"""Where a tool body runs: an opaque execution-context tag, resolved by an injectable :data:`Dispatcher`."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from lamssi_tools.errors import LamssiError

#: A host's execution router: definition, bound callable, validated kwargs.
Dispatcher = Callable[[Any, Callable[..., Any], Mapping[str, Any]], Any]


class UnknownDispatchTarget(LamssiError):
    """A tool declared a dispatch tag no host has claimed."""


def inline_dispatcher(
    definition: Any, fn: Callable[..., Any], kwargs: Mapping[str, Any]
) -> Any:
    """The default: call the body on the current thread.

    Correct for any pure function, and the only safe default for an arbitrary
    host: an agent without a dispatcher cannot know what other
    threads exist.
    """
    return fn(**kwargs)


def strict_dispatcher(
    definition: Any, fn: Callable[..., Any], kwargs: Mapping[str, Any]
) -> Any:
    """Like :func:`inline_dispatcher`, but refuses to silently mis-dispatch.

    Raises when a tool asks for a non-inline context. Useful in tests and in
    headless hosts that would rather fail loudly than run a body that documented
    a thread requirement it is not getting.
    """
    tag = getattr(definition, "dispatch", None)
    if tag is not None:
        raise UnknownDispatchTarget(
            f"tool {getattr(definition, 'name', '?')!r} requires dispatch "
            f"target {tag!r}, but this runtime dispatches inline only"
        )
    return fn(**kwargs)


__all__ = [
    "Dispatcher",
    "UnknownDispatchTarget",
    "inline_dispatcher",
    "strict_dispatcher",
]
