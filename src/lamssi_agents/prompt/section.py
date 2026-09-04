"""ContextBlock: the one section type a system prompt is assembled from."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable

from lamssi_agents.prompt.model import PromptContext, PromptPosition

log = logging.getLogger(__name__)


class ContextBlock:
    """A named block of prompt text, produced by a render callback each turn.

    Every contribution to the system prompt is a ContextBlock: the agent's
    instructions (``position=PromptPosition.INSTRUCTIONS``), feature guidance and
    catalogs, and live application state. There is no separate section class to
    subclass; a feature builds a block with a bound method as its ``render``.

    The callback may accept no arguments or one narrow :class:`PromptContext`, and
    returns the block's text, or an empty string to omit it this turn.

    ``stable=True`` is a provider-cache hint, not value memoization: the callback
    still runs every assembly. Use it only when the text is expected to stay
    byte-stable, so it can join the reusable prompt prefix. Volatile blocks always
    follow stable blocks. Within either group,
    ``position`` selects a named placement without exposing numeric ordering.

    Args:
        name: Stable unique name, used in prompt diagnostics and replacement.
        render: Callable returning the block's text, or empty to omit it. May
            accept ``PromptContext(model_id, tools)`` or no arguments.
        stable: Whether the output may join the provider-cacheable prefix.
        position: Named placement within the stable or volatile group.
        source: Optional provenance label; derived from ``render`` when omitted.

    Raises:
        TypeError: If ``position`` is not a :class:`PromptPosition`, or ``render``
            takes more than one positional argument.
        ValueError: If ``name`` is empty or whitespace.

    Example:
        Add volatile state captured from an application object::

            status = ContextBlock(
                "application-status",
                lambda ctx: f"Connected tools: {sorted(ctx.tools)}",
                stable=False,
                position=PromptPosition.LIVE,
            )
            agent.add_context(status)
    """

    __slots__ = ("name", "position", "cacheable", "source", "_fn", "_takes_context")

    def __init__(
        self,
        name: str,
        render: Callable[..., str],
        *,
        stable: bool = False,
        position: PromptPosition = PromptPosition.CONTEXT,
        source: str = "",
    ) -> None:
        if not name or not name.strip():
            raise ValueError("a context block needs a stable name")
        if not isinstance(position, PromptPosition):
            raise TypeError("position must be a PromptPosition")
        self.name = name.strip()
        self.position = position
        self.cacheable = bool(stable)
        self._fn = render
        self._takes_context = _wants_context(render, self.name)
        self.source = source or _source_of(render)

    @property
    def stable(self) -> bool:
        """Whether this block belongs to the provider-cacheable prompt prefix."""
        return self.cacheable

    def render(self, ctx: PromptContext) -> str:
        return self._fn(ctx) if self._takes_context else self._fn()

    def __repr__(self) -> str:
        kind = "cacheable" if self.cacheable else "volatile"
        return f"<ContextBlock {self.name!r} position={self.position!r} {kind}>"


def _wants_context(fn: Callable[..., str], name: str) -> bool:
    """Whether *fn* takes a positional PromptContext, or is a zero-arg callback."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind(object())
        return True
    except TypeError:
        try:
            signature.bind()
            return False
        except TypeError as exc:
            raise TypeError(
                f"prompt callback {name!r} must accept no arguments or one "
                "positional PromptContext"
            ) from exc


def _source_of(fn: Callable[..., str]) -> str:
    module = getattr(fn, "__module__", "")
    qualified = getattr(fn, "__qualname__", "") or getattr(fn, "__name__", "")
    return ".".join(part for part in (module, qualified) if part) or repr(fn)


def heading(title: str, body: str) -> str:
    """Return ``"## Title\\n\\nbody"``, or ``""`` if *body* is empty.

    Vanishing on empty body matters: an unconditional header plus "(none)"
    reads to the model as a fact about the host, not an absent one.
    """
    body = (body or "").strip()
    if not body:
        return ""
    return f"## {title}\n\n{body}"


def coerce_text(value: object, name: str) -> str:
    """Coerce a section's return value to stripped text, tolerantly.

    A host function returning a path, a number, or ``None`` should degrade
    to usable text rather than crash assembly later inside ``join``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    log.debug(
        "prompt section %r returned %s, not str: coercing", name, type(value).__name__
    )
    try:
        return str(value).strip()
    except Exception as exc:
        log.warning("prompt section %r returned an uncoercible value: %s", name, exc)
        return ""


def normalize_section(section: object):
    """Validate a built context block and return its registry name."""
    if inspect.isclass(section):
        raise TypeError(
            "add_context() requires a ContextBlock instance, "
            f"not the class {section.__name__}; construct it first"
        )
    if hasattr(section, "render"):
        name = getattr(section, "name", "") or type(section).__name__.lower()
        return name, section
    raise TypeError(
        "add_context() requires a ContextBlock instance; "
        f"got {type(section).__name__}"
    )


__all__ = ["ContextBlock", "coerce_text", "heading", "normalize_section"]
