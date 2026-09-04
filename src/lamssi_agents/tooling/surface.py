"""Resolves and enforces which tools an agent may see and call this turn (:class:`ToolSurface`)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Optional

from lamssi_tools import ToolDefinition

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolSurface:
    """A frozen, pre-indexed snapshot of the tools in scope for one turn.

    The ``names``/``by_name`` index is built once in ``__post_init__`` so the
    schema builder, approval check, argument coercion and executor all agree.
    Dispatch resolves a fresh snapshot per access, so an in-turn ``disable_tool``
    takes effect immediately.
    """

    #: In resolution order: the order the schema is built in and the model reads, kept stable.
    defs: tuple[ToolDefinition, ...] = ()
    #: Every name in :attr:`defs`, for the membership test that gates execution.
    names: frozenset[str] = field(init=False)
    #: Name to definition, for callers needing the parameters or ``approval`` field without re-searching the registry.
    by_name: Mapping[str, ToolDefinition] = field(init=False)

    def __post_init__(self) -> None:
        defs = tuple(self.defs)
        object.__setattr__(self, "defs", defs)
        object.__setattr__(self, "names", frozenset(d.name for d in defs))
        # Read-only view: a plain dict here would invite a caller to mutate
        # it and quietly diverge from `names`, the actual boundary.
        object.__setattr__(self, "by_name", MappingProxyType({d.name: d for d in defs}))

    def get(self, name: str) -> Optional[ToolDefinition]:
        """The definition for *name*, or ``None`` when it is out of scope."""
        return self.by_name.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return len(self.defs)


def resolve_surface(
    *,
    all_defs: Iterable[ToolDefinition],
    always_available: Iterable[str] = (),
    disabled: Iterable[str] = (),
    agent_allow: Optional[set[str]] = None,
) -> ToolSurface:
    """Narrow *all_defs* to what this agent may see and call.

    Included when a tool declares ``expose_to_agent``, isn't in *disabled*,
    and passes *agent_allow* (``None`` = unrestricted; an empty set means only
    the *always_available* tools remain). Names in *always_available* stay in
    scope even when the allow-list would exclude them, but still respect
    *disabled* and ``expose_to_agent``.
    """
    always = frozenset(always_available)
    blocked = frozenset(disabled)
    agent_gate = None if agent_allow is None else frozenset(agent_allow) | always

    resolved: list[ToolDefinition] = []
    seen: set[str] = set()
    filtered = 0

    for definition in all_defs:
        name = definition.name
        if name in seen:
            continue
        if not definition.expose_to_agent:
            continue
        if name in blocked:
            filtered += 1
            continue
        if agent_gate is not None and name not in agent_gate:
            filtered += 1
            continue
        seen.add(name)
        resolved.append(definition)

    if filtered:
        log.debug("tool surface: %d tools in scope, %d withheld", len(resolved), filtered)
    return ToolSurface(tuple(resolved))


__all__ = ["ToolSurface", "resolve_surface"]
