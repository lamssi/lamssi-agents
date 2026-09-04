"""Holds an agent's context sections and assembles them into one system prompt."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from lamssi_agents.prompt.model import (
    AssembledPrompt,
    PromptContext,
    PromptPart,
    PromptPosition,
)
from lamssi_agents.prompt.section import coerce_text

log = logging.getLogger(__name__)

_RESERVED_NAMES = frozenset({"instructions"})


class SectionRegistry:
    """The context sections installed on one agent, and their assembly.

    Instance-scoped rather than a module global, so two agents in one process
    never overwrite each other's registrations.
    """

    __slots__ = ("_sections",)

    def __init__(self) -> None:
        self._sections: Dict[str, Any] = {}

    def register(self, name: str, section: Any) -> None:
        """Register one built context section, replacing any earlier one by name.

        Re-registering a name lets an application override a feature's
        contribution explicitly.
        """
        if not name:
            log.warning("refusing to register a prompt section with an empty name")
            return
        if name in _RESERVED_NAMES:
            raise ValueError(
                f"prompt section name {name!r} is reserved for Agent instructions"
            )
        if not hasattr(section, "render"):
            log.warning("prompt section %r has no render method", name)
            return
        if name in self._sections:
            log.debug("prompt section %r re-registered; the later section wins", name)
        self._sections[name] = section

    def names(self) -> Tuple[str, ...]:
        """Every registered name, in registration order."""
        return tuple(self._snapshot())

    def resolve(self) -> List[Any]:
        """Return every registered section in registration order."""
        return list(self._snapshot().values())

    def assemble(
        self,
        *,
        instance_sections: Iterable[Any] = (),
        model: str = "",
        tools: Optional[Iterable[str]] = None,
    ) -> AssembledPrompt:
        """Render the selected sections and join them with a blank line.

        Instance-built sections take precedence over registered context when a
        name repeats. The result carries a per-section character count, for
        tracking prompt bloat.
        """
        ctx = PromptContext(model_id=model, tools=frozenset(tools or ()))

        parts: List[str] = []
        provenance: List[PromptPart] = []
        for section in self._ordered_sections(instance_sections):
            name = _name_of(section)
            try:
                text = coerce_text(section.render(ctx), name)
            except Exception as exc:
                log.warning("prompt section %r failed to render; omitted: %s", name, exc)
                continue
            if not text:
                continue
            parts.append(text)
            provenance.append(PromptPart(
                name=name,
                source=_source_of(section),
                position=_position_of(section),
                cacheable=_cacheable_of(section),
                chars=len(text),
            ))

        prompt = AssembledPrompt(text="\n\n".join(parts), parts=tuple(provenance))
        log.debug(
            "assembled prompt: %d chars from %d sections", prompt.chars, len(provenance)
        )
        return prompt

    def _ordered_sections(
        self, instance_sections: Iterable[Any]
    ) -> List[Any]:
        """Contributing sections in final order, instance sections winning ties.

        Cacheable sections sort ahead of volatile ones for a stable provider
        cache breakpoint.
        """
        sections: List[Any] = []
        seen: Set[str] = set()
        for section in [*instance_sections, *self.resolve()]:
            name = _name_of(section)
            if name in seen:
                log.debug(
                    "prompt section %r contributed more than once; keeping the first",
                    name,
                )
                continue
            seen.add(name)
            sections.append(section)

        sections.sort(
            key=lambda s: (
                not _cacheable_of(s),
                _position_of(s),
                0 if _name_of(s) == "instructions" else 1,
                _name_of(s),
            )
        )
        log.debug("prompt sections: %s", ", ".join(_name_of(s) for s in sections))
        return sections

    def _snapshot(self) -> Dict[str, Any]:
        return dict(self._sections)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._snapshot()

    def __len__(self) -> int:
        return len(self._snapshot())

    def __repr__(self) -> str:
        return f"<SectionRegistry {len(self._sections)} sections>"


# Sections are a structural protocol; read defensively so a malformed
# contribution cannot break sorting for everything else.


def _name_of(section: Any) -> str:
    name = getattr(section, "name", "")
    return name if isinstance(name, str) and name else type(section).__name__


def _position_of(section: Any) -> PromptPosition | int:
    position = getattr(section, "position", PromptPosition.CONTEXT)
    return position if isinstance(position, int) else PromptPosition.CONTEXT


def _cacheable_of(section: Any) -> bool:
    # Defaults to False: guessing cacheable wrongly costs a cache miss every
    # turn, whereas guessing volatile wrongly costs only position.
    return bool(getattr(section, "cacheable", False))


def _source_of(section: Any) -> str:
    declared = getattr(section, "source", "")
    if isinstance(declared, str) and declared:
        return declared
    cls = type(section)
    return f"{cls.__module__}.{cls.__qualname__}"


__all__ = ["SectionRegistry"]
