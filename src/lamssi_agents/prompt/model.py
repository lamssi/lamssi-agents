"""Public data contracts for composing a system prompt."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class PromptPosition(IntEnum):
    """Named placement for a section within its prompt stability group.

    Stable sections are grouped before volatile sections. Position then orders
    sections within either group, so cache behavior and placement remain two
    independent choices.

    Attributes:
        INSTRUCTIONS: The agent's identity and primary instructions.
        GUIDANCE: Operating rules that qualify those instructions.
        REFERENCE: Stable reference material or dynamically activated guidance.
        CONTEXT: Ordinary application and feature context.
        LIVE: Frequently changing state that should appear last.
    """

    INSTRUCTIONS = 0
    GUIDANCE = 5
    REFERENCE = 20
    CONTEXT = 50
    LIVE = 90


@dataclass(frozen=True, slots=True)
class PromptContext:
    """Narrow read-only state supplied while rendering a prompt block.

    Prompt callbacks receive only information needed to tailor text to the
    current request surface. Application objects should be captured explicitly
    in the callback closure rather than retrieved through this context.

    Attributes:
        model_id: Identifier of the model that will receive the prompt.
        tools: Names of tools currently visible to that model after scoping and
            capability filtering.
    """

    model_id: str = ""
    tools: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class PromptPart:
    """Provenance for one rendered prompt section."""

    name: str
    source: str
    position: PromptPosition | int
    cacheable: bool
    chars: int


@dataclass(frozen=True, slots=True)
class AssembledPrompt:
    """A rendered system prompt and its per-section provenance."""

    text: str
    parts: tuple[PromptPart, ...] = ()

    @property
    def blocks(self) -> tuple[tuple[str, int], ...]:
        """Return section names and character counts."""
        return tuple((part.name, part.chars) for part in self.parts)

    @property
    def chars(self) -> int:
        """Return the prompt's character count."""
        return len(self.text)


__all__ = [
    "AssembledPrompt",
    "PromptContext",
    "PromptPart",
    "PromptPosition",
]
