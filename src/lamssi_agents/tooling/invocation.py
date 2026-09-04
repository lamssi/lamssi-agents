"""The validated tool call carried through policy and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from lamssi_tools import ToolDefinition
from lamssi_tools.registry import ToolBinding

@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """One tool request bound to normalized arguments and an implementation."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = ""
    turn: int = 0
    binding: Optional[ToolBinding] = field(default=None, repr=False, compare=False)

    @property
    def id(self) -> str:
        """Return the provider identifier pairing this call with its result."""
        return self.call_id

    @property
    def definition(self) -> Optional[ToolDefinition]:
        """Return the definition selected during preparation, when available."""
        return self.binding.definition if self.binding is not None else None


__all__ = ["ToolInvocation"]
