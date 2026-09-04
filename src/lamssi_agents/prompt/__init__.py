"""Composes the system prompt from named, ordered ContextBlocks."""

from __future__ import annotations

from lamssi_agents.prompt.builder import SectionRegistry
from lamssi_agents.prompt.model import (
    AssembledPrompt,
    PromptContext,
    PromptPart,
    PromptPosition,
)
from lamssi_agents.prompt.section import ContextBlock, heading

__all__ = [
    "AssembledPrompt",
    "ContextBlock",
    "PromptContext",
    "PromptPart",
    "PromptPosition",
    "SectionRegistry",
    "heading",
]
