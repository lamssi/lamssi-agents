"""Provider-agnostic data types shared by every LLM backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    """Provider-agnostic chat message."""

    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: Optional[List[ToolCall]] = None  # assistant messages only
    tool_call_id: Optional[str] = None  # tool-result messages only
    name: Optional[str] = None  # tool name (role="tool")
    # Vision side-channel, kept separate from ``content`` so estimation/compaction/logging see plain text.
    images: Optional[List[str]] = None

    def to_provider_dict(self) -> Dict[str, Any]:
        """Convert to an OpenAI-compatible dict.

        LiteLLM translates this format internally for non-OpenAI providers.
        """
        d: Dict[str, Any] = {"role": self.role, "content": self.content}

        # Fold images into content blocks (OpenAI/Anthropic vision format); text stays first for text-only readers.
        if self.images:
            blocks: List[Dict[str, Any]] = []
            if self.content:
                blocks.append({"type": "text", "text": self.content})
            for url in self.images:
                blocks.append({"type": "image_url", "image_url": {"url": url}})
            d["content"] = blocks

        if self.role == "assistant" and self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]

        if self.role == "tool":
            d["tool_call_id"] = self.tool_call_id or ""
            d["name"] = self.name or ""

        return d


@dataclass
class Usage:
    """Token accounting for a single LLM call (or a running total)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # cached_tokens = cache-read (~0.1x price on Anthropic, 0.5x OpenAI); cache_write_tokens = written this call (~1.25x Anthropic).
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, other: "Usage") -> None:
        """Accumulate *other* into self in-place."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cached_tokens += other.cached_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.reasoning_tokens += other.reasoning_tokens


@dataclass
class StreamDelta:
    """A single chunk from a streaming LLM call."""

    type: str  # text | thinking | tool_call | usage | done | retrying
    text: str = ""
    tool_call: Optional[ToolCall] = None
    finish_reason: Optional[str] = None  # "stop" | "length" | "tool_calls"
    usage: Optional[Usage] = None
    # For type="retrying": {"attempt", "max_retries", "delay", "reason"}.
    retry: Optional[Dict[str, Any]] = None


class ProviderInterrupted(Exception):
    """Raised when an abort event fires during a blocking provider wait."""
