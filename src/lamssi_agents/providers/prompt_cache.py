"""Insert Claude prompt-cache breakpoints; other backends cache automatically and are untouched."""

from __future__ import annotations

from typing import Any, Dict, List


def needs_cache_breakpoints(model: str) -> bool:
    """Whether *model* needs explicit ``cache_control`` markers.

    Route-agnostic: a substring test on "claude"/"anthropic" catches every
    route (direct, anthropic/, openrouter/anthropic/, bedrock, vertex_ai),
    matching LiteLLM's own gate for cache_control forwarding.
    """
    m = (model or "").lower()
    return "claude" in m or "anthropic" in m


def mark_cached(msg: Dict[str, Any]) -> None:
    """Attach an ephemeral ``cache_control`` breakpoint to *msg* in place.

    Placement is role-aware (verified against litellm 1.82.0's Anthropic
    prompt factory and OpenRouter transform): tool/content-less messages get
    it at the message level; text messages get it on the last content block
    (a bare string is promoted to one text block first).
    """
    cc = {"type": "ephemeral"}
    content = msg.get("content")
    if msg.get("role") == "tool" or content in (None, ""):
        msg["cache_control"] = cc
    elif isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content, "cache_control": cc}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1]["cache_control"] = cc
    else:
        msg["cache_control"] = cc


def apply_prompt_caching(msgs: List[Dict[str, Any]], model: str) -> None:
    """Insert cache breakpoints (in place) for models that need them.

    Two breakpoints (Anthropic allows up to four): one at the end of the
    system prompt (caches tool schemas + system together, given Anthropic's
    tools -> system -> messages prefix order), one on the final message (caches
    the whole conversation, re-read at ~0.1x on the next call). No-op for
    auto-caching models.
    """
    if not needs_cache_breakpoints(model):
        return
    if not msgs:
        return
    sys_obj = None
    if msgs[0].get("role") == "system":
        mark_cached(msgs[0])
        sys_obj = id(msgs[0])
    last = msgs[-1]
    if id(last) != sys_obj:          # don't double-mark a lone system msg
        mark_cached(last)
