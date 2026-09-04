"""One provider call: send the prompt, stream the reply, report what was sent."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from lamssi_agents.events import AgentAborted, AgentEventType
from lamssi_agents.history.tokens import rough_tokens
from lamssi_agents.prompt.model import AssembledPrompt
from lamssi_agents.providers import Message, ProviderInterrupted, ToolCall
from lamssi_agents.providers.failures import classify
from lamssi_tools import ToolDefinition

if TYPE_CHECKING:
    from lamssi_agents.agent.loop import RunLoop

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Text, tool calls, and finish reason returned by one provider call."""

    text: str
    tool_calls: List[ToolCall]
    finish_reason: Optional[str]


def run_turn(
    rl: "RunLoop",
    messages: List[Message],
    tools: Optional[List[ToolDefinition]],
    *,
    turn: int,
    tool_count: int,
    schema_chars: int,
    prompt: Optional[AssembledPrompt] = None,
) -> TurnResult:
    """Make one provider call and return its streamed result."""
    sent_chars = _emit_messages_sent(
        rl,
        messages,
        tools,
        turn=turn,
        tool_count=tool_count,
        schema_chars=schema_chars,
        prompt=prompt,
    )
    return _stream(rl, messages, tools, sent_chars=sent_chars)


def _emit_messages_sent(
    rl: "RunLoop",
    messages: List[Message],
    tools: Optional[List[ToolDefinition]],
    *,
    turn: int,
    tool_count: int,
    schema_chars: int,
    prompt: Optional[AssembledPrompt],
) -> int:
    """Emit request diagnostics and return the total character count."""
    system_prompt = messages[0].content if messages and messages[0].role == "system" else ""
    window = _window_breakdown(messages, schema_chars)
    rl.emit(
        AgentEventType.MESSAGES_SENT,
        system_prompt,
        turn=turn,
        history_msg_count=len(rl.conversation.history),
        tool_count=tool_count,
        tool_schema_bytes=schema_chars,
        system_prompt_chars=len(system_prompt),
        prompt_blocks=prompt.blocks if prompt is not None else (),
        messages=messages,
        window=window,
    )
    return int(window["total_chars"])


def _window_breakdown(messages: List[Message], schema_bytes: int) -> Dict[str, Any]:
    """Return request character counts by message role and tool."""
    by_role: Dict[str, int] = {}
    by_tool: Dict[str, int] = {}
    heaviest: List[Tuple[str, int]] = []

    for message in messages:
        size = len(message.content or "")
        if message.tool_calls:
            size += sum(
                len(json.dumps(tc.arguments, default=str)) for tc in message.tool_calls
            )
        by_role[message.role] = by_role.get(message.role, 0) + size
        if message.role == "tool":
            name = message.name or "?"
            by_tool[name] = by_tool.get(name, 0) + size
            heaviest.append((name, size))

    heaviest.sort(key=lambda pair: pair[1], reverse=True)
    total = sum(by_role.values()) + schema_bytes
    return {
        "total_chars": total,
        "est_tokens": rough_tokens(total),
        "by_role": by_role,
        "by_tool": dict(sorted(by_tool.items(), key=lambda kv: kv[1], reverse=True)),
        "tool_schema_chars": schema_bytes,
        "heaviest_results": heaviest[:5],
    }


def _stream(
    rl: "RunLoop", messages: List[Message], tools: Optional[List[ToolDefinition]],
    *, sent_chars: int = 0,
) -> TurnResult:
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    finish_reason: Optional[str] = None

    # One model reference governs the whole call: fitting used the live model too.
    model = rl.model()
    stream = iter(model.stream(messages, tools, abort_event=rl.control.aborted))
    while True:
        try:
            delta = next(stream)
        except StopIteration:
            break
        except AgentAborted:
            raise
        except ProviderInterrupted:
            raise AgentAborted("Agent run aborted during provider call") from None
        except Exception as exc:
            # A provider failure: normalize it to a typed ModelError here, so the
            # loop reads a recovery instead of classifying a raw exception. A bug
            # in the delta handling below stays a plain exception and stays visible.
            raise classify(exc) from exc

        # Checked every delta, not once per turn: a stop press should not wait for a long generation to finish.
        rl.check_abort()
        if delta.type == "text":
            text_parts.append(delta.text)
            rl.emit(AgentEventType.TEXT_DELTA, delta.text)
        elif delta.type == "thinking":
            rl.emit(AgentEventType.THINKING, delta.text)
        elif delta.type == "tool_call" and delta.tool_call:
            tool_calls.append(delta.tool_call)
        elif delta.type == "usage" and delta.usage is not None:
            u = delta.usage
            rl.conversation.tokens.observe(sent_chars, u.prompt_tokens)
            # The sent list can exclude repaired orphan messages.
            if u.prompt_tokens > 0:
                rl.conversation.tokens.anchor = len(rl.conversation.history)
            cumulative = getattr(
                getattr(model, "cumulative_usage", None),
                "total_tokens",
                u.total_tokens,
            )
            rl.emit(
                AgentEventType.USAGE, u,
                prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                total_tokens=u.total_tokens, cached_tokens=u.cached_tokens,
                cache_write_tokens=u.cache_write_tokens,
                reasoning_tokens=u.reasoning_tokens,
                cumulative_total=cumulative,
            )
        elif delta.type == "retrying" and delta.retry is not None:
            # The provider is backing off before a retry; surface it, decide nothing.
            r = delta.retry
            rl.emit(
                AgentEventType.RECOVERING,
                f"{r.get('reason', 'Provider error')} - retrying "
                f"({r.get('attempt')}/{r.get('max_retries')}) in {r.get('delay', 0):.0f}s",
                reason="rate_limit", attempt=r.get("attempt"),
            )
        elif delta.type == "done":
            finish_reason = delta.finish_reason

    return TurnResult("".join(text_parts), tool_calls, finish_reason)


__all__ = ["run_turn", "TurnResult"]
