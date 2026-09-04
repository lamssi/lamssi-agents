"""The ReAct loop: continue, finish, nudge, recover, stop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

from lamssi_agents.agent import conversation, turn as turn_mod
from lamssi_agents import tool_runtime
from lamssi_agents.model import input_limit
from lamssi_agents.events import AgentAborted, AgentEventType
from lamssi_agents.prompt.model import AssembledPrompt
from lamssi_agents.providers import Message
from lamssi_agents.providers.failures import ModelError, Recovery

if TYPE_CHECKING:
    from lamssi_agents.agent.control import RunControl
    from lamssi_agents.agent.conversation import Conversation
    from lamssi_agents.model import Model
    from lamssi_agents.runtime.config import AgentConfig
    from lamssi_agents.tool_runtime import ToolRuntime

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunLoop:
    """The components one run drives: conversation, tool runtime, run control,
    model, config, event sink, prompt assembler, and the before-turn hooks.

    Built once per run. The references are stable while their internal state
    (recorded usage, the growing conversation) changes, which is what the loop
    reads.
    """

    conversation: "Conversation"
    runtime: "ToolRuntime"
    control: "RunControl"
    #: The current model, read live.
    model: Callable[[], "Model"]
    config: "AgentConfig"
    emit: Callable[..., None]
    assemble_prompt: Callable[[], AssembledPrompt]
    #: before-turn checks, each bound to its agent: ``check(turn) -> stop | None``.
    before_turn: List[Callable[[int], Optional[str]]]

    def check_abort(self) -> None:
        """Raise immediately when cancellation has been requested."""
        if self.control.is_aborted:
            raise AgentAborted("Agent run aborted by user")


def run(rl: RunLoop) -> str:
    """Run to a final answer. The user message must already be in the transcript."""
    #: Recovery attempts so far, per reason, so one failure type doesn't exhaust another's budget; loop-local because it derives from control flow, not the messages.
    attempts: dict = {}
    #: Text from length-truncated turns, prepended to the final answer.
    carried = ""

    for turn in range(1, rl.config.max_turns + 1):
        rl.check_abort()
        rl.conversation.turn = turn
        rl.emit(AgentEventType.TURN_START, turn=turn)

        # A policy may end the run here (budget, deadline, kill switch); the loop only knows a string means stop.
        for check in rl.before_turn:
            stop = check(turn)
            if stop is not None:
                return _finalize_text_turn(rl, turn, stop, attempts, carried)

        while True:
            try:
                prepared = _prepare_turn(rl)
            except conversation.ContextWindowExceeded as exc:
                failure = ModelError(
                    Recovery.STOP,
                    "context_preflight",
                    str(exc),
                    "Reduce the fixed system prompt or tool surface, split the latest "
                    "input, or configure the model's real context_window.",
                )
                return _handle_provider_error(rl, failure)

            try:
                response = turn_mod.run_turn(
                    rl, prepared.messages, prepared.tools,
                    turn=turn,
                    tool_count=len(prepared.tools),
                    schema_chars=prepared.schema_chars,
                    prompt=prepared.prompt,
                )
                break
            except AgentAborted:
                raise
            except ModelError as exc:
                failure = exc
                if _try_recover(
                    rl,
                    failure,
                    attempts,
                    prompt=prepared.prompt,
                    schema_chars=prepared.schema_chars,
                ):
                    attempts[failure.reason] = attempts.get(failure.reason, 0) + 1
                    _report_recovery(rl, failure, attempts[failure.reason])
                    continue
                return _handle_provider_error(rl, failure)

        # A completed request clears the compaction-recovery counters; a later,
        # independent overflow in the same run gets its own attempt.
        for reason in list(attempts):
            if reason != _STALLED:
                attempts.pop(reason, None)

        if response.tool_calls:
            attempts.pop(_STALLED, None)          # progress: reset the stall guard
            rl.conversation.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )
            batch = rl.runtime.execute_calls(
                response.tool_calls, rl.conversation.turn
            )
            rl.conversation.extend(batch.messages)
            if batch.aborted:
                raise AgentAborted()
            rl.emit(AgentEventType.TURN_END, turn=turn)
            continue

        outcome, carried = _finish_text_turn(
            rl, turn, response.text, response.finish_reason, attempts, carried
        )
        if outcome is not None:
            return outcome

    return _handle_max_turns(rl)


@dataclass(frozen=True, slots=True)
class PreparedTurn:
    """Everything assembled once for the next provider call."""

    tools: list
    messages: List[Message]
    prompt: AssembledPrompt
    schema_chars: int


def _prepare_turn(rl: RunLoop) -> PreparedTurn:
    """Assemble the prompt, tools, and fitted history for one provider call."""
    tools_defs = rl.runtime.all_defs()
    prompt = rl.assemble_prompt()
    schema_chars = tool_runtime.schema_json_len(tools_defs)
    messages = rl.conversation.fit_request(
        prompt.text,
        overhead_chars=schema_chars,
    )
    return PreparedTurn(
        tools=tools_defs,
        messages=conversation.strip_orphan_tool_messages(messages),
        prompt=prompt,
        schema_chars=schema_chars,
    )


#: Key under which the loop counts consecutive silent replies after a tool result.
_STALLED = "empty_after_tool"


def _finish_text_turn(
    rl: RunLoop, turn: int, clean_text: str, finish_reason: Optional[str],
    attempts: dict, carried: str,
) -> "tuple[Optional[str], str]":
    """End the run, continue after truncation, or nudge a stall.

    Returns ``(final answer or None, carried text)``.
    """
    if finish_reason == "length":
        return _handle_length_finish(rl, turn, clean_text, carried)
    if _is_empty_after_tool(rl, clean_text) and _nudge_after_tool(rl, turn, attempts):
        return None, carried
    return _finalize_text_turn(rl, turn, clean_text, attempts, carried), ""


def _handle_length_finish(
    rl: RunLoop, turn: int, clean_text: str, carried: str
) -> "tuple[Optional[str], str]":
    """Continue a partial reply or report an empty length-cut response."""
    if not clean_text:
        err = _length_diagnosis(rl)
        log.error(err)
        rl.conversation.append(Message(role="assistant", content=err))
        final = f"{carried}\n\n{err}" if carried else err
        rl.emit(AgentEventType.ERROR, err)
        rl.emit(AgentEventType.DONE, final)
        return final, ""

    log.warning("Response truncated: asking the model to continue")
    rl.conversation.append(Message(role="assistant", content=clean_text))
    rl.conversation.append(Message(
        role="user",
        content=(
            "Your response was cut off by the token limit. Do NOT repeat what you "
            "already said. Continue from where you left off, and call the "
            "appropriate tool now if one is needed."
        ),
    ))
    rl.emit(AgentEventType.TURN_END, turn=turn)
    return None, carried + clean_text


def _length_diagnosis(rl: RunLoop) -> str:
    """Explain an empty length-cut response using the recorded token usage."""
    model = rl.model()
    usage = getattr(model, "last_usage", None)
    produced = int(getattr(usage, "completion_tokens", 0) or 0)
    reasoning = int(getattr(usage, "reasoning_tokens", 0) or 0)
    sent = int(getattr(usage, "prompt_tokens", 0) or 0)
    cap = max(1, int(getattr(model, "max_tokens", 8192) or 8192))
    window = input_limit(model)

    if produced >= cap * 0.9:
        return (
            f"The model produced no answer: it spent {produced:,} of its "
            f"{cap:,} output tokens"
            + (f" ({reasoning:,} on reasoning)" if reasoning else "")
            + " and was cut off."
            + (
                f" The input was {sent:,} tokens against a {window:,} window, so "
                f"the context window is not what ran out."
                if sent and window else ""
            )
            + " Raise max_tokens, lower the reasoning effort, or ask for less in "
              "one turn."
        )

    if produced == 0 and (not window or not sent or sent >= window * 0.9):
        return (
            "The input (system prompt + tools + history) leaves no room for a reply"
            + (f" ({sent:,} tokens against a {window:,} window)" if sent and window else "")
            + ". Run /compact, clear the conversation, or reduce the tool count."
        )

    return (
        f"The model was cut off after {produced:,} of {cap:,} output tokens "
        f"without producing text or a complete tool call"
        + (f" ({reasoning:,} on reasoning)" if reasoning else "")
        + f". The input was {sent:,} tokens"
        + (f" against a {window:,} window" if window else "")
        + ". Retrying unchanged will do the same."
    )


def _is_empty_after_tool(rl: RunLoop, clean_text: str) -> bool:
    last = rl.conversation.last
    return not clean_text.strip() and last is not None and last.role == "tool"


def _nudge_after_tool(rl: RunLoop, turn: int, attempts: dict) -> bool:
    """Nudge a silent post-tool response, at most twice."""
    sent = attempts.get(_STALLED, 0)
    if sent >= 2:
        return False
    attempts[_STALLED] = sent + 1
    log.warning("Empty response after a tool result: nudging (%d/2)", sent + 1)
    rl.conversation.append(Message(
        role="user",
        content=(
            "You received a tool result but returned no text and no tool call. "
            "Continue the task now: call the next tool, or give your final answer "
            "if the task is complete. Do not wait for another message."
        ),
    ))
    rl.emit(AgentEventType.TURN_END, turn=turn)
    return True


def _finalize_text_turn(
    rl: RunLoop, turn: int, clean_text: str, attempts: dict, carried: str = ""
) -> str:
    attempts.pop(_STALLED, None)
    rl.conversation.append(Message(role="assistant", content=clean_text))
    full = carried + clean_text
    if full:
        rl.emit(AgentEventType.TEXT_DONE, full)
    rl.emit(AgentEventType.TURN_END, turn=turn)
    rl.emit(AgentEventType.DONE)
    return full


#: A context overflow gets one compacted retry; another rejection is definitive.
MAX_RECOVERIES = 1


def _try_recover(
    rl: RunLoop,
    failure: ModelError,
    attempts: dict,
    *,
    prompt: AssembledPrompt,
    schema_chars: int,
) -> bool:
    """Compact after a context overflow and report whether retrying is useful."""
    if failure.recovery is not Recovery.COMPACT or attempts.get(failure.reason, 0) >= MAX_RECOVERIES:
        return False
    before = list(rl.conversation.history)
    try:
        rl.conversation.fit_request(
            prompt.text,
            overhead_chars=schema_chars,
            force=True,
        )
    except conversation.ContextWindowExceeded:
        return False
    if rl.conversation.history == before:
        log.warning("context overflow, but compaction could not reduce the request - not retrying")
        return False
    return True


def _report_recovery(rl: RunLoop, failure: ModelError, attempt: int) -> None:
    """Report a compaction recovery; the retry sends immediately, with no backoff."""
    log.info("%s (compaction %d/%d)", failure.reason, attempt, MAX_RECOVERIES)
    rl.emit(
        AgentEventType.RECOVERING,
        f"{failure.message} Compacted and retrying ({attempt}/{MAX_RECOVERIES}).",
        reason=failure.reason, attempt=attempt,
    )


def _handle_provider_error(rl: RunLoop, failure: ModelError) -> str:
    err = failure.message
    if failure.hint:
        err = f"{err}\n{failure.hint}"
    log.error("%s: %s", failure.reason, err)
    rl.emit(AgentEventType.ERROR, err)
    # Repair before recording: a failure mid-turn can leave an announced tool call unanswered, rejecting the next request outright.
    rl.conversation.sanitize()
    rl.conversation.append(Message(role="assistant", content=err))
    rl.emit(AgentEventType.DONE, err)
    return err


def _handle_max_turns(rl: RunLoop) -> str:
    limit = rl.config.max_turns
    log.warning("Max turns (%d) reached", limit)
    msg = (
        f"Stopped after {limit} tool-calling turns without reaching a final answer. "
        "Ask the user how to proceed."
    )
    rl.conversation.append(Message(role="assistant", content=msg))
    rl.emit(AgentEventType.TEXT_DONE, msg)
    rl.emit(AgentEventType.DONE)
    return msg
