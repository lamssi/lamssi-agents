"""The domain-free tools and prompt guidance commonly useful to an agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Protocol, runtime_checkable

from lamssi_agents.features.base import Feature
from lamssi_tools import CapabilityContext, Expose, Str, err, tool

if TYPE_CHECKING:
    from lamssi_agents.agent.base import Agent

log = logging.getLogger(__name__)


@runtime_checkable
class AbortSink(Protocol):
    """A host component that can cancel its in-flight work."""

    def abort_all(self) -> None:
        """Stop currently running host work and return promptly."""
        ...


@tool(
    group="system",
    dispatch="worker",
    inject_context=True,
    expose=Expose.HOST | Expose.MCP,
    description="Immediately cancel everything the host is currently running.",
)
def abort(ctx: CapabilityContext, **kw) -> Dict:
    """Ask every registered abort sink to stop."""
    sinks = ctx.get_all(AbortSink)
    if not sinks:
        return {
            "aborted": False,
            "note": "Nothing to abort: no abort sink is configured.",
        }
    failures = []
    for sink in sinks:
        try:
            sink.abort_all()
        except Exception as exc:
            failures.append(str(exc))
    if failures:
        return err(
            "one or more abort sinks failed: " + "; ".join(failures), aborted=False
        )
    return {"aborted": True}


@tool(
    group="system",
    description=(
        "Ask the user one concise question. After their reply you MUST continue with "
        "tool calls: never stop on a text-only response. Use it before a multi-step "
        "job to confirm names or parameters you would otherwise guess."
    ),
    expose=Expose.AGENT,
    approval="never",
    guard_role="always_allowed",
    inject_context=True,
    parameters={"question": Str("The question to put to the user.")},
)
def ask_user(ctx: CapabilityContext, question: str) -> Dict:
    """Put *question* to whoever is running this agent and return their reply."""
    from lamssi_agents.interaction import InteractionKind, request_interaction
    from lamssi_agents.runtime.scope import RunScope

    scope = ctx.get(RunScope)
    if scope is None:
        return err("ask_user is only available inside an agent run.", retriable=False)
    agent = scope.agent
    question = str(question or "")
    if not question:
        return err("No question provided.", retriable=False)

    log.info("ask_user: %s", question[:200])
    response = request_interaction(
        agent._control.interaction.handler,
        agent.emit,
        InteractionKind.QUESTION,
        question,
    )
    if response is None:
        return err(
            "There is no one to ask: this run is unattended or the interaction failed.",
            retriable=False,
            hint="Proceed with a stated assumption, or stop and summarise what you need.",
        )
    agent._runtime.guard.reset_for_new_turn()
    return {"user_response": response.answer}


class SystemTools(Feature):
    """Install ``ask_user`` and ``abort`` plus their loop-guard roles.

    ``ask_user`` is model-visible and reports that nobody is available when the
    Agent has no interaction handler. ``abort`` remains host/MCP-only so a model
    cannot cancel host work by itself. This feature adds no operating
    instructions; install :class:`Guidance` separately when those are wanted.
    """

    name = "system-tools"

    def install(self, agent: Agent) -> None:
        from lamssi_agents.tooling.guard import CORE_GUARD_ROLES

        agent.add_guard_roles(CORE_GUARD_ROLES)
        agent.add_tools(abort, ask_user)


__all__ = ["AbortSink", "SystemTools", "abort", "ask_user"]
