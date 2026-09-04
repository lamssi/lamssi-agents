# SPDX-License-Identifier: MIT
"""05 - watching a run.

Every tool call, result, token count, and error arrives as an event. A host can
use the same stream for progress displays and run logs.

    python examples/05_events.py
"""

from lamssi_agents import Agent, ApprovalPolicy
from lamssi_agents.events import AgentEventType
from lamssi_agents import Files, Guidance, SystemTools

from _support import ScriptedModel, calls, heading, says

heading("Listening to a whole turn")

agent = Agent(features=[SystemTools(), Guidance(), Files(".")], approval=ApprovalPolicy.allow_all())


def watch(event) -> None:
    if event.type is AgentEventType.TOOL_START:
        args = event.metadata.get("arguments") or {}
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"  -> {event.data}({shown})")
    elif event.type is AgentEventType.TOOL_RESULT:
        print(f"  <- {str(event.data)[:70]}")
    elif event.type is AgentEventType.TEXT_DONE:
        print(f"   = {event.data}")
    elif event.type is AgentEventType.MESSAGES_SENT:
        m = event.metadata
        print(f"  [turn {m.get('turn')}: {m.get('tool_count')} tools, "
              f"{m.get('system_prompt_chars')} prompt chars]")


agent.add_event_listener(watch)
agent.use_model(ScriptedModel(
    calls("fs", command="ls"),
    says("There are a handful of Python files here."),
))
agent.chat("what is here?")

heading("The types you are most likely to want")
for name in ("TEXT_DELTA", "TOOL_START", "TOOL_RESULT", "MESSAGES_SENT",
             "USAGE", "RECOVERING", "HISTORY_COMPACTED",
             "ERROR", "DONE"):
    print(f"  AgentEventType.{name}")
