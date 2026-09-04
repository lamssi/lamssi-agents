# SPDX-License-Identifier: MIT
"""06 - what a turn costs, and how to spend less.

Every model call includes the visible tool schemas. This example measures their
size and shows how ``only`` and discovery reduce it.

    python examples/06_cost_control.py
"""

import json

from lamssi_agents import Agent, ApprovalPolicy
from lamssi_agents import Files, Guidance, SystemTools
from lamssi_agents import tool_runtime as tool_mod

from _support import heading


def toll(agent) -> tuple:
    """Return the tool count and fixed prompt characters for one model call."""
    sent = agent.visible_tool_defs()
    schema = sum(
        len(json.dumps(d.to_openai_tool(), separators=(",", ":"))) for d in sent
    )
    prompt = agent.assemble_prompt().chars
    return len(sent), schema + prompt


heading("The toll, per model call")

rows = [
    ("everything", dict()),
    ("only=[five file tools]", dict(only=["read_file", "fs", "write_file", "edit_file", "delete_file"])),
    ("only=[read_file, fs]", dict(only=["read_file", "fs"])),
]
for label, kw in rows:
    count, chars = toll(Agent(features=[SystemTools(), Guidance(), Files(".")], approval=ApprovalPolicy.allow_all(), **kw))
    print(f"  {label:<26} {count:>2} tools   {chars:>6,} chars   ~{chars // 4:>5,} tokens")

print("""
  only=      you choose a fixed list of names. `ask_user` always survives, so a
             narrowed agent can still get itself out of a dead end.
""")

heading("Narrowing the surface does not break the agent")

agent = Agent(features=[SystemTools(), Guidance(), Files(".")], approval=ApprovalPolicy.allow_all(), only=["fs"])
print(f"  in scope     : {agent.available_tool_names()}")

# `ask_user` survives the narrowing, so a scoped agent still has a way out of
# a dead end.
result = tool_mod.invoke_tool_unchecked(agent, "fs", {"command": "ls"})
print(f"  called a deferred tool -> {'ok' if 'error' not in result else result['error']}")
