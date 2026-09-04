# SPDX-License-Identifier: MIT
"""08 - putting your own context in the prompt.

Lamssi assembles the system prompt from ordered blocks. Each block declares its
position and whether the provider can cache it across turns.

    python examples/08_dynamic_context.py
"""

from _support import heading

from lamssi_agents import Agent, ApprovalPolicy, PromptPosition
from lamssi_agents import ContextBlock

STATE = {"stage_mm": 12.5, "laser": "off"}

heading("Two context blocks: one stable, one changing")

agent = Agent(approval=ApprovalPolicy.allow_all(), context=[
    # Reference material. Same every turn, so it is cacheable and sits in the
    # part of the prompt a provider can reuse between calls.
    ContextBlock(
        "instrument-api",
        lambda ctx: "## Instruments\n\n- stage.move(mm)\n- laser.set(on|off)",
        position=PromptPosition.REFERENCE, stable=True,
    ),
    # Live state changes during the run and therefore stays below the provider's
    # cache breakpoint.
    ContextBlock(
        "live-state",
        lambda ctx: f"## Right now\n\n- stage at {STATE['stage_mm']} mm\n"
                    f"- laser {STATE['laser']}",
        position=PromptPosition.LIVE,
    ),
])

prompt = agent.assemble_prompt()
for name, chars in prompt.blocks:
    print(f"  {name:<20} {chars:>5} chars")

heading("The live one re-renders every turn")

print("  before:", "12.5 mm" in agent.assemble_prompt().text)
STATE["stage_mm"] = 40.0
print("  after :", "40.0 mm" in agent.assemble_prompt().text)

print("""
  ContextBlock is the advanced layer for dynamic application state. `position`
  names where the block lands and `stable=True` opts reference material into
  the provider's cacheable prefix.
""")
