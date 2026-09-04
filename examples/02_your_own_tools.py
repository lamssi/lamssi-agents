# SPDX-License-Identifier: MIT
"""02 - using ordinary Python functions as tools.

``Agent`` builds each tool schema from its signature and docstring.

    python examples/02_your_own_tools.py
"""

from lamssi_agents import Agent, ApprovalPolicy

from _support import ScriptedModel, calls, heading, says


def celsius_to_fahrenheit(celsius: float = 0.0) -> dict:
    """Convert a temperature from Celsius to Fahrenheit."""
    return {"fahrenheit": celsius * 9 / 5 + 32}


def repeat(text: str = "", times: int = 1) -> dict:
    """Repeat some text a number of times."""
    return {"result": " ".join([text] * times)}


heading("Two plain functions, handed to an agent")

agent = Agent(
    tools=[celsius_to_fahrenheit, repeat],
    approval=ApprovalPolicy.allow_all(),
)

for definition in agent.visible_tool_defs():
    if definition.name in ("celsius_to_fahrenheit", "repeat"):
        params = ", ".join(f"{p.name}: {p.type}" for p in definition.parameters)
        print(f"  {definition.name}({params})")
        print(f"      {definition.description}")

# Type hints matter. ``times: int`` produces an integer parameter in the schema,
# which keeps a string such as ``"3"`` out of ``range(times)``.

heading("Watching one get called")

agent.use_model(ScriptedModel(
    calls("repeat", text="ping", times=3),
    says("Repeated it three times."),
))
print(agent.chat("say ping three times"))
