# SPDX-License-Identifier: MIT
"""03 - describing and validating tools with ``@tool``.

This example stays offline. It prints the schema produced by the decorator and
calls a few tools through the same validation path used by an agent.

    python examples/03_typed_tools.py
"""

from lamssi_agents import Agent, ApprovalPolicy, tool
from lamssi_agents import tool_runtime as tool_mod
from lamssi_tools import Array, Bool, Expose, Float, Int, Str

from _support import heading


@tool(
    group="lab",
    expose=Expose.AGENT,
    # The tool declares its risk. The Agent's ApprovalPolicy decides how the
    # application handles calls carrying that declaration.
    approval="always",
    # These words help deferred tool discovery. They are not sent to the model.
    keywords="heat warm temperature setpoint",
    parameters={
        "celsius": Float("Target temperature.", ge=-273.15, le=500.0),
        "channel": Int("Which heater.", ge=1, le=4),
    },
)
def set_temperature(
    celsius: float = 20.0,
    channel: int = 1,
) -> dict:
    """Set a heater's target temperature.

    Use this when the user names a temperature they want reached and held.
    """
    return {"channel": channel, "celsius": celsius}


@tool(
    group="lab",
    expose=Expose.AGENT,
    approval="never",
    parameters={"channel": Int("Which heater.", ge=1, le=4)},
)
def read_temperature(channel: int = 1) -> dict:
    """Read a heater's current temperature."""
    return {"channel": channel, "celsius": 21.4}


@tool(
    # The callable keeps its Python name; this is the shorter name shown to the
    # model and used when invoking the tool.
    name="manage_sensor",
    group="lab",
    expose=Expose.AGENT,
    approval="conditional",
    # An inspect call skips approval. Other actions still follow the Agent's
    # approval policy.
    safe_when={"action": "inspect"},
    # Return schemas document results for hosts and logs. They do not coerce the
    # value returned by the Python function.
    returns={
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string"},
            "action": {"type": "string"},
        },
    },
    parameters={
        "sensor_id": Str(
            "Sensor label such as A01.",
            pattern=r"^[A-Z][0-9]{2}$",
            examples=["A01"],
        ),
        "action": Str(
            "Operation to perform.",
            enum=["inspect", "reset"],
        ),
        "dry_run": Bool("Describe a reset without applying it."),
    },
)
def inspect_or_reset_sensor(
    sensor_id: str,
    action: str = "inspect",
    dry_run: bool = True,
) -> dict:
    """Inspect or reset one sensor."""
    return {"sensor_id": sensor_id, "action": action, "dry_run": dry_run}


@tool(
    group="analysis",
    expose=Expose.AGENT,
    approval="never",
    returns={
        "type": "object",
        "properties": {"mean": {"type": "number"}},
    },
    # Large results are clipped before entering conversation history. The hint
    # tells the model how to request a smaller result on its next call.
    truncation=500,
    truncation_side="head",
    truncation_hint="Request fewer readings if the raw values are clipped.",
    parameters={
        "readings": Array(
            "Measurements to summarize.",
            min_items=2,
            max_items=20,
        ),
        "decimals": Int("Decimal places in the mean.", ge=0, le=4),
        "include_values": Bool("Include the original readings in the result."),
    },
)
def summarize_readings(
    readings: list[float],
    decimals: int = 2,
    include_values: bool = False,
) -> dict:
    """Calculate the mean of a short list of readings."""
    result = {"mean": round(sum(readings) / len(readings), decimals)}
    if include_values:
        result["readings"] = readings
    return result


heading("What the model is told")

agent = Agent(
    tools=[
        set_temperature,
        read_temperature,
        inspect_or_reset_sensor,
        summarize_readings,
    ],
    approval=ApprovalPolicy.allow_all(),
)
definitions = {definition.name: definition for definition in agent.visible_tool_defs()}

for name in (
    "set_temperature",
    "read_temperature",
    "manage_sensor",
    "summarize_readings",
):
    definition = definitions[name]
    print(
        f"  {definition.name:<20} approval={definition.approval:<11} "
        f"group={definition.group}"
    )
    if definition.safe_when:
        print(f"      safe without approval when {definition.safe_when}")
    if definition.returns:
        print(f"      return schema: {definition.returns['type']}")
    if definition.truncation:
        print(
            f"      result cap: {definition.truncation} chars, "
            f"keep={definition.truncation_side}"
        )

    for parameter in definition.parameters:
        details = [
            "required" if parameter.required else f"default={parameter.default!r}"
        ]
        if parameter.items:
            details.append(f"items={parameter.items.type.value}")
        for label, value in (
            ("enum", parameter.enum),
            ("example", parameter.examples),
            ("pattern", parameter.pattern),
            ("min", parameter.minimum),
            ("max", parameter.maximum),
            ("min_items", parameter.min_items),
            ("max_items", parameter.max_items),
        ):
            if value is not None:
                details.append(f"{label}={value!r}")
        print(
            f"      {parameter.name}: {parameter.type.value} "
            f"({', '.join(details)})"
        )


heading("Constraints are checked before the function runs")

print(
    "  valid array:",
    tool_mod.invoke_tool_unchecked(
        agent,
        "summarize_readings",
        {"readings": [20.0, 21.0, 22.0]},
    ),
)
print(
    "  too cold  :",
    tool_mod.invoke_tool_unchecked(agent, "set_temperature", {"celsius": -400}),
)
print(
    "  bad label :",
    tool_mod.invoke_tool_unchecked(agent, "manage_sensor", {"sensor_id": "sensor-1"}),
)
print(
    "  too short :",
    tool_mod.invoke_tool_unchecked(agent, "summarize_readings", {"readings": [20.0]}),
)
