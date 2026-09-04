# SPDX-License-Identifier: MIT
"""12 - installing tools and observing their lifecycle with a Feature."""

from __future__ import annotations

from typing import Any

from _support import ScriptedModel, calls, heading, says

from lamssi_agents import Agent, Feature, tool
from lamssi_agents.events import AgentEvent, AgentEventType
from lamssi_agents.tooling import ToolInvocation
from lamssi_tools import Expose, Str


@tool(
    group="weather",
    expose=Expose.AGENT,
    approval="never",
    parameters={"city": Str("Which city.")},
)
def forecast(city: str = "") -> dict:
    """Return today's demonstration forecast for a city."""
    return {"city": city, "summary": "cloudy", "high_c": 17}


class Weather(Feature):
    """Install the forecast tool and record its lifecycle."""

    name = "weather"

    def __init__(self) -> None:
        self.records: list[str] = []

    def install(self, agent: Agent) -> None:
        """Add this feature's tools when the agent is created."""
        agent.add_tools(forecast)

    def before_turn(self, agent: Agent, turn: int) -> str | None:
        """Run before each model call; returning text would end the run."""
        self.records.append(f"before_turn turn={turn}")
        return None

    def before_tool(
        self, call: ToolInvocation, agent: Agent
    ) -> dict[str, Any] | None:
        """Inspect a call and optionally return its result without running it."""
        city = call.arguments.get("city", "")
        self.records.append(f"before_tool {call.name} city={city}")
        if city == "Atlantis":
            return {"error": "City is outside this demonstration."}
        return None

    def after_tool(
        self,
        call: ToolInvocation,
        result: Any,
        is_error: bool,
        agent: Agent,
    ) -> None:
        """Observe a tool after it has actually run."""
        self.records.append(f"after_tool {call.name} error={is_error}")

    def on_event(self, event: AgentEvent) -> None:
        """Observe selected lifecycle events without changing the run."""
        visible_events = {
            AgentEventType.TURN_START,
            AgentEventType.TOOL_START,
            AgentEventType.TOOL_RESULT,
            AgentEventType.DONE,
        }
        if event.type in visible_events:
            self.records.append(f"event {event.type.value}")


heading("Feature lifecycle hooks")

weather = Weather()
agent = Agent(
    model=ScriptedModel(
        calls("forecast", city="Atlantis"),
        calls("forecast", city="Bucharest"),
        says("The Bucharest forecast is cloudy."),
    ),
    features=[weather],
)

result = agent.run("Check Atlantis, then Bucharest.")
print("  result:", result.text)
print("  hook and event order:")
for record in weather.records:
    print("   ", record)

print("  note: after_tool appears only for Bucharest because Atlantis was blocked")
