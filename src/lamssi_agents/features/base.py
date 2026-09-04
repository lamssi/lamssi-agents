"""The one explicit extension seam for optional agent functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lamssi_agents.agent.base import Agent
    from lamssi_agents.events import AgentEvent
    from lamssi_agents.tooling.invocation import ToolInvocation


class Feature:
    """Functionality installed explicitly with ``Agent(features=[...])``.

    ``install`` contributes tools, capabilities, or context. Runtime hooks
    participate in a run. Feature implementations never need to call
    ``super()``.

    Example:
        Package one application capability and tool together::

            class ApplicationFeature(Feature):
                name = "application"

                def install(self, agent):
                    agent.provide(ApplicationApi, self.api)
                    agent.add_tools(inspect_application)

    Note:
        Feature list order is hook order. Return a tool-result dictionary from
        :meth:`before_tool` to block a call safely; raising is treated as a
        fail-closed feature error.
    """

    name = ""

    def install(self, agent: Agent) -> None:
        """Contribute functionality when the feature is installed.

        Args:
            agent: Agent receiving this feature's tools, capabilities, context,
                and configuration.
        """

    def before_turn(self, agent: Agent, turn: int) -> str | None:
        """Optionally stop before a model turn.

        Args:
            agent: Agent about to call its model.
            turn: One-based turn number for the current request.

        Returns:
            Final response text to end the run, or ``None`` to continue.
        """
        return None

    def before_tool(self, call: ToolInvocation, agent: Agent) -> dict[str, Any] | None:
        """Optionally block a validated tool call before approval and execution.

        Args:
            call: Prepared invocation containing the name and normalized arguments.
            agent: Agent processing the call.

        Returns:
            Tool-result dictionary explaining the refusal, or ``None`` to allow
            the remaining safety and approval gates to run.
        """
        return None

    def after_tool(
        self, call: ToolInvocation, result: Any, is_error: bool, agent: Agent
    ) -> None:
        """Observe a completed tool call.

        Args:
            call: Invocation that was attempted.
            result: Final value appended as the tool result.
            is_error: Whether the result represents an execution error.
            agent: Agent that processed the call.
        """

    def on_event(self, event: AgentEvent) -> None:
        """Observe one event emitted by an agent using this feature.

        Args:
            event: Immutable event value including type, data, and metadata.
        """


__all__ = ["Feature"]
