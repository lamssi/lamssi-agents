"""Budget checkpoints supplied by the optional budget feature."""

from __future__ import annotations

from lamssi_agents import (
    Agent,
    ApprovalPolicy,
    InteractionResponse,
    tool,
)
from lamssi_agents.features import Budget
from lamssi_agents.events import AgentEventType
from lamssi_agents.interaction import InteractionKind
from lamssi_agents.providers.models import StreamDelta, ToolCall, Usage
from lamssi_tools import Expose


@tool(expose=Expose.AGENT, approval="never")
def noop() -> dict:
    """Do nothing, so the loop has a reason to take another turn."""
    return {"ok": True}


class Spender:
    """Scripted provider that adds usage on each tool-calling turn."""

    model = name = "scripted"
    is_local = supports_tools = True
    reasoning_effort = None

    def __init__(self, per_call: int = 60_000) -> None:
        self.per_call = per_call
        self._usage = Usage()
        self.calls = 0

    def stream(self, messages, tools=None, **kw):
        self.calls += 1
        self._usage.total_tokens += self.per_call
        if self.calls < 5:
            yield StreamDelta(
                type="tool_call",
                tool_call=ToolCall(id=f"c{self.calls}", name="noop", arguments={}),
            )
            yield StreamDelta(type="done", finish_reason="tool_calls")
            return
        yield StreamDelta(type="text", text=f"turn {self.calls}")
        yield StreamDelta(type="done", finish_reason="stop")

    def set_abort_event(self, event):
        pass

    @property
    def cumulative_usage(self) -> Usage:
        return self._usage


def _agent(*features, answers=None, per_call=60_000):
    agent = Agent(
        model=Spender(per_call),
        features=list(features),
        tools=[noop],
        approval=ApprovalPolicy.allow_all(),
        max_turns=6,
    )
    if answers is not None:
        replies = list(answers)
        def interact(request):
            if request.kind is not InteractionKind.BUDGET_CHECKPOINT:
                return InteractionResponse.continue_()
            reply = replies.pop(0) if replies else ""
            if reply.lower() in {"yes", "y"}:
                return InteractionResponse.continue_()
            if reply:
                return InteractionResponse.cancel()
            return None
        agent.interaction = interact
    return agent


def test_without_the_feature_a_run_is_bounded_only_by_max_turns() -> None:
    """The loop itself no longer knows what a token costs."""
    agent = _agent()
    assert agent.chat("go") == "turn 5"
    assert not agent._before_turn, "nothing registered a per-turn policy"


def test_it_stops_when_the_user_declines() -> None:
    agent = _agent(Budget(every_tokens=50_000), answers=["no"])
    events = []
    agent.add_event_listener(lambda event: events.append(event.type))

    answer = agent.chat("go")

    assert "Stopped at about" in answer and "at your request" in answer
    assert agent.model.calls < 5, "it stopped mid-run rather than at the end"
    assert agent.history[-1].role == "assistant"
    assert agent.history[-1].content == answer
    assert AgentEventType.TEXT_DONE in events
    assert AgentEventType.TURN_END in events
    assert AgentEventType.DONE in events


def test_it_continues_when_the_user_agrees_and_asks_again_later() -> None:
    asked: list = []

    agent = _agent(Budget(every_tokens=50_000), per_call=60_000)
    agent.interaction = lambda request: (
        asked.append(request.prompt) or InteractionResponse.continue_()
    )

    agent.chat("go")

    assert len(asked) >= 2, f"asked once and then never again: {len(asked)}"
    assert "Continue?" in asked[0]


def test_silence_is_not_refusal() -> None:
    """An unattended budget checkpoint does not stop the run."""
    agent = _agent(Budget(every_tokens=50_000))   # no input callback at all

    answer = agent.chat("go")

    assert "Stopped at about" not in answer


def test_an_invalid_interaction_response_is_treated_as_unattended() -> None:
    agent = _agent(Budget(every_tokens=50_000), answers=[""])

    answer = agent.chat("go")

    assert "Stopped at about" not in answer


def test_the_budget_interval_is_owned_by_the_feature() -> None:
    agent = _agent(Budget(every_tokens=50_000), answers=["no"])

    assert "Stopped at about" in agent.chat("go")


def test_the_interval_is_read_live_so_it_can_be_changed_mid_conversation() -> None:
    budget = Budget()
    agent = _agent(budget, answers=["no"])
    assert agent.chat("go") == "turn 5", "0 disables it"

    budget.every_tokens = 50_000
    agent.model.calls = 0          # let the second message run a few turns
    assert "Stopped at about" in agent.chat("again")


def test_the_budget_is_per_message_not_per_agent() -> None:
    """Each chat call starts a new budget interval."""
    agent = _agent(Budget(every_tokens=50_000))
    agent.interaction = lambda request: InteractionResponse.continue_()

    agent.chat("first")
    agent.model.calls = 0
    asked: list = []
    agent.interaction = lambda request: (
        asked.append(request.prompt) or InteractionResponse.continue_()
    )
    agent.chat("second")

    assert asked, "the second message never reached its own first checkpoint"
    assert "across 2 request(s)" in asked[0] or "across 3 request(s)" in asked[0]
