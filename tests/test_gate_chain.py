"""The direct Agent lifecycle hooks and conversation state bag."""

from __future__ import annotations

from pathlib import Path

import pytest

from lamssi_agents import Agent, ApprovalPolicy, Feature, ToolApproval
from lamssi_agents import Files, Guidance, SystemTools
from lamssi_agents.tooling.guard import LoopGuard
from lamssi_agents.agent.conversation import Conversation
from lamssi_agents.providers import Message, ToolCall
from lamssi_agents.tooling import DEFAULT_POLICY
from lamssi_agents.tooling.dedupe import DedupeCache, DedupePolicy
from lamssi_tools import Expose, tool


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def agent(workspace: Path) -> Agent:
    return Agent(features=[SystemTools(), Guidance(), Files(workspace)])


def _run_tool_calls(agent, calls):
    from _scope import run_scope_active
    from lamssi_agents.events import AgentAborted

    with run_scope_active(agent):
        batch = agent._runtime.execute_calls(calls, agent._conversation.turn)
    agent._conversation.extend(batch.messages)
    if batch.aborted:
        raise AgentAborted()


def _call_read(agent: Agent) -> None:
    _run_tool_calls(
        agent, [ToolCall(id="1", name="read_file", arguments={"path": "a.txt"})]
    )


def test_the_feature_chain_contains_only_feature_hooks(agent):
    assert agent._before_tool == []


def test_a_tool_call_resolves_its_surface_once(agent, monkeypatch):
    original = agent._runtime.surface
    calls = 0

    def counted():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(agent._runtime, "surface", counted)
    _call_read(agent)

    assert calls == 1


def test_a_mistyped_safety_check_is_rejected_during_composition():
    @tool(expose=Expose.AGENT, approval="conditional")
    def guarded(path: str) -> dict:
        """Read one guarded path."""
        return {"path": path}

    def check(arguments):
        return bool(arguments.get("route"))

    check.argument_key = "route"

    class BrokenSafety(Feature):
        def install(self, agent):
            agent.add_tools(guarded)
            agent.safe_when["guarded"] = check

    with pytest.raises(ValueError, match="check reads 'route'"):
        Agent(features=[BrokenSafety()])


def test_feature_gates_run_in_install_order(workspace: Path):
    order = []

    class Gate(Feature):
        def __init__(self, label):
            self.label = label

        def before_tool(self, call, agent):
            order.append(self.label)

    target = Agent(
        features=[
            SystemTools(),
            Guidance(),
            Files(workspace),
            Gate("first"),
            Gate("second"),
        ]
    )
    _call_read(target)
    assert order == ["first", "second"]


def test_a_gate_that_blocks_stops_the_ones_after_it(workspace: Path):
    reached = []

    class Blocker(Feature):
        def before_tool(self, call, agent):
            return {"error": "no", "retriable": False}

    class Later(Feature):
        def before_tool(self, call, agent):
            reached.append(call.name)

    target = Agent(
        features=[SystemTools(), Guidance(), Files(workspace), Blocker(), Later()]
    )
    _call_read(target)
    assert reached == []
    assert "no" in (target._conversation.history[-1].content or "")


def test_a_raising_feature_gate_fails_closed_and_answers_the_call(workspace: Path):
    class BrokenSafetyGate(Feature):
        def before_tool(self, call, agent):
            raise RuntimeError("interlock connection failed")

    target = Agent(
        features=[SystemTools(), Guidance(), Files(workspace), BrokenSafetyGate()]
    )

    _call_read(target)

    result = target._conversation.history[-1]
    assert result.role == "tool"
    assert "safety gate failed" in (result.content or "")


def test_a_host_gate_can_refuse_what_approval_would_have_allowed(workspace: Path):
    class OfficeHours(Feature):
        def before_tool(self, call, agent):
            if call.name == "read_file":
                return {"error": "Outside office hours.", "retriable": False}
            return None

    target = Agent(
        features=[SystemTools(), Guidance(), Files(workspace), OfficeHours()]
    )
    _call_read(target)
    assert "Outside office hours" in (target._conversation.history[-1].content or "")


def test_feature_veto_happens_before_approval() -> None:
    approvals = []

    @tool(expose=Expose.AGENT, approval="always")
    def consequential() -> dict:
        """A call that normally requires approval."""
        return {"ran": True}

    class Blocker(Feature):
        def before_tool(self, call, agent):
            return {"error": "blocked by feature", "retriable": False}

    agent = Agent(
        tools=[consequential],
        features=[Blocker()],
        approval=ApprovalPolicy.ask_when_required(
            lambda request: approvals.append(request) or ToolApproval.APPROVE
        ),
    )

    _run_tool_calls(
        agent, [ToolCall(id="1", name="consequential", arguments={})]
    )

    assert approvals == []
    assert "blocked by feature" in (agent._conversation.history[-1].content or "")


def test_gate_order_is_dedupe_then_feature_then_guard_then_approval(
    workspace, monkeypatch
):
    """Dispatch preserves the defined gate order."""
    order = []

    def spy(label):
        def gate(call):
            order.append(label)
            return None

        return gate

    class Recorder(Feature):
        def before_tool(self, call, agent):
            order.append("feature")

    target = Agent(features=[SystemTools(), Guidance(), Files(workspace), Recorder()])
    monkeypatch.setattr(target._runtime, "_dedupe_gate", spy("dedupe"))
    monkeypatch.setattr(target._runtime, "_guard_gate", spy("guard"))
    monkeypatch.setattr(target._runtime, "_approval_gate", spy("approval"))
    _call_read(target)

    assert order == ["dedupe", "feature", "guard", "approval"]


def test_approval_and_execution_share_normalized_arguments() -> None:
    approved = []
    executed = []

    @tool(expose=Expose.AGENT, approval="always")
    def configure(count: int, scope: str = "all") -> dict:
        """Apply one configuration."""
        executed.append((count, scope))
        return {"count": count, "scope": scope}

    agent = Agent(
        tools=[configure],
        approval=ApprovalPolicy.ask_when_required(
            lambda request: (
                approved.append(dict(request.arguments)) or ToolApproval.APPROVE
            )
        ),
    )

    _run_tool_calls(
        agent,
        [ToolCall(id="1", name="configure", arguments={"count": "3"})],
    )

    assert approved == [{"count": 3, "scope": "all"}]
    assert executed == [(3, "all")]


def test_approval_handler_cannot_mutate_nested_execution_arguments() -> None:
    executed = []

    @tool(expose=Expose.AGENT, approval="always")
    def configure(values: list[int]) -> dict:
        """Apply a list of configuration values."""
        executed.append(list(values))
        return {"values": values}

    def approve(request):
        request.arguments["values"].append(99)
        return ToolApproval.APPROVE

    agent = Agent(
        tools=[configure],
        approval=ApprovalPolicy.ask_when_required(approve),
    )

    _run_tool_calls(
        agent,
        [ToolCall(id="1", name="configure", arguments={"values": [1, 2]})],
    )

    assert executed == [[1, 2]]


def test_invalid_boolean_is_rejected_before_approval() -> None:
    approvals = []
    executed = []

    @tool(expose=Expose.AGENT, approval="always")
    def toggle(enabled: bool) -> dict:
        """Set a boolean switch."""
        executed.append(enabled)
        return {"enabled": enabled}

    agent = Agent(
        tools=[toggle],
        approval=ApprovalPolicy.ask_when_required(
            lambda request: approvals.append(request) or ToolApproval.APPROVE
        ),
    )

    _run_tool_calls(
        agent,
        [ToolCall(id="1", name="toggle", arguments={"enabled": "definitely"})],
    )

    assert approvals == []
    assert executed == []
    assert "Invalid arguments" in (agent._conversation.history[-1].content or "")


def test_after_tool_hooks_see_the_result(workspace: Path):
    seen = []

    class Watcher(Feature):
        def after_tool(self, call, result, is_error, agent):
            seen.append((call.name, is_error))

    target = Agent(features=[SystemTools(), Guidance(), Files(workspace), Watcher()])
    _call_read(target)
    assert seen == [("read_file", False)]


def test_a_raising_hook_does_not_break_the_call(workspace: Path):
    class Broken(Feature):
        def after_tool(self, call, result, is_error, agent):
            raise RuntimeError("bookkeeping exploded")

    target = Agent(features=[SystemTools(), Guidance(), Files(workspace), Broken()])
    _call_read(target)
    assert "hello" in (target._conversation.history[-1].content or "")


def test_state_is_built_once_per_conversation():
    transcript = Conversation()
    first = transcript.state(DedupeCache, DedupeCache)
    assert transcript.state(DedupeCache, DedupeCache) is first


def test_the_runtime_owns_the_cache_and_guard():
    runtime = Agent()._runtime
    assert isinstance(runtime.dedupe, DedupeCache)
    assert isinstance(runtime.guard, LoopGuard)


def test_fs_dedupe_lets_a_bigger_max_lines_through():
    """Increasing ``max_lines`` creates a distinct filesystem request."""
    from lamssi_agents.features.files.feature import _DEDUPE

    policy = _DEDUPE["fs"]
    cache = DedupeCache()
    cache.record("fs", {"command": "ls", "max_lines": 10}, policy, turn=1)

    # a bigger cap would show more -> must run, not dedupe
    assert cache.check("fs", {"command": "ls", "max_lines": 50}, policy) is None
    # the truly identical call is still a repeat
    assert cache.check("fs", {"command": "ls", "max_lines": 10}, policy) is not None


def test_invalidation_matches_a_bare_element_host_signature():
    """Invalidation matches bare-element host signatures."""
    cache = DedupeCache()
    read = DedupePolicy(
        signature=lambda args: (args["path"],),  # bare element, not ("path", value)
        invalidated_by=frozenset({"write"}),
        invalidation_key=lambda args: args["path"],
    )
    policies = {"read": read, "write": DedupePolicy()}

    cache.record("read", {"path": "/x"}, read, turn=1)
    assert len(cache) == 1

    invalidated = cache.invalidate("write", {"path": "/x"}, policies)

    assert "read" in invalidated
    assert len(cache) == 0, "a bare-element read entry survived its invalidating write"


def test_a_summarisation_notifies_state_holders():
    cleared = []

    class Custom:
        def on_compacted(self):
            cleared.append("custom")

    transcript = Conversation()
    transcript.history = [Message(role="user", content="go")]
    transcript.state(Custom, Custom)
    assert transcript.on_compacted([]) is False  # a message folded away
    assert cleared == ["custom"]


def test_a_summarisation_clears_the_dedupe_cache():
    runtime = Agent()._runtime
    runtime.dedupe.record("read_file", {"path": "a"}, DEFAULT_POLICY, 1)
    runtime.on_history_compacted(demoted_only=False)
    assert len(runtime.dedupe) == 0


def test_a_demotion_is_a_different_event_from_a_summarisation():
    """Demotion does not emit the transcript-compacted lifecycle event."""
    seen = []

    class Custom:
        def on_compacted(self):
            seen.append("compacted")

        def on_demoted(self):
            seen.append("demoted")

    transcript = Conversation()
    transcript.history = [Message(role="user", content="go")]
    transcript.state(Custom, Custom)

    demoted = transcript.on_compacted(
        [Message(role="user", content="go (body rewritten)")]
    )
    assert demoted is True, "same roles in order means nothing was folded away"
    assert seen == ["demoted"]


def test_a_demotion_keeps_the_dedupe_cache():
    runtime = Agent()._runtime
    runtime.dedupe.record("read_file", {"path": "a"}, DEFAULT_POLICY, 1)
    runtime.on_history_compacted(demoted_only=True)
    assert len(runtime.dedupe) == 1, "the cache still describes a visible call"


def test_a_holder_that_raises_does_not_stop_the_others():
    survived = []

    class Angry:
        def on_cleared(self):
            raise RuntimeError("no")

    class Calm:
        def on_cleared(self):
            survived.append(1)

    transcript = Conversation()
    transcript.state(Angry, Angry)
    transcript.state(Calm, Calm)
    transcript.clear()
    assert survived == [1]


def test_a_new_turn_reaches_the_holders():
    turns = []

    class Counter:
        def on_new_turn(self):
            turns.append(1)

    transcript = Conversation()
    transcript.state(Counter, Counter)
    transcript.begin_request()
    transcript.begin_request()
    assert turns == [1, 1]
