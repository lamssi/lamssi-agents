"""The composition-mutation contract while a run is active.

Compose-only mutation raises during a run; a small live-safe set stays open.
"""

from __future__ import annotations

import threading

import pytest

from lamssi_agents import Agent, ApprovalPolicy, Feature, tool
from lamssi_agents.providers import StreamDelta, ToolCall
from lamssi_tools import CapabilityContext, Expose, ToolRegistry


@tool(expose=Expose.AGENT, approval="never")
def probe() -> dict:
    """A trivial tool used as registration and mount fodder."""
    return {"ok": True}


class _NoopFeature(Feature):
    name = "noop"

    def install(self, agent) -> None:
        pass


#: Mutators that touch state a run reads with no synchronization.
COMPOSE_ONLY = {
    "use": lambda a: a.use(_NoopFeature()),
    "add_tools": lambda a: a.add_tools(probe),
    "provide": lambda a: a.provide(int, 1),
    "set_tool_dispatcher": lambda a: a.set_tool_dispatcher(lambda d, f, k: f(**k)),
    "add_dedupe_policies": lambda a: a.add_dedupe_policies({}),
    "add_safe_when": lambda a: a.add_safe_when({"probe": lambda args: True}),
    "add_guard_roles": lambda a: a.add_guard_roles({"probe": "repeatable"}),
}


@pytest.mark.parametrize("name", sorted(COMPOSE_ONLY))
def test_compose_only_mutation_is_rejected_during_a_run(name: str):
    """Every compose-only mutator raises while a run holds the agent."""
    agent = Agent()
    with agent._control.enter_run():
        with pytest.raises(RuntimeError, match="during a run"):
            COMPOSE_ONLY[name](agent)


@pytest.mark.parametrize("name", sorted(COMPOSE_ONLY))
def test_compose_only_mutation_is_allowed_between_runs(name: str):
    """The same mutators work when no run is active."""
    COMPOSE_ONLY[name](Agent())


def test_mounting_stays_live_during_a_run():
    """A host may mount and unmount an external registry mid-run."""
    agent = Agent()
    external = ToolRegistry()
    with agent._control.enter_run():
        assert agent.mount_tools(external) is agent
        external.add_one(probe._tool_definition, probe, owner="app")
        assert "probe" in agent.available_tool_names()
        assert agent.unmount_tools(external)


def test_tool_access_model_and_abort_stay_live_during_a_run():
    """Disable, enable, model swap, and abort stay available while a run is active."""
    agent = Agent(tools=[probe])
    with agent._control.enter_run():
        agent.disable_tool("probe")
        assert "probe" in agent.disabled_tool_names()
        agent.enable_tool("probe")
        agent.use_model("test-model")
        assert agent.model is not None
        agent.abort()
        assert agent.is_aborted


def test_provide_during_a_live_run_is_rejected():
    """A concurrent provide() during a real run raises instead of racing the ctx."""
    started = threading.Event()
    release = threading.Event()

    @tool(expose=Expose.AGENT, approval="never", inject_context=True)
    def blocker(ctx: CapabilityContext) -> dict:
        ctx.get(int)  # a live capability read, concurrent with the provide below
        started.set()
        release.wait(timeout=5)
        return {"ok": True}

    class _CallBlocker:
        model = "call-blocker"

        def __init__(self) -> None:
            self.turn = 0

        def stream(self, messages, tools=None, **kwargs):
            self.turn += 1
            if self.turn == 1:
                yield StreamDelta(
                    type="tool_call",
                    tool_call=ToolCall(id="1", name="blocker", arguments={}),
                )
                yield StreamDelta(type="done", finish_reason="tool_calls")
                return
            yield StreamDelta(type="text", text="done")
            yield StreamDelta(type="done", finish_reason="stop")

    agent = Agent(
        model=_CallBlocker(), tools=[blocker], approval=ApprovalPolicy.allow_all()
    )
    worker = threading.Thread(target=lambda: agent.chat("go"))
    worker.start()
    try:
        assert started.wait(timeout=5), "the run never reached the blocking tool"
        with pytest.raises(RuntimeError, match="during a run"):
            agent.provide(int, 1)
    finally:
        release.set()
        worker.join(timeout=5)
