"""Live external tool registries without exposing Agent registry ownership."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import pytest

from lamssi_agents import Agent, ApprovalPolicy, ToolApproval, tool
from lamssi_agents import tool_runtime as agent_tools
from lamssi_agents.providers import ToolCall
from lamssi_tools import ToolNameConflictError
from lamssi_tools import CapabilityContext, Expose, ModuleSource, ToolRegistry

from _models import ScriptedModel, calls, says


def _definition(fn):
    return fn._tool_definition


@tool(expose=Expose.AGENT, approval="never")
def app_ping(value: str = "ping") -> dict:
    """Return an application-owned value."""
    return {"value": value}


def test_a_mounted_registry_is_live_and_unmountable() -> None:
    external = ToolRegistry()
    agent = Agent().mount_tools(external)

    assert agent.mount_tools(external) is agent, "mounting is idempotent by identity"
    assert "app_ping" not in agent.available_tool_names()

    external.add_one(_definition(app_ping), app_ping, owner="app:one")
    assert "app_ping" in agent.available_tool_names()
    assert agent.available_tool_names().count("app_ping") == 1
    assert agent_tools.invoke_tool_unchecked(agent, "app_ping", {"value": "live"}) == {
        "value": "live"
    }

    external.remove_owner("app:one")
    assert "app_ping" not in agent.available_tool_names()

    external.add_one(_definition(app_ping), app_ping, owner="app:two")
    assert agent.unmount_tools(external)
    assert not agent.unmount_tools(external)
    assert "app_ping" not in agent.available_tool_names()
    assert external.has_tool("app_ping"), "unmounting must not mutate the host registry"


def test_tool_sources_is_the_constructor_form_of_mounting() -> None:
    external = ToolRegistry()
    agent = Agent(tool_sources=[external])

    external.add_one(_definition(app_ping), app_ping, owner="app")

    assert "app_ping" in agent.available_tool_names()
    assert agent.unmount_tools(external)


def test_duplicate_local_tool_registration_fails_explicitly() -> None:
    agent = Agent(tools=[app_ping])

    with pytest.raises(ValueError, match="already registered.*app_ping"):
        agent.add_tools(app_ping)


def test_mounted_execution_uses_the_external_dispatcher() -> None:
    dispatched: list[str] = []

    def dispatch(definition, fn, kwargs):
        dispatched.append(definition.name)
        return fn(**kwargs)

    external = ToolRegistry(dispatcher=dispatch)
    external.add_one(_definition(app_ping), app_ping, owner="app")
    agent = Agent().mount_tools(external)

    assert agent_tools.invoke_tool_unchecked(agent, "app_ping", {}) == {"value": "ping"}
    assert dispatched == ["app_ping"]


def test_mounted_tools_keep_the_registry_owners_capabilities() -> None:
    class Label:
        def __init__(self, value: str) -> None:
            self.value = value

    @tool(
        name="external_label",
        inject_context=True,
        expose=Expose.AGENT,
        approval="never",
    )
    def external_label(ctx: CapabilityContext) -> str:
        """Return the label owned by this registry."""
        return ctx.require(Label).value

    module = ModuleType("external_context_tools")
    module.external_label = external_label
    external = ToolRegistry()
    external.add(
        ModuleSource(
            module,
            context=CapabilityContext({Label: Label("external")}),
        )
    )
    agent = Agent(capabilities={Label: Label("agent")}).mount_tools(external)

    assert agent_tools.invoke_tool_unchecked(agent, "external_label", {}) == "external"


def test_registry_listeners_run_after_the_registry_lock_is_released() -> None:
    registry = ToolRegistry()
    callback_reads: list[bool] = []

    def registry_is_readable_from_another_thread() -> None:
        finished = threading.Event()

        def read_registry() -> None:
            registry.names()
            finished.set()

        threading.Thread(target=read_registry, daemon=True).start()
        callback_reads.append(finished.wait(timeout=1))

    def on_add(name, definition, fn, owner) -> None:
        registry_is_readable_from_another_thread()

    def on_remove(name, owner) -> None:
        registry_is_readable_from_another_thread()

    registry.add_listener(on_add, on_remove)
    registry.add_one(_definition(app_ping), app_ping, owner="app")
    registry.remove_owner("app")

    assert callback_reads == [True, True]


@tool(
    name="observe_context",
    expose=Expose.AGENT,
    approval="never",
    inject_context=True,
    dispatch="worker",
)
def observe_context(ctx: CapabilityContext) -> dict:
    """Return the run scope and executing thread."""
    from lamssi_agents.runtime import RunScope

    scope = ctx.get(RunScope)
    return {
        "agent": scope.agent if scope is not None else None,
        "session": scope.agent._control if scope is not None else None,
        "thread": threading.current_thread().name,
    }


def test_dispatcher_receives_a_context_bound_callable() -> None:
    """The run scope survives a worker-thread handoff for an owned tool."""
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="app-worker",
    ) as worker:

        def dispatch(definition, fn, kwargs):
            return worker.submit(fn, **kwargs).result(timeout=5)

        agent = Agent()
        agent.set_tool_dispatcher(dispatch)
        agent.add_tools(observe_context)

        result = agent_tools.invoke_tool_unchecked(agent, "observe_context", {})

    assert result["agent"] is agent
    assert result["session"] is agent._control
    assert result["thread"].startswith("app-worker")


@tool(expose=Expose.AGENT, approval="always")
def dangerous_app_tool() -> dict:
    """A mounted operation that must pass Agent approval first."""
    return {"ran": True}


def test_agent_approval_runs_before_a_mounted_registry_dispatcher() -> None:
    dispatched: list[str] = []

    def dispatch(definition, fn, kwargs):
        dispatched.append(definition.name)
        return fn(**kwargs)

    external = ToolRegistry(dispatcher=dispatch)
    external.add_one(_definition(dangerous_app_tool), dangerous_app_tool, owner="app")
    agent = Agent(
        model=ScriptedModel(
            calls(("external-1", "dangerous_app_tool", {})), says("blocked safely")
        )
    ).mount_tools(external)

    result = agent.run("run the dangerous app tool")

    assert result.text == "blocked safely"
    assert dispatched == []
    assert any(
        message.role == "tool" and "needs approval" in (message.content or "")
        for message in agent.history
    )


@tool(name="same_name", expose=Expose.AGENT, approval="never")
def local_same_name() -> str:
    """Local collision fixture."""
    return "local"


@tool(name="same_name", expose=Expose.AGENT, approval="never")
def external_same_name() -> str:
    """External collision fixture."""
    return "external"


def test_an_existing_name_conflict_rejects_the_mount() -> None:
    external = ToolRegistry()
    external.add_one(_definition(external_same_name), external_same_name, owner="app")
    agent = Agent(tools=[local_same_name])

    with pytest.raises(ToolNameConflictError, match="same_name"):
        agent.mount_tools(external)

    assert agent.available_tool_names() == ["same_name"]


def test_a_conflict_added_later_fails_at_live_surface_resolution() -> None:
    external = ToolRegistry()
    agent = Agent(tools=[local_same_name]).mount_tools(external)

    external.add_one(_definition(external_same_name), external_same_name, owner="app")

    with pytest.raises(ToolNameConflictError, match="same_name"):
        agent.available_tool_names()

    external.remove_owner("app")
    assert agent.available_tool_names() == ["same_name"]


def test_unchecked_invocation_returns_a_conflict_error_instead_of_raising() -> None:
    external = ToolRegistry()
    agent = Agent(tools=[local_same_name]).mount_tools(external)
    external.add_one(_definition(external_same_name), external_same_name, owner="app")

    result = agent_tools.invoke_tool_unchecked(agent, "same_name", {})

    assert "Tool name conflict" in result["error"]


def test_registry_reload_is_safe_while_the_agent_reads_the_live_surface() -> None:
    external = ToolRegistry()
    agent = Agent().mount_tools(external)
    failures: list[Exception] = []

    def reload_apps() -> None:
        try:
            for _ in range(250):
                external.add_one(_definition(app_ping), app_ping, owner="app")
                external.remove_owner("app")
        except Exception as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=reload_apps)
    worker.start()
    while worker.is_alive():
        agent.available_tool_names()
    worker.join()

    assert failures == []


def test_registry_replacement_during_approval_does_not_change_execution() -> None:
    ran: list[str] = []

    @tool(name="replaceable", expose=Expose.AGENT, approval="always")
    def old_tool() -> str:
        """The implementation the user was asked to approve."""
        ran.append("old")
        return "old"

    @tool(name="replaceable", expose=Expose.AGENT, approval="always")
    def new_tool() -> str:
        """A replacement loaded while approval is pending."""
        ran.append("new")
        return "new"

    external = ToolRegistry()
    external.add_one(_definition(old_tool), old_tool, owner="app:old")

    def replace_then_approve(request):
        external.remove_owner("app:old")
        external.add_one(_definition(new_tool), new_tool, owner="app:new")
        return ToolApproval.APPROVE

    agent = Agent(
        approval=ApprovalPolicy.ask_when_required(replace_then_approve)
    ).mount_tools(external)

    from _scope import run_scope_active

    with run_scope_active(agent):
        batch = agent._runtime.execute_calls(
            [ToolCall(id="replace-1", name="replaceable", arguments={})],
            agent._conversation.turn,
        )
    agent._conversation.extend(batch.messages)

    assert ran == []
    assert "registry changed" in (agent._conversation.history[-1].content or "")
