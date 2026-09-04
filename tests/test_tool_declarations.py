"""``Agent.add_tools(*callables)``: declaring tools directly."""

from __future__ import annotations

import inspect
from typing import Annotated

import pytest

from lamssi_agents import Agent
from lamssi_tools import (
    Array,
    Bool,
    CapabilityContext,
    CallableSource,
    Expose,
    Int,
    Param,
    Str,
    ToolRegistry,
    tool,
)

PositiveInt = Annotated[int, Param("Positive value.", gt=0)]


def _closure_tool(dep):
    @tool(expose=Expose.ALL, description="returns the value it was built over")
    def held(**_):
        """A tool closed over a live object."""
        return {"dep": dep}

    return held


def test_a_decorated_closure_is_registered_and_carries_its_capture():
    agent = Agent().add_tools(_closure_tool(42))
    names = {t.name for t in agent._tools.list_tools()}
    assert "held" in names


def test_a_plain_function_is_decorated_on_the_way_in():
    def echo(text: str = ""):
        """Echo the text back."""
        return {"echo": text}

    agent = Agent().add_tools(echo)
    assert "echo" in {t.name for t in agent._tools.list_tools()}


def test_parameter_helpers_preserve_a_normal_signature_and_build_constraints():
    @tool(
        expose=Expose.AGENT,
        parameters={
            "query": Str("Text to find.", min_length=1),
            "scores": Array("Scores to inspect.", min_items=1),
            "limit": Int("Maximum results.", ge=1, le=100),
            "exact": Bool("Require an exact match."),
        },
    )
    def search(
        query: str,
        scores: list[float],
        limit: int = 10,
        exact: bool = False,
    ) -> dict:
        return {"query": query, "scores": scores, "limit": limit, "exact": exact}

    signature = inspect.signature(search)
    assert signature.parameters["query"].annotation == "str"
    assert signature.parameters["query"].default is inspect.Parameter.empty
    assert signature.parameters["limit"].default == 10
    assert search("heat", [1.5]) == {
        "query": "heat",
        "scores": [1.5],
        "limit": 10,
        "exact": False,
    }

    parameters = {p.name: p for p in search._tool_definition.parameters}
    assert parameters["query"].min_length == 1
    assert parameters["scores"].items.type.value == "number"
    assert parameters["scores"].min_items == 1
    assert parameters["limit"].minimum == 1
    assert parameters["limit"].maximum == 100
    assert parameters["exact"].default is False


def test_parameter_helpers_reject_names_and_types_that_can_drift():
    with pytest.raises(ValueError, match="unknown argument.*typo"):

        @tool(parameters={"typo": Str("Misspelled.")})
        def unknown(query: str):
            return query

    with pytest.raises(TypeError, match="annotated integer.*describes string"):

        @tool(parameters={"limit": Str("Wrong helper.")})
        def mismatched(limit: int):
            return limit

    def old_style(query):
        return query

    old_style.__annotations__["query"] = Str("Old annotation form.")
    with pytest.raises(TypeError, match="ordinary Python type.*parameters="):
        tool(old_style)


def test_context_injection_requires_a_positional_context_parameter():
    with pytest.raises(TypeError, match="has no context parameter"):

        @tool(inject_context=True)
        def missing_context():
            return None

    with pytest.raises(TypeError, match="must accept a positional value"):

        @tool(inject_context=True)
        def keyword_context(*, ctx):
            return ctx


def test_model_arguments_cannot_be_positional_only():
    with pytest.raises(TypeError, match="positional-only model argument.*value"):

        @tool
        def positional(value, /):
            return value


def test_injected_context_itself_may_be_positional_only():
    @tool(inject_context=True)
    def contextual(ctx, /, value: int):
        return {"ctx": ctx is not None, "value": value}

    registry = ToolRegistry()
    registry.add(
        CallableSource(contextual, context=CapabilityContext())
    )

    assert registry.execute("contextual", {"value": 3}) == {
        "ctx": True,
        "value": 3,
    }


def test_annotated_param_remains_the_advanced_alias_path():
    @tool
    def inspect_value(value: PositiveInt) -> int:
        return value

    parameter = inspect_value._tool_definition.parameters[0]
    assert parameter.type.value == "integer"
    assert parameter.exclusive_minimum == 0


def test_callable_source_registers_functions_without_a_synthetic_module():
    probe = _closure_tool("direct")
    registry = ToolRegistry()

    registry.add(CallableSource(probe))

    assert registry.execute("held", {}) == {"dep": "direct"}


def test_registry_add_is_transactional_when_schema_building_fails(monkeypatch):
    @tool
    def good(value: int):
        return value

    @tool
    def bad(value: int):
        return value

    definition_type = type(good._tool_definition)
    original = definition_type.build_args_model

    def build(definition):
        if definition.name == "bad":
            raise RuntimeError("invalid schema")
        return original(definition)

    monkeypatch.setattr(definition_type, "build_args_model", build)
    registry = ToolRegistry()

    with pytest.raises(RuntimeError, match="invalid schema"):
        registry.add(CallableSource(good, bad), owner="bundle")

    assert registry.registered_tool_defs == []
    assert registry.remove_owner("bundle") == []


def test_two_agents_do_not_share_tools():
    a = Agent().add_tools(_closure_tool("a"))
    b = Agent()
    assert "held" in {t.name for t in a._tools.list_tools()}
    assert "held" not in {t.name for t in b._tools.list_tools()}


def test_a_non_callable_is_rejected():
    import pytest

    with pytest.raises(TypeError):
        Agent().add_tools(object())


def test_no_tools_is_a_no_op():
    agent = Agent().add_tools()
    assert agent.available_tool_names() == []


# expose_tool: per-agent control over which surface a tool reaches


@tool(expose=Expose.AGENT | Expose.MCP, description="a shared, module-level tool")
def shared_probe(**_):
    """Registered into more than one agent, to prove exposure edits stay local."""
    return {"ok": True}


def _agent_sees(agent, name):
    return name in agent.available_tool_names()


def _mcp_sees(agent, name):
    return name in {m["name"] for m in agent._registry.get_all_tool_metas()}


def test_expose_tool_keeps_a_tool_off_mcp():
    agent = Agent().add_tools(shared_probe)
    assert _agent_sees(agent, "shared_probe") and _mcp_sees(agent, "shared_probe")

    agent.expose_tool("shared_probe", Expose.AGENT)
    assert _agent_sees(agent, "shared_probe")
    assert not _mcp_sees(agent, "shared_probe")


def test_expose_tool_can_hide_from_the_model_and_reveal_again():
    agent = Agent().add_tools(shared_probe)

    agent.expose_tool("shared_probe", Expose.HOST)  # off both surfaces
    assert not _agent_sees(agent, "shared_probe")
    assert not _mcp_sees(agent, "shared_probe")

    agent.expose_tool("shared_probe", Expose.AGENT)
    assert _agent_sees(agent, "shared_probe")


def test_expose_tool_does_not_leak_to_another_agent():
    a = Agent().add_tools(shared_probe)
    b = Agent().add_tools(shared_probe)

    a.expose_tool("shared_probe", Expose.HOST)

    assert not _agent_sees(a, "shared_probe")
    assert _agent_sees(b, "shared_probe"), "exposure edit leaked to another agent"
    # The shared @tool declaration itself is untouched.
    assert shared_probe._tool_definition.expose_to_agent is True
    assert shared_probe._tool_definition.expose_to_mcp is True


def test_expose_tool_unknown_name_raises():
    import pytest

    with pytest.raises(ValueError):
        Agent().add_tools(shared_probe).expose_tool("nope", Expose.AGENT)


def test_expose_tool_is_chainable():
    agent = Agent().add_tools(shared_probe).expose_tool("shared_probe", Expose.AGENT)
    assert isinstance(agent, Agent)
