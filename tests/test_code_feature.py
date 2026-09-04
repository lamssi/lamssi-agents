"""Installation and capability requirements for the ``Code`` feature."""

from __future__ import annotations

from pathlib import Path

from lamssi_agents import Agent, Code, Files, Guidance, SystemTools
from lamssi_agents.features.code import CodeExecutor


class _Executor:
    """Enough of a CodeExecutor to be registered and asked for its variables."""

    def __init__(self) -> None:
        self._vars: dict = {}

    def run(self, code: str):
        exec(code, self._vars)  # noqa: S102 - a stand-in, not a sandbox
        return type("R", (), {"ok": True, "stdout": "", "stderr": ""})()

    def variables(self) -> dict:
        return {k: v for k, v in self._vars.items() if not k.startswith("__")}


def _names(agent: Agent) -> set:
    return {t.name for t in agent._tools.list_tools()}


# the kernel ships no interpreter tool

def test_files_no_longer_grants_code_execution(tmp_path: Path):
    agent = Agent(features=[SystemTools(), Guidance(), Files(tmp_path)])
    assert "execute_code" not in _names(agent)

# the feature installs it, the capability makes it visible

def test_the_feature_registers_the_tool(tmp_path: Path):
    agent = Agent(features=[SystemTools(), Guidance(), Code()])
    assert agent._tools.get_tool("execute_code") is not None

def test_registered_but_withheld_without_an_executor(tmp_path: Path):
    """Code execution stays registered but hidden without an executor."""
    agent = Agent(features=[SystemTools(), Guidance(), Code()])
    assert agent._tools.get_tool("execute_code") is not None
    assert "execute_code" not in {d.name for d in agent.visible_tool_defs()}

def test_passing_the_executor_to_the_feature_is_enough(tmp_path: Path):
    """One call instead of installing the tool and providing the capability apart."""
    agent = Agent(features=[SystemTools(), Guidance(), Code(_Executor())])
    assert "execute_code" in {d.name for d in agent.visible_tool_defs()}

def test_a_separately_provided_executor_works_too(tmp_path: Path):
    """A host that registers the capability its own way is not forced through the ctor."""
    agent = Agent(features=[SystemTools(), Guidance(), Code()], capabilities={CodeExecutor: _Executor()})
    assert "execute_code" in {d.name for d in agent.visible_tool_defs()}

def test_the_executor_reaches_the_tool(tmp_path: Path):
    """End to end, because registering a capability nobody reads would still pass above."""
    from lamssi_agents.features.code import execute_code

    agent = Agent(features=[SystemTools(), Guidance(), Code(_Executor())])
    out = execute_code(agent._capabilities, code="answer = 6 * 7\nprint('hi')")

    assert out.get("executed") is True
    assert "hi" in out.get("stdout", "")
    assert out.get("variables", {}).get("answer") == 42

def test_it_still_requires_approval():
    from lamssi_agents.features.code import execute_code

    assert execute_code._tool_definition.approval == "always"
