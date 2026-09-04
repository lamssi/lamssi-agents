"""Keep simple host extension points callable-friendly."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest

from lamssi_agents import Agent, PromptPosition
from lamssi_agents.features.code import CodeExecutor
from lamssi_agents.features.system import AbortSink
from lamssi_tools import CapabilityContext


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    return tmp_path

# capabilities

def test_a_single_method_port_accepts_a_function(project: Path):
    stopped = []
    agent = Agent(
        capabilities={AbortSink: lambda: stopped.append(True)},
    )
    agent._capabilities.get(AbortSink).abort_all()

    assert stopped == [True]

def test_the_function_keeps_its_arguments(project: Path):
    """Pass every declared argument to an adapted callable."""
    @runtime_checkable
    class EditorSink(Protocol):
        def open_in_editor(self, code: str, title: str = "") -> bool: ...

    shown = []
    context = CapabilityContext({
        EditorSink: lambda code, title="": shown.append((code, title)) or True
    })
    adapted = context.require(EditorSink)
    adapted.open_in_editor("x = 1", "demo")

    assert shown == [("x = 1", "demo")]

def test_every_function_adaptable_port_is_covered(project: Path):
    """If a new one-member port appears, it gets this for free: or this fails."""
    adapted = CapabilityContext({AbortSink: lambda *a, **k: "ok"}).require(AbortSink)
    assert adapted.abort_all() == "ok"

def test_a_port_carrying_data_is_not_function_adaptable(project: Path):
    """Protocols with data attributes require an object implementation."""
    @runtime_checkable
    class StatusContributor(Protocol):
        section: str

        def snapshot(self) -> dict: ...

    fn = lambda: {}                                      # noqa: E731
    assert CapabilityContext({StatusContributor: fn}).require(StatusContributor) is fn
    assert "section" in StatusContributor.__annotations__

def test_a_real_implementation_is_never_rewrapped(project: Path):
    """A host class that also defines __call__ is not an invitation to wrap it."""
    class Real:
        def abort_all(self):
            return "real"

        def __call__(self):
            return "WRONG"

    assert CapabilityContext({AbortSink: Real()}).require(AbortSink).abort_all() == "real"

def test_a_multi_method_port_still_wants_an_object(project: Path):
    """Require an object when a protocol declares multiple methods."""
    fn = lambda: None                                    # noqa: E731
    assert CapabilityContext({CodeExecutor: fn}).require(CodeExecutor) is fn
                                                          # where it is used

# prompt sections

def test_a_context_block_wraps_a_function(project: Path):
    from lamssi_agents.prompt import ContextBlock

    agent = Agent()
    def live_state(ctx):
        return "Stage is at 12.5 mm."

    agent.add_context(ContextBlock("live-state", live_state))
    prompt = agent.assemble_prompt()

    assert "live-state" in dict(prompt.blocks)
    assert "Stage is at 12.5 mm." in prompt.text

def test_a_context_block_can_choose_its_position(project: Path):
    from lamssi_agents.prompt import ContextBlock

    agent = Agent()
    agent.add_context(
        ContextBlock(
            "live-state",
            lambda ctx: "from a factory",
            position=PromptPosition.LIVE,
        ),
    )
    prompt = agent.assemble_prompt()
    part = next(part for part in prompt.parts if part.name == "live-state")
    assert part.position is PromptPosition.LIVE
    assert "from a factory" in prompt.text

def test_a_failing_section_does_not_break_assembly(project: Path):
    """Isolate a failing prompt callback to its own block."""
    def boom(ctx):
        raise RuntimeError("nope")

    agent = Agent(instructions="base survives")
    from lamssi_agents.prompt import ContextBlock

    agent.add_context(ContextBlock("boom", boom))
    assert agent.assemble_prompt().text == "base survives"
