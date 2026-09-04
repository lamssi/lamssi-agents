"""Ownership and isolation tests for Agent runtime state."""

from __future__ import annotations

from pathlib import Path

import pytest

from lamssi_agents import (
    Agent,
    ApprovalPolicy,
    Feature,
    Files,
    Guidance,
    Memory,
    Skills,
    SystemTools,
)
from lamssi_agents.agent import RunControl, Conversation
from lamssi_agents import tool_runtime as tool_mod
from lamssi_agents.approval import should_require_approval
from lamssi_agents.features.skills import Skill, SkillRuntime
from lamssi_agents.providers import StreamDelta, Usage
from lamssi_tools import CapabilityContext, Expose, tool

from _models import ScriptedModel, calls, says


class Greeter:
    def greet(self, name: str) -> str:
        raise NotImplementedError


class FriendlyGreeter:
    def greet(self, name: str) -> str:
        return f"hello {name}"


@tool(inject_context=True, expose=Expose.AGENT, approval="never")
def greet(ctx: CapabilityContext, name: str = "world") -> dict:
    """Greet someone through the installed capability."""
    return {"message": ctx.require(Greeter).greet(name)}


class GreetingFeature(Feature):
    name = "greeting"

    def install(self, agent):
        agent.provide(Greeter, FriendlyGreeter())
        agent.add_tools(greet)


class RefusingFeature(Feature):
    """Overrides a hook and nothing else: install() is not even defined."""

    name = "refusing"
    blocked: list = []

    def before_tool(self, call, agent):
        RefusingFeature.blocked.append(call.name)
        return {"error": "refused by policy", "retriable": False}


def test_agent_keeps_runtime_machinery_private() -> None:
    agent = Agent()
    second = Agent()

    for public_name in ("session", "transcript", "tools", "capabilities", "registry"):
        assert not hasattr(agent, public_name)
    assert isinstance(agent._control, RunControl)
    assert isinstance(agent._conversation, Conversation)
    assert agent._control is not second._control
    assert agent._conversation is not second._conversation


def test_one_agent_rejects_overlapping_runs() -> None:
    agent = Agent()
    assert agent._run_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            agent.run("overlap")
    finally:
        agent._run_lock.release()


def test_feature_contributes_a_tool_and_typed_capability() -> None:
    feature = GreetingFeature()
    agent = Agent(features=[feature])

    assert agent.features == [feature]
    assert "greet" in agent.available_tool_names()
    assert agent._capabilities.require(Greeter).greet("Ada") == "hello Ada"
    assert agent._registry.execute("greet", {"name": "Ada"}) == {"message": "hello Ada"}


def test_builtin_features_are_explicit_features(tmp_path) -> None:
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    bare = Agent()
    equipped = Agent(
        features=[SystemTools(), Guidance(), Files(tmp_path), Memory()],
        approval=ApprovalPolicy.allow_all(),
    )

    assert not bare._tools.list_tools()
    assert {"read_file", "write_file", "memory"} <= {
        definition.name for definition in equipped._tools.list_tools()
    }
    result = equipped._tools.execute_binding(
        equipped._tools.resolve("read_file", {"path": "note.txt"})
    )
    assert result["content"] == "hello"


def test_memory_reads_are_free_but_mutations_follow_approval(tmp_path) -> None:
    agent = Agent(
        features=[Memory(tmp_path / "memory")],
        approval=ApprovalPolicy.ask_when_required(lambda request: None),
    )

    assert not should_require_approval(agent, "memory", {"action": "list"})
    assert not should_require_approval(
        agent, "memory", {"action": "recall", "name": "project"}
    )
    assert should_require_approval(
        agent, "memory", {"action": "remember", "name": "project"}
    )
    assert should_require_approval(
        agent, "memory", {"action": "forget", "name": "project"}
    )


def test_memory_catalog_names_what_the_model_can_recall(tmp_path) -> None:
    from lamssi_agents.features.memory import MemoryStore

    path = tmp_path / "memory"
    MemoryStore(path).save(
        "project",
        "The private body",
        type="project",
        description="Current project decisions.",
    )

    agent = Agent(features=[Memory(path)])
    prompt = agent.assemble_prompt()

    assert "memory-catalog" in {part.name for part in prompt.parts}
    assert "`project`" in prompt.text
    assert "Current project decisions." in prompt.text
    assert "The private body" not in prompt.text


def test_memory_names_must_match_the_whole_value(tmp_path) -> None:
    from lamssi_agents.features.memory import MemoryStore

    with pytest.raises(ValueError, match="Invalid memory name"):
        MemoryStore(tmp_path).save("note\n", "body")


def test_agent_drives_the_loop_without_a_runtime_container() -> None:
    agent = Agent(
        model=ScriptedModel(calls(("call-1", "greet", {"name": "Ada"})), says("done")),
        features=[GreetingFeature()],
        approval=ApprovalPolicy.allow_all(),
    )

    assert agent.chat("say hello") == "done"
    assert any(message.role == "tool" for message in agent._conversation.history)


def test_non_features_are_rejected_plainly() -> None:
    with pytest.raises(TypeError, match="inherit Feature"):
        Agent(features=[object()])  # type: ignore[list-item]


def test_a_hook_is_registered_even_when_install_is_overridden() -> None:
    """Feature hooks register even when ``install`` is overridden."""

    class Both(RefusingFeature):
        name = "both"

        def install(self, agent):
            agent.add_tools(greet)

    agent = Agent(features=[Both()])

    assert len(agent._before_tool) == 1, (
        "the feature's before_tool was not registered"
    )




# the skill catalog offers only the routes that exist


def _catalog(agent) -> str:
    prompt = agent.build_system_prompt()
    start = prompt.find("## Available Skills")
    return prompt[start:] if start >= 0 else ""


def _skilled(tmp_path, **kw):
    from lamssi_agents import Skills

    root = tmp_path / "skills" / "surveying"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: surveying\ndescription: Survey a tree.\n---\nWalk it, then report.\n",
        encoding="utf-8",
    )
    return Agent(
        features=[
            SystemTools(),
            Guidance(),
            Files(tmp_path),
            Skills(tmp_path / "skills", allow_model_loading=True),
        ],
        **kw,
    )


def test_the_catalog_gives_a_path_so_a_skill_can_be_read_rather_than_pinned(tmp_path):
    """Route A: the model fetches the body itself."""
    agent = _skilled(tmp_path, approval=ApprovalPolicy.allow_all())

    catalog = _catalog(agent)
    path = next(
        l.strip() for l in catalog.splitlines() if l.strip().endswith("SKILL.md")
    )

    assert Path(path).is_file(), "the catalog advertised a path that does not resolve"
    result = tool_mod.invoke_tool_unchecked(agent, "read_file", {"path": path})
    assert "Walk it, then report." in result["content"]


def test_the_catalog_advertises_only_the_routes_installed(tmp_path):
    """Naming a route the host did not install teaches the model a dead end."""
    agent = _skilled(tmp_path)
    both = _catalog(agent)
    assert "load_skill" in both and "read_file" in both

    agent.disable_tool("load_skill")
    read_only = _catalog(agent)
    assert "load_skill(" not in read_only
    assert "Read the file at the path below" in read_only


def test_no_route_means_no_catalog(tmp_path):
    """Neither tool installed, so naming procedures it cannot reach is waste."""
    from lamssi_agents import Skills

    root = tmp_path / "s" / "x"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: x\ndescription: d\n---\nbody\n", encoding="utf-8"
    )

    agent = Agent(features=[Skills(tmp_path / "s")])

    assert _catalog(agent) == ""


def test_skill_loading_is_host_owned_unless_exposed_to_the_model(tmp_path):
    """Hosts can always pin; one explicit option exposes the loader tool."""
    from lamssi_agents import Skills

    root = tmp_path / "skills" / "surveying"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: surveying\ndescription: d\n---\nbody\n", encoding="utf-8"
    )
    paths = tmp_path / "skills"

    host_only = Agent(
        features=[SystemTools(), Guidance(), Files(tmp_path), Skills(paths)]
    )
    runtime = host_only.get(SkillRuntime)
    assert "load_skill" not in host_only.available_tool_names()
    assert runtime.load("surveying")["status"] == "loaded"
    assert "### Skill: surveying" in host_only.build_system_prompt()

    model_loading = Agent(
        features=[
            SystemTools(),
            Guidance(),
            Files(tmp_path),
            Skills(paths, allow_model_loading=True),
        ]
    )
    assert "load_skill" in model_loading.available_tool_names()


def test_skills_accept_programmatic_entries_and_custom_loaders(tmp_path):
    direct = Skill("direct", "A programmatic skill.", "Direct body")

    def loader(root, *, source):
        assert root == tmp_path
        return [Skill("loaded", "A loaded skill.", "Loaded body", source=source)]

    agent = Agent(features=[Skills(tmp_path, entries=[direct], loader=loader)])
    runtime = agent.get(SkillRuntime)

    assert [skill.name for skill in runtime.list()] == ["loaded", "direct"]
    assert runtime.load("direct")["status"] == "loaded"


def test_builtin_skills_are_an_explicit_feature_option():
    agent = Agent(features=[Skills(include_builtin=True)])

    assert "code-assistance" in {skill.name for skill in agent.get(SkillRuntime).list()}


def test_skill_public_api_hides_catalog_implementation():
    import lamssi_agents.features.skills as skills_api

    assert set(skills_api.__all__) == {"Skill", "SkillRuntime", "Skills"}
    assert not hasattr(skills_api, "SkillCatalog")


def test_the_tool_implies_pinning_because_that_is_its_whole_job(tmp_path):
    from lamssi_agents import Skills

    root = tmp_path / "s" / "x"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: x\ndescription: d\n---\nb\n", encoding="utf-8"
    )

    agent = Agent(
        features=[
            SystemTools(),
            Guidance(),
            Files(tmp_path),
            Skills(tmp_path / "s", allow_model_loading=True),
        ]
    )

    assert "active-skills" in agent._prompt.names()


def test_the_tool_is_absent_when_the_catalog_found_nothing(tmp_path):
    """A tool that can only answer "no skills available" costs a schema to say so."""
    from lamssi_agents import Skills

    empty = tmp_path / "empty"
    empty.mkdir()
    agent = Agent(
        features=[
            SystemTools(),
            Guidance(),
            Files(tmp_path),
            Skills(empty, allow_model_loading=True),
        ]
    )

    assert "load_skill" not in agent.available_tool_names()


class DirectModel:
    """What a host talking to an API itself would write. No factory involved."""

    model, name = "my-api/v1", "direct"
    is_local = supports_tools = True
    reasoning_effort = None

    def __init__(self) -> None:
        self._usage = Usage()
        self.abort_event = None

    def stream(self, messages, tools=None, **kw):
        self.abort_event = kw.get("abort_event")
        yield StreamDelta(type="text", text="hi from my own client")
        yield StreamDelta(type="done", finish_reason="stop")

    @property
    def cumulative_usage(self):
        return self._usage


def test_a_supplied_model_is_used_directly() -> None:
    """A supplied model receives the agent's cancellation state and label."""
    import sys

    sys.modules.pop("litellm", None)
    model = DirectModel()
    agent = Agent(model=model, approval=ApprovalPolicy.allow_all())

    assert "litellm" not in sys.modules
    assert agent.model is model
    assert agent.model_id == "my-api/v1", "the label came off the model"
    assert agent.chat("go") == "hi from my own client"
    assert model.abort_event is agent._control.aborted, (
        "call-scoped abort was not passed"
    )


def test_use_model_swaps_mid_conversation() -> None:
    agent = Agent(model=DirectModel(), approval=ApprovalPolicy.allow_all())
    agent.chat("first")

    replacement = DirectModel()
    agent.use_model(replacement)
    agent.chat("second")

    assert agent.model is replacement
    assert replacement.abort_event is agent._control.aborted
    assert len(agent._conversation.history) == 4, "the conversation survived the swap"


# the skill catalog is described in the prompt, or absent when there is none


def test_a_skill_catalog_is_described_without_being_asked_for(tmp_path):
    """Installing skills exposes their catalog in the prompt."""
    skill = tmp_path / "skills" / "calibrate"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: calibrate\ndescription: How to calibrate the rig.\n---\n\nSteps.\n",
        encoding="utf-8",
    )

    agent = Agent(
        features=[
            SystemTools(),
            Guidance(),
            Files(tmp_path),
            Skills(tmp_path / "skills"),
        ],
    )

    assert agent.get(SkillRuntime) is not None, (
        "the skill feature should have a runtime"
    )
    prompt = agent.assemble_prompt()

    assert "skill-catalog" in dict(prompt.blocks)
    assert "calibrate" in prompt.text, "the model cannot load what it cannot see"
    assert "How to calibrate the rig." in prompt.text


def test_no_catalog_means_no_block_and_no_tool(tmp_path):
    """A host without skills pays nothing for either half being possible."""
    agent = Agent(
        approval=ApprovalPolicy.allow_all(), features=[SystemTools(), Guidance()]
    )

    assert agent.get(SkillRuntime) is None
    assert "skill-catalog" not in dict(agent.assemble_prompt().blocks)
    assert "load_skill" not in {d.name for d in agent.visible_tool_defs()}
