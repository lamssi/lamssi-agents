"""Observable safety invariants for the tool pipeline.

These tests assert outcomes such as execution, approval, and call/result pairing.
They avoid depending on the internal order of gates. The run scope is read from
the injected capability context only where its propagation is itself under test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from lamssi_agents import (
    Agent,
    ApprovalPolicy,
    ToolApproval,
    ToolApprovalResult,
)
from lamssi_agents.providers.models import Message
from lamssi_tools import CapabilityContext, Expose, Str, tool

from _models import ScriptedModel, calls, says

# Tools used to record executed calls

ran: List[str] = []

@tool(
    expose=Expose.AGENT,
    approval="always",
    description="A gated tool. Records that its body executed.",
    parameters={"n": Str("An arbitrary marker.")},
)
def gated(n: str = "x") -> Dict[str, Any]:
    ran.append(f"gated:{n}")
    return {"ok": True, "n": n}

@tool(
    expose=Expose.AGENT,
    approval="never",
    description="An ungated tool. Records that its body executed.",
    parameters={"n": Str("An arbitrary marker.")},
)
def ungated(n: str = "x") -> Dict[str, Any]:
    ran.append(f"ungated:{n}")
    return {"ok": True, "n": n}

@tool(
    expose=Expose.AGENT,
    approval="never",
    inject_context=True,
    description="Reports whether the run scope is active on its own thread.",
)
def session_probe(ctx: CapabilityContext) -> Dict[str, Any]:
    # Reads the run scope from the injected capability context, because that
    # propagation across the host's dispatch seam is itself under test here.
    from lamssi_agents.runtime import RunScope

    scope = ctx.get(RunScope)
    session = scope.agent._control if scope is not None else None
    ran.append("session_probe")
    return {
        "session_present": session is not None,
        "session_id": id(session) if session else None,
    }

@tool(
    expose=Expose.AGENT,
    approval="never",
    guard_role="always_allowed",
    description="Stands in for ask_user: the tool every guard message points at.",
    parameters={"n": Str("An arbitrary marker.")},
)
def always_allowed_probe(n: str = "x") -> Dict[str, Any]:
    ran.append(f"always_allowed_probe:{n}")
    return {"ok": True, "n": n}

@tool(
    expose=Expose.AGENT,
    approval="never",
    guard_role="repeatable",
    description="Stands in for a live-state read: repeating it is meaningful.",
    parameters={"n": Str("An arbitrary marker.")},
)
def repeatable_probe(n: str = "x") -> Dict[str, Any]:
    ran.append(f"repeatable_probe:{n}")
    return {"ok": True, "n": n, "reading": len(ran)}

@pytest.fixture(autouse=True)
def _reset_ran():
    ran.clear()
    yield
    ran.clear()

@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    return tmp_path

def build(project: Path, **config) -> Any:
    """An agent with only the probe tools: no features or domain code."""
    agent = Agent(
        max_turns=8,
        tools=(gated, ungated, session_probe, always_allowed_probe, repeatable_probe),
    )
    if config:
        agent._config = agent._config.merged(**config).normalised()
    return agent

def tool_results(agent: Agent) -> List[Message]:
    return [m for m in agent._conversation.history if m.role == "tool"]

def announced_ids(agent: Agent) -> List[str]:
    out: List[str] = []
    for m in agent._conversation.history:
        for tc in getattr(m, "tool_calls", None) or []:
            out.append(tc.id if hasattr(tc, "id") else tc["id"])
    return out

def answered_ids(agent: Agent) -> List[str]:
    return [m.tool_call_id for m in tool_results(agent)]

# invariant 1: exactly one result per announced call

@pytest.mark.parametrize(
    "name, args, approve",
    [
        ("ungated", {"n": "a"}, None),                       # plain success
        ("gated", {"n": "a"}, ToolApproval.APPROVE),          # approved
        ("gated", {"n": "a"}, ToolApproval.REJECT),           # rejected
        ("nonexistent_tool", {}, None),                       # out of scope
        ("gated", {"n": "a"}, "raise"),                       # handler explodes
    ],
)
def test_every_announced_call_gets_exactly_one_result(project, name, args, approve):
    """Every announced tool call receives exactly one result message."""
    agent = build(project)
    agent.use_model(ScriptedModel(calls(("c1", name, args)), says("done")))

    if approve == "raise":
        def cb(request):
            raise RuntimeError("approval handler exploded")
        agent.approval = ApprovalPolicy.ask_when_required(cb)
    elif approve is not None:
        agent.approval = ApprovalPolicy.ask_when_required(lambda request: approve)

    try:
        agent.chat("go")
    except Exception:
        # A raising handler may abort the run; the transcript must still pair.
        pass

    assert announced_ids(agent) == ["c1"]
    assert answered_ids(agent) == ["c1"], (
        f"announced {announced_ids(agent)} but answered {answered_ids(agent)}"
    )

def test_abort_mid_batch_still_answers_every_announced_call(project):
    """ABORT unwinds the run. The calls it never reached still need results."""
    agent = build(project)
    agent.use_model(ScriptedModel(
        calls(("c1", "gated", {"n": "a"}), ("c2", "gated", {"n": "b"}),
              ("c3", "gated", {"n": "c"})),
        says("done"),
    ))
    agent.approval = ApprovalPolicy.ask_when_required(
        lambda request: ToolApproval.ABORT
    )

    try:
        agent.chat("go")
    except Exception:
        pass

    assert sorted(answered_ids(agent)) == ["c1", "c2", "c3"], (
        "an aborted batch left announced calls unanswered"
    )
    assert ran == [], "a body ran despite the abort"

# invariant 2: approval fails closed

def test_no_handler_blocks_a_gated_call(project):
    """A gated call is blocked when no approval handler is available."""
    agent = build(project)
    agent.use_model(ScriptedModel(calls(("c1", "gated", {})), says("done")))
    agent.chat("go")

    assert ran == [], "a gated tool ran with nobody to approve it"
    assert len(tool_results(agent)) == 1

def test_a_raising_handler_blocks_rather_than_runs(project):
    agent = build(project)
    agent.use_model(ScriptedModel(calls(("c1", "gated", {})), says("done")))

    def cb(request):
        raise RuntimeError("the UI thread died")

    agent.approval = ApprovalPolicy.ask_when_required(cb)
    try:
        agent.chat("go")
    except Exception:
        pass

    assert ran == [], "a gated tool ran after its approval handler raised"

@pytest.mark.parametrize("decision", [False, None, "deny", "no", 0, "APPROVE"])
def test_an_unrecognised_decision_blocks(project, decision):
    """Only the exact approve decision permits tool execution."""
    agent = build(project)
    agent.use_model(ScriptedModel(calls(("c1", "gated", {})), says("done")))
    agent.approval = ApprovalPolicy.ask_when_required(lambda request: decision)

    agent.chat("go")

    assert ran == [], f"approval handler returned {decision!r} and the tool ran anyway"

def test_a_malformed_approval_result_blocks(project):
    agent = build(project)
    agent.use_model(ScriptedModel(calls(("c1", "gated", {})), says("done")))
    agent.approval = ApprovalPolicy.ask_when_required(
        lambda request: ToolApprovalResult(decision="yes")  # type: ignore[arg-type]
    )

    agent.chat("go")

    assert ran == [], "an invalid ToolApprovalResult.decision ran the tool"

def test_approve_still_runs_the_tool(project):
    """The exact approve decision runs the gated tool."""
    agent = build(project)
    agent.use_model(ScriptedModel(calls(("c1", "gated", {})), says("done")))
    agent.approval = ApprovalPolicy.ask_when_required(
        lambda request: ToolApproval.APPROVE
    )
    agent.chat("go")
    assert ran == ["gated:x"]

    ran.clear()
    agent2 = build(project)
    agent2.use_model(ScriptedModel(calls(("c1", "gated", {})), says("done")))
    agent2.approval = ApprovalPolicy.ask_when_required(
        lambda request: ToolApprovalResult(ToolApproval.APPROVE)
    )
    agent2.chat("go")
    assert ran == ["gated:x"], "ToolApprovalResult(APPROVE) must also run"

def test_approval_never_is_not_prompted(project):
    """A tool that declares it needs no approval is never promoted into one."""
    asked: List[str] = []
    agent = build(project)
    agent.use_model(ScriptedModel(calls(("c1", "ungated", {})), says("done")))
    agent.approval = ApprovalPolicy.ask_when_required(
        lambda request: (asked.append(request.tool), ToolApproval.APPROVE)[1]
    )

    agent.chat("go")

    assert asked == [], f"approval was requested for a never-gated tool: {asked}"
    assert ran == ["ungated:x"]

# invariant 3: a blocked call is never prompted for

def test_an_out_of_scope_call_never_reaches_the_approval_handler(project):
    """Out-of-scope calls are rejected before approval handling."""
    asked: List[str] = []
    agent = build(project)
    agent.use_model(ScriptedModel(calls(("c1", "not_a_tool", {})), says("done")))
    agent.approval = ApprovalPolicy.ask_when_required(
        lambda request: (asked.append(request.tool), ToolApproval.APPROVE)[1]
    )

    agent.chat("go")

    assert asked == [], f"approval requested for an unavailable tool: {asked}"

def test_a_disabled_tool_is_not_reachable_by_naming_it(project):
    """Disabled in-process tools cannot be invoked by name."""
    agent = build(project)
    agent.disable_tool("ungated")
    agent.use_model(ScriptedModel(calls(("c1", "ungated", {})), says("done")))

    agent.chat("go")

    assert ran == [], "a disabled tool ran when the model named it"
    assert len(tool_results(agent)) == 1

# invariant 4: session authority is isolated

def test_two_agents_do_not_share_authority(project):
    """Two agents must not share conversational authority."""
    a1, a2 = build(project), build(project)

    a1.disable_tool("ungated")

    assert "ungated" in a1.disabled_tool_names()
    assert "ungated" not in a2.disabled_tool_names(), (
        "disabling a tool in one session disabled it in another"
    )
    assert a1._control is not a2._control
    assert a1._conversation is not a2._conversation

# invariant 5: the Session reaches the tool body

def test_the_session_is_active_on_the_thread_that_runs_the_body(project):
    """Tool execution binds the active session on its running thread."""
    agent = build(project)
    agent.use_model(ScriptedModel(calls(("c1", "session_probe", {})), says("done")))

    agent.chat("go")

    result = tool_results(agent)[0].content
    assert "session_present" in result and "true" in result.lower(), (
        f"the tool body ran with no active Session: {result}"
    )

# repeat handling: cache before prompt, and batch visibility

def test_a_repeat_is_answered_without_asking_the_user_again(project):
    """A cached repeat is answered before loop and approval gates."""
    asked: List[str] = []
    agent = build(project)
    agent.use_model(ScriptedModel(
        calls(("c1", "gated", {"n": "a"})),
        calls(("c2", "gated", {"n": "a"})),
        says("done"),
    ))
    agent.approval = ApprovalPolicy.ask_when_required(
        lambda request: (asked.append(request.tool), ToolApproval.APPROVE)[1]
    )

    agent.chat("go")

    assert ran == ["gated:a"], f"the repeat re-executed the body: {ran}"
    assert len(asked) == 1, f"the user was prompted for a cached repeat: {asked}"

@pytest.mark.parametrize("name", ["always_allowed_probe", "repeatable_probe"])
def test_a_repeat_exempt_tool_is_not_answered_from_the_cache(project, name):
    """Repeat-exempt roles bypass both loop and dedupe gates."""
    agent = build(project)
    agent.use_model(ScriptedModel(
        calls(("c1", name, {"n": "a"})),
        calls(("c2", name, {"n": "a"})),
        calls(("c3", name, {"n": "a"})),
        says("done"),
    ))

    agent.chat("go")

    assert ran == [f"{name}:a"] * 3, (
        f"a repeat-exempt tool was answered from cache instead of running: {ran}"
    )

def test_an_ordinary_repeat_is_still_cached(project):
    """Regular tool calls remain subject to repeat caching."""
    agent = build(project)
    agent.use_model(ScriptedModel(
        calls(("c1", "ungated", {"n": "a"})),
        calls(("c2", "ungated", {"n": "a"})),
        says("done"),
    ))

    agent.chat("go")

    assert ran == ["ungated:a"], f"a NORMAL-role repeat re-executed: {ran}"

def test_tool_lifecycle_events_carry_the_call_id(project):
    """Tool lifecycle events share the provider's call id."""
    seen: List[tuple] = []
    agent = build(project)
    agent.use_model(ScriptedModel(
        calls(("c1", "gated", {"n": "a"}), ("c2", "ungated", {"n": "b"})),
        says("done"),
    ))
    agent.approval = ApprovalPolicy.ask_when_required(
        lambda request: ToolApproval.APPROVE
    )
    agent.add_event_listener(
        lambda e: seen.append((e.type.name, (e.metadata or {}).get("tool_call_id")))
    )

    agent.chat("go")

    for kind in ("TOOL_START", "TOOL_APPROVAL", "TOOL_RESULT"):
        ids = [cid for name, cid in seen if name == kind]
        assert ids, f"no {kind} event was emitted"
        assert all(i is not None for i in ids), f"{kind} carried no tool_call_id"
    starts = [cid for name, cid in seen if name == "TOOL_START"]
    assert starts == ["c1", "c2"], f"start events not correlated per call: {starts}"

def test_a_same_batch_duplicate_executes_once(project):
    """Identical calls in one provider batch execute only once."""
    agent = build(project)
    agent.use_model(ScriptedModel(
        calls(("c1", "ungated", {"n": "dup"}), ("c2", "ungated", {"n": "dup"})),
        says("done"),
    ))

    agent.chat("go")

    assert ran == ["ungated:dup"], f"a same-batch duplicate ran twice: {ran}"
    assert sorted(answered_ids(agent)) == ["c1", "c2"]

# per-agent instructions

def test_explicit_instructions_are_per_agent(tmp_path):
    """Reading one agent's instructions cannot change another agent."""
    override = tmp_path / "prompts_a"
    override.mkdir()
    (override / "system.md").write_text("IDENTITY FROM RUNTIME A", encoding="utf-8")

    agent_a = Agent(
        instructions=(override / "system.md").read_text(encoding="utf-8"),
    )
    agent_b = Agent(
        instructions="IDENTITY FROM RUNTIME B",
    )

    a_prompt = agent_a.instructions
    b_prompt = agent_b.instructions

    assert a_prompt == "IDENTITY FROM RUNTIME A"
    assert "IDENTITY FROM RUNTIME A" not in b_prompt, (
        "agent A's explicit instructions reached agent B"
    )
    assert b_prompt == "IDENTITY FROM RUNTIME B"

# Default tool surface

#: Tools that access external state.
WORLD_TOOLS = frozenset({
    "write_file", "edit_file", "delete_file",
    "run_bash", "run_powershell", "execute_code", "memory",
})

def test_the_convenience_constructor_grants_nothing_that_touches_the_world(project):
    """A bare Agent grants no tools that touch the outside world."""
    agent = Agent()
    granted = {d.name for d in agent.visible_tool_defs()} & WORLD_TOOLS

    assert not granted, f"Agent() silently granted {sorted(granted)}"

def test_the_workspace_pack_is_available_when_asked_for(project):
    """Explicitly enabling Files installs workspace tools."""
    from lamssi_agents import Files

    agent = Agent(features=[Files(project)])
    names = {d.name for d in agent.visible_tool_defs()}

    assert "read_file" in names and "write_file" in names, (
        f"workspace_tools=True did not install the pack: {sorted(names)}"
    )
