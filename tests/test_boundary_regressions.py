"""Regressions for boundaries the refactor introduced or left implicit.

Each test pins a specific defect a code review found, so a later change that
re-breaks the boundary fails here instead of silently.
"""

from __future__ import annotations

from typing import Any, Dict, List

from lamssi_agents import Agent
from lamssi_agents.features.base import Feature
from lamssi_agents.providers import Message

from _models import ScriptedModel, says


# F3: compaction and clearing invalidate the calibrator's provider anchor.

def _conv(*contents: str):
    from lamssi_agents.agent.conversation import Conversation

    c = Conversation()
    c.history = [Message(role="user", content=x) for x in contents]
    return c


def test_clear_resets_the_token_anchor():
    c = _conv("a", "b")
    c.tokens.anchor = 2
    c.clear()
    assert c.tokens.anchor == -1


def test_a_demotion_resets_the_token_anchor():
    """Same length, rewritten bodies: the provider measurement is stale."""
    c = _conv("a", "b")
    c.tokens.anchor = 2
    c.on_compacted([Message(role="user", content="A"), Message(role="assistant", content="B")])
    assert c.tokens.anchor == -1


def test_a_summarisation_resets_the_token_anchor():
    c = _conv("a", "b", "c", "d")
    c.tokens.anchor = 4
    c.on_compacted([Message(role="user", content="summary")])
    assert c.tokens.anchor == -1


# F2: a mid-run model swap governs the provider call it precedes.


def test_a_mid_run_model_swap_governs_the_provider_call():
    """A before_turn hook that swaps the model makes THAT model stream this turn.

    The loop reads the model live; a stale capture would stream the old model
    while fitting used the new one.
    """
    old = ScriptedModel(says("old-answer"), name="OLD")
    new = ScriptedModel(says("new-answer"), name="NEW")

    class SwapModel(Feature):
        name = "swap-model"

        def before_turn(self, agent: Any, turn: int):
            if turn == 1:
                agent.use_model(new)
            return None

    agent = Agent(old, features=[SwapModel()])
    result = agent.chat("go")

    assert result == "new-answer", "the live model must stream, not the run-start capture"
    assert new.calls == 1 and old.calls == 0


# Guard rules set through the public setter reach the live guard.

def test_guard_rules_setter_resyncs_the_live_guard():
    from dataclasses import replace

    from lamssi_agents.tooling.guard import DEFAULT_GUARD_RULES

    agent = Agent()
    agent.guard_rules = replace(DEFAULT_GUARD_RULES, err_streak_limit=1)
    assert agent.rules.err_streak_limit == 1
    assert agent._runtime.guard.rules.err_streak_limit == 1, (
        "the live guard must reflect the setter, not the construction snapshot"
    )


# sanitize keeps a non-contiguous tool result instead of fabricating a duplicate.

def test_sanitize_keeps_a_non_contiguous_tool_result_without_duplicating():
    from lamssi_agents.agent.conversation import Conversation
    from lamssi_agents.providers import ToolCall

    c = Conversation()
    c.history = [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="A", name="t", arguments={}),
                ToolCall(id="B", name="t", arguments={}),
            ],
        ),
        Message(role="tool", tool_call_id="A", name="t", content="rA"),
        Message(role="assistant", content="thinking"),
        Message(role="tool", tool_call_id="B", name="t", content="rB"),
    ]
    c.sanitize()
    ids = [m.tool_call_id for m in c.history if m.role == "tool"]
    assert ids.count("B") == 1, f"fabricated a duplicate result: {ids}"
    assert "rB" in [m.content for m in c.history if m.role == "tool"]


# A write into a read-only reference dir is external (approval-gated), not in-zone.

def test_write_into_a_read_only_reference_dir_is_not_in_zone(tmp_path):
    from lamssi_agents.features.files.space import FileSpace, ReadableDir

    ws = tmp_path / "ws"
    ws.mkdir()
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "guide.txt").write_text("x", encoding="utf-8")
    sp = FileSpace(
        project_root=lambda: ws,
        readable_dirs=[ReadableDir("manuals", ref, "manuals")],
    )

    read = sp.resolve(str(ref / "guide.txt"))
    assert read.error is None and read.root.startswith("reference")

    write = sp.resolve(str(ref / "guide.txt"), allow_external=True, write=True)
    assert write.external is True, (
        "a write into a read-only reference dir must be external, not in-zone"
    )


__all__: List[Dict[str, Any]] = []
