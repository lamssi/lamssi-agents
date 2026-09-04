"""Tests for shared parsing, compaction, path, and result utilities."""

from __future__ import annotations

import pytest

from lamssi_agents.providers import Message, StreamDelta


# memory frontmatter

@pytest.mark.parametrize(
    "description",
    [
        "Remember: use tabs",
        "budget #1 priority",
        "@handle and [bracket] leading",
        "line one\nline two",
        "plain text",
    ],
)
def test_memory_description_roundtrips(tmp_path, description):
    from lamssi_agents.features.memory import MemoryStore

    store = MemoryStore(tmp_path)
    store.save("note", "the body", type="feedback", description=description)

    loaded = store.load("note")
    assert loaded is not None
    assert loaded.description == description
    assert loaded.type == "feedback"
    assert loaded.content == "the body"


# compaction focus

class _CapturingProvider:
    """Records the user prompt handed to ``stream`` for one summary call."""

    def __init__(self):
        self.prompt = ""

    def stream(self, messages, **_kw):
        self.prompt = messages[-1].content
        yield StreamDelta(type="text", text="updated recap")
        yield StreamDelta(type="done", finish_reason="stop")


def test_compaction_focus_on_fresh_path():
    from lamssi_agents.history import compaction

    focus = "the DdF-42 migration decision"
    provider = _CapturingProvider()
    fresh = [
        Message(role="user", content="please migrate the table"),
        Message(role="assistant", content="done"),
    ]
    compaction._llm_summarise(provider, fresh, focus=focus)
    assert "## Focus" in provider.prompt
    assert focus in provider.prompt


def test_compaction_focus_on_update_path():
    """Keep focus guidance when extending an existing summary."""
    from lamssi_agents.history import compaction

    focus = "the DdF-42 migration decision"
    provider = _CapturingProvider()
    with_previous = [
        Message(role="user", content=compaction.frame_summary("an earlier summary")),
        Message(role="user", content="please migrate the table"),
        Message(role="assistant", content="done"),
    ]
    compaction._llm_summarise(provider, with_previous, focus=focus)
    assert "## Focus" in provider.prompt, "focus block missing on the update path"
    assert focus in provider.prompt


# path containment


def test_within_is_the_containment_check(tmp_path):
    from lamssi_agents.features.files.space import _within

    root = tmp_path
    assert _within(root, root)
    assert _within(root / "a" / "b", root)
    assert not _within(root.parent, root)


# helpers: err() and call_signature()

def test_err_reproduces_the_hand_written_shapes():
    from lamssi_tools import err

    assert err("boom") == {"error": "boom"}
    assert err("boom", retriable=False) == {"error": "boom", "retriable": False}
    assert err("boom", retriable=True, hint="try x") == {
        "error": "boom",
        "retriable": True,
        "hint": "try x",
    }
    assert err("boom", retriable=False, available=["a", "b"]) == {
        "error": "boom",
        "retriable": False,
        "available": ["a", "b"],
    }


def test_call_signature_is_order_independent_and_shared():
    from lamssi_agents.tooling.guard import LoopGuard, call_signature

    a = call_signature("read_file", {"path": "x", "n": 1})
    b = call_signature("read_file", {"n": 1, "path": "x"})
    assert a == b
    assert LoopGuard._tool_key("read_file", {"path": "x", "n": 1}) == a
