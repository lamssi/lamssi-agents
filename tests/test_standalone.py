"""Built-in features complete a real turn without application integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from _models import ScriptedModel, calls, says


def _script_read_then_answer(path: str) -> list:
    """Two turns: read the file, then answer from its result."""
    return [
        calls(("c1", "read_file", {"path": path})),
        says("The file says hello."),
    ]


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "notes.txt").write_text(
        "hello from an isolated project\n", encoding="utf-8"
    )
    return tmp_path


def test_builtin_features_run_a_full_turn(project: Path) -> None:
    """Drive built-in features through a real tool call against a real file."""
    from lamssi_agents import Agent
    from lamssi_agents import Files, Guidance, SystemTools

    model = ScriptedModel(*_script_read_then_answer("notes.txt"))
    agent = Agent(
        model=model,
        max_turns=6,
        features=[SystemTools(), Guidance(), Files(project)],
    )

    answer = agent.chat("What does notes.txt say?")

    assert "hello" in answer.lower() or "file says" in answer.lower()
    # The tool actually ran: its result is in the transcript.
    tool_results = [message for message in agent.history if message.role == "tool"]
    assert tool_results, "read_file never produced a tool result"
    assert "isolated project" in tool_results[0].content
