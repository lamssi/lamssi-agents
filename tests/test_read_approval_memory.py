"""Declared reference directories and run-scoped read approvals.

Writes still require approval, and runtime grants remain local to one conversation.
"""

from __future__ import annotations

from pathlib import Path

from lamssi_agents import Agent, ApprovalPolicy, ToolApproval
from _scope import run_scope_active
from lamssi_agents.features.files.space import FileSpace, ReadableDir

def _space(ws: Path, **kw) -> FileSpace:
    return FileSpace(project_root=lambda: ws, **kw)

# declared readable dirs

def test_a_declared_readable_dir_reads_free_but_writes_ask(tmp_path: Path):
    ws = tmp_path / "ws"; ws.mkdir()
    ref = tmp_path / "ref"; ref.mkdir()
    (ref / "lib.py").write_text("x = 1", encoding="utf-8")

    sp = _space(ws, readable_dirs=[ReadableDir("ref", ref, "a library")])

    assert sp.call_is_free({"path": str(ref / "lib.py")}, key="path") is True
    assert sp.call_is_free({"path": str(ref / "lib.py")}, key="path", write=True) is False
    route = sp.resolve(str(ref / "lib.py"))
    assert route.error is None and route.base == ref

def test_files_feature_readable_dirs_reach_the_file_space(tmp_path: Path):
    from lamssi_agents import Agent
    from lamssi_agents import Files

    ws = tmp_path / "ws"; ws.mkdir()
    ref = tmp_path / "ref"; ref.mkdir()
    (ref / "note.md").write_text("hi", encoding="utf-8")

    agent = Agent(approval=ApprovalPolicy.allow_all(), features=[Files(ws, read_only=[ref])])
    assert agent.get(FileSpace).call_is_free(
        {"path": str(ref / "note.md")}, key="path"
    ) is True

# granted on approval, for one run

def test_approving_a_read_pins_its_directory(tmp_path: Path):
    ws = tmp_path / "ws"; ws.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (outside / "a.py").write_text("a", encoding="utf-8")
    (outside / "sub").mkdir()
    (outside / "sub" / "b.py").write_text("b", encoding="utf-8")

    sp = _space(ws)
    with run_scope_active(Agent()):
        assert sp.call_is_free({"path": str(outside / "a.py")}, key="path") is False

        assert sp.remember_read_approval("read_file", {"path": str(outside / "a.py")}) == outside

        # the sibling, and a file deeper in the tree, are both free to read
        assert sp.call_is_free({"path": str(outside / "a.py")}, key="path") is True
        assert sp.call_is_free({"path": str(outside / "sub" / "b.py")}, key="path") is True
        # but a write under it still asks
        assert sp.call_is_free(
            {"path": str(outside / "sub" / "b.py")}, key="path", write=True
        ) is False

def test_a_nested_read_pins_the_directory_that_holds_it(tmp_path: Path):
    """The grant is the file's directory, so a sibling read rides on it."""
    ws = tmp_path / "ws"; ws.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (outside / "a.py").write_text("a", encoding="utf-8")
    (outside / "b.py").write_text("b", encoding="utf-8")

    sp = _space(ws)
    with run_scope_active(Agent()):
        sp.remember_read_approval("read_file", {"path": str(outside / "a.py")})
        assert sp.call_is_free({"path": str(outside / "b.py")}, key="path") is True

def test_a_grant_does_not_reach_another_conversation_on_the_same_runtime(tmp_path: Path):
    """External read grants remain scoped to one conversation."""
    ws = tmp_path / "ws"; ws.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (outside / "a.py").write_text("a", encoding="utf-8")

    sp = _space(ws)                      # one agent capability
    first, second = Agent(), Agent()

    with run_scope_active(first):
        sp.remember_read_approval("read_file", {"path": str(outside / "a.py")})
        assert sp.call_is_free({"path": str(outside / "a.py")}, key="path") is True

    with run_scope_active(second):
        assert sp.call_is_free({"path": str(outside / "a.py")}, key="path") is False

    # and with no run at all, nothing is free outside the workspace
    assert sp.call_is_free({"path": str(outside / "a.py")}, key="path") is False

def test_nothing_is_granted_outside_a_run(tmp_path: Path):
    """No run, nothing to pin the grant to: and nothing silently stored."""
    ws = tmp_path / "ws"; ws.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    sp = _space(ws)
    assert sp.remember_read_approval("read_file", {"path": str(outside / "a.py")}) is None
    assert sp.approved_dirs() == ()

def test_a_denylisted_directory_is_never_remembered(tmp_path: Path):
    sp = _space(tmp_path / "ws")
    with run_scope_active(Agent()):
        # the resolved directory is under a sensitive dir part → never pinned
        assert sp.remember_read_approval("read_file", {"path": "C:/x/.ssh/known_hosts"}) is None
        assert sp.approved_dirs() == ()

def test_the_workspace_is_not_remembered_it_is_already_free(tmp_path: Path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "in.py").write_text("x", encoding="utf-8")
    sp = _space(ws)
    with run_scope_active(Agent()):
        # a workspace read never reaches approval, so remembering it is redundant
        assert sp.remember_read_approval("read_file", {"path": "in.py"}) is None

def test_a_write_tool_is_not_remembered(tmp_path: Path):
    ws = tmp_path / "ws"; ws.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    sp = _space(ws)
    with run_scope_active(Agent()):
        # write_file is not a read tool, so approving it grants nothing standing
        assert sp.remember_read_approval("write_file", {"path": str(outside / "x.py")}) is None

# the grant is made by the pack, not by whoever asked

def test_any_front_end_that_approves_a_read_makes_the_grant(tmp_path: Path):
    """Read approval creates the same grant through every front end."""
    from lamssi_agents import Agent
    from lamssi_agents import Files
    from lamssi_agents import approval
    from lamssi_agents.providers import ToolCall

    ws = tmp_path / "ws"; ws.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (outside / "a.py").write_text("A", encoding="utf-8")
    (outside / "b.py").write_text("B", encoding="utf-8")

    asked = []
    agent = Agent(
        approval=ApprovalPolicy.ask_when_required(
            lambda request: (
                asked.append(request.tool),
                ToolApproval.APPROVE,
            )[1]
        ),
        features=[Files(ws)],
    )
    with run_scope_active(agent):
        batch = agent._runtime.execute_calls(
            [ToolCall(id="1", name="read_file", arguments={"path": str(outside / "a.py")})],
            agent._conversation.turn,
        )
    agent._conversation.extend(batch.messages)

    assert asked == ["read_file"], "the out-of-scope read had to be approved"
    from lamssi_agents.features.files import ReadGrants

    assert outside.resolve() in agent.conversation_state(
        ReadGrants, ReadGrants
    ).directories

    with run_scope_active(agent):
        assert not approval.should_require_approval(
            agent, "read_file", {"path": str(outside / "b.py")}
        ), "the sibling read should ride on the grant"
        assert approval.should_require_approval(
            agent, "write_file", {"path": str(outside / "c.py")}
        ), "a write there is not covered by a read grant"
