"""Diagnostics for missing skill resources and empty filtered searches."""

from __future__ import annotations

from pathlib import Path

import pytest

from lamssi_agents import Agent, ApprovalPolicy, Files, Guidance, Skills, SystemTools
from lamssi_agents import tool_runtime as tool_mod
from lamssi_agents.features.skills import SkillRuntime

# a skill's bundled files


@pytest.fixture
def skill_with_a_reference(tmp_path: Path) -> Path:
    root = tmp_path / "skills" / "app-creation"
    (root / "reference").mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: app-creation\ndescription: Build apps.\n"
        "allowed-tools: read_file\n---\n\nRead the reference first.\n",
        encoding="utf-8",
    )
    (root / "reference" / "api.md").write_text(
        "def position_feedback(self, widget_id, axes, name): ...\n", encoding="utf-8"
    )
    (root / "reference" / "logo.png").write_bytes(b"\x89PNG not text")
    return tmp_path / "skills"


def test_a_skill_reports_the_files_it_ships(skill_with_a_reference: Path):
    """Loaded skills report readable absolute paths for bundled resources."""
    agent = Agent(features=[Skills(skill_with_a_reference)])
    found = agent.get(SkillRuntime).load("app-creation")["resources"]

    assert [r["name"] for r in found] == ["reference/api.md"], (
        "SKILL.md itself is not a resource, and a binary is not readable"
    )
    assert Path(found[0]["path"]).is_absolute(), "a relative path is what failed before"
    assert Path(found[0]["path"]).is_file()


def test_the_reported_path_actually_opens(skill_with_a_reference: Path, tmp_path: Path):
    """A reported skill resource path opens outside the project root."""
    project = tmp_path / "workspace"
    project.mkdir()
    agent = Agent(
        approval=ApprovalPolicy.allow_all(),
        features=[
            SystemTools(),
            Guidance(),
            Files(project),
            Skills(skill_with_a_reference),
        ],
    )
    resource = agent.get(SkillRuntime).load("app-creation")["resources"][0]

    result = tool_mod.invoke_tool_unchecked(
        agent, "read_file", {"path": resource["path"]}
    )

    assert "error" not in result, result
    assert "position_feedback" in result["content"]


def test_load_skill_hands_the_paths_to_the_model(
    skill_with_a_reference: Path, tmp_path: Path
):
    """A resource nothing mentions is a resource nobody opens."""
    project = tmp_path / "workspace"
    project.mkdir()
    agent = Agent(
        features=[
            SystemTools(),
            Guidance(),
            Files(project),
            Skills(skill_with_a_reference, allow_model_loading=True),
        ],
    )
    result = tool_mod.invoke_tool_unchecked(
        agent, "load_skill", {"name": "app-creation"}
    )

    assert result["status"] == "loaded"
    assert [r["name"] for r in result["resources"]] == ["reference/api.md"]
    assert "read_file" in result["resources_hint"]


# Empty lookup diagnostics


@pytest.fixture
def project_with_a_stub(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "app.py").write_text(
        "self.position_feedback('stages_fb', axes='XYZ')\n", encoding="utf-8"
    )
    return tmp_path


def fs(agent, command: str):
    return tool_mod.invoke_tool_unchecked(agent, "fs", {"command": command})


@pytest.fixture
def agent(project_with_a_stub: Path):
    return Agent(
        approval=ApprovalPolicy.allow_all(),
        features=[SystemTools(), Guidance(), Files(project_with_a_stub)],
    )


def test_a_grep_that_matches_nothing_says_it_looked(agent):
    """Files were read and none contained it, so the query is the thing to change."""
    result = fs(agent, "grep -rn quantum_flux_capacitor .")

    assert result["count"] == 0
    assert result["files_scanned"] > 0, "it read files, and the result must say so"
    assert result["hint"], "a bare count of 0 sends the model round again"


def test_a_filter_that_matched_no_files_says_that_instead(agent):
    """Identify an include filter that matched no files."""
    result = fs(agent, "grep -rn position_feedback --include=*.rs .")

    assert result["count"] == 0
    assert result["files_scanned"] == 0, "no file matched the filter"
    assert result["hint"] != "", "the model cannot tell which argument was wrong"


def test_a_find_that_matches_nothing_hands_back_a_pasteable_command(agent):
    """Return a command the caller can retry directly."""
    result = fs(agent, "find . -name '*.rs'")

    assert result["count"] == 0
    assert "find" in result["hint"], f"the hint is not pasteable: {result['hint']}"


def test_find_ipath_is_case_insensitive(agent):
    result = fs(agent, "find . -ipath '*PKG/APP.PY'")

    assert result["lines"] == ["pkg/app.py"]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("grep -l position_feedback .", "pkg/app.py"),
        ("grep -c position_feedback .", "pkg/app.py:1"),
        ("grep -e position_feedback .", "pkg/app.py:1:"),
    ],
)
def test_grep_modes_keep_their_output_contract(agent, command, expected):
    result = fs(agent, command)

    assert any(line.startswith(expected) for line in result["lines"]), result


def test_a_path_that_does_not_exist_fails_forward(agent):
    """`retriable: False` reads as "this tool is broken" and closes the cycle."""
    result = fs(agent, "ls no_such_dir")

    assert "error" in result
    assert result["retriable"] is True, "the argument was wrong, not the tool"
    assert result["hint"], "a refusal with no next step is where the loop starts"


# one read tool, routing on what the file is


def _reader(tmp_path):
    from lamssi_agents import Agent, Files, Guidance, SystemTools

    return Agent(
        features=[SystemTools(), Guidance(), Files(tmp_path)],
        approval=ApprovalPolicy.allow_all(),
    )


def test_prose_and_config_come_back_as_text(tmp_path):
    """Plain text and configuration files return readable text."""
    from lamssi_agents import tool_runtime as tool_mod

    (tmp_path / "notes.txt").write_text("Remember to call the lab.\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        '{"model": "gemma", "retries": 3}', encoding="utf-8"
    )
    agent = _reader(tmp_path)

    notes = tool_mod.invoke_tool_unchecked(agent, "read_file", {"path": "notes.txt"})
    config = tool_mod.invoke_tool_unchecked(agent, "read_file", {"path": "config.json"})

    assert "Remember to call the lab." in notes["content"]
    assert '"retries": 3' in config["content"], (
        "a config must come back readable, not summarised"
    )


def test_a_line_range_always_returns_text(tmp_path):
    """How you see a data file's raw rows: asking for lines means lines."""
    from lamssi_agents import tool_runtime as tool_mod

    (tmp_path / "d.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    agent = _reader(tmp_path)

    sliced = tool_mod.invoke_tool_unchecked(
        agent, "read_file", {"path": "d.csv", "start_line": 1, "end_line": 2}
    )

    assert "content" in sliced and "a,b" in sliced["content"]


def test_a_data_file_falls_back_to_text_when_the_extra_is_absent(tmp_path):
    """Data files fall back to text when table dependencies are absent."""
    from lamssi_agents import tool_runtime as tool_mod

    (tmp_path / "d.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    agent = _reader(tmp_path)

    result = tool_mod.invoke_tool_unchecked(agent, "read_file", {"path": "d.csv"})

    assert "error" not in result
    assert result.get("kind") == "table" or "a,b" in result.get("content", "")


def test_reading_a_workspace_file_never_prompts(tmp_path):
    """Reading a workspace file does not require approval."""
    from lamssi_agents.tooling import needs_approval

    agent = _reader(tmp_path)
    definition = agent._tools.get_tool("read_file")

    for args in ({"path": "notes.txt"}, {"path": "d.csv", "max_rows": 10}):
        assert not needs_approval(
            ApprovalPolicy.reject_when_required(),
            "read_file",
            args,
            definition,
            rules=agent.safe_when,
        ), f"a workspace read prompted: {args}"
