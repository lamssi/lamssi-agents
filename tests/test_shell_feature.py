"""Platform-specific shell installation, schemas, and command behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from lamssi_agents import Agent, Files, Guidance, Shell, SystemTools
from lamssi_agents.features.shell import (
    _execute,
    detect_shell,
    run_bash,
    run_powershell,
)
from lamssi_tools import CapabilityContext


def _names(agent: Agent) -> set:
    return {t.name for t in agent._tools.list_tools()}


# the kernel ships no shell

def test_files_no_longer_grants_a_shell(tmp_path: Path):
    """Asking for file access must not also hand over command execution."""
    agent = Agent(features=[SystemTools(), Guidance(), Files(tmp_path)])
    assert not {"run_bash", "run_powershell"} & _names(agent)

def test_no_feature_means_no_shell_tool(tmp_path: Path):
    agent = Agent(features=[SystemTools(), Guidance()])
    assert not {"run_bash", "run_powershell"} & _names(agent)

# exactly one, chosen at install

def test_exactly_one_shell_tool_is_installed(tmp_path: Path):
    """Never both: the model is not asked to pick a dialect."""
    agent = Agent(features=[SystemTools(), Guidance(), Shell()])
    assert len({"run_bash", "run_powershell"} & _names(agent)) == 1

def test_the_detected_shell_is_the_one_installed(tmp_path: Path):
    agent = Agent(features=[SystemTools(), Guidance(), Shell()])
    expected = "run_bash" if detect_shell() == "bash" else "run_powershell"
    assert expected in _names(agent)

@pytest.mark.parametrize("prefer,expected", [
    ("bash", "run_bash"),
    ("powershell", "run_powershell"),
])
def test_a_host_can_pin_the_choice(tmp_path: Path, prefer, expected):
    """A host can select its preferred shell explicitly."""
    agent = Agent(features=[SystemTools(), Guidance(), Shell(prefer=prefer)])
    assert expected in _names(agent)
    assert len({"run_bash", "run_powershell"} & _names(agent)) == 1

def test_both_can_be_installed_deliberately(tmp_path: Path):
    """A host can install both shell features explicitly."""
    agent = Agent(features=[SystemTools(), Guidance(), Shell(prefer="bash"), Shell(prefer="powershell")])
    assert {"run_bash", "run_powershell"} <= _names(agent)

def test_an_unknown_preference_is_refused():
    with pytest.raises(ValueError, match="bash"):
        Shell(prefer="zsh")


# each description states its own shell's rules

def test_the_powershell_description_rules_out_the_chain_operators():
    """The sentence that would have saved the run this test's docstring describes."""
    text = run_powershell._tool_definition.description
    assert "&&" in text and "parse error" in text
    assert "a; if ($?) { b }" in text, "says it is broken without saying what to write"

def test_neither_description_hedges_about_which_shell_it_is():
    for fn, wrong in ((run_bash, "powershell"), (run_powershell, "bash")):
        text = fn._tool_definition.description.lower()
        assert wrong not in text.replace("not bash", ""), (
            f"{fn.__name__} describes a shell it does not run"
        )

def test_both_still_require_approval():
    for fn in (run_bash, run_powershell):
        assert fn._tool_definition.approval == "always"


# the WSL trap

def test_detection_falls_back_to_powershell_only_without_bash(monkeypatch):
    monkeypatch.setattr("lamssi_agents.features.shell.bash_binary", lambda: None)
    monkeypatch.setattr("sys.platform", "win32")
    assert detect_shell() == "powershell"


def test_a_missing_working_directory_never_starts_a_process(monkeypatch):
    started = []
    monkeypatch.setattr(
        "lamssi_agents.features.shell.subprocess.run",
        lambda *args, **kwargs: started.append((args, kwargs)),
    )

    result = _execute(
        CapabilityContext(),
        "ignored",
        "definitely-missing-directory",
        10,
        ["ignored"],
        "test",
    )

    assert started == []
    assert "does not exist" in result["error"]
