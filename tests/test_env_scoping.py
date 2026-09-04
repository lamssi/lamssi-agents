"""Credential filtering for shell subprocesses.

Tests inspect child-process output and cover shells that load user profiles.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from lamssi_agents.redaction import (
    allow_env,
    clear_env_allowances,
    forget_secrets,
    is_secret_name,
    register_secret,
    safe_environ,
)
from lamssi_agents.features.shell import detect_shell, run_bash, run_powershell
from lamssi_tools import CapabilityContext

#: Shaped like the real thing, never a live credential.
FAKE_ANTHROPIC = "sk-ant-api03-ENVSCOPETESTKEY1234567890abcdefGHIJ"
FAKE_OPAQUE = "b81c4d0af62e4937bb15e8c7d024af93"

#: Whichever tool this machine would actually install: the platform no longer
#: decides it, since a Windows box with Git Bash gets bash.
_SHELL = detect_shell()

#: One command per shell that prints the whole environment. Deliberately the
#: thing an attacker would actually ask for.
DUMP_ENV = (
    'Get-ChildItem Env: | ForEach-Object { "$($_.Name)=$($_.Value)" }'
    if _SHELL == "powershell"
    else "env"
)

@pytest.fixture(autouse=True)
def clean_registry():
    """No leakage between tests: both registries are process-global."""
    forget_secrets()
    clear_env_allowances()
    yield
    forget_secrets()
    clear_env_allowances()

def shell(command: str) -> dict:
    run = run_powershell if _SHELL == "powershell" else run_bash
    return run(CapabilityContext(), command=command, timeout=30)

# the artefact: what the child actually receives

def test_a_provider_key_does_not_reach_a_shell_command(monkeypatch):
    """Provider credentials do not reach a child shell environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)

    out = shell(DUMP_ENV)

    assert out.get("exit_code") == 0, out
    assert FAKE_ANTHROPIC not in out["stdout"]
    assert "ANTHROPIC_API_KEY" not in out["stdout"]

def test_a_key_under_an_innocuous_name_does_not_reach_it_either(monkeypatch):
    """Credential-shaped values are removed under neutral variable names."""
    monkeypatch.setenv("HELPER_CONFIG", FAKE_ANTHROPIC)

    assert FAKE_ANTHROPIC not in shell(DUMP_ENV)["stdout"]

def test_a_registered_secret_does_not_reach_it_whatever_it_looks_like(monkeypatch):
    """Registered secrets are removed regardless of name or format."""
    monkeypatch.setenv("MY_GATEWAY", FAKE_OPAQUE)
    register_secret(FAKE_OPAQUE)

    assert FAKE_OPAQUE not in shell(DUMP_ENV)["stdout"]

def test_ordinary_commands_still_work():
    """Environment scrubbing preserves variables required by ordinary commands."""
    out = shell("echo hello-from-the-shell")

    assert out.get("exit_code") == 0, out
    assert "hello-from-the-shell" in out["stdout"]
    assert "PATH" in {name.upper() for name in safe_environ()}

# the shell's own profile, which would undo all of it

def _argv_of(run, command: str, monkeypatch) -> list:
    """Capture the argv that *run* would spawn."""
    seen: dict = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run(CapabilityContext(), command=command, timeout=30)
    return seen["argv"]

def test_the_posix_shell_does_not_read_the_user_profile(monkeypatch):
    """Use a non-login shell so profiles cannot restore filtered variables."""
    argv = _argv_of(run_bash, "true", monkeypatch)

    assert argv[1:] == ["-c", "true"]
    assert "-lc" not in argv, "a login shell re-sources the profile"

def test_the_windows_shell_does_not_read_the_user_profile(monkeypatch):
    """Same measure, already in place on the other platform: pinned so it
    stays that way."""
    assert "-NoProfile" in _argv_of(run_powershell, "echo hi", monkeypatch)

def test_bash_is_never_the_wsl_launcher(monkeypatch):
    """Bash execution excludes the Windows WSL launcher."""
    argv = _argv_of(run_bash, "true", monkeypatch)
    assert Path(argv[0]).parent.name.lower() != "system32", (
        f"resolved WSL's launcher: {argv[0]}"
    )

# which names count

@pytest.mark.parametrize("name", [
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY",
    "GROQ_API_KEY", "HF_TOKEN", "GITHUB_TOKEN", "GH_TOKEN",
    "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS", "SSH_AUTH_SOCK", "DB_PASSWORD",
    "CLIENT_SECRET", "NPM_TOKEN", "DOCKER_PASSWORD", "STRIPE_SECRET_KEY",
])
def test_credential_names_are_recognised(name: str):
    assert is_secret_name(name), f"{name} should be treated as a credential"
    assert name not in safe_environ({name: "some-value-here"})

@pytest.mark.parametrize("name", [
    "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TEMP", "TMPDIR", "LANG",
    "VIRTUAL_ENV", "PYTHONPATH", "PATHEXT", "COMSPEC", "PROCESSOR_ARCHITECTURE",
    # The ones a substring match would get wrong, which is why names are
    # split into parts rather than searched.
    "TOKENIZERS_PARALLELISM",   # a real Hugging Face setting
    "KEYBOARD_LAYOUT",
    "SESSIONNAME",              # Windows
    "AUTHOR",
    "MONKEYPATCH_HOME",
])
def test_ordinary_names_survive(name: str):
    assert not is_secret_name(name), f"{name} is not a credential"
    assert safe_environ({name: "ordinary-value"}) == {name: "ordinary-value"}

# which values count

def test_a_connection_string_with_a_password_is_dropped():
    """``DATABASE_URL`` names nothing, and carries a password anyway."""
    env = {"DATABASE_URL": "postgres://admin:hunter2@db.internal:5432/app"}

    assert safe_environ(env) == {}

def test_a_plain_url_survives():
    """The same rule must not eat every endpoint setting."""
    env = {"API_BASE": "https://api.example.com/v1", "PROXY": "http://proxy:3128"}

    assert safe_environ(env) == env

def test_a_vendor_shaped_value_is_dropped_under_any_name():
    assert safe_environ({"SOMETHING_HARMLESS": FAKE_ANTHROPIC}) == {}

def test_an_unregistered_opaque_value_is_not_guessed_at():
    """Neutral opaque values require explicit secret registration."""
    env = {"BUILD_ID": FAKE_OPAQUE}

    assert safe_environ(env) == env

# the host's opt-in

def test_a_host_can_allow_one_variable_through():
    """A host can exempt one named environment variable."""
    env = {"GH_TOKEN": "ghp_realish", "ANTHROPIC_API_KEY": FAKE_ANTHROPIC}

    allow_env("gh_token")

    assert safe_environ(env) == {"GH_TOKEN": "ghp_realish"}

def test_keep_works_per_call():
    env = {"GH_TOKEN": "x-value-here"}

    assert safe_environ(env, keep=("GH_TOKEN",)) == env
    assert safe_environ(env) == {}

# the model is told, but only when it matters

def test_a_failing_command_explains_the_missing_credentials(monkeypatch):
    """Failed commands report that credentials were removed from the environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)

    out = shell("exit 3")

    assert out["exit_code"] == 3
    assert "ANTHROPIC_API_KEY" in out["note"]
    assert "not retry" in out["note"].lower() or "do not retry" in out["note"].lower()
    assert FAKE_ANTHROPIC not in out["note"], "name the variable, never the value"

def test_a_succeeding_command_says_nothing_about_it(monkeypatch):
    """The note is diagnostic, not a per-turn tax: a successful command pays
    nothing for it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)

    assert "note" not in shell("echo ok")

# the source is never modified

def test_the_real_environment_is_left_alone(monkeypatch):
    """Environment filtering does not mutate the process environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)

    safe_environ()

    assert os.environ["ANTHROPIC_API_KEY"] == FAKE_ANTHROPIC
