"""Build and test the wheel in an isolated environment.

The slow test verifies imports, packaged data, and one complete tool-calling turn.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.slow

_UV = shutil.which("uv")
requires_uv = pytest.mark.skipif(_UV is None, reason="uv is not installed")


def _run(
    *argv: str, cwd: Path | None = None, env: dict | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        cwd=str(cwd or REPO),
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    """Build the distribution and return the wheel."""
    out = tmp_path_factory.mktemp("dist")
    proc = _run(_UV, "build", "--out-dir", str(out))
    assert proc.returncode == 0, f"uv build failed:\n{proc.stdout}\n{proc.stderr}"
    wheels = list(out.glob("lamssi_agents-*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"
    return wheels[0]


@pytest.fixture(scope="module")
def lonely_python(tmp_path_factory, wheel: Path) -> Path:
    """A virtual environment containing the framework and nothing else of ours."""
    venv = tmp_path_factory.mktemp("venv")
    proc = _run(_UV, "venv", str(venv))
    assert proc.returncode == 0, f"uv venv failed:\n{proc.stderr}"

    python = venv / (
        "Scripts/python.exe" if sys.platform.startswith("win") else "bin/python"
    )
    proc = _run(_UV, "pip", "install", "--python", str(python), str(wheel))
    assert proc.returncode == 0, f"install failed:\n{proc.stdout}\n{proc.stderr}"
    return python


def _in_venv(python: Path, source: str) -> dict:
    """Run source against the installed wheel and parse its result marker."""
    import os

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["PYTHONNOUSERSITE"] = "1"
    proc = subprocess.run(
        [str(python), "-c", textwrap.dedent(source)],
        cwd=str(REPO.parent),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"failed in the clean venv:\n{proc.stdout}\n{proc.stderr}"
    )
    line = next(l for l in proc.stdout.splitlines() if l.startswith("@@RESULT@@"))
    return json.loads(line[len("@@RESULT@@") :])


@requires_uv
def test_the_wheel_exposes_every_declared_package(lonely_python: Path):
    result = _in_venv(
        lonely_python,
        """
        import importlib.util, json
        packages = {}
        for name in ("lamssi_agents", "lamssi_tools", "lamssi_cli", "lamssi_packages"):
            spec = importlib.util.find_spec(name)
            packages[name] = None if spec is None else spec.origin
        print("@@RESULT@@" + json.dumps(packages))
    """,
    )
    assert all(result.values()), f"wheel omitted packages: {result}"
    assert all("site-packages" in path for path in result.values()), result


@requires_uv
def test_import_is_inert_from_the_wheel(lonely_python: Path):
    result = _in_venv(
        lonely_python,
        """
        import json, sys
        import lamssi_agents
        loaded = sorted(m for m in sys.modules if m.startswith("lamssi_agents."))
        heavy = [m for m in ("docling", "pandas", "numpy", "PIL", "litellm")
                 if m in sys.modules]
        print("@@RESULT@@" + json.dumps({
            "version": lamssi_agents.__version__,
            "file": lamssi_agents.__file__,
            "handlers_loaded": any(
                m.startswith("lamssi_agents.features.files.")
                and m.rsplit(".", 1)[-1] in {"read", "write", "search", "document", "table"}
                for m in loaded
            ),
            "optional_deps": heavy,
        }))
    """,
    )
    assert "site-packages" in result["file"], (
        f"imported the checkout rather than the wheel: {result['file']}"
    )
    assert result["version"] == "0.1.0"
    assert not result["handlers_loaded"], "a bare import pulled in the file tools"
    assert not result["optional_deps"], (
        f"a bare import pulled in {result['optional_deps']}"
    )


@requires_uv
def test_data_files_shipped(lonely_python: Path):
    """The wheel contains bundled skill data files."""
    result = _in_venv(
        lonely_python,
        """
        import json
        from lamssi_agents import Agent, Skills
        from lamssi_agents.features.skills import SkillRuntime
        agent = Agent(features=[Skills(include_builtin=True)])
        runtime = agent.get(SkillRuntime)
        print("@@RESULT@@" + json.dumps({
            "skills": sorted(s.name for s in runtime.list()),
        }))
    """,
    )
    assert "code-assistance" in result["skills"], "the bundled skills are missing"


@requires_uv
def test_a_full_turn_runs_from_the_wheel(lonely_python: Path, tmp_path: Path):
    """The headline claim, demonstrated: a core-only runtime dispatches a real tool."""
    (tmp_path / "notes.txt").write_text(
        "hello from an isolated wheel\n", encoding="utf-8"
    )

    result = _in_venv(
        lonely_python,
        f"""
        import json
        from pathlib import Path
        from lamssi_agents import Agent
        from lamssi_agents import Files, Guidance, SystemTools
        from lamssi_agents.providers.models import StreamDelta, ToolCall, Usage

        project = Path(r"{tmp_path}")

        class Scripted:
            model = name = "scripted"
            is_local = supports_tools = True
            reasoning_effort = None
            def __init__(self):
                self._s = [
                    [StreamDelta(type="tool_call",
                                 tool_call=ToolCall(id="1", name="read_file",
                                                    arguments={{"path": "notes.txt"}})),
                     StreamDelta(type="done", finish_reason="tool_calls")],
                    [StreamDelta(type="text", text="The file says hello."),
                     StreamDelta(type="done", finish_reason="stop")],
                ]
                self._u = Usage()
            def stream(self, messages, tools=None, **kw): yield from self._s.pop(0)
            def check_connectivity(self): return True, "scripted"
            @property
            def cumulative_usage(self): return self._u

        agent = Agent(
            model=Scripted(),
            features=[SystemTools(), Guidance(), Files(project)],
        )
        answer = agent.chat("What does notes.txt say?")

        tool_results = [m.content for m in agent.history if m.role == "tool"]
        print("@@RESULT@@" + json.dumps({{
            "answer": answer,
            "tool_ran": bool(tool_results) and "isolated wheel" in tool_results[0],
            "tools": agent.available_tool_names(),
            "roles": [m.role for m in agent.history],
        }}))
    """,
    )

    assert result["tool_ran"], "read_file did not return the file's content"
    assert result["answer"] == "The file says hello."
    assert result["roles"] == ["user", "assistant", "tool", "assistant"]
    assert "read_file" in result["tools"]


@requires_uv
def test_the_cli_entry_point_works(lonely_python: Path):
    """``lamssi-agent --list-hosts`` runs with no host installed."""
    proc = subprocess.run(
        [str(lonely_python), "-m", "lamssi_cli", "--list-hosts"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "none" in proc.stdout
    assert "Only the null host is installed" in proc.stdout
