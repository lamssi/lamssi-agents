"""Run every documented example in its own subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

#: These call a live model. Skipped by default: a test suite that needs an API
#: key is a test suite that does not run.
NEEDS_A_MODEL = {"01_hello.py", "16_ask_model_and_vision.py"}

#: Prevent the desktop examples from opening a window and blocking the suite.
CHILD_ENV = {**os.environ, "LAMSSI_EXAMPLE_HEADLESS": "1"}


def example_files() -> list:
    return sorted(p.name for p in EXAMPLES.glob("[0-9]*.py"))


def test_the_examples_directory_exists():
    assert EXAMPLES.is_dir(), "examples/ is where a new user starts"
    assert (EXAMPLES / "README.md").is_file()


def test_the_project_has_the_licence_it_declares():
    """The declared license file exists for wheel packaging."""
    import tomllib

    root = EXAMPLES.parent
    declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    named = declared["project"]["license"]["file"]
    text = (root / named).read_text(encoding="utf-8")

    assert "MIT License" in text
    assert "Lamssi Single Member P.C" in text


@pytest.mark.parametrize("name", example_files())
def test_every_example_runs_cleanly(name: str):
    """Run each example once and verify every runtime promise together."""
    if name in NEEDS_A_MODEL and not os.environ.get("LAMSSI_RUN_MODEL_EXAMPLES"):
        pytest.skip("needs a live model; set LAMSSI_RUN_MODEL_EXAMPLES=1")

    before = {p.name for p in EXAMPLES.iterdir()}
    result = subprocess.run(
        [sys.executable, name],
        cwd=EXAMPLES,
        capture_output=True,
        text=True,
        timeout=180,
        encoding="utf-8",
        errors="replace",
        env=CHILD_ENV,
    )
    assert result.returncode == 0, (
        f"{name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    offenders = sorted({c for c in result.stdout if ord(c) > 127})
    assert not offenders, (
        f"{name} printed non-ASCII {offenders}, which raises on a cp1252 console"
    )
    new = {p.name for p in EXAMPLES.iterdir()} - before - {"__pycache__"}
    assert not new, f"{name} left {sorted(new)} in examples/"


@pytest.mark.parametrize("name", example_files())
def test_every_example_is_safe_to_copy(name: str):
    """Copied examples retain their licence and never impose host configuration."""
    import ast

    source = (EXAMPLES / name).read_text(encoding="utf-8")
    head = source.splitlines()[0]
    assert head == "# SPDX-License-Identifier: MIT", name
    tree = ast.parse(source)
    offenders = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AugAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "environ"
    ]
    assert not offenders, (
        f"{name} writes to os.environ at {', '.join(offenders)}; an example reads "
        f"configuration, it does not impose it"
    )


def test_the_readme_indexes_every_example():
    """An example nobody can find is an example nobody reads."""
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    missing = [name for name in example_files() if name not in readme]
    assert not missing, f"not listed in examples/README.md: {missing}"
