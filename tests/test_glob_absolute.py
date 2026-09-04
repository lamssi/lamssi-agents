"""Absolute glob patterns in workspaces and approved external directories."""

from __future__ import annotations

from pathlib import Path

from lamssi_tools import CapabilityContext

from lamssi_agents.features.files.search import fs
from lamssi_agents.features.files.space import FileSpace, ReadableDir

def _ctx(space: FileSpace) -> CapabilityContext:
    ctx = CapabilityContext()
    ctx.register(FileSpace, space)
    return ctx

def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "core").mkdir(parents=True)
    (ws / "core" / "a.py").write_text("def foo(): return 1", encoding="utf-8")
    (ws / "core" / "b.py").write_text("x = 2", encoding="utf-8")
    return ws

def _posix(path: Path) -> str:
    """``fs`` command lines use forward slashes on every platform."""
    return str(path).replace("\\", "/")

def test_an_absolute_glob_into_the_workspace_lists_relative(tmp_path: Path):
    ws = _workspace(tmp_path)
    ctx = _ctx(FileSpace(project_root=lambda: ws))

    out = fs(ctx, command=f"ls {_posix(ws / 'core')}/*")

    assert "error" not in out, out
    assert sorted(out["lines"]) == ["core/a.py", "core/b.py"]

def test_an_absolute_glob_into_a_readable_dir_lists_it(tmp_path: Path):
    ws = _workspace(tmp_path)
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "lib.py").write_text("class Bar: pass", encoding="utf-8")
    ctx = _ctx(
        FileSpace(project_root=lambda: ws, readable_dirs=[ReadableDir("ref", ref, "")])
    )

    out = fs(ctx, command=f"ls {_posix(ref)}/*.py")

    assert "error" not in out, out
    assert out["lines"] == ["lib.py"]

def test_an_absolute_pattern_greps_file_contents(tmp_path: Path):
    ws = _workspace(tmp_path)
    ctx = _ctx(FileSpace(project_root=lambda: ws))

    out = fs(ctx, command=f"grep foo {_posix(ws / 'core')}")

    assert "error" not in out, out
    assert any("core/a.py" in line for line in out["lines"]), out["lines"]

def test_an_absolute_glob_outside_every_zone_lists_absolute(tmp_path: Path):
    ws = _workspace(tmp_path)
    ext = tmp_path / "elsewhere"
    ext.mkdir()
    (ext / "z.py").write_text("z = 9", encoding="utf-8")
    ctx = _ctx(FileSpace(project_root=lambda: ws))

    # The body globs fine; whether it prompted was settled by the approval gate before.
    out = fs(ctx, command=f"ls {_posix(ext)}/*.py")

    assert "error" not in out, out
    assert out["lines"] == [_posix(ext / "z.py")]

def test_a_relative_glob_is_unaffected(tmp_path: Path):
    ws = _workspace(tmp_path)
    ctx = _ctx(FileSpace(project_root=lambda: ws))

    out = fs(ctx, command="ls core/*.py")

    assert sorted(out["lines"]) == ["core/a.py", "core/b.py"]
