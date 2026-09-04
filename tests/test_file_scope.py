"""File scope for workspaces, reference directories, external paths, and secrets."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lamssi_agents.features.files.read import read_file
from lamssi_agents.features.files.search import fs_paths
from lamssi_agents.features.files.space import (
    FileSpace,
    ReadableDir,
    _within,
    is_denied,
    suggest_near_match,
)
from lamssi_agents.features.files.write import (
    _writable,
    delete_file,
    edit_file,
    write_file,
)
from lamssi_tools import CapabilityContext


def _space(root: Path) -> FileSpace:
    return FileSpace(project_root=lambda: root)


# the denylist: never, even in the workspace


@pytest.mark.parametrize(
    "name", [".env", ".env.local", "id_rsa", "server.pem", "app.key", "aws-credentials"]
)
def test_a_secret_in_the_workspace_is_refused_for_reading(tmp_path: Path, name: str):
    """Secret filename patterns are denied with or without a leading dot."""
    (tmp_path / name).write_text("SECRET", encoding="utf-8")
    route = _space(tmp_path).resolve(name)
    assert route.error is not None, f"{name} should be denied"
    assert "denylist" in route.error["error"].lower()
    assert route.target is None
    assert is_denied(tmp_path / name) is True


@pytest.mark.parametrize("name", [".env", "id_rsa", "server.pem"])
def test_a_secret_is_refused_for_writing_too(tmp_path: Path, name: str):
    """Apply the secrets denylist to writes, including non-dot files."""
    target, _route, err = _writable(_space(tmp_path), name)
    assert target is None and err is not None
    assert "denylist" in err["error"].lower()


def test_a_file_under_a_dot_ssh_directory_is_refused(tmp_path: Path):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    (ssh / "known_hosts").write_text("x", encoding="utf-8")
    route = _space(tmp_path).resolve(".ssh/known_hosts")
    assert route.error is not None and "denylist" in route.error["error"].lower()


def test_an_ordinary_file_is_not_denied(tmp_path: Path):
    (tmp_path / "config.py").write_text("x", encoding="utf-8")
    route = _space(tmp_path).resolve("config.py")
    assert route.error is None and route.target is not None
    assert is_denied(tmp_path / "config.py") is False


# the free check: the workspace is the only write-free zone


def test_the_workspace_is_free_and_everything_else_asks(tmp_path: Path):
    """Workspace access is free while external writes still require approval."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    ref = tmp_path / "ref"  # a reference dir OUTSIDE the workspace
    ref.mkdir()
    space = FileSpace(
        project_root=lambda: workspace,
        readable_dirs=[ReadableDir("reference", ref.resolve(), "")],
    )
    outside = str((tmp_path / "outside.txt").resolve())
    in_ref = str((ref / "note.md").resolve())

    # in-workspace relative path is free to read AND write: the only write-free zone
    assert space.call_is_free({"path": "src/x.py"}, key="path") is True
    assert space.call_is_free({"path": "src/x.py"}, key="path", write=True) is True
    # a bare absolute path outside every zone is never free
    assert space.call_is_free({"path": outside}, key="path") is False
    # a readable dir is free to read, but a write there is not free (it prompts)
    assert space.call_is_free({"path": in_ref}, key="path") is True
    assert space.call_is_free({"path": in_ref}, key="path", write=True) is False


def test_a_missing_path_never_bypasses_approval(tmp_path: Path):
    space = _space(tmp_path)

    assert space.is_free("") is False
    assert space.call_is_free({}, key="path") is False
    assert space.call_is_free({"other": "note.txt"}, key="path") is False


def test_fs_approval_uses_the_same_source_parsers_as_execution():
    assert fs_paths("grep -e needle src tests") == ["src", "tests"]
    assert fs_paths("find src -name '*.py' -maxdepth 2") == ["src"]
    assert fs_paths("tree docs -L 3 | head -20") == ["docs"]
    assert fs_paths("grep -e") is None


@pytest.mark.skipif(os.name != "nt", reason="Windows path casing")
def test_windows_containment_ignores_path_casing(tmp_path: Path):
    root = tmp_path.resolve()
    target = root / "Folder" / "file.txt"

    assert _within(Path(str(target).swapcase()), root)


# writes reach outside the workspace (approval-gated upstream)


def test_write_can_target_outside_the_workspace(tmp_path: Path):
    outside = (tmp_path.parent / "ext_write.txt").resolve()
    target, route, err = _writable(_space(tmp_path), str(outside))
    assert err is None, err
    assert route.external and target == outside


# a readable dir exposes its whole tree (prefix restriction removed)


def test_a_readable_dir_is_fully_readable(tmp_path: Path):
    """A registered reference directory is readable throughout its tree."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    ref = tmp_path / "framework"
    (ref / "core" / "deep").mkdir(parents=True)
    (ref / "core" / "deep" / "engine.py").write_text("x", encoding="utf-8")
    ref = ref.resolve()
    space = FileSpace(
        project_root=lambda: workspace,
        readable_dirs=[ReadableDir("framework", ref, "")],
    )

    route = space.resolve(str(ref / "core" / "deep" / "engine.py"))

    assert route.error is None
    assert route.target is not None
    assert route.base == ref
    assert route.root == "reference:framework"
    assert route.display() == "core/deep/engine.py"


def test_read_file_labels_a_declared_reference_root(
    tmp_path: Path,
):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    target = reference / "guide.md"
    target.write_text("reference text", encoding="utf-8")
    space = FileSpace(
        project_root=lambda: workspace,
        readable_dirs=[ReadableDir("manual", reference.resolve(), "")],
    )
    ctx = CapabilityContext({FileSpace: space})

    result = read_file(ctx, path=str(target))

    assert result["root"] == "reference:manual"
    assert result["path"] == "guide.md"


def test_a_protected_name_above_the_workspace_does_not_block_deletion(
    tmp_path: Path,
):
    parent = tmp_path / "plugins"
    workspace = parent / "workspace"
    workspace.mkdir(parents=True)
    target = workspace / "ordinary.txt"
    target.write_text("delete me", encoding="utf-8")
    space = FileSpace(
        project_root=lambda: workspace,
        protected_paths=("plugins",),
    )

    result = delete_file(CapabilityContext({FileSpace: space}), path="ordinary.txt")

    assert result["status"] == "deleted"
    assert not target.exists()


def test_a_protected_workspace_component_still_blocks_deletion(tmp_path: Path):
    protected = tmp_path / "plugins"
    protected.mkdir()
    target = protected / "app.py"
    target.write_text("keep me", encoding="utf-8")
    space = FileSpace(
        project_root=lambda: tmp_path,
        protected_paths=("plugins",),
    )

    result = delete_file(
        CapabilityContext({FileSpace: space}),
        path="plugins/app.py",
    )

    assert "protected" in result["error"].lower()
    assert target.exists()


def test_protected_workspace_components_block_writes_and_edits(tmp_path: Path):
    protected = tmp_path / ".lamssi"
    protected.mkdir()
    target = protected / "state.json"
    target.write_text("original", encoding="utf-8")
    context = CapabilityContext(
        {FileSpace: FileSpace(project_root=lambda: tmp_path, protected_paths=(".lamssi",))}
    )

    write_result = write_file(
        context,
        path=".lamssi/state.json",
        content="replaced",
    )
    edit_result = edit_file(
        context,
        path=".lamssi/state.json",
        old_string="original",
        new_string="edited",
    )

    assert "protected" in write_result["error"].lower()
    assert "protected" in edit_result["error"].lower()
    assert target.read_text(encoding="utf-8") == "original"


def test_deleting_a_symlink_keeps_its_target(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("keep me", encoding="utf-8")
    link = workspace / "outside-link.txt"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    space = FileSpace(project_root=lambda: workspace)
    result = delete_file(
        CapabilityContext({FileSpace: space}),
        path="outside-link.txt",
    )

    assert result == {
        "status": "deleted",
        "path": "outside-link.txt",
        "kind": "link",
    }
    assert not link.exists()
    assert not link.is_symlink()
    assert target.read_text(encoding="utf-8") == "keep me"


# the "did you mean?" hint stays inside the sandbox


def test_a_missing_file_in_the_root_does_not_scan_outside_it(tmp_path: Path):
    """Missing-file suggestions stay inside the workspace."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    neighbour = tmp_path / "other"
    neighbour.mkdir()
    (neighbour / "README.md").write_text("not yours\n", encoding="utf-8")

    # Must not raise, and must not point at the neighbour.
    assert suggest_near_match(workspace / "README.md", workspace) is None


def test_a_sibling_directory_inside_the_root_is_still_suggested(tmp_path: Path):
    """The clamp must not cost the case the sweep exists for."""
    workspace = tmp_path / "ws"
    (workspace / "alpha").mkdir(parents=True)
    (workspace / "beta").mkdir()
    (workspace / "beta" / "app.py").write_text("x = 1\n", encoding="utf-8")

    assert (
        suggest_near_match(workspace / "alpha" / "app.py", workspace) == "beta/app.py"
    )


def test_read_file_reports_not_found_rather_than_raising(tmp_path: Path):
    """Return a useful result when the requested file is missing."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    neighbour = tmp_path / "other-project"
    neighbour.mkdir()
    (neighbour / "README.md").write_text("not yours\n", encoding="utf-8")

    ctx = CapabilityContext()
    ctx.register(FileSpace, _space(workspace))
    result = read_file(ctx, path="README.md")

    assert "error" in result
    assert "not found" in result["error"].lower()
    assert "other-project" not in str(result)
