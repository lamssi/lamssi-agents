"""Configuration and installation for the sandboxed Files feature."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
)

from lamssi_agents.features.base import Feature
from lamssi_agents.features.files.hooks import as_write_hook
from lamssi_agents.tooling.dedupe import DedupePolicy, arg_subset_signature

_READ_INVALIDATORS = frozenset({"write_file", "edit_file", "delete_file"})
_DEDUPE = {
    "read_file": DedupePolicy(
        signature=arg_subset_signature(
            "path", "start_line", "end_line", "sheet", "max_rows"
        ),
        invalidated_by=_READ_INVALIDATORS,
        invalidation_key=lambda args: args.get("path"),
    ),
    "fs": DedupePolicy(
        # max_lines caps output like read_file's max_rows/end_line do, so it
        # belongs in the signature: a larger cap is a real re-request, not a repeat.
        signature=arg_subset_signature("command", "max_lines"),
        invalidated_by=_READ_INVALIDATORS,
    ),
}


if TYPE_CHECKING:
    from lamssi_agents.agent.base import Agent


def _free_check(space: Any, key: str, *, write: bool) -> Callable:
    """Build an approval check for one path argument."""

    def check(arguments: Mapping[str, Any]) -> bool:
        return space.call_is_free(dict(arguments or {}), key=key, write=write)

    check.__name__ = f"{key}_is_free"
    check.argument_key = key  # type: ignore[attr-defined]
    return check


def safe_when(space: Any) -> Dict[str, Callable]:
    """Return the Files feature's per-call in-scope checks."""
    from .search import fs_call_is_free

    return {
        "read_file": _free_check(space, "path", write=False),
        "fs": fs_call_is_free(space),
        "write_file": _free_check(space, "path", write=True),
        "edit_file": _free_check(space, "path", write=True),
    }


def remember_read_grant(space: Any) -> Callable[[str, Mapping[str, Any]], None]:
    """Build the observer that remembers an approved external read."""

    def hook(tool: str, arguments: Mapping[str, Any]) -> None:
        space.remember_read_approval(tool, dict(arguments or {}))

    hook.__name__ = "remember_read_grant"
    return hook


class Files(Feature):
    """Install the file tools, rooted at one workspace.

    Files only. Running a command is :class:`~lamssi_agents.features.shell.Shell`'s
    job and running Python is :class:`~lamssi_agents.features.code.Code`'s: file
    access does not grant execution.

    Args:
        root: Writable workspace directory, a callable returning the current
            workspace, or ``None`` to resolve :func:`Path.cwd` when used. Tools
            cannot escape this root for writes.
        read_only: Additional readable directories. Each entry is a path or
            ``(path, description)`` tuple shown to the model as a reference root.
        on_write: Callbacks invoked after a successful write, edit, or delete.
        protected_paths: Workspace-relative path components that file tools may
            not modify or delete. Defaults to ``.git`` and ``.lamssi``.

    Raises:
        ValueError: If a fixed ``root`` is not an existing directory.

    Example:
        Give the agent one workspace plus read-only documentation::

            Files(
                "./project",
                read_only=[("../manuals", "Product manuals")],
            )
    """

    name = "files"

    def __init__(
        self,
        root: Any = None,
        *,
        read_only: Sequence[Any] = (),
        on_write: Sequence[Any] = (),
        protected_paths: Sequence[str] = (".git", ".lamssi"),
    ) -> None:
        self.root = root
        self.read_only = tuple(read_only)
        self.on_write = tuple(on_write)
        self.protected_paths = tuple(protected_paths)

    def install(self, agent: "Agent") -> None:
        from .read import read_file
        from .search import fs
        from .space import FileSpace, ReadableDir
        from .write import delete_file, edit_file, write_file

        if self.root is not None:
            # A callable is kept as one: a host that switches project mid-process
            # would otherwise keep resolving paths against the directory it started in.
            root_of: Callable[[], Any]
            if callable(self.root):
                root_of = self.root
            else:
                resolved = Path(self.root).expanduser().resolve()
                if not resolved.is_dir():
                    raise ValueError(f"workspace does not exist: {resolved}")
                root_of = partial(_fixed, resolved)
        else:
            root_of = Path.cwd

        directories = []
        for entry in self.read_only:
            path, hint = (
                (entry + ("",))[:2] if isinstance(entry, tuple) else (entry, "")
            )
            resolved = Path(path).expanduser().resolve()
            directories.append(ReadableDir(resolved.name, resolved, hint))

        space = FileSpace(
            project_root=root_of,
            readable_dirs=directories,
            protected_paths=self.protected_paths,
        )
        agent.provide(FileSpace, space)
        agent.add_tools(read_file, fs, write_file, edit_file, delete_file)
        space.write_hooks.add(as_write_hook(_record_run_file))
        for hook in self.on_write:
            space.write_hooks.add(as_write_hook(hook))

        agent.add_safe_when(safe_when(space))
        agent.on_approved_call(remember_read_grant(space))
        agent.add_dedupe_policies(_DEDUPE)


def _fixed(value: Path) -> Path:
    """A root that does not move, as a callable like the host-supplied kind."""
    return value


def _record_run_file(event: Any) -> None:
    """Contribute one changed path to the structured run output."""
    path = str(event.rel_path).replace("\\", "/")
    event.control.record_output("files_written", path, unique=True)


__all__ = [
    "Files",
]
