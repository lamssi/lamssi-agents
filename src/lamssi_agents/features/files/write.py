"""The Files feature's write, edit, and delete tools."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from lamssi_agents.agent.control import RunControl
from lamssi_tools import CapabilityContext, Expose, Str, err, tool

from .hooks import make_event
from .space import FileSpace, display_path, is_denied

log = logging.getLogger(__name__)


def _writable(space: FileSpace, path: str) -> tuple:
    """Resolve *path* for writing and return ``(target, route, error)``."""
    route = space.resolve(path, allow_external=True, write=True)
    if route.error is not None:
        return None, None, route.error
    if route.target is None:
        return None, None, err(f"Path {path!r} could not be resolved.", retriable=False)
    return route.target, route, None


def _after_write(
    ctx: CapabilityContext,
    space: FileSpace,
    rel: str,
    content: str,
    kind: str,
    action: str,
) -> Dict[str, Any]:
    """Notify the host and run the write hooks."""
    control = ctx.get(RunControl)

    # One path: everything reacts to a write via the hook chain, not a separate host callback.
    event = make_event(rel, content, kind, control, action)
    return space.write_hooks.fire(event) if event is not None else {}


@tool(
    group="files",
    expose=Expose.ALL,
    inject_context=True,
    description=(
        "Use this to create a new file, or to replace one completely. To change part "
        "of an existing file, use edit_file: overwriting loses everything you did not "
        "retype. "
        "Parent directories are created automatically. "
        "Overwriting replaces the whole file, so read it first unless you intend to "
        "discard what is there."
    ),
    # Safe in the workspace; a write outside it prompts.
    approval="conditional",
    keywords="save create new store put output export",
    parameters={
        "path": Str("Path to the file, relative to the workspace, or absolute."),
        "content": Str("Complete file content to write."),
    },
)
def write_file(
    ctx: CapabilityContext,
    path: str = "",
    content: str = "",
    **kw,
) -> Dict:
    if not path:
        return err("path is required", retriable=False)
    if content is None:
        content = ""
    space = ctx.require(FileSpace)

    target, route, problem = _writable(space, path)
    if problem is not None:
        return problem

    rel = route.display(target)
    problem = _protected_problem(
        ctx,
        rel,
        Path(rel).parts if route.root == "workspace" else (),
    )
    if problem is not None:
        return problem

    is_existing = target.is_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(content, encoding="utf-8")
    except Exception as e:
        return err(f"Write failed: {e}", retriable=False)

    log.info(
        "%s %s (%d lines)",
        "Updated" if is_existing else "Created",
        rel,
        len(content.splitlines()),
    )

    result: Dict[str, Any] = {
        "status": "updated" if is_existing else "created",
        "path": rel,
        "lines": len(content.splitlines()),
    }
    result.update(
        _after_write(
            ctx, space, rel, content, "write", "update" if is_existing else "create"
        )
    )
    return result


def _describe_difference(sent: str, found: str) -> str:
    """State how *sent* differs from *found*; the model can't spot it itself."""
    # Most common slip: escaped quotes in a literal compared byte for byte: checked first.
    unescaped = sent.replace('\\"', '"').replace("\\'", "'")
    if unescaped == found:
        return (
            'Your old_string escaped its quotes: it contains \\" where the file '
            'has ". old_string is matched literally, not parsed as a Python '
            "string, so send exactly the characters in the file."
        )
    if sent.strip() == found.strip():
        return (
            "The two differ only in leading or trailing whitespace. Copy the "
            "indentation from the file exactly."
        )

    at = next(
        (i for i, (a, b) in enumerate(zip(sent, found, strict=False)) if a != b),
        min(len(sent), len(found)),
    )
    return (
        f"They are identical up to character {at}, then yours has "
        f"{sent[at : at + 12]!r} where the file has {found[at : at + 12]!r}."
    )


def _no_match_payload(old_string: str, text: str) -> Dict[str, Any]:
    """Build a no-match error with the closest line when available."""
    payload: Dict[str, Any] = err("old_string not found in file", retriable=False)
    file_lines = text.splitlines()
    file_set = set(file_lines)
    candidate = ""
    for ln in old_string.splitlines():
        if not ln.strip():
            continue
        if ln not in file_set:
            candidate = ln
            break
    if not candidate:
        return payload
    from difflib import get_close_matches

    matches = get_close_matches(candidate, file_lines, n=1, cutoff=0.6)
    if not matches or matches[0] == candidate:
        return payload
    payload["hint"] = (
        f"This line from your old_string is not in the file: {candidate!r}. "
        f"The closest line in the file is: {matches[0]!r}. "
        + _describe_difference(candidate, matches[0])
    )
    return payload


@tool(
    group="files",
    expose=Expose.ALL,
    inject_context=True,
    description=(
        "Use this to change part of a file you already know the contents of: it is "
        "cheaper and safer than rewriting the whole thing. To create a file, or to "
        "replace one outright, use write_file. "
        "Replaces an exact text match. old_string must match exactly "
        "one location: include surrounding context to disambiguate. Set new_string "
        "to an empty string to delete the matched text."
    ),
    # Safe in the workspace; an edit outside it prompts.
    approval="conditional",
    keywords="change modify update rename replace fix patch amend refactor tweak",
    parameters={
        "path": Str("Path to the file, relative to the workspace, or absolute."),
        "old_string": Str("Exact text to find; must match exactly once."),
        "new_string": Str("Replacement text."),
    },
)
def edit_file(
    ctx: CapabilityContext,
    path: str = "",
    old_string: str = "",
    new_string: str = "",
    **kw,
) -> Dict:
    if not path:
        return err("path is required", retriable=False)
    if not old_string:
        return err("old_string is required", retriable=False)
    space = ctx.require(FileSpace)

    target, route, problem = _writable(space, path)
    if problem is not None:
        return problem
    rel = route.display(target)
    problem = _protected_problem(
        ctx,
        rel,
        Path(rel).parts if route.root == "workspace" else (),
    )
    if problem is not None:
        return problem
    if not target.is_file():
        return err(f"File not found: {path}", retriable=False)

    try:
        text = target.read_text("utf-8")
    except Exception as e:
        return err(f"Read error: {e}", retriable=False)

    count = text.count(old_string)
    if count == 0:
        return _no_match_payload(old_string, text)
    if count > 1:
        return err(
            f"old_string matched {count} locations: add more context.",
            retriable=False,
        )

    new_text = text.replace(old_string, new_string, 1)
    try:
        target.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return err(f"Write failed: {e}", retriable=False)

    log.info("Edited %s", rel)

    result: Dict[str, Any] = {
        "status": "edited",
        "path": rel,
        "lines": len(new_text.splitlines()),
    }
    result.update(_after_write(ctx, space, rel, new_text, "edit", "edit"))
    return result


@tool(
    group="files",
    expose=Expose.ALL,
    inject_context=True,
    description=(
        "Delete a file or directory. Set recursive=true to remove a non-empty "
        "directory. A symbolic link is removed without deleting its target."
    ),
    # Destructive, and no argument pattern makes a delete safe, so it always prompts.
    approval="always",
    keywords="remove erase rm unlink discard clean",
    parameters={
        "path": Str(
            "Path to the file or directory, relative to the workspace, or absolute."
        )
    },
)
def delete_file(
    ctx: CapabilityContext,
    path: str = "",
    recursive: bool = False,
    **kw,
) -> Dict:
    if not path:
        return err("path is required", retriable=False)
    space = ctx.require(FileSpace)

    try:
        candidate = Path(path).expanduser()
        requested = Path(
            os.path.abspath(
                candidate if candidate.is_absolute() else space.workspace() / candidate
            )
        )
    except (OSError, RuntimeError, ValueError):
        return err(f"Path {path!r} could not be resolved.", retriable=False)
    is_symlink = requested.is_symlink()
    is_junction = bool(
        getattr(requested, "is_junction", lambda: False)()
    )
    is_link = is_symlink or is_junction

    target, route, problem = _writable(space, str(requested) if is_link else path)
    if problem is not None:
        return problem
    if is_link and is_denied(requested):
        return err(
            f"Access to {path!r} is refused: it matches the sensitive-path denylist.",
            retriable=False,
        )

    base = route.base
    if base is not None and target == base and not is_link:
        return err(
            "Refusing to delete the workspace directory itself.", retriable=False
        )

    if is_link:
        rel = display_path(requested, space.workspace())
        try:
            workspace_parts = requested.relative_to(space.workspace()).parts
        except ValueError:
            workspace_parts = ()
    else:
        rel = route.display(target)
        workspace_parts = Path(rel).parts if route.root == "workspace" else ()
    problem = _protected_problem(ctx, rel, workspace_parts)
    if problem is not None:
        return problem

    if not is_link and not target.exists():
        return err(f"Path not found: {rel}", retriable=False)

    import shutil

    try:
        if is_link:
            if is_junction and not is_symlink:
                requested.rmdir()
            else:
                requested.unlink()
            kind = "link"
        elif target.is_dir():
            if any(target.iterdir()) and not recursive:
                return err(
                    f"{rel!r} is a non-empty directory: pass recursive=true to "
                    "delete it and its contents.",
                    retriable=False,
                )
            shutil.rmtree(target)
            kind = "directory"
        else:
            target.unlink()
            kind = "file"
    except Exception as e:
        return err(f"Delete failed: {e}", retriable=False)

    log.info("Deleted %s %s", kind, rel)
    result: Dict[str, Any] = {"status": "deleted", "path": rel, "kind": kind}
    result.update(_after_write(ctx, space, rel, "", "delete", "delete"))
    return result


def _protected_paths(ctx: CapabilityContext) -> frozenset:
    """Top-level names this tool must never delete.

    Taken from the file-space policy installed by ``Files``.
    """
    space = ctx.get(FileSpace)
    if space is None:
        return frozenset({".git"})
    return space.protected_paths()


def _protected_problem(
    ctx: CapabilityContext,
    rel: str,
    workspace_parts: tuple[str, ...],
) -> Dict[str, Any] | None:
    """Return an error when a workspace path contains a protected component."""
    protected = _protected_paths(ctx)
    if not protected.intersection(workspace_parts):
        return None
    return err(
        f"Refusing to modify a protected path: {rel}",
        retriable=False,
        protected=sorted(protected),
    )
