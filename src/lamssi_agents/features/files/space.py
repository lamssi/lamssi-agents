"""Workspace resolution and access rules used by the Files feature."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lamssi_agents.features.files.hooks import HookChain
from lamssi_agents.runtime.scope import active_agent
from lamssi_tools import err

log = logging.getLogger(__name__)

#: Credential/secret paths that must never be read or written; matched on basename or a sensitive directory component. Cannot be approved around.
_DENY_NAME_PATTERNS = (
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "id_dsa*",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "*credentials*",
    "*secret*.json",
    "secrets.*",
)
_DENY_DIR_PARTS = frozenset({".ssh", ".gnupg", ".aws", ".azure", ".kube", "gcloud"})

#: Read tools mapped to their path/glob argument, used to pin an approved read's directory.
_READ_TOOL_KEYS = {
    "read_file": "path",
}


def is_denied(target: Path) -> bool:
    """Whether *target* is a credential/secret path the file tools must refuse.

    Checked per file (not just per call) because search reads files directly,
    so this is what stops a wide search from surfacing a private key.
    """
    name = target.name.lower()
    if any(fnmatch(name, pat) for pat in _DENY_NAME_PATTERNS):
        return True
    return bool({p.lower() for p in target.parts} & _DENY_DIR_PARTS)


def _granted_dirs() -> tuple:
    """Read grants belonging to the agent in progress, or ``()`` outside a run."""
    agent = active_agent()
    if agent is None:
        return ()
    grants = agent.conversation_state(ReadGrants, ReadGrants)
    return grants.directories


class ReadGrants:
    """External read directories approved for one Agent conversation."""

    __slots__ = ("_directories",)

    def __init__(self) -> None:
        self._directories: set[Path] = set()

    @property
    def directories(self) -> Tuple[Path, ...]:
        return tuple(self._directories)

    def grant(self, directory: Path) -> None:
        self._directories.add(Path(directory).resolve())

    def on_new_turn(self) -> None:
        self._directories.clear()

    on_cleared = on_new_turn


@dataclass(frozen=True, slots=True)
class ReadableDir:
    """A reference tree the agent may read without prompting.

    Named and hinted only for the prompt section; the model reads by
    absolute path, not by name.
    """

    name: str
    path: Path
    hint: str = ""

    def describe(self) -> str:
        return (
            f"{self.path.as_posix()}: {self.hint}"
            if self.hint
            else self.path.as_posix()
        )


@dataclass(frozen=True, slots=True)
class Resolution:
    """The outcome of resolving one path or glob."""

    #: The resolved file, or ``None`` on error.
    target: Optional[Path] = None
    #: A ready-to-return error payload, or ``None``.
    error: Optional[Dict[str, Any]] = None
    #: True when the path is absolute and outside every free zone: resolved but approval-gated.
    external: bool = False
    #: The free zone the path landed in, for display; ``None`` for an external path.
    base: Optional[Path] = None
    #: Stable route classification returned to the model.
    root: str = "external"

    def display(self, target: Optional[Path] = None) -> str:
        """A path relative to its named root, or absolute when it has none."""
        t = target if target is not None else self.target
        return display_path(t, self.base)


class FileSpace:
    """The workspace, the read-free reference dirs, and the write-hook chain."""

    def __init__(
        self,
        *,
        project_root: Optional[Callable[[], Path]] = None,
        readable_dirs: Sequence[ReadableDir] = (),
        protected_paths: Sequence[str] = (".git",),
    ) -> None:
        # Resolved per call, not captured: a captured path would pin the agent to the first project.
        self._project_root = project_root or Path.cwd
        self._readable: Tuple[ReadableDir, ...] = tuple(readable_dirs)
        self._protected_paths = frozenset(
            str(p).replace("\\", "/") for p in protected_paths
        )
        self.write_hooks = HookChain()

    def workspace(self) -> Path:
        """The project directory: read and write, no prompt."""
        try:
            return Path(os.fspath(self._project_root())).resolve()
        except Exception as exc:
            log.debug("project root provider raised: %s", exc)
            return Path.cwd().resolve()

    def readable_dirs(self) -> Tuple[ReadableDir, ...]:
        """The reference trees the agent may read freely. For the prompt section."""
        return self._readable

    def protected_paths(self) -> frozenset[str]:
        """Workspace-relative paths file deletion must never remove."""
        return self._protected_paths

    def add_readable(self, name: str, path: Path, hint: str = "") -> None:
        """Register a read-free reference directory."""
        self._readable = (
            *self._readable,
            ReadableDir(name, Path(path).resolve(), hint),
        )

    def _free_base(self, target: Path, *, write: bool) -> Optional[Path]:
        """Return the approval-free base containing *target*, if any."""
        ws = self.workspace()
        if _within(target, ws):
            return ws
        if not write:
            for rd in self._readable:
                if _within(target, rd.path):
                    return rd.path
            for granted in _granted_dirs():
                if _within(target, granted):
                    return granted
        return None

    def _root_for_base(self, base: Path) -> str:
        """Classify one free-zone base without leaking routing policy to tools."""
        if base == self.workspace():
            return "workspace"
        for readable in self._readable:
            if base == readable.path:
                return f"reference:{readable.name}"
        return "external"

    def remember_read_approval(self, tool_name: str, arguments: Any) -> Optional[Path]:
        """Grant the approved external read directory for the active run."""
        key = _READ_TOOL_KEYS.get(tool_name)
        if key is None:
            return None
        value = str((arguments or {}).get(key) or "")
        if not value:
            return None
        resolved = _expand_under(self.workspace(), value)
        if resolved is None:
            return None
        directory = resolved if resolved.is_dir() else resolved.parent
        if is_denied(directory) or _within(directory, self.workspace()):
            return None
        agent = active_agent()
        if agent is None:
            return None
        agent.conversation_state(ReadGrants, ReadGrants).grant(directory)
        return directory

    def approved_dirs(self) -> Tuple[Path, ...]:
        """Directories the active run has approved reads into."""
        return tuple(_granted_dirs())

    def resolve(
        self, path: str, *, allow_external: bool = False, write: bool = False
    ) -> Resolution:
        """Resolve *path* to a file.

        Relative paths are taken under the workspace. An absolute path is free
        if it lands in a free zone, otherwise external (approval-gated when
        ``allow_external``) or refused if it matches the denylist. A read-only
        reference directory is a free zone for reads but not for writes.
        """
        candidate = Path(path)

        if candidate.is_absolute():
            try:
                target = candidate.expanduser().resolve()
            except (OSError, ValueError):
                return Resolution(error=self._bad_path(path))
            if is_denied(target):
                return Resolution(error=self._denied(path))
            base = self._free_base(target, write=write)
            if base is not None:
                return Resolution(
                    target=target,
                    base=base,
                    root=self._root_for_base(base),
                )
            if allow_external:
                return Resolution(target=target, external=True, root="external")
            return Resolution(error=self._outside(path))

        ws = self.workspace()
        try:
            target = (ws / path).resolve()
        except (OSError, ValueError):
            return Resolution(error=self._bad_path(path))
        if not _within(target, ws):
            return Resolution(error=self._outside(path))
        if is_denied(target):
            return Resolution(error=self._denied(path))
        return Resolution(target=target, base=ws, root="workspace")

    def resolve_glob(self, pattern: str) -> Resolution:
        """Resolve the base directory and scope classification for a glob."""
        anchor = Path(_glob_anchor(pattern))
        ws = self.workspace()
        if not anchor.is_absolute():
            try:
                resolved = (ws / anchor).resolve()
            except (OSError, ValueError):
                return Resolution(error=self._bad_path(pattern))
            if not _within(resolved, ws):
                return Resolution(error=self._outside(pattern))
            return Resolution(base=ws, root="workspace")
        try:
            resolved = anchor.expanduser().resolve()
        except (OSError, ValueError):
            resolved = anchor
        base = self._free_base(resolved, write=False)
        if base is not None:
            return Resolution(base=base, root=self._root_for_base(base))
        return Resolution(external=True, root="external")

    def is_free(self, path: str, *, write: bool = False) -> bool:
        """Return whether *path* is allowed without approval."""
        if not path:
            return False
        resolved = _expand_under(self.workspace(), path)
        if resolved is None:
            return False
        if is_denied(resolved):
            return False
        return self._free_base(resolved, write=write) is not None

    def call_is_free(
        self, arguments: Dict[str, Any], *, key: str, write: bool = False
    ) -> bool:
        """:meth:`is_free` for the argument at *key* of a tool call.

        ``True`` means the call stays in a free zone and the approval prompt can be
        skipped. For a glob argument the wildcard-free anchor is what is checked.
        """
        if not arguments:
            return False
        value = str(arguments.get(key) or "")
        if not value:
            return False
        return self.is_free(value, write=write)

    def _outside(self, path: str) -> Dict[str, Any]:
        return err(
            f"Path {path!r} is outside the workspace.",
            hint=(
                "Use a path relative to the workspace, or an absolute path: an absolute "
                "path outside the workspace and the reference dirs is read/written only "
                "with the user's approval."
            ),
            retriable=False,
        )

    def _bad_path(self, path: str) -> Dict[str, Any]:
        return err(f"Path {path!r} could not be resolved.", retriable=False)

    def _denied(self, path: str) -> Dict[str, Any]:
        return err(
            f"Access to {path!r} is refused: it matches the sensitive-path denylist "
            "(credentials, private keys, .env and similar). This boundary cannot be "
            "approved around.",
            retriable=False,
        )

    def __repr__(self) -> str:
        return f"<FileSpace readable={[d.name for d in self._readable]}>"


def _within(target: Path, root: Path) -> bool:
    """Return whether *target* is *root* or one of its descendants."""
    target_text = os.path.normcase(str(target))
    root_text = os.path.normcase(str(root))
    return target_text == root_text or target_text.startswith(root_text + os.sep)


def has_glob(s: str) -> bool:
    """Whether *s* contains a glob metacharacter (``*``, ``?`` or ``[``)."""
    return any(ch in s for ch in "*?[")


def _glob_anchor(pattern: str) -> str:
    """The longest leading part of *pattern* containing no wildcard."""
    norm = pattern.replace("\\", "/").rstrip("/")
    if not norm:
        return "."
    parts: List[str] = []
    for part in norm.split("/"):
        if has_glob(part):
            break
        parts.append(part)
    return "/".join(parts) or "."


def _looks_like_path_arg(value: str) -> bool:
    """A crude "is this a path, not a glob" test, for :meth:`FileSpace.is_free`."""
    return not has_glob(value)


def _expand_under(workspace: Path, raw: str) -> Optional[Path]:
    """Resolve a path or glob anchor under *workspace*."""
    anchor = raw if _looks_like_path_arg(raw) else _glob_anchor(raw)
    try:
        p = Path(anchor)
        return (
            p.expanduser().resolve()
            if p.is_absolute()
            else (workspace / anchor).resolve()
        )
    except (OSError, ValueError):
        return None


def path_is_hidden_or_pycache(parts: Sequence[str]) -> bool:
    return any(
        part == "__pycache__" or (part.startswith(".") and part not in (".", ".."))
        for part in parts
    )


def suggest_near_match(target: Path, root: Path) -> Optional[str]:
    """A sibling with the same basename, when *target* does not exist.

    Saves the two extra turns of being told the file is missing, then
    listing the directory to find the typo.

    Every candidate is confined to *root*. The sibling-directory sweep below
    starts one level above *target*, which for a path directly in the root is the
    root's own parent: so unclamped it scans directories the agent may not read
    and names files outside the sandbox.
    """
    parent = target.parent
    name = target.name.lower()
    if not parent.exists() or not name or not _within(parent, root):
        return None
    try:
        for entry in parent.iterdir():
            if entry.name.lower() == name and entry != target:
                return _relative(entry, root)
    except (OSError, PermissionError):
        return None
    grandparent = parent.parent
    if grandparent.exists() and _within(grandparent, root):
        try:
            for sub in grandparent.iterdir():
                if not sub.is_dir():
                    continue
                cand = sub / target.name
                if cand.is_file():
                    return _relative(cand, root)
        except (OSError, PermissionError):
            pass
    return None


def display_path(path: Optional[Path], base: Optional[Path]) -> str:
    """A path to show the model: relative to *base*, else absolute (posix).

    The single copy of the model-facing path convention, shared by
    :meth:`Resolution.display` and the search tools.
    """
    if path is None:
        return ""
    if base is not None:
        try:
            return str(path.relative_to(base)).replace("\\", "/")
        except ValueError:
            pass
    return str(path).replace("\\", "/")


def _relative(path: Path, root: Path) -> Optional[str]:
    """Return *path* relative to *root*, or ``None`` when it lies outside."""
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return None


__all__ = [
    "FileSpace",
    "Resolution",
    "ReadableDir",
    "ReadGrants",
    "display_path",
    "is_denied",
    "path_is_hidden_or_pycache",
    "suggest_near_match",
]
