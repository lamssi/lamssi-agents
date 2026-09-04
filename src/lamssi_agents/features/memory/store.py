"""File-backed storage for the optional Memory feature."""

from __future__ import annotations

import builtins
import logging
import re
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lamssi_agents._frontmatter import dump_frontmatter, parse_frontmatter

log = logging.getLogger(__name__)


# user: about the person; project: ongoing work; feedback: corrections to apply;
# reference: pointers to external resources.
VALID_TYPES = ("user", "project", "feedback", "reference")
INDEX_FILENAME = "MEMORY.md"
INDEX_HEADER = (
    "# Memory Index\n\n"
    "Persistent notes carried across conversations. "
    "Each entry links to a file under this directory.\n"
)


@dataclass
class Memory:
    name: str
    description: str
    type: str = "user"
    content: str = ""
    file_path: str = ""

    def render(self) -> str:
        """Return the Markdown frontmatter and body written to disk."""
        return dump_frontmatter(
            {"name": self.name, "description": self.description, "type": self.type},
            self.content,
        )


_NAME_RE = re.compile(r"[a-zA-Z][\w\-]{0,63}$")


def _validate_name(name: str) -> str:
    """Return a valid filesystem-safe memory name."""
    if not name or not _NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid memory name: {name!r}. "
            "Use a short identifier (letters, digits, _ or -)."
        )
    return name


class MemoryStore:
    """File-based memory store rooted at a single directory."""

    def __init__(self, base_dir: str | Path | Callable[[], str | Path]):
        self._base_dir = base_dir
        self._lock = threading.RLock()

    @property
    def base_dir(self) -> Path:
        value = self._base_dir() if callable(self._base_dir) else self._base_dir
        return Path(value)

    def _path(self, name: str) -> Path:
        return self.base_dir / f"{name}.md"

    def _index_path(self) -> Path:
        return self.base_dir / INDEX_FILENAME

    def _ensure_dir(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        content: str,
        type: str = "user",
        description: str = "",
    ) -> Memory:
        """Create or overwrite a memory and refresh the index."""
        _validate_name(name)
        if type not in VALID_TYPES:
            raise ValueError(f"Invalid type {type!r}. Use one of {VALID_TYPES}.")

        with self._lock:
            self._ensure_dir()
            mem = Memory(
                name=name,
                description=(description or "").strip(),
                type=type,
                content=content or "",
                file_path=str(self._path(name).resolve()),
            )
            self._write(self._path(name), mem.render())
            self._rebuild_index()
        log.info("Saved memory %s (%s)", name, type)
        return mem

    def load(self, name: str) -> Memory | None:
        """Read one memory, returning ``None`` when it does not exist."""
        _validate_name(name)
        path = self._path(name)
        with self._lock:
            if not path.is_file():
                return None
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                log.warning("Cannot read %s: %s", path, exc)
                return None
        meta, body = parse_frontmatter(text)
        return Memory(
            name=str(meta.get("name") or name),
            description=str(meta.get("description", "")),
            type=str(meta.get("type", "user")),
            content=body,
            file_path=str(path.resolve()),
        )

    def delete(self, name: str) -> bool:
        """Remove a memory and refresh the index."""
        _validate_name(name)
        path = self._path(name)
        with self._lock:
            if not path.is_file():
                return False
            path.unlink()
            self._rebuild_index()
        log.info("Forgot memory %s", name)
        return True

    def list(self, type: str | None = None) -> builtins.list[Memory]:
        """Return every memory, optionally filtered by type."""
        with self._lock:
            if not self.base_dir.is_dir():
                return []
            out: builtins.list[Memory] = []
            for path in sorted(self.base_dir.glob("*.md")):
                if path.name == INDEX_FILENAME:
                    continue
                try:
                    _validate_name(path.stem)
                except ValueError:
                    log.debug("skipping malformed memory filename: %s", path.name)
                    continue
                memory = self.load(path.stem)
                if memory and (type is None or memory.type == type):
                    out.append(memory)
            return out

    @staticmethod
    def _write(path: Path, content: str) -> None:
        """Atomically replace a file with UTF-8 content."""
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(content)
                temporary = Path(handle.name)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _rebuild_index(self) -> None:
        """Rewrite ``MEMORY.md`` with one entry for each stored memory."""
        memories = self.list()
        lines: builtins.list[str] = [INDEX_HEADER]
        if not memories:
            lines.append("_(no memories yet)_\n")
        else:
            for memory_type in VALID_TYPES:
                bucket = [memory for memory in memories if memory.type == memory_type]
                if not bucket:
                    continue
                lines.append(f"## {memory_type.title()}\n")
                for memory in bucket:
                    description = memory.description or "_(no description)_"
                    lines.append(
                        f"- [{memory.name}]({memory.name}.md): {description}"
                    )
                lines.append("")
        self._write(self._index_path(), "\n".join(lines).rstrip() + "\n")


__all__ = ["Memory", "MemoryStore"]
