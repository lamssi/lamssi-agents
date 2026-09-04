"""Skill definitions, Markdown discovery, and lookup for the Skills feature."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lamssi_agents._frontmatter import parse_frontmatter

log = logging.getLogger(__name__)

_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent / "builtin"


@dataclass(frozen=True, slots=True)
class Skill:
    """One discoverable procedure loaded by :class:`Skills`."""

    name: str
    description: str
    instructions: str = ""
    allowed_tools: tuple[str, ...] = ()
    source: str = "host"
    file_path: Path | None = None


def _string_list(value: Any) -> tuple[str, ...]:
    """Normalize a list or comma/whitespace-delimited string."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(token for token in re.split(r"[,\s]+", value) if token)
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _load_skill(path: Path, *, source: str) -> Skill | None:
    """Load one skill Markdown file, returning ``None`` when unreadable."""
    if not path.is_file():
        return None
    try:
        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    except OSError as exc:
        log.warning("cannot read skill %s: %s", path, exc)
        return None

    return Skill(
        name=str(metadata.get("name") or path.stem),
        description=str(metadata.get("description") or "").strip(),
        instructions=body,
        allowed_tools=_string_list(
            metadata.get("allowed-tools")
            or metadata.get("allowed_tools")
            or metadata.get("tools")
        ),
        source=source,
        file_path=path.resolve(),
    )


def _load_directory(directory: Path, *, source: str) -> list[Skill]:
    """Load flat ``*.md`` and ``<name>/SKILL.md`` layouts."""
    if not directory.is_dir():
        return []

    loaded: list[Skill] = []
    for path in sorted(directory.glob("*.md")):
        if path.name.startswith(("_", ".")):
            continue
        skill = _load_skill(path, source=source)
        if skill is not None:
            loaded.append(skill)

    for child in sorted(directory.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        for path in (child / "SKILL.md", child / f"{child.name}.md"):
            if path.is_file():
                skill = _load_skill(path, source=source)
                if skill is not None:
                    loaded.append(skill)
                break
    return loaded


class _SkillCatalog:
    """Feature-internal collection loaded from roots and explicit entries."""

    __slots__ = ("_entries", "_loaded", "_loader", "_roots", "_skills")

    def __init__(
        self,
        *,
        entries: Iterable[Skill] = (),
        loader: Callable[..., Iterable[Skill]] | None = None,
    ) -> None:
        self._skills: dict[str, Skill] = {}
        self._roots: list[tuple[Path, str]] = []
        self._entries = tuple(entries)
        self._loader = loader or _load_directory
        self._loaded = False

    def add_root(self, path: str | Path, *, source: str = "host") -> None:
        """Add a root; later roots and explicit entries win by name."""
        entry = (Path(path), source)
        if entry not in self._roots:
            self._roots.append(entry)
            self._loaded = False

    def load(self, *, force: bool = False) -> None:
        """Load every configured root once, unless *force* is true."""
        if self._loaded and not force:
            return
        self._skills.clear()
        for root, source in self._roots:
            if not root.is_dir():
                log.debug("skill root does not exist: %s", root)
                continue
            try:
                for skill in self._loader(root, source=source) or ():
                    self.register(skill)
            except Exception as exc:  # noqa: BLE001 - isolate a host loader
                log.warning("could not load skills from %s: %s", root, exc)
        for skill in self._entries:
            self.register(skill)
        self._loaded = True
        log.info("Skills ready (%d)", len(self._skills))

    def register(self, skill: Skill) -> None:
        """Add or replace one skill by name."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Resolve a name, tolerating hyphen/underscore differences."""
        self.load()
        direct = self._skills.get(name)
        if direct is not None:
            return direct
        normalized = name.lower().replace("-", "_")
        return next(
            (
                skill
                for skill in self._skills.values()
                if skill.name.lower().replace("-", "_") == normalized
            ),
            None,
        )

    def list(self) -> Sequence[Skill]:
        """Return discovered skills in load order."""
        self.load()
        return tuple(self._skills.values())

    def __len__(self) -> int:
        return len(self.list())

    def __repr__(self) -> str:
        return f"<SkillCatalog {len(self)} skill(s)>"


__all__ = ["Skill"]
