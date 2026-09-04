"""Conversation-local state and operations for the Skills feature."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lamssi_tools import err

from .catalog import Skill, _SkillCatalog

if TYPE_CHECKING:
    from lamssi_agents.agent.base import Agent

log = logging.getLogger(__name__)


class _SkillPins:
    """Skill names pinned to one conversation, in load order."""

    __slots__ = ("names",)

    def __init__(self) -> None:
        self.names: list[str] = []

    def on_cleared(self) -> None:
        self.names.clear()


_ONE_SHOT_PATTERNS = (
    re.compile(r"^\s*fix\b", re.IGNORECASE),
    re.compile(r"^\s*show\s+(me\s+)?\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+(is|does|are)\b", re.IGNORECASE),
    re.compile(r"^\s*explain\b", re.IGNORECASE),
    re.compile(r"^\s*tell\s+me\b", re.IGNORECASE),
)

_RESOURCE_SUFFIXES = frozenset({".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".py"})
_MAX_RESOURCES = 20


class SkillRuntime:
    """The Skills feature as installed on one Agent conversation."""

    __slots__ = ("_agent", "_catalog")

    def __init__(
        self,
        agent: Agent,
        catalog: _SkillCatalog,
    ) -> None:
        self._agent = agent
        self._catalog = catalog

    @property
    def active(self) -> tuple[str, ...]:
        """Return pinned skill names in load order."""
        return tuple(self._pins())

    def list(self) -> list[Skill]:
        """Return every discovered skill."""
        return list(self._catalog.list())

    def get(self, name: str) -> Skill | None:
        """Return a skill by name, including separator-tolerant matches."""
        return self._catalog.get(name)

    def load(self, name: str) -> dict[str, Any]:
        """Pin one skill to this conversation and describe what became available."""
        name = (name or "").strip()
        if not name:
            return err("name is required", retriable=False)
        skill = self.get(name)
        if skill is None:
            return err(
                f"No skill named {name!r}",
                retriable=False,
                available=sorted(item.name for item in self.list()),
            )

        pins = self._pins()
        if skill.name in pins:
            return {
                "name": skill.name,
                "status": "already_active",
                "note": "Already pinned and visible in your prompt.",
            }

        pins.append(skill.name)
        log.info("Skill pinned: %s", skill.name)
        result: dict[str, Any] = {
            "name": skill.name,
            "status": "loaded",
            "description": skill.description,
            "note": "Its full instructions are now in your prompt.",
            "source": skill.source,
        }

        resources = _skill_resources(skill)
        if resources:
            result["resources"] = resources
            result["resources_hint"] = (
                "Files shipped with this skill. Read one with read_file(path=<the "
                "full path above>)."
            )

        if _looks_one_shot(self._last_user_message()):
            result["overload_hint"] = (
                "This looks like a one-shot request. Apply the skill lightly and "
                "act directly rather than running an unnecessary workflow."
            )
        return result

    def unload(self, name: str) -> bool:
        """Unpin one skill from this conversation."""
        skill = self.get((name or "").strip())
        canonical = skill.name if skill is not None else (name or "").strip()
        try:
            self._pins().remove(canonical)
        except ValueError:
            return False
        return True

    def _pins(self) -> list[str]:
        return self._agent.conversation_state(_SkillPins, _SkillPins).names

    def _last_user_message(self) -> str:
        for message in reversed(self._agent.history):
            if message.role == "user" and message.content:
                return message.content
        return ""

    def __repr__(self) -> str:
        return (
            f"<SkillRuntime {len(self._catalog)} skill(s), {len(self.active)} active>"
        )


def _looks_one_shot(text: str) -> bool:
    return bool(
        text
        and len(text) <= 200
        and any(pattern.search(text) for pattern in _ONE_SHOT_PATTERNS)
    )


def _skill_resources(skill: Skill) -> list[dict[str, str]]:
    """Return readable files shipped beside a skill's ``SKILL.md``."""
    if not skill.file_path:
        return []
    try:
        directory = Path(skill.file_path).resolve().parent
        if not directory.is_dir():
            return []
        found = [
            {"name": path.relative_to(directory).as_posix(), "path": str(path)}
            for path in sorted(directory.rglob("*"))
            if path.is_file()
            and path.name != "SKILL.md"
            and path.suffix.lower() in _RESOURCE_SUFFIXES
        ]
    except (OSError, ValueError) as exc:
        log.debug("could not list resources for skill %r: %s", skill.file_path, exc)
        return []

    if len(found) > _MAX_RESOURCES:
        log.debug(
            "skill %r ships %d resources; listing the first %d",
            skill.name,
            len(found),
            _MAX_RESOURCES,
        )
        return found[:_MAX_RESOURCES]
    return found


__all__ = ["SkillRuntime"]
