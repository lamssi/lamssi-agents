"""Install skill discovery, prompt sections, and optional loading."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from lamssi_agents.features.base import Feature

from .catalog import Skill, _BUILTIN_SKILLS_DIR, _SkillCatalog
from .prompt import active_skills_block, skill_catalog_block
from .runtime import SkillRuntime
from .tools import load_skill

if TYPE_CHECKING:
    from lamssi_agents.agent.base import Agent


class Skills(Feature):
    """Install discoverable procedures for the current conversation.

    ``Skills(paths)`` lists summaries and file paths. Hosts can pin a body
    through :class:`SkillRuntime`; ``allow_model_loading=True`` also gives the
    model the ``load_skill`` tool.

    Args:
        *roots: Directories containing flat Markdown files or ``SKILL.md``
            directories.
        entries: Skill objects supplied directly by the host.
        loader: Optional directory loader for custom file layouts.
        include_builtin: Include the procedures shipped with lamssi-agents.
        allow_model_loading: Expose ``load_skill`` to the model. Host code can
            always pin a skill through :class:`SkillRuntime`.

    Example:
        Advertise project skills while keeping loading under host control::

            from lamssi_agents.features.skills import SkillRuntime

            agent = Agent(features=[Skills("./skills")])
            agent.get(SkillRuntime).load("release-check")
    """

    name = "skills"

    def __init__(
        self,
        *roots: str | Path,
        entries: Iterable[Skill] = (),
        loader: Callable[..., Iterable[Skill]] | None = None,
        include_builtin: bool = False,
        allow_model_loading: bool = False,
    ) -> None:
        self.roots = tuple(Path(root) for root in roots)
        self.entries = tuple(entries)
        self.loader = loader
        self.include_builtin = include_builtin
        self.allow_model_loading = allow_model_loading

    def install(self, agent: Agent) -> None:
        catalog = _SkillCatalog(entries=self.entries, loader=self.loader)
        if self.include_builtin:
            catalog.add_root(_BUILTIN_SKILLS_DIR, source="builtin")
        for path in self.roots:
            catalog.add_root(path)
        catalog.load()
        self._bind(agent, catalog, add_tool=True)

    def _bind(
        self,
        agent: Agent,
        catalog: _SkillCatalog,
        *,
        add_tool: bool,
    ) -> None:
        runtime = SkillRuntime(agent, catalog)
        agent.provide(SkillRuntime, runtime)
        agent.add_context(skill_catalog_block(runtime))
        agent.add_context(active_skills_block(runtime))
        if add_tool and self.allow_model_loading and catalog.list():
            agent.add_tools(load_skill)


__all__ = ["Skills"]
