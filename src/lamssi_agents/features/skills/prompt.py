"""Prompt sections contributed only by the Skills feature."""

from __future__ import annotations

import logging

from lamssi_agents.prompt.model import PromptContext, PromptPosition
from lamssi_agents.prompt.section import ContextBlock, coerce_text, heading

from .catalog import Skill
from .runtime import SkillRuntime

log = logging.getLogger(__name__)

_ACTIVE_SKILL_HEADER = "## Active Skill: follow it verbatim"


def active_skills_block(runtime: SkillRuntime) -> ContextBlock:
    """The ``active-skills`` block: bodies pinned to this Agent conversation."""

    def render(ctx: PromptContext) -> str:
        blocks: list[str] = []
        for name in runtime.active:
            skill = runtime.get(name)
            if skill is None:
                log.debug("active skill %r no longer resolves", name)
                continue
            body = _render_skill(skill)
            if body:
                blocks.append(f"{_ACTIVE_SKILL_HEADER}\n\n{body}")
        return "\n\n".join(blocks)

    return ContextBlock(
        "active-skills",
        render,
        position=PromptPosition.GUIDANCE,
        stable=False,
        source="lamssi_agents.features.skills.active-skills",
    )


def skill_catalog_block(runtime: SkillRuntime) -> ContextBlock:
    """The ``skill-catalog`` block: the small routing table for discovered skills."""

    def render(ctx: PromptContext) -> str:
        lead = _route_help(ctx)
        if not lead:
            return ""
        rows = []
        for skill in sorted(runtime.list(), key=lambda item: item.name):
            if not skill.name:
                continue
            row = f"- `{skill.name}`: {(skill.description or '').strip()}"
            if skill.file_path:
                row += f"\n  {skill.file_path}"
            rows.append(row)
        return (
            heading("Available Skills", lead + "\n\n" + "\n".join(rows)) if rows else ""
        )

    return ContextBlock(
        "skill-catalog",
        render,
        position=PromptPosition.REFERENCE,
        stable=True,
        source="lamssi_agents.features.skills.skill-catalog",
    )


def _route_help(ctx: PromptContext) -> str:
    head = (
        "Each entry is a procedure to bring in when its summary matches the job. "
        "You have the summary only; do not improvise its steps."
    )
    pin = "load_skill" in ctx.tools
    read = "read_file" in ctx.tools
    if pin and read:
        return head + (
            " Either `load_skill(name=…)`, which pins it for the conversation, or "
            "`read_file` at the path below for a one-off."
        )
    if pin:
        return head + " Call `load_skill(name=…)` to get its steps."
    if read:
        return head + " Read the file at the path below to get its steps."
    return ""


def _render_skill(skill: Skill) -> str:
    lines = [f"### Skill: {skill.name}", "", skill.description.strip()]
    if skill.allowed_tools:
        lines += ["", f"**Tools:** {', '.join(skill.allowed_tools)}"]
    if skill.instructions:
        lines += ["", skill.instructions.strip()]
    return coerce_text("\n".join(lines), f"skill:{skill.name}")


__all__ = ["active_skills_block", "skill_catalog_block"]
