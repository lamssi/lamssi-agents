"""Tool contributed by the optional Skills feature."""

from __future__ import annotations

from typing import Any

from lamssi_tools import CapabilityContext, Expose, Str, tool

from .runtime import SkillRuntime


@tool(
    group="system",
    expose=Expose.AGENT,
    approval="never",
    guard_role="always_allowed",
    inject_context=True,
    requires=SkillRuntime,
    description=(
        "Load a skill's full instructions. Call this first when the skill catalog "
        "lists one matching the task. Hyphens and underscores are interchangeable."
    ),
    parameters={"name": Str("Skill name, from the catalog.")},
)
def load_skill(
    ctx: CapabilityContext,
    name: str = "",
) -> dict[str, Any]:
    """Pin one skill to the current conversation."""
    return ctx.require(SkillRuntime).load(name)


__all__ = ["load_skill"]
