"""``lamssi_agents.tooling``: per-tool policy (dedupe, approval, surface, guard rules) owned by the tool rather than the kernel."""

from __future__ import annotations

from lamssi_agents.approval import (
    DECLARATIONS,

    declared_approval,
    declared_safe,
    is_safe,
    needs_approval,
    pattern_check,
)
from lamssi_agents.tooling.dedupe import (
    DEFAULT_POLICY,
    DedupeCache,
    DedupePolicy,
    arg_subset_signature,
    default_repeat_hint,
    full_arg_signature,
)
from lamssi_agents.tooling.guard import (
    CORE_GUARD_ROLES,
    DEFAULT_GUARD_RULES,
    GuardMessages,
    GuardRole,
    GuardRules,
)
from lamssi_agents.tooling.invocation import ToolInvocation
from lamssi_agents.tooling.surface import ToolSurface, resolve_surface

__all__ = [
    "GuardRole", "GuardRules", "GuardMessages",
    "CORE_GUARD_ROLES", "DEFAULT_GUARD_RULES",
    "DedupeCache", "DedupePolicy", "DEFAULT_POLICY",
    "full_arg_signature", "arg_subset_signature", "default_repeat_hint",
    "DECLARATIONS", "needs_approval", "is_safe",
    "pattern_check", "declared_safe", "declared_approval",
    "ToolSurface", "resolve_surface",
    "ToolInvocation",
]
