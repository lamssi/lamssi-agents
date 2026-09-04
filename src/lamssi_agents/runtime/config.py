"""Frozen, per-runtime configuration, constructed once and threaded explicitly to the code that needs it."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields, replace
from typing import Any, Optional

log = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Everything tunable about how an agent runs.

    Frozen: derive a variant with :meth:`merged` rather than mutating, so a value
    cannot change under a turn that is midway through using it.
    """

    #: Hard backstop on tool-calling turns in one run.
    max_turns: int = 200
    #: Combined system-prompt-plus-history size that triggers compaction.
    history_budget_tokens: int = 80_000
    #: Messages kept verbatim below the summary.
    keep_recent: int = 24
    #: Hard clip on one tool result; keep well below history_budget_tokens: a cap that size can trigger compaction on a single call and evict the cached prefix.
    max_tool_result_chars: int = 8_000
    #: Fraction of the model's context window that triggers auto-compaction. Used
    #: only when reserve_tokens is 0.
    autocompact_fraction: float = 0.85
    #: If > 0, trigger at (window - reserve_tokens) instead of the fraction: a fixed
    #: headroom for the reply.
    reserve_tokens: int = 0
    #: Built-in compaction strategy: ``"summarise"`` or ``"ladder"``.
    compaction: str = "summarise"

    #: Prefix :meth:`from_env` reads; per-config so two embeddings can be tuned independently.
    env_prefix: str = "LAMSSI_"

    def merged(self, **overrides: Any) -> "AgentConfig":
        """A copy with *overrides* applied; ``None`` values are ignored.

        Ignoring ``None`` is what lets callers forward optional arguments
        straight through without each one needing a presence check.
        """
        clean = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(clean) - {f.name for f in fields(self)}
        if unknown:
            raise TypeError(f"unknown config field(s): {sorted(unknown)}")
        return replace(self, **clean) if clean else self

    @classmethod
    def from_env(
        cls,
        prefix: str = "LAMSSI_",
        *,
        base: Optional["AgentConfig"] = None,
    ) -> "AgentConfig":
        """Read ``<PREFIX><FIELD>`` for each field, over *base* or the defaults.

        An unparseable value is logged and ignored, not raised, so a typo does not
        stop the agent from starting.
        """
        start = base or cls()
        env_prefix = prefix or start.env_prefix
        out: dict[str, Any] = {}

        for f in fields(cls):
            if f.name == "env_prefix":
                continue
            raw = os.environ.get(f"{env_prefix}{f.name.upper()}")
            if raw is None:
                continue
            try:
                if f.type in ("int", int):
                    out[f.name] = int(raw)
                elif f.type in ("float", float):
                    out[f.name] = float(raw)
                else:
                    out[f.name] = raw
            except ValueError:
                log.warning(
                    "Ignoring invalid %s%s=%r (expected %s)",
                    env_prefix, f.name.upper(), raw, f.type,
                )

        return replace(start, env_prefix=env_prefix, **out)

    def normalised(self) -> "AgentConfig":
        """Clamp every value into a range the runtime can actually honour."""
        return replace(
            self,
            max_turns=max(1, int(self.max_turns)),
            history_budget_tokens=max(1000, int(self.history_budget_tokens)),
            keep_recent=max(1, int(self.keep_recent)),
            max_tool_result_chars=max(1000, int(self.max_tool_result_chars)),
            # Must leave room for the reply, so never the whole window.
            autocompact_fraction=max(0.5, min(0.95, float(self.autocompact_fraction))),
            reserve_tokens=max(0, int(self.reserve_tokens)),
        )


__all__ = ["AgentConfig"]
