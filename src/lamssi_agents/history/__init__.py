"""History management: truncation of individual tool results and compaction of old messages, keeping a conversation inside a context window."""

from __future__ import annotations

from lamssi_agents.history.compaction import (
    COMPACTION_STRATEGIES,
    DEFAULT_HISTORY_BUDGET_TOKENS,
    DEFAULT_KEEP_RECENT,
    SUMMARY_MAX_CHARS,
    Compactor,
    compress_history,
    get_compaction_strategy,
    summarise_only,
)
from lamssi_agents.history.truncation import (
    DEFAULT_MAX_TOOL_RESULT_CHARS,
    Truncator,
    clip_result,
    truncate_tool_result,
)

__all__ = [
    "truncate_tool_result",
    "clip_result",
    "Truncator",
    "compress_history",
    "DEFAULT_MAX_TOOL_RESULT_CHARS",
    "DEFAULT_HISTORY_BUDGET_TOKENS",
    "DEFAULT_KEEP_RECENT",
    "SUMMARY_MAX_CHARS",
    "Compactor",
    "summarise_only",
    "get_compaction_strategy",
    "COMPACTION_STRATEGIES",
]
