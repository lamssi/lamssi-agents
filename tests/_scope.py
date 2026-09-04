"""Test helper: bind an agent as the active run scope for a block."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from lamssi_tools.context import capability_context_active


@contextmanager
def run_scope_active(agent: Any) -> Iterator[None]:
    """Bind *agent*'s capability context so a tool body sees its RunScope."""
    with capability_context_active(agent._capabilities):
        yield
