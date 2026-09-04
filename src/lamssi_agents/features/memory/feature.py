"""Persistent agent notes as an explicit feature."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lamssi_agents.features.base import Feature
from lamssi_agents.prompt.model import PromptContext, PromptPosition
from lamssi_agents.prompt.section import ContextBlock, heading

from .store import MemoryStore
from .tools import memory

if TYPE_CHECKING:
    from lamssi_agents.agent.base import Agent


def _memory_catalog_block(store: MemoryStore) -> ContextBlock:
    """The ``memory-catalog`` block: a live index the model can recall by name."""

    def render(ctx: PromptContext) -> str:
        if "memory" not in ctx.tools:
            return ""
        memories = store.list()
        if not memories:
            return ""
        rows = [
            f"- `{item.name}` ({item.type}): {item.description or '(no description)'}"
            for item in memories
        ]
        return heading(
            "Available Memories",
            'Use `memory(action="recall", name=...)` to read one.\n\n'
            + "\n".join(rows),
        )

    return ContextBlock(
        "memory-catalog",
        render,
        position=PromptPosition.CONTEXT,
        stable=False,
        source="lamssi_agents.features.memory.memory-catalog",
    )


class Memory(Feature):
    """Install persistent named notes and their catalog prompt.

    Args:
        path: Storage directory or callable returning one. Relative paths resolve
            inside the installed :class:`Files` workspace, or from the current
            directory when Files is absent. Defaults to ``.lamssi/memory``.

    Note:
        Recall and list operations are read-only. Remember and forget operations
        remain subject to the agent's approval policy.
    """

    name = "memory"

    def __init__(self, path: Any = ".lamssi/memory") -> None:
        self.path = path

    def install(self, agent: Agent) -> None:
        from lamssi_agents.features.files import FileSpace

        store = agent.get(MemoryStore)
        if store is None:
            if callable(self.path):
                location = self.path
            else:
                configured = Path(self.path)
                if configured.is_absolute():
                    location = configured
                else:
                    space = agent.get(FileSpace)
                    location = (
                        (lambda: space.workspace() / configured)
                        if space is not None
                        else Path.cwd() / configured
                    )
            store = MemoryStore(location)
            agent.provide(MemoryStore, store)
        agent.add_context(_memory_catalog_block(store))
        agent.add_tools(memory)


__all__ = ["Memory"]
