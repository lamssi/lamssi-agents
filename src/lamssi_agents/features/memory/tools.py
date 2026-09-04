"""Tool exposed by the optional Memory feature."""

from __future__ import annotations

from typing import Any

from lamssi_tools import CapabilityContext, Expose, Str, err, tool

from .store import MemoryStore

_ACTIONS = ("remember", "recall", "forget", "list")


def _store_or_error(
    ctx: CapabilityContext,
) -> tuple[MemoryStore | None, dict[str, Any] | None]:
    store = ctx.get(MemoryStore)
    if store is None:
        return None, err(
            "No memory store is configured for this host.",
            retriable=False,
            hint="Carry the fact in your reply instead.",
        )
    return store, None


@tool(
    group="system",
    expose=Expose.ALL,
    inject_context=True,
    approval="conditional",
    safe_when={"action": ["recall", "list"]},
    description=(
        "Use this when you learn something that will still matter in a later "
        "conversation: a preference the user stated, a decision and its reason, "
        "where something lives. Not for what you can re-read from the code. "
        "Notes that persist between conversations. Actions: remember (name and "
        "content, optionally a type and a one-line description), recall (name), "
        "forget (name), list (optionally filtered by type). Types are user, project, "
        "feedback, reference. Save sparingly: only facts that are not already "
        "recoverable from the code or its history. The index is already in your "
        "prompt; recall a body only when the index suggests it is relevant."
    ),
    keywords="remember recall note preference remind forget store persist",
    parameters={
        "action": Str("", enum=["remember", "recall", "forget", "list"]),
        "name": Str("Required for remember, recall and forget."),
        "content": Str("Markdown body. Required for remember."),
        "type": Str("", enum=["", "user", "project", "feedback", "reference"]),
        "description": Str("One-line index summary, for remember."),
    },
)
def memory(
    ctx: CapabilityContext,
    action: str = "list",
    name: str = "",
    content: str = "",
    type: str = "",
    description: str = "",
    **_,
) -> dict[str, Any]:
    """Run one memory operation."""
    if action not in _ACTIONS:
        return err(f"Unknown action '{action}'. Valid: {', '.join(_ACTIONS)}")

    store, store_error = _store_or_error(ctx)
    if store_error:
        return store_error

    if action == "list":
        items: list[dict[str, str]] = [
            {"name": item.name, "type": item.type, "description": item.description}
            for item in store.list(type=type or None)
        ]
        return {"memories": items, "count": len(items)}

    if not name:
        return err("name is required")

    if action == "recall":
        try:
            item = store.load(name)
        except ValueError as exc:
            return err(str(exc))
        if item is None:
            return err(f"No memory named '{name}'.")
        return {
            "name": item.name,
            "type": item.type,
            "description": item.description,
            "content": item.content,
        }

    if action == "forget":
        try:
            deleted = store.delete(name)
        except ValueError as exc:
            return err(str(exc))
        if not deleted:
            return err(f"No memory named '{name}'.")
        return {"forgot": name}

    if not content:
        return err("content is required for remember")
    try:
        item = store.save(
            name,
            content,
            type=type or "user",
            description=description,
        )
    except ValueError as exc:
        return err(str(exc))
    return {"saved": item.name, "type": item.type, "path": item.file_path}


__all__ = ["memory"]
