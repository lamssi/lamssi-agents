"""Headless REPL for the lamssi_agents agent (``lamssi-agent`` entry point)."""

from __future__ import annotations

from typing import Any

__all__ = ["main"]


# Lazy: an eager import of __main__ makes `python -m lamssi_cli` warn
# that the module is already in sys.modules.
def __getattr__(name: str) -> Any:
    if name == "main":
        from lamssi_cli.__main__ import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
