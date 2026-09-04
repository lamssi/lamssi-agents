"""Sources that collect ``@tool`` callables from objects, modules, and directories."""

from __future__ import annotations

import hashlib
import importlib
import logging
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

from lamssi_tools.context import CapabilityContext
from lamssi_tools.models import ToolDefinition

log = logging.getLogger(__name__)

ToolPair = tuple[ToolDefinition, Callable[..., Any]]


class ToolSource(Protocol):
    """A producer of ``(ToolDefinition, callable)`` pairs."""

    def collect(self) -> Iterable[ToolPair]: ...


def _iter_decorated(
    items: Iterable[tuple[str, Any]],
) -> Iterator[tuple[ToolDefinition, Any]]:
    """Yield decorated callables once per tool name."""
    seen: set[str] = set()
    for _, attr in items:
        if not callable(attr):
            continue
        fn = getattr(attr, "__func__", attr)
        td: ToolDefinition | None = getattr(fn, "_tool_definition", None)
        if td is None or td.name in seen:
            continue
        seen.add(td.name)
        yield td, attr


def _wants_context(fn: Any) -> bool:
    target = getattr(fn, "__func__", fn)
    return bool(getattr(target, "_tool_inject_context", False))


class InstanceSource:
    """Tools defined as ``@tool``-decorated methods on *obj*.

    Yields bound methods so the receiver is implicit at call time.
    """

    def __init__(
        self, obj: Any, *, context: CapabilityContext | None = None
    ) -> None:
        self._obj = obj
        self._context = context

    def collect(self) -> Iterator[ToolPair]:
        def _items() -> Iterator[tuple[str, Any]]:
            for name in dir(self._obj):
                try:
                    yield name, getattr(self._obj, name)
                except Exception as exc:  # noqa: BLE001 - descriptors may raise anything
                    log.debug("skipping unreadable tool attribute %s: %s", name, exc)
                    continue

        for td, fn in _iter_decorated(_items()):
            bound = _inject_capability_context(td, fn, self._context)
            if bound is not None:
                yield td, bound


def _inject_capability_context(
    td: ToolDefinition, fn: Any, context: CapabilityContext | None
) -> Callable[..., Any] | None:
    """Bind *context* when requested, or skip the callable when unavailable."""
    if not _wants_context(fn):
        return fn
    if context is None:
        log.debug(
            "tool %r requires a CapabilityContext but none was provided: skipping",
            td.name,
        )
        return None

    def invoke(**kwargs: Any) -> Any:
        return fn(context, **kwargs)

    return invoke


class CallableSource:
    """A collection of already-decorated tool callables."""

    def __init__(
        self,
        *callables: Callable[..., Any],
        context: CapabilityContext | None = None,
    ) -> None:
        self._callables = callables
        self._context = context

    def collect(self) -> Iterator[ToolPair]:
        for td, fn in _iter_decorated(
            (getattr(fn, "__name__", "tool"), fn) for fn in self._callables
        ):
            handler = _inject_capability_context(td, fn, self._context)
            if handler is not None:
                yield td, handler


class ModuleSource:
    """Tools defined as ``@tool``-decorated callables in Python modules.

    Args:
        *modules: Modules to scan.
        context: Bound as the first positional argument of every handler declared
            with ``@tool(inject_context=True)``. Handlers wanting a context are
            skipped when none is given.
    """

    def __init__(
        self, *modules: Any, context: CapabilityContext | None = None
    ) -> None:
        self._modules = modules
        self._context = context

    def collect(self) -> Iterator[ToolPair]:
        seen: set[str] = set()
        for mod in self._modules:
            for td, fn in _iter_decorated(vars(mod).items()):
                if td.name in seen:
                    continue
                handler = _inject_capability_context(td, fn, self._context)
                if handler is None:
                    continue
                seen.add(td.name)
                yield td, handler


class DirectorySource:
    """Tools defined in ``.py`` files inside *path*.

    Loads each file as a one-off module and scans it for ``@tool`` callables,
    skipping files starting with ``_`` or that fail to load. Executing a file
    is arbitrary code execution, so this is opt-in via ``trust=True``.

    Args:
        path: Directory to scan, or ``None`` for a no-op source.
        namespace: Module name prefix; each file gets a unique suffix to avoid
            ``sys.modules`` collisions between same-named files.
        trust: Must be True to execute anything.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        context: CapabilityContext | None = None,
        namespace: str = "lamssi_tools.external",
        trust: bool = False,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._context = context
        self._namespace = namespace
        self._trust = trust

    def collect(self) -> Iterator[ToolPair]:
        d = self._path
        if d is None or not d.is_dir():
            return
        if not self._trust:
            log.warning(
                "DirectorySource(%s) collected nothing: loading these files would "
                "execute them, so it requires trust=True",
                d,
            )
            return

        import sys

        for py_file in sorted(d.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            source_id = hashlib.sha1(
                str(py_file.resolve()).encode("utf-8")
            ).hexdigest()[:8]
            mod_name = f"{self._namespace}.{source_id}.{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(mod_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                # Registered before exec so a dataclass or pickle inside the file
                # can resolve its own module by name.
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)  # type: ignore[unionattr]
            except Exception as exc:  # noqa: BLE001 - isolate any plugin import failure
                sys.modules.pop(mod_name, None)
                log.warning("Failed to load tools from %s: %s", py_file, exc)
                continue

            yield from ModuleSource(mod, context=self._context).collect()


__all__ = [
    "CallableSource",
    "DirectorySource",
    "InstanceSource",
    "ModuleSource",
    "ToolPair",
    "ToolSource",
]
