"""Runtime tool registry: binds ``ToolDefinition`` to callables, with argument validation and dispatched execution."""

from __future__ import annotations

import logging
import threading
from contextvars import copy_context
from dataclasses import dataclass, field
from functools import wraps
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type

from pydantic import BaseModel

from lamssi_tools.dispatch import Dispatcher, inline_dispatcher
from lamssi_tools.models import (
    Expose,
    ToolDefinition,
    ToolExecutionError,
    strip_pydantic_urls,
)

log = logging.getLogger(__name__)

__all__ = ["ToolRegistry", "ToolBinding", "ToolExecutionError"]


def _capture_call_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Carry the caller's context variables across host dispatch."""
    context = copy_context()

    @wraps(fn)
    def bound(*args: Any, **kwargs: Any) -> Any:
        return context.run(fn, *args, **kwargs)

    return bound


@dataclass(frozen=True, slots=True)
class ToolBinding:
    """A validated call bound to the exact registry entry that resolved it."""

    registry: "ToolRegistry" = field(repr=False, compare=False)
    generation: int
    definition: ToolDefinition
    arguments: Mapping[str, Any]
    function: Callable[..., Any] = field(repr=False, compare=False)
    dispatcher: Dispatcher = field(repr=False, compare=False)


class ToolRegistry:
    """Binds ``ToolDefinition`` to a Python callable, with schema generation, argument validation, and dispatched execution.

    Every tool is tagged with an *owner* so a whole bundle (a host object, a
    handler pack, a user-scripts directory) can be removed in one call.
    """

    # Callback signatures
    OnAdd = Callable[[str, ToolDefinition, Callable[..., Any], str], None]
    OnRemove = Callable[[str, str], None]

    def __init__(self, *, dispatcher: Optional[Dispatcher] = None) -> None:
        self._lock = threading.RLock()
        self._defs: Dict[str, ToolDefinition] = {}
        self._callables: Dict[str, Callable[..., Any]] = {}
        self._args_models: Dict[str, Type[BaseModel]] = {}
        self._generation = 0

        # owner -> ordered list of tool names
        self._owners: Dict[str, List[str]] = {}
        # tool name -> owner (reverse lookup)
        self._tool_owner: Dict[str, str] = {}

        # Where a body runs. Inline unless a host installs its own router.
        self._dispatch: Dispatcher = dispatcher or inline_dispatcher

        # A list, not one slot, since two subsystems may each want to mirror registrations into their own surface.
        self._listeners: List[
            Tuple[Optional["ToolRegistry.OnAdd"], Optional["ToolRegistry.OnRemove"]]
        ] = []


    def set_dispatcher(self, dispatcher: Optional[Dispatcher]) -> None:
        """Install the host's execution router (``None`` restores inline)."""
        with self._lock:
            self._dispatch = dispatcher or inline_dispatcher
            self._generation += 1


    def add_listener(
        self,
        on_add: Optional["ToolRegistry.OnAdd"] = None,
        on_remove: Optional["ToolRegistry.OnRemove"] = None,
    ) -> Callable[[], None]:
        """Subscribe to tool add/remove events; returns an unsubscribe callable.

        Lets a host mirror registered callables into its own command surface;
        each callback receives the *owner* tag so a subscriber can filter by bundle.
        """
        entry = (on_add, on_remove)
        with self._lock:
            self._listeners.append(entry)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._listeners.remove(entry)
                except ValueError:
                    pass

        return _unsubscribe

    def add(self, source: Any, *, owner: str = "default") -> List[str]:
        """Register every tool produced by *source*, tagged with *owner*.

        Args:
            source: Any object with a ``collect()`` method yielding
                ``(ToolDefinition, callable)`` pairs, e.g.
                :class:`~lamssi_tools.sources.InstanceSource`.
            owner: Group tag for bulk removal via :meth:`remove_owner`.

        Returns:
            Names registered.

        Raises:
            ValueError: If any produced name is already registered or duplicated
                within the source. Registration is all-or-nothing for conflicts.
        """
        entries = list(source.collect())
        names = [definition.name for definition, _ in entries]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ValueError(
                "Tool names repeated by one source: " + ", ".join(duplicate_names)
            )
        prepared = [
            (definition, fn, definition.build_args_model())
            for definition, fn in entries
        ]

        added_entries: List[Tuple[ToolDefinition, Callable[..., Any]]] = []
        with self._lock:
            conflicts = sorted(name for name in names if name in self._defs)
            if conflicts:
                raise ValueError(
                    "Tool already registered: " + ", ".join(conflicts)
                )
            # Ensure the owner key exists even with zero tools, so callers can
            # tear it down idempotently later.
            self._owners.setdefault(owner, [])

            for td, fn, args_model in prepared:
                self._install(td, fn, args_model, owner)
                added_entries.append((td, fn))
            if added_entries:
                self._generation += 1
            listeners = list(self._listeners)

        self._notify_added(added_entries, owner, listeners)
        names = [definition.name for definition, _ in added_entries]
        if names:
            log.info("Owner %r: registered %d tool(s)", owner, len(names))
        return names

    def add_one(
        self,
        tool_def: ToolDefinition,
        fn: Callable[..., Any],
        *,
        owner: str = "default",
    ) -> None:
        """Register a single ``(ToolDefinition, callable)`` pair.

        Like :meth:`add`, raises ``ValueError`` when the name is already present.
        """
        args_model = tool_def.build_args_model()
        with self._lock:
            if tool_def.name in self._defs:
                raise ValueError(f"Tool already registered: {tool_def.name}")
            self._owners.setdefault(owner, [])
            self._install(tool_def, fn, args_model, owner)
            self._generation += 1
            listeners = list(self._listeners)
        self._notify_added([(tool_def, fn)], owner, listeners)
        log.debug("Registered tool: %s (owner=%s)", tool_def.name, owner)

    def remove_owner(self, owner: str) -> List[str]:
        """Remove every tool tagged with *owner*. Returns removed names."""
        with self._lock:
            names = self._owners.pop(owner, [])
            for name in names:
                self._defs.pop(name, None)
                self._callables.pop(name, None)
                self._args_models.pop(name, None)
                self._tool_owner.pop(name, None)
            if names:
                self._generation += 1
            listeners = list(self._listeners)
        self._notify_removed(names, owner, listeners)
        if names:
            log.info("Owner %r: unregistered %d tool(s)", owner, len(names))
        return list(names)

    def _install(
        self,
        td: ToolDefinition,
        fn: Callable[..., Any],
        args_model: Any,
        owner: str,
    ) -> None:
        self._defs[td.name] = td
        self._callables[td.name] = fn
        self._args_models[td.name] = args_model
        self._owners.setdefault(owner, []).append(td.name)
        self._tool_owner[td.name] = owner

    @staticmethod
    def _notify_added(
        entries: List[Tuple[ToolDefinition, Callable[..., Any]]],
        owner: str,
        listeners: List[
            Tuple[Optional["ToolRegistry.OnAdd"], Optional["ToolRegistry.OnRemove"]]
        ],
    ) -> None:
        """Notify an add snapshot after releasing the registry lock."""
        for definition, fn in entries:
            for on_add, _ in listeners:
                if on_add is None:
                    continue
                try:
                    on_add(definition.name, definition, fn, owner)
                except Exception as exc:
                    log.error("on_add listener failed for '%s': %s", definition.name, exc)

    @staticmethod
    def _notify_removed(
        names: List[str],
        owner: str,
        listeners: List[
            Tuple[Optional["ToolRegistry.OnAdd"], Optional["ToolRegistry.OnRemove"]]
        ],
    ) -> None:
        """Notify a removal snapshot after releasing the registry lock."""
        for name in names:
            for _, on_remove in listeners:
                if on_remove is None:
                    continue
                try:
                    on_remove(name, owner)
                except Exception as exc:
                    log.error("on_remove listener failed for '%s': %s", name, exc)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        with self._lock:
            return name in self._defs

    def set_exposure(self, name: str, expose: Expose) -> ToolDefinition:
        """Change which surfaces a registered tool reaches, on this registry only.

        Stores a copy carrying the new ``expose_to_agent`` / ``expose_to_mcp`` flags,
        so the module-level ``@tool`` definition other registries share is left
        untouched. Takes effect at the next surface resolution. HOST is implicit: the
        host may always call a registered tool through :meth:`execute`, whatever the
        mask says, so only the AGENT and MCP bits are stored.

        Not an add or a remove, so add/remove listeners do not fire: the tool is the
        same registration, reaching a different surface.

        Returns:
            The replacement definition.

        Raises:
            ValueError: If no tool of that name is registered.
        """
        with self._lock:
            if name not in self._defs:
                raise ValueError(f"set_exposure: no tool named {name!r} is registered")
            e = Expose(int(expose))
            td = self._defs[name].model_copy(update={
                "expose_to_agent": bool(e & Expose.AGENT),
                "expose_to_mcp": bool(e & Expose.MCP),
            })
            self._defs[name] = td
            self._generation += 1
            return td


    def get_all_tool_metas(self) -> List[Dict[str, Any]]:
        """Flat list of all tools with metadata for MCP/ZMQ; only ``expose_to_mcp=True`` tools, since exposure outside the process is opt-in."""
        tools: List[Dict[str, Any]] = []
        with self._lock:
            definitions = list(self._defs.items())

        for name, td in definitions:
            if not td.expose_to_mcp:
                continue
            schema = td.input_schema()
            tools.append({
                "name": name,
                "description": td.description,
                "input_schema": schema,
                "group": td.group,
            })

        return tools


    def resolve(self, name: str, arguments: Mapping[str, Any]) -> ToolBinding:
        """Validate arguments and bind a call to the current registry entry."""
        with self._lock:
            if name not in self._defs:
                raise ToolExecutionError(f"Unknown tool: {name}")
            args_model = self._args_models[name]
            definition = self._defs[name]
            function = self._callables[name]
            dispatcher = self._dispatch
            generation = self._generation

        try:
            validated = args_model.model_validate(dict(arguments))
        except Exception as exc:
            raise ToolExecutionError(
                f"Invalid arguments for {name}: {strip_pydantic_urls(str(exc))}"
            ) from exc

        return ToolBinding(
            registry=self,
            generation=generation,
            definition=definition,
            arguments=MappingProxyType(validated.model_dump()),
            function=function,
            dispatcher=dispatcher,
        )

    def execute_binding(self, binding: ToolBinding) -> Any:
        """Execute one previously resolved binding without looking up its name again."""
        if binding.registry is not self:
            raise ToolExecutionError("Tool binding belongs to a different registry")
        
        with self._lock:
            if binding.generation != self._generation:
                raise ToolExecutionError(
                    f"Tool registry changed after '{binding.definition.name}' was resolved; "
                    "the call was not executed."
                )

        try:
            return binding.dispatcher(
                binding.definition,
                _capture_call_context(binding.function),
                dict(binding.arguments),
            )
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool '{binding.definition.name}' raised: {exc}"
            ) from exc

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Resolve and execute a tool by name.

        Raises:
            ToolExecutionError: If the tool is unknown, arguments are invalid,
                or the tool itself raises.
        """
        return self.execute_binding(self.resolve(name, arguments))


    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """Return the :class:`ToolDefinition` for *name*, or ``None``."""
        with self._lock:
            return self._defs.get(name)

    def iter_entries(self):
        """Iterate over ``(name, ToolDefinition, callable, owner)`` tuples; a snapshot, safe to mutate the registry while iterating it."""
        with self._lock:
            return [
                (n, self._defs[n], self._callables[n], self._tool_owner[n])
                for n in list(self._defs)
            ]

    def owner_suffixes(self, prefix: str) -> List[str]:
        """For owners shaped ``"<prefix><name>"``, the ``<name>`` parts."""
        with self._lock:
            return sorted(
                o[len(prefix):]
                for o in self._owners
                if o.startswith(prefix) and o != prefix
            )

    def names(
        self,
        *,
        owner: Optional[str] = None,
        owner_prefix: Optional[str] = None,
        exclude_owner_prefix: Optional[str] = None,
    ) -> set:
        """Tool names narrowed by owner identity or prefix."""
        with self._lock:
            out = set()
            for name, own in self._tool_owner.items():
                if owner is not None and own != owner:
                    continue
                if owner_prefix is not None and not own.startswith(owner_prefix):
                    continue
                if exclude_owner_prefix is not None and own.startswith(exclude_owner_prefix):
                    continue
                out.add(name)
            return out


    @property
    def registered_tool_defs(self) -> List[ToolDefinition]:
        """All registered ToolDefinition objects."""
        with self._lock:
            return list(self._defs.values())

    @property
    def generation(self) -> int:
        """Monotonic version of execution-affecting registry state."""
        with self._lock:
            return self._generation

    def __len__(self) -> int:
        with self._lock:
            return len(self._defs)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._defs
