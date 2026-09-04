"""One registry surface over a local registry plus live mounted registries.

Resolves and runs tools across the owner's own registry and externally owned
registries a host mounts live. Mounts stay live by reference: a host adding or
removing a tool changes the next resolution.

Local tools run under the owner's capability context; mounted tools run under
their own registry's context.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

from lamssi_tools.context import CapabilityContext, capability_context_active
from lamssi_tools.errors import err
from lamssi_tools.models import ToolDefinition, ToolExecutionError, strip_pydantic_urls
from lamssi_tools.registry import ToolBinding, ToolRegistry

log = logging.getLogger(__name__)

__all__ = ["MountedRegistry", "ToolNameConflictError"]


class ToolNameConflictError(RuntimeError):
    """Two live tool registries currently publish the same name."""


_Entry = Tuple[str, ToolRegistry, bool]


class MountedRegistry:
    """A live tool surface: one local registry plus mounted ones.

    Resolution order is local, then external mounts. A name published by two
    registries at once is a conflict, not a silent shadow.
    """

    __slots__ = ("_local", "_capabilities", "_mounts", "_lock")

    def __init__(
        self,
        local: ToolRegistry,
        *,
        capabilities: Optional[CapabilityContext] = None,
    ) -> None:
        self._local = local
        self._capabilities = capabilities
        self._mounts: List[ToolRegistry] = []
        self._lock = threading.RLock()

    def mount(self, registry: ToolRegistry) -> bool:
        """Mount *registry* by identity; return whether it was newly added."""
        with self._lock:
            if any(existing is registry for existing in self._mounts):
                return False
            self._mounts.append(registry)
            return True

    def unmount(self, registry: ToolRegistry) -> bool:
        """Unmount an external registry by identity."""
        with self._lock:
            for index, existing in enumerate(self._mounts):
                if existing is registry:
                    del self._mounts[index]
                    return True
        return False

    def list_tools(self) -> List[ToolDefinition]:
        """Every published definition, in resolution order.

        Raises:
            ToolNameConflictError: If two registries publish the same name.
        """
        definitions: List[ToolDefinition] = []
        owners: dict[str, str] = {}
        for label, registry, _ in self._entries():
            for definition in registry.registered_tool_defs:
                previous = owners.get(definition.name)
                if previous is not None:
                    raise self._conflict(definition.name, previous, label)
                owners[definition.name] = label
                definitions.append(definition)
        return definitions

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """The definition for *name*, or ``None`` when no registry publishes it."""
        registry = self._registry_for(name)
        return registry.get_definition(name) if registry is not None else None

    def resolve(self, name: str, arguments) -> ToolBinding:
        """Validate *arguments* and bind *name* to the registry that owns it.

        Raises:
            ToolExecutionError: If no registry publishes *name*, or the arguments
                fail validation.
            ToolNameConflictError: If two registries publish *name*.
        """
        registry = self._registry_for(name)
        if registry is None:
            raise ToolExecutionError(f"Unknown tool: {name}")
        return registry.resolve(name, arguments)

    def execute_binding(self, binding: ToolBinding):
        """Run a previously resolved *binding*, or return an error payload.

        Rejects a binding whose registry has since been unloaded or replaced.
        Runs the binding under the owning registry's capability context: the
        owner's own for local tools, the host registry's own for a
        mounted tool. Never raises; a failure becomes an error payload.
        """
        name = binding.definition.name
        try:
            registry = self._registry_for(name)
        except ToolNameConflictError as exc:
            return err(str(exc), retriable=False)
        if registry is None:
            return err(
                f"Unknown tool: {name}",
                retriable=False,
                hint="The tool may have been unloaded. Use a tool from the current list.",
            )
        if registry is not binding.registry:
            return err(
                f"Tool '{name}' changed after it was resolved; the call was not executed.",
                retriable=False,
            )
        capabilities = self._capabilities_for(registry)
        try:
            if capabilities is None:
                return registry.execute_binding(binding)
            with capability_context_active(capabilities):
                return registry.execute_binding(binding)
        except ToolExecutionError as exc:
            return err(strip_pydantic_urls(str(exc)), retriable=False)
        except Exception as exc:
            log.error("Tool %r raised: %s", name, exc, exc_info=True)
            return err(f"{type(exc).__name__}: {strip_pydantic_urls(str(exc))}")

    def _entries(self) -> List[_Entry]:
        """``(label, registry, uses_owner_capabilities)`` in resolution order."""
        entries: List[_Entry] = [("local registry", self._local, True)]
        with self._lock:
            mounts = list(self._mounts)
        for index, registry in enumerate(mounts, start=1):
            entries.append((f"mounted registry {index}", registry, False))
        return entries

    def _registry_for(self, name: str) -> Optional[ToolRegistry]:
        found: Optional[ToolRegistry] = None
        found_label = ""
        for label, registry, _ in self._entries():
            if not registry.has_tool(name):
                continue
            if found is not None:
                raise self._conflict(name, found_label, label)
            found = registry
            found_label = label
        return found

    def _capabilities_for(
        self, registry: ToolRegistry
    ) -> Optional[CapabilityContext]:
        for _, source, uses_owner_capabilities in self._entries():
            if source is registry:
                return self._capabilities if uses_owner_capabilities else None
        return None

    @staticmethod
    def _conflict(name: str, first: str, second: str) -> ToolNameConflictError:
        return ToolNameConflictError(
            f"Tool name conflict: {name!r} is published by both {first} and "
            f"{second}. Unmount one source or rename the tool."
        )

    def __len__(self) -> int:
        return len(self.list_tools())

    def __repr__(self) -> str:
        with self._lock:
            mounts = len(self._mounts)
        return f"<MountedRegistry local={len(self._local)} mounts={mounts}>"
