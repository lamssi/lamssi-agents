"""CLI-only host discovery and the directory-backed default host."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

from lamssi_agents.runtime.config import AgentConfig
from lamssi_tools import LamssiError

log = logging.getLogger(__name__)

HOST_ENTRY_POINT_GROUP = "lamssi_agents.hosts"
NULL_HOST_NAME = "none"


@runtime_checkable
class HostBootstrap(Protocol):
    """Application lifecycle needed only by the bundled CLI."""

    name: str

    def resolve_root(self, arg: Optional[str]) -> Optional[Path]: ...

    def create_agent(self, *, root: Optional[Path], config: object) -> object: ...

    def wait_ready(self, timeout: float = 5.0) -> None: ...

    def status_lines(self) -> Sequence[Tuple[str, str]]: ...

    def shutdown(self) -> None: ...


class UnknownHost(LamssiError):
    """No host bootstrap is registered under the requested name."""


class NullHost:
    """A CLI host that installs file, memory, and system features over a directory."""

    name = NULL_HOST_NAME

    def __init__(self, *, state_dir_name: str = ".lamssi") -> None:
        self._root: Optional[Path] = None
        self._state_dir_name = state_dir_name

    def resolve_root(self, arg: Optional[str]) -> Optional[Path]:
        root = Path(arg).expanduser().resolve() if arg else Path.cwd().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"not a directory: {root}")
        self._root = root
        return root

    def create_agent(self, *, root: Optional[Path], config: object):
        from lamssi_agents import Agent, Files, Guidance, Memory, SystemTools

        self._root = root or self._root or Path.cwd().resolve()
        cfg = config if isinstance(config, AgentConfig) else AgentConfig()
        return Agent(
            config=cfg,
            features=[
                SystemTools(),
                Guidance(),
                Files(lambda: self._root),
                Memory(
                    lambda: (self._root or Path.cwd())
                    / self._state_dir_name
                    / "memory"
                ),
            ],
        )

    def wait_ready(self, timeout: float = 5.0) -> None:
        return None

    def status_lines(self) -> Sequence[Tuple[str, str]]:
        root = self._root or Path.cwd()
        return (("project", str(root)), ("workspace tools", "installed"))

    def shutdown(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"<NullHost root={self._root}>"


def available_hosts(group: str = HOST_ENTRY_POINT_GROUP) -> dict[str, str]:
    from importlib.metadata import entry_points

    out = {NULL_HOST_NAME: "lamssi_cli.hosts:NullHost"}
    try:
        for entry in entry_points(group=group):
            out[entry.name] = entry.value
    except Exception as exc:
        log.debug("host discovery unavailable: %s", exc)
    return out


def load_host(
    spec: Optional[str] = None,
    group: str = HOST_ENTRY_POINT_GROUP,
) -> HostBootstrap:
    if not spec or spec == NULL_HOST_NAME:
        return NullHost()
    if ":" in spec:
        return _load_from_path(spec)

    from importlib.metadata import entry_points

    try:
        for entry in entry_points(group=group):
            if entry.name == spec:
                return _instantiate(entry.load(), spec)
    except Exception as exc:
        log.debug("host entry-point lookup failed: %s", exc)

    known = ", ".join(sorted(available_hosts(group)))
    raise UnknownHost(f"unknown host {spec!r}; available: {known}")


def _load_from_path(spec: str) -> HostBootstrap:
    import importlib

    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise UnknownHost(f"could not import host module {module_name!r}: {exc}") from exc
    try:
        target = getattr(module, attr)
    except AttributeError as exc:
        raise UnknownHost(f"{module_name!r} has no attribute {attr!r}") from exc
    return _instantiate(target, spec)


def _instantiate(target: object, spec: str) -> HostBootstrap:
    host = target() if callable(target) else target
    for required in (
        "name",
        "create_agent",
        "resolve_root",
        "wait_ready",
        "status_lines",
        "shutdown",
    ):
        if not hasattr(host, required):
            raise UnknownHost(
                f"host {spec!r} does not satisfy HostBootstrap (missing {required!r})"
            )
    return host  # type: ignore[return-value]


__all__ = [
    "HostBootstrap",
    "NullHost",
    "UnknownHost",
    "load_host",
    "available_hosts",
    "HOST_ENTRY_POINT_GROUP",
    "NULL_HOST_NAME",
]
