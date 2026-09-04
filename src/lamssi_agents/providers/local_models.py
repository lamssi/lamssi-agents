"""Probe a local model server (LM Studio, Ollama) for its loaded context window."""

from __future__ import annotations

import json
import ipaddress
import logging
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

log = logging.getLogger(__name__)

#: Loopback only: probing a network-reachable host's API is not this library's business.
#: Short enough that a wrong/no server fails fast without stalling startup.
_TIMEOUT_S = 1.0

#: Ollama's built-in window; a model resident at this almost certainly means OLLAMA_CONTEXT_LENGTH was never set.
_OLLAMA_DEFAULT_NUM_CTX = 4096


@dataclass(frozen=True, slots=True)
class LocalModelInfo:
    """What a local server says about one model. Every field is optional."""

    #: The window this instance was loaded with, which is the hard wall.
    context_window: int = 0
    #: Tool-calling support if the server said; ``None`` differs from ``False`` (said no).
    supports_tools: Optional[bool] = None
    #: Which server answered, for the log line. Empty when nothing did.
    source: str = ""

    def __bool__(self) -> bool:
        return self.context_window > 0 or self.supports_tools is not None


def is_local_base(api_base: Optional[str]) -> bool:
    """Whether *api_base* points at this machine."""
    if not api_base:
        return False
    try:
        hostname = urlsplit(api_base).hostname
    except ValueError:
        return False
    if hostname is None:
        return False
    hostname = hostname.rstrip(".").lower()
    if hostname in {"localhost", "0.0.0.0"}:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def probe(api_base: Optional[str], model: str) -> LocalModelInfo:
    """Ask the server at *api_base* about *model*. Never raises.

    Tries LM Studio's native REST API first, then Ollama's. Both are additive to the
    OpenAI-compatible endpoint the provider already talks to, so the same ``api_base``
    reaches them once ``/v1`` is stripped.
    """
    if not is_local_base(api_base):
        return LocalModelInfo()

    root = _root_of(api_base)
    name = _bare_model_name(model)

    for probe_one in (_probe_lm_studio, _probe_ollama):
        try:
            info = probe_one(root, name)
        except Exception as exc:            # a shape this does not recognise
            log.debug("local model probe raised against %s: %s", root, exc)
            continue
        if info:
            log.info(
                "%s reports %s: context_window=%s%s",
                info.source, name,
                f"{info.context_window:,}" if info.context_window else "unknown",
                "" if info.supports_tools is None else f", tools={info.supports_tools}",
            )
            return info
    return LocalModelInfo()


def _probe_lm_studio(root: str, model: str) -> LocalModelInfo:
    """Read LM Studio model metadata from ``GET /api/v0/models``."""
    body = _get_json(f"{root}/api/v0/models")
    entry = _match(body.get("data") if isinstance(body, dict) else None, model)
    if entry is None:
        return LocalModelInfo()

    window = _positive_int(entry.get("loaded_context_length")) or _positive_int(
        entry.get("max_context_length")
    )
    caps = entry.get("capabilities")
    tools = ("tool_use" in caps) if isinstance(caps, (list, tuple)) else None
    return LocalModelInfo(context_window=window, supports_tools=tools, source="LM Studio")


def _probe_ollama(root: str, model: str) -> LocalModelInfo:
    """Read Ollama metadata and the loaded model's context window."""
    body = _get_json(f"{root}/api/show", payload={"model": model})
    if not isinstance(body, dict):
        return LocalModelInfo()

    window = _resident_window(root, model) or _num_ctx(body.get("parameters"))
    if window:
        _warn_if_defaulted(model, window, body.get("model_info"))

    caps = body.get("capabilities")
    tools = ("tools" in caps) if isinstance(caps, (list, tuple)) else None
    return LocalModelInfo(context_window=window, supports_tools=tools, source="Ollama")


def _resident_window(root: str, model: str) -> int:
    """Return the resident model's context window from ``POST /api/ps``."""
    try:
        body = _get_json(f"{root}/api/ps")
    except Exception as exc:
        log.debug("ollama /api/ps did not answer: %s", exc)
        return 0

    rows = body.get("models") if isinstance(body, dict) else None
    if not isinstance(rows, Sequence):
        return 0
    wanted = model if ":" in model else f"{model}:latest"
    for entry in rows:
        if isinstance(entry, dict) and wanted in (entry.get("model"), entry.get("name")):
            return _positive_int(entry.get("context_length"))
    return 0


def _warn_if_defaulted(model: str, window: int, info: Any) -> None:
    """Warn when Ollama uses its small default window for a larger model."""
    supported = _architecture_window(info)
    if window <= _OLLAMA_DEFAULT_NUM_CTX < supported:
        log.warning(
            "Ollama serves %s with a %s-token window; the model supports %s. An "
            "oversize prompt is truncated silently; set OLLAMA_CONTEXT_LENGTH and "
            "restart the server.",
            model, f"{window:,}", f"{supported:,}",
        )


def _architecture_window(info: Any) -> int:
    """The ``<arch>.context_length`` the GGUF declares: a ceiling, not a wall."""
    if not isinstance(info, dict):
        return 0
    for key, value in info.items():
        if isinstance(key, str) and key.endswith(".context_length"):
            return _positive_int(value)
    return 0


def _get_json(url: str, payload: Optional[Dict[str, Any]] = None) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        return json.loads(response.read() or b"null")


def _root_of(api_base: str) -> str:
    """Strip an OpenAI-compatible ``/v1`` suffix from an API base URL."""
    root = api_base.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def _bare_model_name(model: str) -> str:
    """Strip a known provider routing prefix from a model name."""
    for prefix in ("openai/", "ollama/", "huggingface/", "local/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def _match(entries: Any, model: str) -> Optional[Dict[str, Any]]:
    """Match a model entry by exact id, then by final path segment."""
    if not isinstance(entries, Sequence):
        return None
    rows = [e for e in entries if isinstance(e, dict)]
    for entry in rows:
        if entry.get("id") == model:
            return entry
    tail = model.rsplit("/", 1)[-1]
    for entry in rows:
        entry_id = entry.get("id")
        if isinstance(entry_id, str) and entry_id.rsplit("/", 1)[-1] == tail:
            return entry
    return None


def _num_ctx(parameters: Any) -> int:
    """``num_ctx`` out of Ollama's parameters blob, which is newline-separated text."""
    if isinstance(parameters, dict):
        return _positive_int(parameters.get("num_ctx"))
    if isinstance(parameters, str):
        for line in parameters.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "num_ctx":
                return _positive_int(parts[1])
    return 0


def _positive_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


__all__ = ["LocalModelInfo", "probe", "is_local_base"]
