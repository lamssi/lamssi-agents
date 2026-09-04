"""Query each provider backend for the models it can actually serve."""

from __future__ import annotations

import json
import logging
from functools import partial
from typing import Callable, List, Tuple

log = logging.getLogger(__name__)

# Curated fallback lists used only when the live /v1/models call fails; most recent on top.
ANTHROPIC_FALLBACK: List[str] = [
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]
OPENAI_FALLBACK: List[str] = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "o1",
    "o1-mini",
]
OPENROUTER_FALLBACK: List[str] = [
    "openrouter/anthropic/claude-3.5-sonnet",
    "openrouter/anthropic/claude-3.5-haiku",
    "openrouter/openai/gpt-4o",
    "openrouter/openai/gpt-4o-mini",
    "openrouter/google/gemini-2.0-flash-001",
    "openrouter/meta-llama/llama-3.3-70b-instruct",
]


def with_v1(base_url: str) -> str:
    """*base_url* normalised to exactly one trailing ``/v1`` segment.

    A bare ``server_url`` and an already-normalised ``api_base`` both work.
    :func:`local_models._root_of` is the inverse.
    """
    url = (base_url or "").rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def models_endpoint(base_url: str) -> str:
    """The ``/v1/models`` URL for an OpenAI-compatible server."""
    return with_v1(base_url) + "/models"


def http_get_json(url: str, headers: dict, timeout: float = 5.0) -> dict:
    """Tiny stdlib HTTP GET that returns the parsed JSON body.

    Raises ``RuntimeError`` on an HTTP error, carrying the status code plus
    the first 200 chars of the body so callers can surface a useful message
    (401/403 = bad key, 404 = wrong endpoint, ...).
    """
    import urllib.request
    import urllib.error

    req = urllib.request.Request(url, headers={"Accept": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")[:200]
        except Exception:
            body = ""
        raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc


# One fetch routine per backend; source_for_model picks it without I/O.

def _cloud_fallback(name: str, fallback: List[str], exc: Exception) -> Tuple[List[str], str]:
    """The curated fallback list, labelled by why the live call failed."""
    msg = str(exc)
    reason = "invalid API key" if ("401" in msg or "403" in msg) else msg[:60]
    return list(fallback), f"{name} ({reason}: fallback list)"


def _fetch_keyed_cloud(name, endpoint, fallback, headers, extract, api_key) -> Tuple[List[str], str]:
    """Keyed ``GET /v1/models`` then extract ids; fall back with no key or on error."""
    key = (api_key or "").strip()
    if not key:
        return list(fallback), f"{name} (no API key: fallback list)"
    try:
        ids = extract(http_get_json(endpoint, headers(key)))
        return ids, f"{name} ({len(ids)} models: key OK)"
    except Exception as exc:
        return _cloud_fallback(name, fallback, exc)


def _anthropic_ids(data: dict) -> List[str]:
    ids = [e.get("id", "").strip() for e in (data.get("data") or [])]
    return sorted(n for n in ids if n)


def fetch_anthropic(api_key: str) -> Tuple[List[str], str]:
    """Anthropic's served models, or the curated fallback list."""
    return _fetch_keyed_cloud(
        "Anthropic", "https://api.anthropic.com/v1/models", ANTHROPIC_FALLBACK,
        lambda key: {"x-api-key": key, "anthropic-version": "2023-06-01"},
        _anthropic_ids, api_key,
    )


def _openai_ids(data: dict) -> List[str]:
    # Chat-capable families only; the raw catalogue also lists embeddings/tts/whisper.
    ids = [e.get("id", "").strip() for e in (data.get("data") or [])]
    ids = [n for n in ids if any(n.startswith(p) for p in ("gpt-", "o1", "o3", "o4", "chatgpt-"))]
    return sorted(set(ids))


def fetch_openai(api_key: str) -> Tuple[List[str], str]:
    """OpenAI's chat-capable models, or the curated fallback list."""
    return _fetch_keyed_cloud(
        "OpenAI", "https://api.openai.com/v1/models", OPENAI_FALLBACK,
        lambda key: {"Authorization": f"Bearer {key}"}, _openai_ids, api_key,
    )


def fetch_openrouter(api_key: str) -> Tuple[List[str], str]:
    """OpenRouter's models: its /models is public; the key only gates completions."""
    key = (api_key or "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        data = http_get_json("https://openrouter.ai/api/v1/models", headers)
        # ids look like "anthropic/claude-3.5-sonnet"; LiteLLM needs the openrouter/ prefix.
        ids = sorted({
            f"openrouter/{e['id'].strip()}"
            for e in (data.get("data") or [])
            if isinstance(e, dict) and e.get("id")
        })
        if not ids:
            return list(OPENROUTER_FALLBACK), "OpenRouter (empty list: fallback)"
        label = (f"OpenRouter ({len(ids)} models: key OK)" if key
                 else f"OpenRouter ({len(ids)} models: set a key to use)")
        return ids, label
    except Exception as exc:
        return _cloud_fallback("OpenRouter", OPENROUTER_FALLBACK, exc)


def fetch_local(server_url: str) -> Tuple[List[str], str]:
    """Models a local OpenAI-compatible server reports; empty if unreachable."""
    url = (server_url or "").rstrip("/")
    if not url:
        return [], "local (no server URL configured)"
    try:
        data = http_get_json(models_endpoint(url), {}, timeout=3.0)
        entries = data.get("data") or data.get("models") or []
        names = sorted(
            n for n in (
                (e.get("id") or e.get("name") or "").strip()
                for e in entries if isinstance(e, dict)
            ) if n
        )
        return names, f"local ({url}: {len(names)} models)"
    except Exception as exc:
        log.debug("local model fetch failed for %s: %s", url, exc)
        return [], f"local ({url} unreachable: {str(exc)[:60]})"


def fetch_unlisted(prefix: str) -> Tuple[List[str], str]:
    """No catalogue for a backend LiteLLM routes: an empty list, not an invented one."""
    name = prefix or "unknown"
    return [], f"{name} (no catalogue: LiteLLM routes it, type the name)"


def source_for_model(
    model: str,
    *,
    api_key: str = "",
    api_base: str = "",
) -> Callable[[], Tuple[List[str], str]]:
    """The catalogue fetcher for one configured backend: a zero-arg callable.

    Calling it returns ``(model_names, source_label)``; picking it does no network
    I/O, only calling it does.

    ``api_base`` is checked first: an OpenAI-compatible local model carries an
    ``openai/`` prefix, so matching by id first would be wrong.
    """
    if api_base:
        return partial(fetch_local, api_base)
    name = (model or "").lower()
    if name.startswith("openrouter/"):
        return partial(fetch_openrouter, api_key)
    if name.startswith(("claude", "anthropic/")):
        return partial(fetch_anthropic, api_key)
    if name.startswith(("gpt", "o1", "o3", "o4", "chatgpt-", "openai/")):
        return partial(fetch_openai, api_key)
    prefix = name.split("/", 1)[0] if "/" in name else ""
    return partial(fetch_unlisted, prefix)
