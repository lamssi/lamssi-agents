"""``ask_model``: a tool-less, single-shot LLM call for a typed value, for use inside a host's code sandbox."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Callable, Optional

from lamssi_agents.model import ModelInput, model_id, resolve_model
from lamssi_agents.providers import Message

log = logging.getLogger("AskModel")


class AskModelError(RuntimeError):
    """LLM call failed or its reply could not be coerced (and no default was given)."""


class _Required:
    """Sentinel for an omitted ``default``, distinct from an explicit ``default=None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<required>"


_REQUIRED: Any = _Required()

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_JSON_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
_TRUE_TOKENS = frozenset({"true", "yes", "y", "1", "on", "enabled"})
_FALSE_TOKENS = frozenset({"false", "no", "n", "0", "off", "disabled"})


# Tool-less and single-shot (never the ReAct loop), so it cannot recurse into
# code execution and deadlock the thread running the sandbox script.
def build_ask_model(
    model: ModelInput,
    *,
    abort_event: Optional[threading.Event] = None,
    history_limit: int = 200,
) -> Callable[..., Any]:
    """Build an ``ask_model`` callable a host can inject into its own sandbox.

    ``model`` has the same meaning as :class:`Agent`'s model argument: a model id
    or a configured adapter. *abort_event* lets the host interrupt a blocking
    model call. *history_limit* bounds the call log exposed as ``ask_model.history``.
    """
    adapter = resolve_model(model)
    resolved_model = model_id(adapter)

    history: list = []

    def ask_model(
        prompt: str,
        *,
        type: type = str,
        default: Any = _REQUIRED,
        system: Optional[str] = None,
        callback: Optional[Callable[[Any], Any]] = None,
        image: Any = None,
    ) -> Any:
        """Ask an LLM and return its answer coerced to *type*.

        Args:
            prompt: The question/instruction; embed live values inline.
            type: Desired return type: ``str``, ``float``, ``int``, ``bool``,
                ``dict`` or ``list``. The model is told to answer in that shape.
            default: Returned if the call fails or the reply can't be coerced.
                If omitted, failure raises :class:`AskModelError` instead.
            system: Override the system persona for this call.
            callback: Called once with the returned value, ``callback(value)``,
                for single-line act-on-result loops. Exceptions propagate, and
                it also fires on the ``default`` fallback.
            image: A numpy array, PIL image, file path, raw bytes, ``data:``
                URL, or list of any of those. Requires a vision-capable model.

        Returns:
            The coerced value (plain data, carries no authority). Every call is
            also appended to ``ask_model.history``.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise AskModelError("ask_model(prompt=...) requires a non-empty string prompt.")

        raw_text = ""
        error: Optional[str] = None
        n_images = 0
        try:
            image_urls = None
            if image is not None:
                from lamssi_agents.vision import to_image_urls
                image_urls = to_image_urls(image)
                n_images = len(image_urls)
            sys_content = _system_prompt(system, type)
            if image_urls:
                sys_content += " An image is attached: base your answer on it."
            messages = [
                Message(role="system", content=sys_content),
                Message(role="user", content=prompt, images=image_urls),
            ]
            chunks: list[str] = []
            for delta in adapter.stream(messages, tools=None, abort_event=abort_event):
                if delta.type == "text" and delta.text:
                    chunks.append(delta.text)
            raw_text = "".join(chunks).strip()
            if not raw_text:
                raise AskModelError("LLM returned an empty response.")
            value = _coerce(raw_text, type)
            ok = True
        except Exception as exc:
            error = str(exc)
            if default is _REQUIRED:
                _record(history, history_limit, prompt, resolved_model, type,
                        raw_text, None, callback, False, error, n_images)
                if isinstance(exc, AskModelError):
                    raise
                raise AskModelError(f"ask_model failed: {exc}") from exc
            log.warning("ask_model failed (%s): returning default %r", exc, default)
            value = default
            ok = False

        _record(history, history_limit, prompt, resolved_model, type,
                raw_text, value, callback, ok, error, n_images)

        # Callback runs after recording, so history reflects the LLM result
        # even if the callback then fails.
        if callback is not None:
            callback(value)
        return value

    # Exposed to sandbox/agent code as ``ask_model.history`` (clear via ``.clear()``).
    ask_model.history = history
    return ask_model


def _record(history, limit, prompt, model, type_, response,
            result, callback, ok, error, n_images=0) -> None:
    """Record one bounded ``ask_model`` history entry."""
    cb_name = None
    if callback is not None:
        cb_name = getattr(callback, "__name__", None) or repr(callback)
    history.append({
        "ts": time.time(),
        "prompt": prompt,
        "model": model,
        "type": getattr(type_, "__name__", str(type_)),
        "response": response,
        "result": result,
        "callback": cb_name,
        "images": n_images,
        "ok": ok,
        "error": error,
    })
    if len(history) > limit:
        del history[: len(history) - limit]
    log.info(
        "ask_model[%s] model=%s ok=%s result=%r%s",
        getattr(type_, "__name__", type_), model, ok, result,
        f" callback={cb_name}" if cb_name else "",
    )


def _system_prompt(system: Optional[str], type_: type) -> str:
    base = system or (
        "You are a precise assistant returning a value for another program. "
        "Be precise and concise."
    )
    if type_ in (float, int):
        return base + (
            " Respond with ONLY a single number: digits only, no units, "
            "no words, no explanation."
        )
    if type_ is bool:
        return base + " Respond with ONLY 'true' or 'false'."
    if type_ in (dict, list):
        return base + (
            " Respond with ONLY valid JSON: no markdown code fences, no prose."
        )
    return base


def _coerce(text: str, type_: type) -> Any:
    """Parse *text* into *type_* or raise :class:`AskModelError`."""
    if type_ is str:
        return text

    if type_ is bool:
        for tok in re.findall(r"[a-z01]+", text.lower()):
            if tok in _TRUE_TOKENS:
                return True
            if tok in _FALSE_TOKENS:
                return False
        raise AskModelError(f"Could not parse a boolean from {text!r}.")

    if type_ in (int, float):
        m = _NUM_RE.search(text)
        if not m:
            raise AskModelError(f"Could not parse a number from {text!r}.")
        val = float(m.group())
        return round(val) if type_ is int else val

    if type_ in (dict, list):
        m = _JSON_RE.search(text)
        payload = m.group(1) if m else text
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AskModelError(f"Could not parse JSON from {text!r}: {exc}") from exc
        if not isinstance(parsed, type_):
            raise AskModelError(
                f"Expected {type_.__name__} but the model returned "
                f"{type(parsed).__name__}."
            )
        return parsed

    raise AskModelError(
        f"Unsupported ask_model type {type_!r}; use str, float, int, bool, dict or list."
    )
