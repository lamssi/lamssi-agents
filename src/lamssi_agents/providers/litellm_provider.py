"""The universal LLM provider, backed by LiteLLM."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterator, List, Optional, Tuple

from lamssi_agents.providers import local_models
from lamssi_agents.providers.errors import clean_model_error
from lamssi_agents.providers.model_catalog import models_endpoint, source_for_model
from lamssi_agents.providers.models import (
    Message,
    ProviderInterrupted,
    StreamDelta,
    ToolCall,
    Usage,
)
from lamssi_agents.providers.prompt_cache import apply_prompt_caching
from lamssi_agents.providers.protocol import Model
from lamssi_agents.providers.usage import extract_usage
from lamssi_tools import ToolDefinition, build_tools_openai_schema

#: How long a stream may go with no data before it's treated as dead: a gap,
#: not a total (a healthy local model can legitimately take minutes per token).
STREAM_STALL_SECONDS = 120.0

# OpenAI-compatible clients require a non-empty key even when local auth is disabled (LM Studio, Ollama).
_LOCAL_API_KEY = "local"

#: Reasoning-buffer flush thresholds.
_THINK_FLUSH_CHARS = 160
_THINK_FLUSH_SECONDS = 0.10


def _decode_tool_arguments(raw: Any, tool_name: str) -> Dict[str, Any]:
    """Decode one provider tool payload, refusing malformed or non-object JSON."""
    if raw in (None, ""):
        return {}
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Provider returned invalid JSON arguments for tool {tool_name!r}."
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"Provider returned non-object arguments for tool {tool_name!r}."
        )
    return value


def _reasoning_text(delta: Any) -> str:
    """Extract reasoning text from supported provider delta fields."""
    for field_name in ("reasoning_content", "reasoning"):
        value = getattr(delta, field_name, None)
        if isinstance(value, str) and value:
            return value

    blocks = getattr(delta, "thinking_blocks", None)
    if isinstance(blocks, (list, tuple)):
        parts = [
            text
            for block in blocks
            if isinstance(block, dict)
            and isinstance(text := block.get("thinking") or block.get("text"), str)
            and text
        ]
        if parts:
            return "".join(parts)
    return ""


def _interruptible_chunks(
    stream: Any,
    abort_event: threading.Event | None,
) -> Iterator[Any]:
    """Wrap a blocking provider stream with abort and stall checks."""
    chunks: "queue.Queue[Any]" = queue.Queue(maxsize=512)
    stop = threading.Event()
    done = object()
    failed = object()

    def put(item: Any) -> bool:
        while not stop.is_set():
            try:
                chunks.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for chunk in stream:
                if not put(chunk) or (abort_event and abort_event.is_set()):
                    return
        except Exception as exc:
            put((failed, exc))
            return
        put(done)

    worker = threading.Thread(
        target=produce,
        daemon=True,
        name="LLMStreamReader",
    )
    worker.start()
    last_chunk_at = time.monotonic()
    try:
        while True:
            if abort_event is not None and abort_event.is_set():
                raise ProviderInterrupted("Stream aborted by user")
            try:
                item = chunks.get(timeout=0.1)
            except queue.Empty:
                if time.monotonic() - last_chunk_at > STREAM_STALL_SECONDS:
                    raise TimeoutError(
                        "The model stopped sending: no data for "
                        f"{STREAM_STALL_SECONDS:.0f}s. The request timed out."
                    ) from None
                continue
            last_chunk_at = time.monotonic()
            if item is done:
                return
            if isinstance(item, tuple) and item and item[0] is failed:
                raise item[1]
            yield item
    finally:
        stop.set()
        try:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        except Exception:
            pass
        worker.join(timeout=0.2)


@dataclass(slots=True)
class _StreamParser:
    """Accumulate provider chunks and emit LamSSI stream deltas."""

    tool_calls: Dict[int, Dict[str, str]] = field(default_factory=dict)
    thinking: List[str] = field(default_factory=list)
    thinking_chars: int = 0
    thinking_at: float = field(default_factory=time.monotonic)
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    started: bool = False

    def _flush_thinking(self) -> Optional[StreamDelta]:
        if not self.thinking:
            return None
        delta = StreamDelta(type="thinking", text="".join(self.thinking))
        self.thinking.clear()
        self.thinking_chars = 0
        self.thinking_at = time.monotonic()
        return delta

    def _feed_reasoning(self, delta: Any) -> Optional[StreamDelta]:
        """Buffer reasoning and return a display delta when its batch is ready."""
        thought = _reasoning_text(delta)
        if not thought:
            return None
        self.started = True
        self.thinking.append(thought)
        self.thinking_chars += len(thought)
        ready = self.thinking_chars >= _THINK_FLUSH_CHARS
        ready = ready or time.monotonic() - self.thinking_at >= _THINK_FLUSH_SECONDS
        return self._flush_thinking() if ready else None

    def _feed_content(self, delta: Any) -> List[StreamDelta]:
        """Emit answer text after any reasoning buffered ahead of it."""
        content = getattr(delta, "content", None)
        if not content:
            return []
        self.started = True
        emitted = []
        pending = self._flush_thinking()
        if pending is not None:
            emitted.append(pending)
        emitted.append(StreamDelta(type="text", text=content))
        return emitted

    def _feed_tool_calls(self, delta: Any) -> None:
        """Merge streamed tool-call fragments by provider index."""
        for fragment in getattr(delta, "tool_calls", None) or ():
            self.started = True
            buffer = self.tool_calls.setdefault(
                fragment.index,
                {"id": "", "name": "", "arguments": ""},
            )
            if fragment.id:
                buffer["id"] = fragment.id
            function = fragment.function
            if function:
                if function.name:
                    buffer["name"] = function.name
                if function.arguments:
                    buffer["arguments"] += function.arguments

    def feed(self, chunk: Any) -> List[StreamDelta]:
        """Consume one raw chunk and return any immediately visible deltas."""
        emitted: List[StreamDelta] = []
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            self.usage = extract_usage(chunk_usage)
            self.started = True

        choices = getattr(chunk, "choices", None)
        if not choices:
            return emitted
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            pending = self._feed_reasoning(delta)
            if pending is not None:
                emitted.append(pending)
            emitted.extend(self._feed_content(delta))
            self._feed_tool_calls(delta)

        reason = getattr(choice, "finish_reason", None)
        if reason:
            self.started = True
            self.finish_reason = "length" if reason == "length" else reason
        return emitted

    def finish(self) -> Iterator[StreamDelta]:
        """Flush buffered reasoning, tool calls, usage, and the terminal delta."""
        pending = self._flush_thinking()
        if pending is not None:
            yield pending
        for index, buffer in self.tool_calls.items():
            yield StreamDelta(
                type="tool_call",
                tool_call=ToolCall(
                    id=buffer["id"] or f"call_{index}",
                    name=buffer["name"],
                    arguments=_decode_tool_arguments(
                        buffer["arguments"], buffer["name"]
                    ),
                ),
            )
        if self.usage is not None:
            yield StreamDelta(type="usage", usage=self.usage)
        yield StreamDelta(
            type="done",
            finish_reason=self.finish_reason,
            usage=self.usage,
        )


log = logging.getLogger(__name__)


#: Loggers LiteLLM/deps attach to; quietened because this library runs inside a
#: host app whose own logging a noisy dependency would otherwise drown out.
class LiteLLMModel(Model):
    """Universal model adapter backed by **LiteLLM**.

    Pass this object to ``Agent(model=...)`` when the call needs explicit endpoint,
    credential, sampling, or context-window configuration. A plain string passed
    to ``Agent`` is shorthand for ``LiteLLMModel(the_string)``.

    Args:
        model: LiteLLM model identifier. Uses the same ``provider/model``
            convention as LiteLLM itself (``gpt-4o``, ``claude-sonnet-4-20250514``,
            ``ollama/qwen2.5-14b``). For an OpenAI-compatible server such as LM
            Studio, use ``openai/<server-model-id>`` with ``api_base``; ``openai``
            describes the wire protocol, not the company serving the model.
        api_key: Model service API key. Loopback endpoints use a harmless local
            placeholder when omitted; other endpoints fall back to environment
            variables.
        api_base: Override the server URL (for local / custom endpoints).
        supports_tools: Explicit native tool-calling support. ``None`` uses local
            server metadata when available and otherwise assumes support.
        reasoning_effort: Optional LiteLLM reasoning level: ``low``, ``medium``,
            ``high``, or ``disable``.
        context_window: Explicit total/input context limit when automatic model
            metadata is unavailable or wrong. Must be positive.
        temperature: Default sampling temperature, clamped to ``0`` through ``2``.
        max_tokens: Default maximum output tokens. On servers reporting a total
            context window, this amount is reserved before fitting input.

    Raises:
        ImportError: If LiteLLM is not installed.
        ValueError: If ``reasoning_effort`` or ``context_window`` is invalid.

    Example:
        Connect to an OpenAI-compatible LM Studio server::

            model = LiteLLMModel(
                "openai/qwen3-8b",
                api_base="http://127.0.0.1:1234/v1",
                context_window=32_768,
                temperature=0.2,
            )
            agent = Agent(model=model, instructions="Help with this project.")
    """

    # The adapter is the only retry authority: bounded attempts, capped linear backoff, honoring Retry-After.
    _MAX_RETRIES: int = 5
    _RETRY_BASE_DELAY: float = 10.0
    _MAX_RETRY_DELAY: float = 60.0

    # LiteLLM's unified reasoning_effort translates per-provider; dropped via drop_params on non-reasoning models.
    _REASONING_LEVELS = frozenset({"low", "medium", "high", "disable"})

    # Each call receives its own abort event so a shared adapter can serve concurrent runs without collision.
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        supports_tools: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        context_window: Optional[int] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        try:
            import litellm as _litellm
        except ImportError as exc:
            raise ImportError("pip install litellm") from exc

        self._litellm = _litellm
        self.model = model
        self._api_base = api_base
        self._is_local = local_models.is_local_base(api_base)
        # Cloud/remote endpoints keep ``None`` so LiteLLM can resolve their own provider env vars.
        self._api_key = api_key or (_LOCAL_API_KEY if self._is_local else None)
        if self._api_key:
            from lamssi_agents.redaction import register_secret

            register_secret(self._api_key)
        self._reasoning_effort = self._validate_reasoning(reasoning_effort)
        self.temperature = max(0.0, min(2.0, float(temperature)))
        self.max_tokens = max(1, int(max_tokens))

        # Remembered answer from `model_catalog()`; None until first asked.
        self._catalog: Optional[Tuple[List[str], str]] = None

        # Cumulative token usage across every call made through this adapter instance; see reset_usage().
        self._cumulative_usage = Usage()
        self._last_usage = Usage()
        self._usage_lock = threading.Lock()

        # Ask the local server first: LiteLLM's DB only knows named models and defaults local ones to 32k.
        served = local_models.probe(api_base, model)
        self._supports_tools = self._resolve_tool_support(supports_tools, served)
        self._configure_context_window(served, context_window)
        #: True once a request larger than the believed window was accepted.
        self._window_widened = False

    @staticmethod
    def _resolve_tool_support(
        explicit: Optional[bool],
        served: local_models.LocalModelInfo,
    ) -> bool:
        """Apply explicit > detected > enabled precedence for tool support."""
        if explicit is not None:
            return bool(explicit)
        if served.supports_tools is not None:
            return served.supports_tools
        return True

    @staticmethod
    def _context_override(value: Optional[int]) -> int:
        """Validate an optional explicit context window, returning zero if absent."""
        if value is None:
            return 0
        try:
            window = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("context_window must be a positive integer") from exc
        if window <= 0:
            raise ValueError("context_window must be a positive integer")
        return window

    def _configure_context_window(
        self,
        served: local_models.LocalModelInfo,
        explicit: Optional[int],
    ) -> None:
        """Resolve and report explicit, local-probed, or LiteLLM window metadata."""
        override = self._context_override(explicit)
        detected = served.context_window
        detected_as = ""
        if not detected:
            detected, detected_as = self._detect_context_window(
                self.model, self._litellm
            )

        self._context_window = override or detected or 32_000
        self._window_is_total = bool(served.context_window) or (
            self._is_local and bool(override)
        )
        if not (override or detected):
            log.warning(
                "LiteLLMModel: model=%s is unknown to the LiteLLM DB and no local "
                "server answered; guessing context_window=%d. Pass context_window= to "
                "set it.",
                self.model,
                self._context_window,
            )
            return

        detail = ""
        if override:
            detail = " (override)"
        elif served.context_window:
            detail = f" (from {served.source})"
        elif detected_as != self.model:
            detail = f" (LiteLLM DB, as {detected_as!r})"
        log.info(
            "LiteLLMModel: model=%s, context_window=%d tokens%s",
            self.model,
            self._context_window,
            detail,
        )

    @classmethod
    def _validate_reasoning(cls, value: Optional[str]) -> Optional[str]:
        """Coerce *value* to a valid ``reasoning_effort`` or raise."""
        if value is None or value == "":
            return None
        norm = str(value).strip().lower()
        if norm in cls._REASONING_LEVELS:
            return norm
        raise ValueError(
            f"reasoning_effort must be one of "
            f"{sorted(cls._REASONING_LEVELS) + [None]}, got {value!r}"
        )

    @property
    def reasoning_effort(self) -> Optional[str]:
        return self._reasoning_effort

    def set_reasoning_effort(self, value: Optional[str]) -> None:
        """Change the sticky reasoning level for subsequent calls."""
        self._reasoning_effort = self._validate_reasoning(value)
        log.info("reasoning_effort set to %r", self._reasoning_effort)

    @staticmethod
    def _detect_context_window(model: str, _litellm) -> Tuple[int, str]:
        """Find the context window while progressively removing route prefixes."""
        candidate = model
        while True:
            try:
                info = _litellm.get_model_info(candidate)
                ctx = info.get("max_input_tokens") or info.get("max_tokens") or 0
                if ctx:
                    return int(ctx), candidate
            except Exception:
                pass
            if "/" not in candidate:
                return 0, ""
            candidate = candidate.split("/", 1)[1]

    _TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

    # Terminal quota and billing errors.
    _TERMINAL_TOKENS: tuple = (
        "insufficient_quota",
        "quota_exceeded",
        "billing_hard_limit_reached",
        "payment_required",
        "exceeded_current_quota",
        "you exceeded your current quota",
        "please check your plan and billing",
    )

    def _is_terminal(self, exc: Exception) -> bool:
        """True for a quota / billing error retrying cannot fix."""
        msg = str(exc).lower()
        return any(tok in msg for tok in self._TERMINAL_TOKENS)

    def _is_transient(self, exc: Exception) -> bool:
        """True for a rate-limit or server error worth retrying."""
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        if isinstance(status, int):
            return status in self._TRANSIENT_STATUS or status >= 500
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        msg = str(exc).lower()
        return any(
            m in msg
            for m in (
                "rate_limit",
                "429",
                "overloaded",
                "timed out",
                "timeout",
                "connection refused",
                "connection reset",
                "connection error",
                "api connection",
                "temporarily unavailable",
                "service unavailable",
            )
        )

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        """Whether to retry after *attempt* (1-based): transient, not terminal, budget left."""
        return (
            attempt < self._MAX_RETRIES
            and self._is_transient(exc)
            and not self._is_terminal(exc)
        )

    def _as_error(self, exc: Exception) -> Exception:
        """The error to raise on give-up: an actionable quota message, else *exc*."""
        return (
            RuntimeError(self._terminal_message(exc)) if self._is_terminal(exc) else exc
        )

    def _terminal_message(self, exc: Exception) -> str:
        """An actionable message for a quota / billing error."""
        msg = str(exc).lower()
        hint = ""
        if "openai" in msg or self.model.lower().startswith(("gpt", "o1", "o3", "o4")):
            hint = " Top up or raise the limit at https://platform.openai.com/account/billing"
        elif "anthropic" in msg or self.model.lower().startswith("claude"):
            hint = " Check usage at https://console.anthropic.com/settings/billing"
        return (
            f"Quota / billing limit reached on {self.model} - not retrying."
            f"{hint} Original: {clean_model_error(exc)}"
        )

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        """Backoff before a retry: the server's Retry-After, else capped linear."""
        after = self._retry_after_header(exc)
        if after is not None:
            return after
        return min(self._RETRY_BASE_DELAY * attempt, self._MAX_RETRY_DELAY)

    def _retry_after_header(self, exc: Exception) -> Optional[float]:
        """The server's Retry-After in seconds, clamped, if it sent one."""
        for src in (getattr(exc, "response", None), exc):
            headers = getattr(src, "headers", None) or {}
            try:
                value = headers.get("retry-after") or headers.get("Retry-After")
            except AttributeError:
                continue
            if value:
                try:
                    return max(0.0, min(float(value), self._MAX_RETRY_DELAY))
                except (TypeError, ValueError):
                    pass
        return None

    def _wait(
        self,
        delay: float,
        abort_event: threading.Event | None = None,
    ) -> None:
        """Abort-aware backoff; an abort raises ProviderInterrupted."""
        if abort_event is not None:
            if abort_event.wait(timeout=delay):
                raise ProviderInterrupted("Retry wait aborted by user")
        else:
            time.sleep(delay)

    def _retry_or_raise(self, exc: Exception, attempt: int) -> float:
        """Return a retry delay or raise a normalized provider error."""
        if not self._should_retry(exc, attempt):
            raise self._as_error(exc)
        return self._retry_delay(exc, attempt)

    # Prompt-caching switch.
    _prompt_caching: bool = True

    def _apply_prompt_caching(self, msgs: List[Dict[str, Any]]) -> None:
        """Insert cache breakpoints (in place), honouring the kill switch."""
        if self._prompt_caching:
            apply_prompt_caching(msgs, self.model)

    def _common_kwargs(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Build the kwargs dict for ``litellm.completion()``."""
        msgs = [m.to_provider_dict() for m in messages]

        self._apply_prompt_caching(msgs)

        kw: Dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "drop_params": True,
        }
        if self._api_key:
            kw["api_key"] = self._api_key
        if self._api_base:
            kw["api_base"] = self._api_base

        if tools and self._supports_tools:
            kw["tools"] = build_tools_openai_schema(tools)
            kw["tool_choice"] = "auto"

        # Some OpenAI models reject tools combined with reasoning_effort.
        if self._reasoning_effort is not None:
            if tools and self._tool_reasoning_conflict():
                if not self._tool_reasoning_warned:
                    log.warning(
                        "Dropping reasoning_effort=%r for model %r: "
                        "OpenAI's /v1/chat/completions rejects "
                        "tools + reasoning_effort together. Set "
                        "reasoning_effort=None in AI settings, or "
                        "switch to a non-gpt-5 model, to silence "
                        "this warning.",
                        self._reasoning_effort,
                        self.model,
                    )
                    self._tool_reasoning_warned = True
            else:
                kw["reasoning_effort"] = self._reasoning_effort

        return kw

    # Prevent repeated compatibility warnings.
    _tool_reasoning_warned: bool = False

    # Confirmed model families that reject tools with reasoning_effort.
    _NO_TOOLS_REASONING_PREFIXES: tuple[str, ...] = (
        "gpt-5",  # gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.5, etc.
        "openai/gpt-5",  # local-server routing prefix
    )

    def _tool_reasoning_conflict(self) -> bool:
        """True when this provider's model rejects tools + reasoning."""
        name = (self.model or "").lower()
        return any(name.startswith(p) for p in self._NO_TOOLS_REASONING_PREFIXES)

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        abort_event: threading.Event | None = None,
    ) -> Iterator[StreamDelta]:
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        kw = self._common_kwargs(messages, tools, temperature, max_tokens)
        kw["stream"] = True
        kw["stream_options"] = {"include_usage": True}

        parser = _StreamParser()
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                stream_iter = self._litellm.completion(**kw)
                for chunk in _interruptible_chunks(stream_iter, abort_event):
                    yield from parser.feed(chunk)
                break
            except Exception as exc:
                if parser.started:
                    raise
                delay = self._retry_or_raise(exc, attempt)
                yield StreamDelta(
                    type="retrying",
                    retry={
                        "attempt": attempt,
                        "max_retries": self._MAX_RETRIES,
                        "delay": delay,
                        "reason": clean_model_error(exc),
                    },
                )
                self._wait(delay, abort_event)

        if parser.usage is not None:
            self._record_usage(parser.usage)
        yield from parser.finish()

    @property
    def supports_tools(self) -> bool:
        """Whether native tool schemas are sent to this model."""
        return self._supports_tools

    @property
    def context_window(self) -> int:
        """Detected or configured context-window metadata in tokens."""
        return self._context_window

    @property
    def max_input_tokens(self) -> int:
        """Usable input limit after reserving output on total-window servers."""
        if not self._window_is_total:
            return self._context_window
        return max(1, self._context_window - self.max_tokens)

    def _widen_window_if_exceeded(self, prompt_tokens: int) -> None:
        """Widen a guessed context window after a larger request succeeds."""
        if prompt_tokens <= self._context_window:
            return
        was, self._context_window = self._context_window, prompt_tokens
        if not self._window_widened:
            self._window_widened = True
            log.warning(
                "context_window for %s was %d, but a %d-token request was accepted; "
                "widening. Pass context_window= to set it and stop guessing.",
                self.model,
                was,
                prompt_tokens,
            )

    def _record_usage(self, usage: Usage) -> None:
        """Update usage counters and log the latest and cumulative totals."""
        with self._usage_lock:
            self._last_usage = usage
            self._cumulative_usage.add(usage)
            cum = self._cumulative_usage

        self._widen_window_if_exceeded(usage.prompt_tokens)

        extras = []
        if usage.cached_tokens:
            extras.append(f"cache_read={usage.cached_tokens}")
        if usage.cache_write_tokens:
            extras.append(f"cache_write={usage.cache_write_tokens}")
        if usage.reasoning_tokens:
            extras.append(f"reasoning={usage.reasoning_tokens}")
        extras_str = (" " + " ".join(extras)) if extras else ""

        log.info(
            "[tokens] model=%s in=%d out=%d total=%d%s  cum: in=%d out=%d total=%d",
            self.model,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            extras_str,
            cum.prompt_tokens,
            cum.completion_tokens,
            cum.total_tokens,
        )

    @property
    def last_usage(self) -> Usage:
        """Token usage from the most recent provider call."""
        with self._usage_lock:
            return replace(self._last_usage)

    @property
    def cumulative_usage(self) -> Usage:
        """Token usage summed across every call on this provider instance."""
        with self._usage_lock:
            return replace(self._cumulative_usage)

    def reset_usage(self) -> None:
        """Zero the cumulative and last-call usage counters."""
        with self._usage_lock:
            self._cumulative_usage = Usage()
            self._last_usage = Usage()

    @property
    def name(self) -> str:
        """Adapter name used in logs and status UIs."""
        return "LiteLLM"

    @property
    def is_local(self) -> bool:
        """Whether ``api_base`` points to a loopback endpoint."""
        return self._is_local

    def available_models(self, *, refresh: bool = False) -> List[str]:
        """Model ids this provider's backend reports, asked of the backend.

        Args:
            refresh: Re-query instead of the remembered answer (e.g. after
                changing a key, or behind a "refresh" button).

        Returns:
            The model ids, or a curated fallback list when unreachable. Empty
            for a backend with no catalogue that LiteLLM can still route.

        See :meth:`model_catalog` for the source label alongside the names.
        """
        return list(self.model_catalog(refresh=refresh)[0])

    def model_catalog(self, *, refresh: bool = False) -> Tuple[List[str], str]:
        """:meth:`available_models`, plus a label saying where the list came from.

        Args:
            refresh: Re-query instead of returning the remembered answer.

        Returns:
            ``(model_ids, label)``, e.g. ``"Anthropic (48 models: key OK)"``.
            Remembered after the first call, since this does network I/O.
        """
        if refresh or self._catalog is None:
            fetch = source_for_model(
                self.model,
                api_key=self._api_key or "",
                api_base=self._api_base or "",
            )
            self._catalog = fetch()
        return self._catalog

    def check_connectivity(self) -> tuple[bool, str]:
        """Verify the target endpoint is reachable."""
        if not self._api_base:
            # Cloud provider: assume reachable; API key errors surface later.
            return True, ""

        url = models_endpoint(self._api_base)
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3):
                pass
            return True, ""
        except urllib.error.HTTPError as exc:
            # A 4xx means the server answered (e.g. 401 auth) and is reachable; only 5xx counts as unreachable.
            if 400 <= exc.code < 500:
                return True, ""
            return False, f"Cannot reach {self._api_base}: HTTP {exc.code}"
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            return False, f"Cannot reach {self._api_base}: {reason}"
        except Exception as exc:
            return False, f"Cannot reach {self._api_base}: {exc}"
