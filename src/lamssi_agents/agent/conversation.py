"""The Conversation: its messages, token calibration, and history operations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from lamssi_agents.events import AgentEventType
from lamssi_agents.history.tokens import TokenCalibrator, estimate_tokens
from lamssi_agents.model import input_limit
from lamssi_agents.providers import Message

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HistoryServices:
    """The model, strategy, config, and event sink that history fitting and
    compaction read, as live accessors.

    ``notify_compacted`` tells the tool runtime that a summarisation invalidated
    its safety state.
    """

    model: Callable[[], Any]
    summary_model: Callable[[], Any]
    compactor: Callable[[], Any]
    config: Callable[[], Any]
    emit: Callable[..., None]
    abort_event: Any
    fixed_overhead: Callable[[], int]
    notify_compacted: Callable[[bool], None]


def _null_services() -> HistoryServices:
    """Services for a bare Conversation: message ops work, fitting does not."""
    return HistoryServices(
        model=lambda: None,
        summary_model=lambda: None,
        compactor=lambda: (lambda history, **_: history),
        config=lambda: None,
        emit=lambda *a, **k: None,
        abort_event=None,
        fixed_overhead=lambda: 0,
        notify_compacted=lambda demoted: None,
    )


class Conversation:
    """One conversation: its messages, turn counter, token calibration, and
    per-conversation feature state.

    The single mutable owner of the message list; other components return
    messages for it to append and never edit the list directly. Fits and
    compacts its own history through the injected :class:`HistoryServices`.
    """

    __slots__ = ("history", "turn", "tokens", "_state", "_svc")

    def __init__(self, services: Optional[HistoryServices] = None) -> None:
        self.history: List[Message] = []
        #: 1-based turn counter within the current run, for dedupe bookkeeping.
        self.turn: int = 0
        self.tokens = TokenCalibrator()
        #: Per-conversation feature state, keyed by the type that owns it.
        self._state: Dict[type, Any] = {}
        self._svc = services if services is not None else _null_services()

    def state(self, key: type, factory: Any) -> Any:
        """This conversation's state for *key*, built on first use."""
        value = self._state.get(key)
        if value is None:
            value = self._state[key] = factory()
        return value

    def _notify(self, event: str) -> None:
        """Send *event* to every conversation-state holder that defines it."""
        for holder in list(self._state.values()):
            hook = getattr(holder, event, None)
            if hook is None:
                continue
            try:
                hook()
            except Exception as exc:
                log.warning(
                    "conversation state %s.%s raised: %s",
                    type(holder).__name__, event, exc, exc_info=True,
                )

    def append(self, message: Message) -> None:
        self.history.append(message)

    def extend(self, messages: Any) -> None:
        self.history.extend(messages)

    @property
    def last(self) -> Optional[Message]:
        return self.history[-1] if self.history else None

    def last_user_message(self) -> str:
        """The most recent user message's text, or ``""``."""
        for m in reversed(self.history):
            if m.role == "user" and m.content:
                return m.content
        return ""

    def begin_request(self) -> None:
        """Prepare for a new user message: reset the turn and notify holders."""
        self.turn = 0
        self._notify("on_new_turn")

    def clear(self) -> None:
        """Drop the conversation and everything derived from it."""
        self.history.clear()
        self.turn = 0
        self.tokens.on_cleared()
        self._notify("on_cleared")

    def on_compacted(self, compacted: List[Message]) -> bool:
        """Adopt a compacted history and notify state holders.

        Returns whether the change was a demotion (message bodies only), so the
        caller can invalidate tool-safety state only on a summarisation.
        """
        log.info(
            "History compacted: %d -> %d messages", len(self.history), len(compacted)
        )
        demoted_only = _same_messages(self.history, compacted)
        self.history = compacted
        event = "on_demoted" if demoted_only else "on_compacted"
        # The token calibrator is core Conversation state, not a _state holder,
        # so notify it directly: its char-anchor is now stale (the ratio survives).
        getattr(self.tokens, event)()
        self._notify(event)
        return demoted_only

    def sanitize(self) -> None:
        """Add missing tool results and remove orphaned results, in place."""
        history = self.history
        if not history:
            return
        # A call is answered if a result for it exists anywhere, so a result that
        # is not contiguous with its call is not mistaken for a missing one.
        answered: set = {
            getattr(m, "tool_call_id", None) for m in history if m.role == "tool"
        }
        answered.discard(None)
        i = 0
        while i < len(history):
            msg = history[i]
            if msg.role != "assistant" or not msg.tool_calls:
                i += 1
                continue
            j = i + 1
            while j < len(history) and history[j].role == "tool":
                j += 1
            missing = {tc.id for tc in msg.tool_calls if tc.id} - answered
            if missing:
                log.warning("Patching %d unanswered tool call(s) at %d", len(missing), i)
                names = {tc.id: tc.name for tc in msg.tool_calls if tc.id}
                synthetic = [
                    Message(
                        role="tool",
                        content=json.dumps(
                            {"error": "Tool execution was cancelled or never completed."}
                        ),
                        tool_call_id=tc_id,
                        name=names.get(tc_id, "unknown"),
                    )
                    for tc_id in missing
                ]
                history[j:j] = synthetic
                i = j + len(synthetic)
            else:
                i = j
        history[:] = strip_orphan_tool_messages(history)

    # History sizing and compaction. These read the run through self._svc, never
    # an Agent, so the model, strategy, and config can change between requests.

    def context_usage(self, *, fixed_chars: Optional[int] = None) -> "ContextUsage":
        """Estimate the next complete request, using provider usage when available."""
        model = self._svc.model()
        window = input_limit(model)
        last = getattr(model, "last_usage", None)
        anchor_tokens = int(getattr(last, "prompt_tokens", 0) or 0)
        anchor_message = self.tokens.anchor

        if anchor_tokens > 0 and 0 <= anchor_message <= len(self.history):
            trailing = self.history[anchor_message:]
            return ContextUsage(
                used=anchor_tokens + estimate_tokens(trailing, self.tokens),
                window=window,
                anchored=True,
            )

        overhead = fixed_chars
        if overhead is None:
            try:
                overhead = self._svc.fixed_overhead()
            except Exception as exc:
                log.debug("context usage could not size the fixed request: %s", exc)
                overhead = 0

        used = self.tokens.estimate(overhead) + estimate_tokens(self.history, self.tokens)
        return ContextUsage(used=used, window=window)

    def _autocompact_budget(self) -> int:
        """Return the configured automatic compaction budget."""
        config = self._svc.config()
        window = input_limit(self._svc.model())
        if window <= 0:
            return config.history_budget_tokens
        if config.reserve_tokens > 0:
            return max(1, window - config.reserve_tokens)
        return int(config.autocompact_fraction * window)

    def _compact(
        self,
        history: List[Message],
        *,
        budget_tokens: int,
        keep_recent: int,
        system_prompt: str = "",
        overhead_chars: int = 0,
        used_estimate: Optional[int] = None,
        focus: str = "",
    ) -> tuple[List[Message], int]:
        """Run the configured strategy once and measure its history result."""
        compacted = self._svc.compactor()(
            history,
            model=self._svc.summary_model(),
            fallback_model=self._svc.model(),
            budget_tokens=budget_tokens,
            keep_recent=keep_recent,
            calibrator=self.tokens,
            system_prompt=system_prompt,
            overhead_chars=overhead_chars,
            used_estimate=used_estimate,
            focus=focus,
            abort_event=self._svc.abort_event,
        )
        return compacted, estimate_tokens(compacted, self.tokens)

    def _emit_compacting(
        self, *, displayed_tokens: int, messages_before: int, history_tokens: int
    ) -> None:
        """Announce a potentially slow compaction pass before entering it."""
        self._svc.emit(
            AgentEventType.HISTORY_COMPACTING,
            f"~{displayed_tokens:,} tokens",
            messages_before=messages_before,
            tokens_before=history_tokens,
        )

    def _adopt(self, result: "CompactionResult", compacted: List[Message]) -> None:
        """Adopt compacted history, tell the runtime, and emit the result."""
        demoted_only = self.on_compacted(compacted)
        self._svc.notify_compacted(demoted_only)
        log.info("compaction: %s", result)
        self._svc.emit(
            AgentEventType.HISTORY_COMPACTED, str(result),
            messages_before=result.messages_before,
            messages_after=result.messages_after,
            tokens_saved=result.tokens_saved,
        )

    def fit_request(
        self,
        system_prompt: str,
        *,
        overhead_chars: int = 0,
        force: bool = False,
    ) -> List[Message]:
        """Compact when needed, adopt a smaller result, and enforce the hard limit."""
        config = self._svc.config()

        before = list(self.history)
        tokens_before = estimate_tokens(before, self.tokens)
        fixed_chars = len(system_prompt) + overhead_chars
        fixed_tokens = self.tokens.estimate(fixed_chars)
        usage = self.context_usage(fixed_chars=fixed_chars)

        budget = self._autocompact_budget()
        window = input_limit(self._svc.model())
        original = _measure_fit(
            before,
            original=before,
            fixed_tokens=fixed_tokens,
            measured_tokens=usage.used,
            calibrator=self.tokens,
        )
        hard_fit = force or (window > 0 and original.request_tokens > window)
        needs_compaction = force or original.request_tokens > budget or hard_fit

        if not needs_compaction:
            return [Message(role="system", content=system_prompt)] + self.history

        self._emit_compacting(
            displayed_tokens=original.request_tokens,
            messages_before=len(before),
            history_tokens=tokens_before,
        )

        target_budget = min(budget, window) if hard_fit and window > 0 else budget
        compacted, _ = self._compact(
            before,
            budget_tokens=1 if force else target_budget,
            keep_recent=2 if force else config.keep_recent,
            system_prompt=system_prompt,
            overhead_chars=overhead_chars,
            used_estimate=original.request_tokens,
        )
        candidate = _measure_fit(
            compacted,
            original=before,
            fixed_tokens=fixed_tokens,
            measured_tokens=usage.used,
            calibrator=self.tokens,
        )
        useful_saving = worth_adopting(tokens_before, candidate.history_tokens)
        candidate_is_smaller = candidate.request_tokens < original.request_tokens
        adopted = candidate.messages != before and candidate_is_smaller and (
            useful_saving or hard_fit
        )
        selected = candidate if adopted else original

        result = CompactionResult(
            messages_before=len(before),
            messages_after=len(selected.messages),
            tokens_before=tokens_before,
            tokens_after=selected.history_tokens if adopted else tokens_before,
            adopted=adopted,
        )

        if result:
            self._adopt(result, selected.messages)

        # Never send a request known to exceed the model window.
        if window > 0 and selected.request_tokens > window:
            raise ContextWindowExceeded(selected.request_tokens, window)

        if not result and original.request_tokens > budget and not force:
            log.warning(
                "compaction declined: request ~%d tokens against a soft budget of %d, "
                "and the best pass saved only ~%d history tokens. The request still "
                "fits the known hard window, or that window is unknown.",
                original.request_tokens,
                budget,
                max(0, tokens_before - candidate.history_tokens),
            )
            self._svc.emit(
                AgentEventType.HISTORY_COMPACTED,
                f"declined - over budget by ~{original.request_tokens - budget} tokens",
                messages_before=len(before),
                messages_after=len(before),
                tokens_saved=0,
            )

        return [Message(role="system", content=system_prompt)] + self.history

    def force_compaction(
        self,
        *,
        budget_tokens: int = 1,
        keep_recent: Optional[int] = None,
        focus: str = "",
    ) -> "CompactionResult":
        """Shrink the history now, whatever the estimate says.

        Called both when the provider rejects a request as too large and when a
        host wants to compact deliberately (before a long run, at a task boundary).

        A budget of 1 runs every pass: demotion then summarisation, for the
        strongest shrink; pass a real number to compact *to* a size. ``keep_recent``
        overrides the configured tail for this call only.
        """
        before = list(self.history)
        tokens_before = estimate_tokens(before, self.tokens)

        budget = max(1, int(budget_tokens))
        if tokens_before > budget:
            self._emit_compacting(
                displayed_tokens=tokens_before,
                messages_before=len(before),
                history_tokens=tokens_before,
            )

        compacted, tokens_after = self._compact(
            before,
            budget_tokens=budget,
            keep_recent=(
                self._svc.config().keep_recent
                if keep_recent is None
                else max(1, int(keep_recent))
            ),
            focus=focus,
        )

        # Same rule the automatic path uses: see :func:`worth_adopting`.
        saved = tokens_before - tokens_after
        adopted = worth_adopting(tokens_before, tokens_after)

        result = CompactionResult(
            messages_before=len(before),
            messages_after=len(compacted),
            tokens_before=tokens_before,
            tokens_after=tokens_after if adopted else tokens_before,
            adopted=adopted,
        )
        if result:
            self._adopt(result, compacted)
        else:
            log.info(
                "compaction not worth adopting (%d messages, ~%d tokens; the pass would "
                "have %s)",
                len(before), tokens_before,
                f"saved only ~{saved}" if saved >= 0 else f"ADDED ~{-saved}",
            )
        return result

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        return f"<Conversation {len(self.history)} message(s)>"


def _same_messages(before: List[Message], after: List[Message]) -> bool:
    """Return whether only message bodies changed, preserving roles and count."""
    if len(before) != len(after):
        return False
    return [m.role for m in before] == [m.role for m in after]


class ContextWindowExceeded(RuntimeError):
    """The next request still cannot fit after request fitting."""

    def __init__(self, estimated_tokens: int, window_tokens: int) -> None:
        self.estimated_tokens = estimated_tokens
        self.window_tokens = window_tokens
        super().__init__(
            f"The next request is estimated at ~{estimated_tokens:,} tokens after "
            f"fitting, above the model's "
            f"{window_tokens:,}-token context window. It was not sent."
        )


@dataclass(frozen=True, slots=True)
class ContextUsage:
    """Estimated size of the next request against the model's input window."""

    used: int
    window: int
    anchored: bool = False

    @property
    def fraction(self) -> float:
        return (self.used / self.window) if self.window > 0 else 0.0

    @property
    def percent(self) -> int:
        return round(self.fraction * 100)

    @property
    def remaining(self) -> int:
        return max(0, self.window - self.used) if self.window > 0 else 0

    def __str__(self) -> str:
        if self.window <= 0:
            return f"~{_short_tokens(self.used)} / unknown"
        return (
            f"~{_short_tokens(self.used)} / {_short_tokens(self.window)} "
            f"({self.percent}%)"
        )


def _short_tokens(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 10_000:
        return f"{tokens // 1000}k"
    if tokens >= 1_000:
        return f"{tokens / 1000:.1f}k"
    return str(tokens)


def strip_orphan_tool_messages(messages: List[Message]) -> List[Message]:
    """Drop tool results without a preceding owning call."""
    valid_ids: set = set()
    out: List[Message] = []
    dropped = 0
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            valid_ids.update(tc.id for tc in m.tool_calls if tc.id)
            out.append(m)
        elif m.role == "tool":
            tcid = getattr(m, "tool_call_id", None)
            if tcid and tcid in valid_ids:
                out.append(m)
            else:
                dropped += 1
        else:
            out.append(m)
    if dropped:
        log.warning("Dropped %d orphan tool message(s) before send", dropped)
    return out


@dataclass(frozen=True, slots=True)
class _Fit:
    """The three values needed to compare one fitted history."""

    messages: List[Message]
    history_tokens: int
    request_tokens: int


def _measure_fit(
    messages: List[Message],
    *,
    original: List[Message],
    fixed_tokens: int,
    measured_tokens: int,
    calibrator: Any,
) -> _Fit:
    """Measure one history candidate against the same request inputs."""
    history_tokens = estimate_tokens(messages, calibrator)
    request_tokens = fixed_tokens + history_tokens
    if messages == original:
        request_tokens = max(measured_tokens, request_tokens)
    return _Fit(
        messages=messages,
        history_tokens=history_tokens,
        request_tokens=request_tokens,
    )


#: Minimum absolute and relative savings required for adoption.
_MIN_SAVING_TOKENS = 200
_MIN_SAVING_FRACTION = 0.05


def worth_adopting(tokens_before: int, tokens_after: int) -> bool:
    """Return whether compaction saved enough tokens to adopt its result."""
    saved = tokens_before - tokens_after
    return saved >= max(_MIN_SAVING_TOKENS, int(tokens_before * _MIN_SAVING_FRACTION))


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """What one compaction pass achieved. Falsy when it achieved nothing."""

    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    #: Whether the result was actually adopted; only an adopted pass changed the transcript.
    adopted: bool = False

    @property
    def messages_removed(self) -> int:
        return self.messages_before - self.messages_after

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    def __bool__(self) -> bool:
        """Return whether the transcript changed."""
        return self.adopted

    def __str__(self) -> str:
        """Return an ASCII summary safe for common Windows consoles."""
        return (
            f"{self.messages_before} -> {self.messages_after} messages, "
            f"~{self.tokens_before:,} -> ~{self.tokens_after:,} tokens"
        )


__all__ = [
    "Conversation",
    "HistoryServices",
    "strip_orphan_tool_messages",
    "ContextUsage",
    "ContextWindowExceeded",
    "CompactionResult",
    "worth_adopting",
]
