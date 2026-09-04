"""JSONL conversation logging."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from lamssi_agents.events import AgentEvent, AgentEventType
from lamssi_agents.redaction import redact

log = logging.getLogger("ConversationLog")


def _looks_ok(preview: str) -> bool:
    """Return whether a tool-result preview lacks a top-level error."""
    body = preview.strip()
    if body.startswith("{"):
        try:
            parsed = json.loads(body)
        except ValueError:
            pass  # truncated preview, or not JSON after all
        else:
            if isinstance(parsed, dict):
                return "error" not in parsed
    return '"error"' not in preview


RemoteSink = Callable[[Dict[str, Any]], None]


def _serialise_message(msg: Any) -> Dict[str, Any]:
    """Serialize the provider-visible fields of a model message."""
    out: Dict[str, Any] = {"role": getattr(msg, "role", None)}
    content = getattr(msg, "content", None)
    if content:
        out["content"] = content
    name = getattr(msg, "name", None)
    if name:
        out["name"] = name
    tcid = getattr(msg, "tool_call_id", None)
    if tcid:
        out["tool_call_id"] = tcid
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": getattr(tc, "id", None),
                "name": getattr(tc, "name", None),
                "arguments": getattr(tc, "arguments", None),
            }
            for tc in tool_calls
        ]
    return out


class ConversationLogger:
    """Append-only JSONL logger that hooks the agent's event stream.

    One file per session: ``<log_dir>/<session_id>.jsonl``. The
    session_id is generated on construction and rolled when the file
    is closed (e.g. after ``agent.clear_history()``).

    Constructing with ``debug=True`` returns a :class:`DebugConversationLogger`
    (via ``__new__`` dispatch), so callers get the richer logger without
    knowing the subclass exists.
    """

    def __new__(cls, *args: Any, debug: bool = False, **kwargs: Any):
        # debug=True on this exact class transparently builds the debug variant.
        target = DebugConversationLogger if debug and cls is ConversationLogger else cls
        return super().__new__(target)

    def __init__(
        self,
        log_dir: Union[str, Path],
        *,
        model: str = "",
        adapter: str = "",
        remote: Optional[RemoteSink] = None,
        debug: bool = False,
    ) -> None:
        """Create a logger.

        Args:
            debug: When ``True``, an instance of :class:`DebugConversationLogger`
                is built instead, adding token-growth / prompt-capture analytics.
        """
        # log_dir=None means no on-disk file (records still flow to a remote sink).
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.adapter = adapter
        self.remote = remote
        self.debug = debug

        self.session_id = self._new_session_id()
        self.path = self._session_path()
        self._tool_starts: Dict[str, float] = {}  # tool_name -> t0 (for duration_ms)
        #: Messages already written, so a snapshot logs only what was appended.
        self._logged_msg_count = 0
        self._closed = False

        self._write(
            {
                "kind": "session_start",
                "model": self.model,
                "adapter": self.adapter,
                "debug": self.debug,
            }
        )
        log.info(
            "Conversation log opened: %s%s",
            self.path,
            "  (debug)" if debug else "",
        )

    def __call__(self, event: AgentEvent) -> None:
        """Event-callback signature; the agent calls this for every event."""
        if self._closed:
            return
        try:
            self._handle(event)
        except Exception as exc:  # pragma: no cover
            log.warning("ConversationLogger error: %s", exc)

    def attach(self, agent) -> None:
        """Register this logger as an event listener on *agent*."""
        agent.add_event_listener(self)

    def set_metadata(self, *, model: str = "", adapter: str = "") -> None:
        """Update model metadata for subsequent records."""
        if model:
            self.model = model
        if adapter:
            self.adapter = adapter

    def roll_session(self) -> None:
        """Close the current file and start a new session."""
        self._write({"kind": "session_end"})
        self.session_id = self._new_session_id()
        self.path = self._session_path()
        self._tool_starts.clear()
        self._logged_msg_count = 0
        self._write(
            {
                "kind": "session_start",
                "model": self.model,
                "adapter": self.adapter,
                "debug": self.debug,
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self._write({"kind": "session_end"})
        self._closed = True
        log.info("Conversation log closed: %s", self.path)

    def _handle(self, event: AgentEvent) -> None:
        et = event.type

        if et == AgentEventType.USER_MESSAGE:
            self._write({"kind": "user", "text": str(event.data or "")})

        elif et == AgentEventType.TEXT_DONE:
            self._write({"kind": "assistant", "text": str(event.data or "")})

        elif et == AgentEventType.TOOL_START:
            name = str(event.data or "")
            args = event.metadata.get("arguments", {})
            self._tool_starts[name] = time.monotonic()
            self._write({"kind": "tool_call", "tool": name, "args": args})

        elif et == AgentEventType.TOOL_RESULT:
            self._handle_tool_result(event)

        elif et == AgentEventType.TOOL_REJECTED:
            self._write(
                {
                    "kind": "tool_rejected",
                    "tool": event.metadata.get("tool_name", ""),
                    "reason": str(event.data or ""),
                }
            )

        elif et == AgentEventType.ERROR:
            self._write({"kind": "error", "message": str(event.data or "")})

        elif et == AgentEventType.ABORTED:
            self._write({"kind": "aborted"})

        elif et == AgentEventType.HISTORY_COMPACTED:
            self._write(
                {
                    "kind": "compacted",
                    "detail": str(event.data or ""),
                    "messages_before": event.metadata.get("messages_before"),
                    "messages_after": event.metadata.get("messages_after"),
                    "tokens_saved": event.metadata.get("tokens_saved"),
                }
            )
            # History was rewritten, so the next snapshot cannot be a delta.
            self._logged_msg_count = 0

        elif et == AgentEventType.DONE:
            self._handle_done(event)

        # TEXT_DELTA is unlogged (TEXT_DONE has the full message). Approval and
        # interaction events are host coordination; tool records hold outcomes.

    def _handle_tool_result(self, event: AgentEvent) -> None:
        name = event.metadata.get("tool_name", "")
        t0 = self._tool_starts.pop(name, None)
        duration_ms = int((time.monotonic() - t0) * 1000) if t0 else None
        preview = str(event.data or "")
        rec: Dict[str, Any] = {
            "kind": "tool_result",
            "tool": name,
            "ok": _looks_ok(preview),
            "duration_ms": duration_ms,
        }
        self._record_tool_payload(rec, preview)
        self._write(rec)

    def _record_tool_payload(self, rec: Dict[str, Any], preview: str) -> None:
        """Attach the tool result body to *rec*: a short preview by default."""
        rec["preview"] = preview[:500]

    def _handle_done(self, event: AgentEvent) -> None:
        self._write({"kind": "turn_end"})

    def _write(self, fields: Dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "session_id": self.session_id,
            **fields,
        }
        # Redact after serialising: the flattened string exposes nested JSON fields too.
        line = redact(json.dumps(record, separators=(",", ":"), default=str))

        if self.path is not None:
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as exc:  # pragma: no cover
                log.warning("Failed to write %s: %s", self.path, exc)

        if self.remote is not None:
            try:
                # Send the already-redacted form: this sink posts over the network.
                self.remote(json.loads(line))
            except Exception as exc:
                log.warning("Remote sink error: %s", exc)

    def _session_path(self) -> Optional[Path]:
        if self.log_dir is None:
            return None
        return self.log_dir / f"{self.session_id}.jsonl"

    @staticmethod
    def _new_session_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{ts}-{uuid.uuid4().hex[:8]}"


class DebugConversationLogger(ConversationLogger):
    """Base logger plus the heavy analytics for chasing token growth.

    On top of the normal turn stream it captures: the full system prompt
    when it changes (hash-compared otherwise), per-turn ``messages_sent``
    snapshots, concatenated THINKING deltas, USAGE deltas, and full
    (untruncated) tool results.

    Built automatically by ``ConversationLogger(..., debug=True)``.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        *,
        model: str = "",
        adapter: str = "",
        remote: Optional[RemoteSink] = None,
        debug: bool = True,
    ) -> None:
        super().__init__(
            log_dir,
            model=model,
            adapter=adapter,
            remote=remote,
            debug=True,
        )
        self._last_prompt_hash: Optional[str] = None
        self._prev_total_tokens: int = 0
        self._thinking_buffer: list[str] = []
        self._current_turn: int = 0

    def roll_session(self) -> None:
        self._flush_thinking()  # goes to the outgoing session's file
        super().roll_session()
        self._last_prompt_hash = None
        self._prev_total_tokens = 0
        self._current_turn = 0

    def _handle(self, event: AgentEvent) -> None:
        et = event.type

        if et == AgentEventType.TURN_START:
            self._flush_thinking()  # close out previous turn's buffer
            self._current_turn = int(event.metadata.get("turn", 0))

        elif et == AgentEventType.MESSAGES_SENT:
            self._log_messages_sent(event)

        elif et == AgentEventType.THINKING:
            # Buffered and flushed once per turn, avoiding thousands of one-token writes.
            self._thinking_buffer.append(str(event.data or ""))

        elif et == AgentEventType.USAGE:
            self._log_usage(event)

        else:
            super()._handle(event)

    def _record_tool_payload(self, rec: Dict[str, Any], preview: str) -> None:
        rec["result"] = preview  # full, untruncated

    def _handle_done(self, event: AgentEvent) -> None:
        self._flush_thinking()  # flush any buffered thinking from the turn
        super()._handle_done(event)

    def _log_usage(self, event: AgentEvent) -> None:
        md = event.metadata
        total = int(md.get("total_tokens") or 0)
        self._write(
            {
                "kind": "usage",
                "turn": self._current_turn,
                "prompt": md.get("prompt_tokens"),
                "completion": md.get("completion_tokens"),
                "cached": md.get("cached_tokens"),
                "cache_write": md.get("cache_write_tokens"),
                "reasoning": md.get("reasoning_tokens"),
                "total": total,
                "delta_total": total - self._prev_total_tokens,
                "cumulative_total": md.get("cumulative_total"),
            }
        )
        self._prev_total_tokens = total

    def _log_messages_sent(self, event: AgentEvent) -> None:
        """Capture per-turn system-prompt snapshot + sizing metrics."""
        md = event.metadata
        prompt = str(event.data or "")
        h = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()[:12]
        rec: Dict[str, Any] = {
            "kind": "messages_sent",
            "turn": md.get("turn"),
            "history_msg_count": md.get("history_msg_count"),
            "tool_count": md.get("tool_count"),
            "tool_schema_bytes": md.get("tool_schema_bytes"),
            "system_prompt_chars": md.get("system_prompt_chars"),
            "system_prompt_hash": h,
        }
        # Full prompt only on the first turn or when changed; identical turns get just the hash.
        if h != self._last_prompt_hash:
            rec["system_prompt"] = prompt
            rec["system_prompt_changed"] = self._last_prompt_hash is not None
            self._last_prompt_hash = h

        # Only messages appended since the last snapshot; a `compacted` record resets the count so a rewrite logs in full.
        messages = md.get("messages") or ()
        if messages:
            if len(messages) <= self._logged_msg_count:
                self._logged_msg_count = 0  # history shrank: not an append
            fresh = list(messages)[self._logged_msg_count :]
            rec["messages_from"] = self._logged_msg_count
            rec["messages"] = [_serialise_message(m) for m in fresh]
            self._logged_msg_count = len(messages)
        self._write(rec)

    def _flush_thinking(self) -> None:
        if not self._thinking_buffer:
            return
        self._write(
            {
                "kind": "thinking",
                "turn": self._current_turn,
                "text": "".join(self._thinking_buffer),
            }
        )
        self._thinking_buffer.clear()
