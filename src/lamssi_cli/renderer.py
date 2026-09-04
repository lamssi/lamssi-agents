"""Turn ``AgentEvent`` streams into terminal output."""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, List, Optional

from lamssi_cli.ansi import bold, cyan, dim, green, magenta, red, yellow
from lamssi_cli.spinner import _Activity
from lamssi_agents.conversation_log import _serialise_message as _serialise_message_for_cli
from lamssi_agents.events import AgentEvent, AgentEventType

# role → coloured label, shared by /messages and /history rendering.
ROLE_LABELS = {
    "system": magenta("system"),
    "user": bold("user"),
    "assistant": green("agent"),
    "tool": cyan("tool"),
}


class Renderer:
    def __init__(
        self,
        activity: _Activity,
        *,
        verbose: bool = False,
        truncate: bool = False,
        show_messages: bool = False,
    ) -> None:
        self.activity = activity
        self.verbose = verbose
        # Display only: nothing here touches the history the model sees.
        self.truncate = truncate
        self.show_messages = show_messages  # print full LLM payload each turn

        self._streaming_text: bool = False       # mid TEXT_DELTA stream?
        self._text_streamed_this_turn: bool = False  # any TEXT_DELTA since TURN_START
        self._last_messages_sent: List[Dict[str, Any]] = []  # for /messages

        #: Getter for the agent's ContextUsage, or None for no gauge; set once the REPL has an agent.
        self.gauge: Optional[Callable[[], Any]] = None

    def on_event(self, e: AgentEvent) -> None:
        t = e.type

        if t == AgentEventType.TEXT_DELTA:
            self.activity.pause()
            if not self._streaming_text:
                sys.stdout.write(green("agent: "))
                self._streaming_text = True
            self._text_streamed_this_turn = True
            sys.stdout.write(e.data or "")
            sys.stdout.flush()
            return
        if t == AgentEventType.TEXT_DONE:
            # USAGE fires between TEXT_DELTA and TEXT_DONE, clearing the streaming line; fall back to the final text if nothing streamed.
            if not self._text_streamed_this_turn and e.data:
                self.activity.pause()
                print(green("agent: ") + str(e.data))
            else:
                self._end_text_line()
            self.activity.update("waiting", reset_clock=True)
            return

        if t == AgentEventType.THINKING:
            # Not printed: rendering every token risks blocking the model's stream thread; note() avoids update()'s spinner lock.
            self.activity.note("thinking")
            return

        if t == AgentEventType.MESSAGES_SENT:
            # Always cache so /messages can dump it on demand.
            msgs = e.metadata.get("messages") or []
            self._last_messages_sent = [
                _serialise_message_for_cli(m) for m in msgs
            ]
            if self.show_messages:
                self.activity.pause()
                self._end_streaming_lines()
                print(magenta(f"  -- messages sent (turn {e.metadata.get('turn')}) --"))
                for i, m in enumerate(self._last_messages_sent):
                    self.render_message(i, m)
                self._render_window(e.metadata.get("window"))
                print(magenta(f"  -- /messages ({len(self._last_messages_sent)} msgs) --"))
                self.activity.update("calling model", reset_clock=False)
            return

        if t == AgentEventType.TURN_START:
            self.activity.pause()
            self._end_streaming_lines()
            self._text_streamed_this_turn = False
            n = e.metadata.get("turn", "?")
            print(dim(f"  -- turn {n} --"))
            self.activity.update(f"calling model (turn {n})")
            return

        if t == AgentEventType.TOOL_START:
            self.activity.pause()
            self._end_streaming_lines()
            args = e.metadata.get("arguments", {})
            print(cyan(f"  → {e.data}{dim('(' + self._fmt_args(args) + ')')}"))
            self.activity.update(f"running {e.data}")
            return

        if t == AgentEventType.TOOL_RESULT:
            self.activity.pause()
            self._end_streaming_lines()
            body = e.data or ""
            if self.truncate and len(body) > 200:
                body = body[:197] + "..."
            print(dim(self._indent_pretty(body, prefix="    ")))
            self.activity.update("calling model")
            return

        if t == AgentEventType.HISTORY_COMPACTING:
            self.activity.pause()
            self._end_streaming_lines()
            n = e.metadata.get("messages_before")
            label = f"compacting history ({n} messages)" if n else "compacting history"
            print(dim(f"  -- {label} --"))
            self.activity.update("compacting history")
            return

        if t == AgentEventType.USAGE:
            self.activity.pause()
            self._end_streaming_lines()
            md = e.metadata
            # Reasoning tokens never reach the transcript, so a turn can exhaust its budget with nothing to show; print only when non-zero.
            reasoning = md.get("reasoning_tokens") or 0
            # Window comes last: `in` is this call's cost, window shows how close the next call is to not fitting.
            print(dim(
                f"    [tokens] in={md.get('prompt_tokens')} "
                f"out={md.get('completion_tokens')} "
                + (f"reasoning={reasoning} " if reasoning else "")
                + f"total={md.get('total_tokens')} "
                f"cache_read={md.get('cached_tokens')} "
                f"cache_write={md.get('cache_write_tokens')}"
                f"{self.context_gauge(prefix='   window ')}"
            ))
            self.activity.update("processing", reset_clock=False)
            return

        if t == AgentEventType.ERROR:
            self.activity.pause()
            self._end_streaming_lines()
            print(red(f"  ! {e.data}"))
            return

        if t == AgentEventType.ABORTED:
            self.activity.pause()
            self._end_streaming_lines()
            print(yellow(f"  ⏹ {e.data}"))
            return

        if t == AgentEventType.DONE:
            self._end_streaming_lines()
            self.activity.stop()
            return

        if self.verbose:
            self.activity.pause()
            self._end_streaming_lines()
            print(dim(f"  · {t.value}  {e.metadata or ''}"))

    def context_gauge(self, *, prefix: str = "", suffix: str = "") -> str:
        """Render the context gauge, e.g. ``~4.2k / 32k (13%)``.

        Never raises and never returns a partial line: this can print mid
        ``input()``. Rendering failures therefore return an empty gauge.
        Returns "" when no gauge is available.
        """
        if self.gauge is None:
            return ""
        try:
            usage = self.gauge()
        except Exception:
            return ""
        if usage is None or getattr(usage, "window", 0) <= 0:
            return ""
        return f"{prefix}{usage}{suffix}"

    def flush_streams(self) -> None:
        """Terminate any in-progress streamed text line."""
        self._end_streaming_lines()

    def _end_text_line(self) -> None:
        if self._streaming_text:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._streaming_text = False

    def _end_streaming_lines(self) -> None:
        self._end_text_line()

    def _fmt_args(self, args: Dict[str, Any]) -> str:
        """Render tool args as one-line JSON, shortened when ``/truncate`` is on."""
        s = json.dumps(args, default=str, ensure_ascii=False)
        if self.truncate and len(s) > 100:
            return s[:97] + "..."
        return s

    @staticmethod
    def _indent_pretty(body: str, *, prefix: str = "    ") -> str:
        """Pretty-print JSON if possible; otherwise indent the raw body."""
        body = body.strip()
        if body.startswith(("{", "[")):
            try:
                obj = json.loads(body)
                body = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
            except (json.JSONDecodeError, ValueError):
                pass
        return "\n".join(prefix + line for line in body.splitlines())

    def _render_window(self, window: Optional[Dict[str, Any]]) -> None:
        """Print a breakdown of where the characters in this request went, by role and tool."""
        if not window:
            return

        total = window.get("total_chars") or 0
        if total <= 0:
            return

        def pct(n: int) -> str:
            return f"{n / total:5.1%}"

        print(magenta(
            f"  -- window: {total:,} chars ≈ {window.get('est_tokens', 0):,} tokens --"
        ))
        for role, size in sorted(
            (window.get("by_role") or {}).items(), key=lambda kv: kv[1], reverse=True
        ):
            print(dim(f"       {role:<12} {size:>9,}  {pct(size)}"))

        schema = window.get("tool_schema_chars") or 0
        if schema:
            print(dim(f"       {'tool schema':<12} {schema:>9,}  {pct(schema)}"))

        by_tool = window.get("by_tool") or {}
        if by_tool:
            print(dim("       by tool:"))
            for name, size in list(by_tool.items())[:5]:
                print(dim(f"         {name:<18} {size:>9,}  {pct(size)}"))

    def render_message(self, i: int, m: Dict[str, Any]) -> None:
        """Pretty-print one provider-bound message for /messages output."""
        role = m.get("role", "?")
        label = ROLE_LABELS.get(role, role)
        head = f"  [{i:>2}] {label}"
        tail = ""
        if m.get("name"):
            tail += dim(f"  name={m['name']}")
        if m.get("tool_call_id"):
            tail += dim(f"  tool_call_id={m['tool_call_id']}")
        body = m.get("content") or ""
        if self.truncate and len(body) > 400:
            body = body[:397] + "..."
        # Indent body so multi-line content lines up under the label.
        if body:
            body_lines = body.splitlines()
            body_pretty = "\n".join("       " + line for line in body_lines)
            print(f"{head}{tail}\n{dim(body_pretty)}")
        else:
            print(f"{head}{tail}  {dim('(empty)')}")
        for tc in m.get("tool_calls", []) or []:
            args_str = tc.get("arguments")
            if isinstance(args_str, dict):
                args_str = json.dumps(args_str, default=str, ensure_ascii=False)
            print(cyan(f"       → tool_call {tc.get('name')}({args_str or ''})  ") + dim(f"id={tc.get('id')}"))

    def render_messages(self) -> None:
        """Dump the last message array sent to the LLM (``/messages``)."""
        if not self._last_messages_sent:
            print(dim("  no messages yet: send something first"))
            return
        print()
        print(magenta(f"  -- last messages sent ({len(self._last_messages_sent)} msgs) --"))
        for i, m in enumerate(self._last_messages_sent):
            self.render_message(i, m)
        print()

    def render_history(self, hist: List[Any]) -> None:
        """One line per history message (``/history``)."""
        if not hist:
            print(dim("  empty"))
            return
        print()
        for i, m in enumerate(hist):
            label = ROLE_LABELS.get(m.role, m.role)
            body = (m.content or "")[:160].replace("\n", " ")
            tag = ""
            if getattr(m, "tool_calls", None):
                tag = dim(f" [{len(m.tool_calls)} tool_call(s)]")
            elif getattr(m, "name", None):
                tag = dim(f" [{m.name}]")
            print(f"  {i:>2}. {label}{tag}  {dim(body)}")
        print()
