"""The interactive REPL: read-eval loop, interactive callbacks, shutdown."""

from __future__ import annotations

import json
import time
from typing import Optional

from lamssi_agents.approval import ApprovalRequest, ToolApproval, ToolApprovalResult
from lamssi_agents.agent import Agent
from lamssi_agents.interaction import (
    InteractionDecision,
    InteractionKind,
    InteractionRequest,
    InteractionResponse,
)
from lamssi_cli import commands
from lamssi_cli.ansi import bold, dim, magenta, readline_safe, yellow
from lamssi_cli.boot import BootedSession
from lamssi_cli.renderer import Renderer
from lamssi_cli.spinner import _Activity

try:                                    # POSIX line editing; absent on stock Windows
    import readline  # noqa: F401
    _READLINE = True
except ImportError:
    _READLINE = False


class AgentREPL:
    def __init__(self, renderer: Renderer, activity: _Activity) -> None:
        self.renderer = renderer
        self._activity = activity
        self.session: Optional[BootedSession] = None

    def attach(self, session: BootedSession) -> None:
        """Bind the session returned by :func:`boot`."""
        self.session = session
        # The renderer is built before the agent, so it reads context usage lazily through a callable gauge.
        self.renderer.gauge = lambda: self.agent.context_usage if self.agent else None

    @property
    def agent(self) -> Optional[Agent]:
        return self.session.agent if self.session is not None else None

    def interact(self, request: InteractionRequest) -> InteractionResponse:
        self._activity.pause()
        self.renderer.flush_streams()
        print(magenta(f"  {request.kind.value}: {request.prompt}"))
        try:
            reply = input(magenta("    your reply › ")).strip()
        except (EOFError, KeyboardInterrupt):
            reply = ""
        self._activity.update("calling model", reset_clock=True)
        if request.kind is InteractionKind.QUESTION:
            return InteractionResponse.answered(reply)
        allowed = reply.lower() in {"y", "yes", "continue", "go", "ok", "allow"}
        return InteractionResponse(
            decision=(
                InteractionDecision.CONTINUE if allowed
                else InteractionDecision.CANCEL
            )
        )

    def approve_tool(self, request: ApprovalRequest):
        name = request.tool
        args = dict(request.arguments)
        self._activity.pause()
        self.renderer.flush_streams()
        print(yellow(f"  approval needed: {name}  {dim(json.dumps(args, default=str)[:200])}"))
        # Loop until a clear answer or interrupt: silent rejects confuse user and model.
        while True:
            try:
                choice = input(yellow("    [a]pprove / [r]eject / a[b]ort › ")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return ToolApprovalResult(ToolApproval.ABORT, reason="user interrupted")
            if choice.startswith("a"):
                self._activity.update(f"running {name}", reset_clock=True)
                return ToolApprovalResult(ToolApproval.APPROVE)
            if choice.startswith("r"):
                return ToolApprovalResult(ToolApproval.REJECT, reason="user rejected")
            if choice.startswith("b"):
                return ToolApprovalResult(ToolApproval.ABORT)
            print(dim("    please type a, r, or b"))

    def _prompt(self) -> str:
        """Build the prompt string, e.g. ``~4.2k / 32k (13%) you › ``.

        Drawn fresh on every read so the context gauge stays current.
        """
        prompt = f"{dim(self.renderer.context_gauge(suffix='  '))}{bold('you › ')}"
        return readline_safe(prompt) if _READLINE else prompt

    def run(self) -> int:
        while True:
            try:
                line = input(self._prompt())
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                if not commands.dispatch(self, line):
                    return 0
                continue

            self._activity.start("calling model (Ctrl+C to abort)")
            try:
                self.agent.chat(line)
            except KeyboardInterrupt:
                # First Ctrl+C aborts cleanly; spinner keeps spinning while the in-flight call unwinds.
                self._activity.update("aborting (Ctrl+C again to force)")
                self.agent.abort()
                try:
                    # A second Ctrl+C raises here and bails to the REPL.
                    time.sleep(2.0)
                except KeyboardInterrupt:
                    pass
            finally:
                self._activity.stop()

    def shutdown(self) -> None:
        """Release whatever the boot acquired. Safe to call more than once."""
        if self.session is not None:
            self.session.shutdown()
