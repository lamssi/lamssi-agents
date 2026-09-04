"""Boot a host and return its agent."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from lamssi_agents.approval import ApprovalHandler, ApprovalPolicy
from lamssi_agents.agent import Agent
from lamssi_agents.events import (
    AgentEventCallback,
)
from lamssi_agents.interaction import InteractionHandler
from lamssi_agents.model import LiteLLMModel
from lamssi_cli.hosts import HostBootstrap, load_host
from lamssi_agents.runtime import AgentConfig
from lamssi_cli.ansi import dim, green, red, yellow

log = logging.getLogger(__name__)


@dataclass
class BootedSession:
    """What a successful boot produced."""

    agent: Agent
    host: HostBootstrap

    def shutdown(self) -> None:
        try:
            self.agent.set_conversation_log_dir(None)
        except Exception as exc:
            log.debug("closing the conversation log raised: %s", exc)
        try:
            self.host.shutdown()
        except Exception as exc:
            log.debug("host shutdown raised: %s", exc)


def boot(
    *,
    host: Optional[str] = None,
    workspace: Optional[str] = None,
    model: Optional[str] = None,
    server_url: Optional[str] = None,
    max_turns: Optional[int] = None,
    token_checkpoint: Optional[int] = None,
    log_dir: Optional[str] = None,
    debug_log: bool = False,
    verbose: bool = False,
    on_event: Optional[AgentEventCallback] = None,
    interaction: Optional[InteractionHandler] = None,
    on_tool_approval: Optional[ApprovalHandler] = None,
) -> Optional[BootedSession]:
    """Resolve a host, create its agent, and attach callbacks.

    Returns:
        ``None`` after printing a reason, if any step fails.
    """
    t0 = time.perf_counter()
    print()

    try:
        bootstrap = load_host(host)
    except Exception as exc:
        print(red(f"  {exc}"))
        return None

    try:
        root = bootstrap.resolve_root(workspace)
    except Exception as exc:
        print(red(f"  {exc}"))
        return None

    # Environment first, then the flags, so a value passed explicitly on the command
    # line wins over one exported in a shell profile.
    config = AgentConfig.from_env().merged(max_turns=max_turns)

    try:
        agent = bootstrap.create_agent(root=root, config=config)
    except Exception as exc:
        print(red(f"  could not build the agent: {exc}"))
        if verbose:
            log.exception("create_agent failed")
        return None

    from lamssi_agents.features import Budget

    checkpoint = token_checkpoint
    if checkpoint is None:
        try:
            checkpoint = int(os.environ.get("LAMSSI_TOKEN_CHECKPOINT", "200000"))
        except ValueError:
            checkpoint = 200_000
    if checkpoint > 0:
        agent.use(Budget(checkpoint))

    print(f"  {dim('host')}         {green('✓')}  {bootstrap.name}")
    for label, detail in bootstrap.status_lines():
        print(f"  {dim(label):<21}{green('✓')}  {detail}")

    try:
        bootstrap.wait_ready(5.0)
    except Exception as exc:
        print(yellow(f"  host not fully ready: {exc}"))

    n_tools = len(agent.available_tool_names())
    print(f"  {dim('tools')}        {green('✓')}  {n_tools} available")

    # The CLI has one explicit model input: the flag, then LAMSSI_MODEL.
    resolved_model = (model or os.environ.get("LAMSSI_MODEL", "")).strip()
    if not resolved_model:
        print(red("  no model configured. Pass --model, or set one in the host's settings."))
        return None

    try:
        if server_url:
            base = server_url.rstrip("/")
            if not base.endswith("/v1"):
                base += "/v1"
            value = LiteLLMModel(
                f"openai/{resolved_model}",
                api_base=base,
                api_key="local",
            )
        else:
            value = resolved_model
        agent.use_model(value)
    except Exception as exc:
        print(red(f"  could not load model {resolved_model!r}: {exc}"))
        return None

    if on_event is not None:
        agent.add_event_listener(on_event)
    if interaction is not None:
        agent.interaction = interaction
    if on_tool_approval is not None:
        agent.approval = ApprovalPolicy.ask_when_required(on_tool_approval)

    print(
        f"  {dim('agent')}        {green('✓')}  "
        f"{agent.model_adapter_name}  {dim(resolved_model)}"
    )
    print(
        f"  {dim('approve')}      {green('✓')}  {agent.approval.name}  "
        f"{dim('(/approve to change)')}"
    )

    if log_dir:
        try:
            agent.set_conversation_log_dir(log_dir, debug=debug_log)
            mode = "debug" if debug_log else "standard"
            print(f"  {dim('log')}          {green('✓')}  {mode}  {dim(str(agent.conversation_logger.path))}")
        except Exception as exc:
            print(yellow(f"  log          ! could not open: {exc}"))

    print()
    print(dim(f"  ready in {time.perf_counter() - t0:.1f}s. Type /help (Ctrl+C aborts a turn)."))
    print()
    return BootedSession(agent=agent, host=bootstrap)
__all__ = ["boot", "BootedSession"]
