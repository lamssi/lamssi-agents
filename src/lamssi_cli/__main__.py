"""Entry point: parse arguments, boot a host, run the REPL."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from lamssi_cli.boot import boot
from lamssi_cli.renderer import Renderer
from lamssi_cli.repl import AgentREPL
from lamssi_cli.spinner import _Activity


def _utf8_output() -> None:
    """Configure console output for UTF-8 where supported."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    _utf8_output()
    p = argparse.ArgumentParser(
        prog="lamssi-agent",
        description="A terminal agent over any host application, or none.",
    )
    p.add_argument(
        "--host",
        help=(
            "Host to boot: a registered name, an explicit 'module:attr' path, or "
            "'none' (the default) for the current directory with the core tools."
        ),
    )
    p.add_argument("--list-hosts", action="store_true", help="List the available hosts and exit.")
    p.add_argument("--workspace", help="Project root, if the host takes one.")
    p.add_argument("--model", help="Model name. Overrides the environment and the host's settings.")
    p.add_argument(
        "--server-url",
        help=(
            "Base URL of the model server, for a local model. Defaults to LM Studio's "
            "http://127.0.0.1:1234; Ollama is http://127.0.0.1:11434. Only needed for "
            "a non-default host or port: an 'ollama/…' model already knows its own."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Stream extra agent events and debug logs.")
    p.add_argument("--truncate", action="store_true",
                   help="Shorten long tool results and call arguments on screen.")
    p.add_argument("--max-turns", type=int, default=None,
                   help="Backstop on tool-calling turns per message.")
    p.add_argument("--token-checkpoint", type=int, default=None,
                   help="Ask whether to continue after every N tokens (0 disables).")
    p.add_argument("--log-dir", help="Write a conversation log to this directory.")
    p.add_argument("--debug-log", action="store_true",
                   help="Verbose log: system prompts, thinking, per-turn token deltas.")
    p.add_argument("--no-spinner", action="store_true",
                   help="Disable the activity spinner.")
    p.add_argument("--show-messages", action="store_true",
                   help="Print every message array sent to the model. Heavy.")
    args = p.parse_args(argv)

    if args.list_hosts:
        return _list_hosts()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    activity = _Activity(enabled=not args.no_spinner)
    renderer = Renderer(
        activity,
        verbose=args.verbose,
        truncate=args.truncate,
        show_messages=args.show_messages,
    )
    repl = AgentREPL(renderer, activity)

    session = boot(
        host=args.host,
        workspace=args.workspace,
        model=args.model,
        server_url=args.server_url,
        max_turns=args.max_turns,
        token_checkpoint=args.token_checkpoint,
        log_dir=args.log_dir,
        debug_log=args.debug_log,
        verbose=args.verbose,
        on_event=renderer.on_event,
        interaction=repl.interact,
        on_tool_approval=repl.approve_tool,
    )
    if session is None:
        return 2

    repl.attach(session)
    try:
        return repl.run()
    finally:
        repl.shutdown()


def _list_hosts() -> int:
    from lamssi_cli.hosts import available_hosts

    hosts = available_hosts()
    print("\nAvailable hosts:\n")
    for name, target in sorted(hosts.items()):
        note = "  (default: current directory, core tools)" if name == "none" else ""
        print(f"  {name:<16} {target}{note}")
    if len(hosts) == 1:
        print("\nOnly the null host is installed. A host registers itself by publishing")
        print("an entry point in the 'lamssi_agents.hosts' group.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
