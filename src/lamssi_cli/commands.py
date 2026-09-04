"""Slash commands for the REPL as a ``{name: handler}`` dispatch table."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from lamssi_cli.ansi import bold, cyan, dim, green, red


def cmd_help(repl, rest: str) -> None:
    rows = [
        ("/help", "this list"),
        ("/tools", "show the tool surface this turn"),
        ("/prompt", "show the rendered system prompt"),
        ("/skills", "list available skills"),
        ("/load <name>", "load and pin a skill"),
        ("/unload <name>", "unload a skill"),
        ("/compact [focus]", "summarise the history now, keeping the recent tail"),
        ("/clear", "clear conversation history"),
        ("/history", "show conversation history"),
        ("/messages", "dump the last message array sent to the LLM"),
        ("/show-messages", "toggle printing full LLM payload every turn"),
        ("/usage", "cumulative token usage"),
        ("/model [name]", "show or switch model"),
        ("/approve [required|all|allow|reject]", "show or set what needs your say-so"),
        ("/verbose", "toggle verbose event stream"),
        ("/truncate", "toggle shortening long tool results on screen"),
        (
            "/log on [path] [debug]",
            "start a session log (debug: + prompts + thinking + tokens)",
        ),
        ("/log off", "stop logging"),
        ("/log show", "show log path + size"),
        ("/q", "quit"),
    ]
    print()
    for cmd, desc in rows:
        print(f"  {cyan(cmd.ljust(22))}{dim(desc)}")
    print()


def cmd_tools(repl, rest: str) -> None:
    tools = repl.agent.visible_tool_defs()
    full = repl.agent.all_tool_defs()
    runtime = _skill_runtime(repl)
    active_skills = runtime.active if runtime is not None else ()
    baseline = (
        "essentials + always-available"
        if not active_skills
        else f"skills: {', '.join(active_skills)}"
    )
    print()
    print(f"  {bold('Active surface')} ({baseline}): {len(tools)} tools:")
    for t in sorted(tools, key=lambda t: t.name):
        print(f"    {green('●')} {t.name}")
    hidden = len(full) - len(tools)
    if hidden > 0:
        print(
            dim(
                f"  {hidden} more agent-visible tools available (load a skill to widen)"
            )
        )
    print()


def cmd_prompt(repl, rest: str) -> None:
    prompt = repl.agent.build_system_prompt()
    print()
    print(dim("-" * 60))
    print(prompt)
    print(dim("-" * 60))
    print(dim(f"  {len(prompt)} chars, ~{len(prompt) // 4} tokens"))
    print()


def cmd_skills(repl, rest: str) -> None:
    runtime = _skill_runtime(repl)
    skills = runtime.list() if runtime is not None else []
    if not skills:
        print(dim("  no skills are installed"))
        return
    active = set(runtime.active)
    print()
    for s in skills:
        marker = green("● loaded") if s.name in active else dim("○")
        print(
            f"  {marker}  {bold(s.name)} {dim('(' + s.source + ')')}  {s.description}"
        )
    print()


def cmd_load(repl, rest: str) -> None:
    name = rest
    if not name:
        print(red("  usage: /load <skill_name>"))
        return
    runtime = _skill_runtime(repl)
    if runtime is None:
        print(red("  no skills are installed"))
        return
    result = runtime.load(name)
    if "error" in result:
        print(red(f"  {result['error']}"))
        if result.get("available"):
            print(dim(f"  available: {', '.join(result['available'])}"))
        return
    print(green(f"  ● loaded {result['name']}"))


def cmd_unload(repl, rest: str) -> None:
    name = rest
    runtime = _skill_runtime(repl)
    if runtime is not None and runtime.unload(name):
        print(green(f"  unloaded {name}"))
    else:
        print(dim(f"  {name} not loaded"))


def _skill_runtime(repl):
    """Return the optional Skills runtime without teaching Agent about Skills."""
    if repl.agent is None:
        return None
    from lamssi_agents.features.skills import SkillRuntime

    return repl.agent.get(SkillRuntime)


def cmd_clear(repl, rest: str) -> None:
    repl.agent.clear_history()
    print(dim("  history cleared"))


def cmd_usage(repl, rest: str) -> None:
    if repl.agent.model is None:
        print(dim("  no usage data yet"))
        return
    u = repl.agent.usage
    print()
    print(f"  {bold('Cumulative usage')}")
    print(f"    prompt      {u.prompt_tokens}")
    print(f"    completion  {u.completion_tokens}")
    print(f"    cache_read  {u.cached_tokens}")
    print(f"    cache_write {u.cache_write_tokens}")
    print(f"    total       {u.total_tokens}")
    print()


def cmd_model(repl, rest: str) -> None:
    name = rest
    if not name:
        print(f"  model: {repl.agent.model_id}  ({repl.agent.model_adapter_name})")
        return
    try:
        repl.agent.use_model(name)
        print(green(f"  ● switched to {name}  ({repl.agent.model_adapter_name})"))
    except Exception as exc:
        print(red(f"  failed: {exc}"))


def cmd_approve(repl, rest: str) -> None:
    from lamssi_agents import ApprovalPolicy

    if not rest:
        print(f"  approval: {repl.agent.approval.name}")
        return
    choice = rest.strip().lower()
    policies = {
        "required": lambda: ApprovalPolicy.ask_when_required(repl.approve_tool),
        "all": lambda: ApprovalPolicy.ask_for_everything(repl.approve_tool),
        "allow": ApprovalPolicy.allow_all,
        "reject": ApprovalPolicy.reject_when_required,
    }
    factory = policies.get(choice)
    if factory is None:
        print(red("  invalid policy: use required, all, allow or reject"))
        return
    repl.agent.approval = factory()
    print(green(f"  ● approval = {repl.agent.approval.name}"))


def cmd_verbose(repl, rest: str) -> None:
    repl.renderer.verbose = not repl.renderer.verbose
    print(dim(f"  verbose = {repl.renderer.verbose}"))


def cmd_compact(repl, rest: str) -> None:
    """``/compact [focus]``: summarise the history now, keeping the recent tail."""
    agent = repl.agent
    # The start line comes from the HISTORY_COMPACTING event, so manual, automatic, and reactive paths announce identically.
    try:
        result = agent.compact(focus=rest.strip() or None)
    except Exception as exc:
        print(red(f"  compaction failed: {exc}"))
        return
    if not result:
        print(dim(f"  nothing to compact: {result}"))
        return
    print(green(f"  ● {result}"))
    print(dim(f"    now {agent.context_usage}"))


def cmd_truncate(repl, rest: str) -> None:
    repl.renderer.truncate = not repl.renderer.truncate
    print(
        dim(
            f"  truncate = {repl.renderer.truncate}  (long tool results shortened on screen only)"
        )
    )


def cmd_show_messages(repl, rest: str) -> None:
    repl.renderer.show_messages = not repl.renderer.show_messages
    print(
        dim(
            f"  show-messages = {repl.renderer.show_messages}  (print full LLM payload each turn)"
        )
    )


def cmd_log(repl, rest: str) -> None:
    """``/log on [path] [debug]`` / ``/log off`` / ``/log show``."""
    parts = rest.split() if rest else []
    sub = parts[0].lower() if parts else "show"

    if sub == "off":
        repl.agent.set_conversation_log_dir(None)
        print(dim("  logging stopped"))
        return

    if sub == "show":
        clog = repl.agent.conversation_logger
        if clog is None:
            print(dim("  no log active"))
            return
        try:
            size = clog.path.stat().st_size
        except Exception:
            size = 0
        mode = "debug" if clog.debug else "standard"
        print(f"  {green('●')} {mode}  {clog.path}  {dim(f'({size:,} bytes)')}")
        return

    if sub == "on":
        path = None
        debug = False
        for tok in parts[1:]:
            if tok.lower() == "debug":
                debug = True
            else:
                path = tok
        if path is None:
            agent = repl.agent
            if agent is None:
                print(red("  no agent: pass a path: /log on <dir>"))
                return
            from lamssi_agents.features.files import FileSpace

            space = agent.get(FileSpace)
            root = space.workspace() if space is not None else Path.cwd()
            path = str(root / ".lamssi" / "conversations")
        try:
            repl.agent.set_conversation_log_dir(path, debug=debug)
            clog = repl.agent.conversation_logger
            mode = "debug" if debug else "standard"
            print(f"  {green('●')} {mode}  {clog.path}")
        except Exception as exc:
            print(red(f"  failed: {exc}"))
        return

    print(red(f"  unknown /log subcommand: {sub}"))
    print(dim("  available: on, off, show"))


def cmd_history(repl, rest: str) -> None:
    repl.renderer.render_history(repl.agent.history)


def cmd_messages(repl, rest: str) -> None:
    repl.renderer.render_messages()


def cmd_quit(repl, rest: str) -> bool:
    return False


# name / alias → handler
COMMANDS: Dict[str, Callable] = {
    "help": cmd_help,
    "h": cmd_help,
    "?": cmd_help,
    "tools": cmd_tools,
    "prompt": cmd_prompt,
    "skills": cmd_skills,
    "load": cmd_load,
    "unload": cmd_unload,
    "clear": cmd_clear,
    "usage": cmd_usage,
    "model": cmd_model,
    "approve": cmd_approve,
    "verbose": cmd_verbose,
    "compact": cmd_compact,
    "truncate": cmd_truncate,
    "log": cmd_log,
    "history": cmd_history,
    "messages": cmd_messages,
    "show-messages": cmd_show_messages,
    "trace": cmd_show_messages,
    "q": cmd_quit,
    "quit": cmd_quit,
    "exit": cmd_quit,
}


def dispatch(repl, line: str) -> bool:
    """Run a ``/command`` line. Return True to stay in the REPL, False to exit."""
    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    handler = COMMANDS.get(cmd)
    if handler is None:
        print(red(f"  unknown command: /{cmd}"))
        print(dim("  type /help"))
        return True
    return handler(repl, rest) is not False
