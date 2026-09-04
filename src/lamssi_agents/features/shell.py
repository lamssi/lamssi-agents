"""The ``Shell`` feature: one command tool, for the shell this machine actually has."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Dict, List, Optional

from lamssi_agents.features.base import Feature
from lamssi_agents.redaction import safe_environ
from lamssi_tools import CapabilityContext, Expose, Int, Str, err, tool

if TYPE_CHECKING:
    from lamssi_agents.agent.base import Agent

log = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 8000  # per-stream truncation cap

#: Shared by both tools, since only the argv and the label differ between them.
_COMMON = dict(
    group="execute",
    # 12k chars covers a build-log tail (the pathological case) and keeps the end,
    # where the exit status and any error live.
    truncation=12_000,
    truncation_side="tail",
    expose=Expose.AGENT,
    inject_context=True,
    approval="always",
)


@lru_cache(maxsize=1)
def bash_binary() -> Optional[str]:
    """Path to a bash that understands this machine's paths, or ``None``.

    Resolved on first use, not at import, so importing still does nothing. On
    Windows, PATH's ``bash`` is WSL's launcher, which runs in a different filesystem
    namespace: every absolute path the model was given would be silently wrong there.
    Git for Windows' bash is the one that matches, so look beside ``git`` first and
    refuse anything under System32.
    """
    if not sys.platform.startswith("win"):
        return which("bash")

    git = which("git")
    if git:
        # ...\Git\cmd\git.exe -> ...\Git\bin\bash.exe
        candidate = Path(git).parent.parent / "bin" / "bash.exe"
        if candidate.is_file() and _windows_bash_starts(str(candidate)):
            return str(candidate)

    found = which("bash")
    if (
        found
        and Path(found).parent.name.lower() != "system32"
        and _windows_bash_starts(found)
    ):
        return found
    return None


def _windows_bash_starts(binary: str) -> bool:
    """Whether a Windows bash executable can start successfully."""
    try:
        probe = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            timeout=5,
            check=False,
            env=safe_environ(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def detect_shell() -> str:
    """``"bash"`` when this machine has a usable one, else ``"powershell"``."""
    if bash_binary():
        return "bash"
    return "powershell" if sys.platform.startswith("win") else "bash"


def _truncate(text: str, cap: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= cap:
        return text
    head = text[: cap // 2]
    tail = text[-cap // 2 :]
    dropped = len(text) - len(head) - len(tail)
    return f"{head}\n... [{dropped} chars truncated] ...\n{tail}"


def _resolve_cwd(ctx: CapabilityContext, cwd: str) -> Optional[Path]:
    """Where to run the command. Relative paths are taken against the project root."""
    from lamssi_agents.features.files import FileSpace

    space = ctx.get(FileSpace)
    root = space.workspace() if space is not None else Path.cwd()
    if cwd:
        p = Path(cwd).expanduser()
        if not p.is_absolute():
            p = root / p
        return p if p.is_dir() else None
    return root


def _execute(
    ctx: CapabilityContext,
    command: str,
    cwd: str,
    timeout: int,
    argv: List[str],
    label: str,
) -> Dict:
    """Run *argv* and shape the result. Shared by both shell tools."""
    workdir = _resolve_cwd(ctx, cwd)
    if workdir is None:
        return err(
            f"Working directory does not exist: {cwd}",
            retriable=False,
            cwd=cwd,
        )
    # Build a scrubbed child environment because the model controls the command.
    env = safe_environ()
    withheld = sorted(set(os.environ) - set(env))
    env["NO_COLOR"] = "1"  # suppress ANSI escapes for cleaner model parsing

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            # Decode as UTF-8, not the locale default (cp1252 on Windows),
            # so non-ASCII command output doesn't crash the reader thread.
            encoding="utf-8",
            errors="replace",
            cwd=str(workdir) if workdir else None,
            env=env,
            timeout=max(1, int(timeout)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return err(
            f"timeout after {timeout}s",
            retriable=False,
            shell=label,
            command=command,
            stdout=_truncate(exc.stdout or ""),
            stderr=_truncate(exc.stderr or ""),
        )
    except FileNotFoundError as exc:
        return err(f"shell not found: {exc}", retriable=False, shell=label)

    out: Dict = {
        "shell": label,
        "command": command,
        "cwd": str(workdir) if workdir else os.getcwd(),
        "exit_code": proc.returncode,
        "stdout": _truncate(proc.stdout or ""),
        "stderr": _truncate(proc.stderr or ""),
    }
    if proc.returncode != 0:
        out["error"] = f"exit code {proc.returncode}"
        out["retriable"] = False
        if withheld:
            # Only on failure: an unexplained auth error otherwise invites the model
            # to hunt for the key and pass it on the command line: what scoping prevents.
            shown = ", ".join(withheld[:6])
            more = f" and {len(withheld) - 6} more" if len(withheld) > 6 else ""
            out["note"] = (
                f"This command ran without the host's credential environment "
                f"variables ({shown}{more}). If it failed to authenticate, that "
                f"is why: it is a deliberate restriction, not a misconfiguration. "
                f"Do not retry, and do not try to obtain the credential and pass "
                f"it yourself. Report it and let the user run the command."
            )
    return out


# Shell-specific tool declaration.
@tool(
    **_COMMON,
    parameters={
        "command": Str("Bash command. Run with -c, so no profile is sourced."),
        "cwd": Str(
            "Working directory. A relative path resolves against the project root; "
            "empty means the project root."
        ),
        "timeout": Int("Hard timeout in seconds (default 60)."),
    },
    truncation_hint=(
        "Re-run piping through a filter: grep for the error, or tail the last lines."
    ),
    description=(
        "Run a single bash command. Returns stdout, stderr and the exit code, each "
        "truncated. Always requires approval. Use it for build, install and "
        "version-control tasks; for Python logic prefer execute_code."
    ),
    keywords="command terminal bash sh git npm pip execute invoke",
)
def run_bash(
    ctx: CapabilityContext,
    command: str = "",
    cwd: str = "",
    timeout: int = 60,
) -> Dict:
    """Run *command* in bash. Approval is always required."""
    if not command or not command.strip():
        return err("No command provided", retriable=False)
    # `-c`, not `-lc`: a login shell sources the profile, which would re-export the
    # credentials `safe_environ` just stripped (PATH is unaffected either way).
    return _execute(
        ctx, command, cwd, timeout, [bash_binary() or "bash", "-c", command], "bash"
    )


@tool(
    **_COMMON,
    parameters={
        "command": Str("Windows PowerShell 5.1 command."),
        "cwd": Str(
            "Working directory. A relative path resolves against the project root; "
            "empty means the project root."
        ),
        "timeout": Int("Hard timeout in seconds (default 60)."),
    },
    truncation_hint=(
        "Re-run piping through a filter: Select-String for the error, or "
        "Select-Object -Last 40."
    ),
    description=(
        "Run a single Windows PowerShell 5.1 command. Returns stdout, stderr and the "
        "exit code, each truncated. Always requires approval. Use it for build, "
        "install and version-control tasks; for Python logic prefer execute_code. "
        "This is powershell.exe, not pwsh 7 and not bash: `&&` and `||` are parse "
        "errors, so chain with `a; b`, or `a; if ($?) { b }` to run b only when a "
        "succeeded. There is no head, tail, which or touch -- use `Select-Object "
        "-First N` / `-Last N`, `(Get-Command x).Source`, `New-Item`. Discard output "
        "with `2>$null`, not `2>/dev/null`. Set a variable with `$env:NAME = 'v'`, "
        "not `NAME=v cmd`. Never call Read-Host or another prompting cmdlet: stdin "
        "is closed and the call will block until the timeout."
    ),
    keywords="command terminal powershell pwsh git npm pip execute invoke",
)
def run_powershell(
    ctx: CapabilityContext,
    command: str = "",
    cwd: str = "",
    timeout: int = 60,
) -> Dict:
    """Run *command* in Windows PowerShell. Approval is always required."""
    if not command or not command.strip():
        return err("No command provided", retriable=False)
    argv = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        command,
    ]
    return _execute(ctx, command, cwd, timeout, argv, "powershell")


class Shell(Feature):
    """Install one shell tool, for the shell this machine has.

    Not installing this feature means no shell tool at all: the kernel ships none.

    Args:
        prefer: ``"bash"`` or ``"powershell"`` to require that shell. Empty
            selects a working installed shell automatically.

    Raises:
        ValueError: If ``prefer`` is not empty, ``bash``, or ``powershell``.
    """

    name = "shell"

    def __init__(self, prefer: str = "") -> None:
        if prefer and prefer not in ("bash", "powershell"):
            raise ValueError(f"prefer must be 'bash' or 'powershell', got {prefer!r}")
        self.prefer = prefer

    def install(self, agent: "Agent") -> None:
        chosen = self.prefer or detect_shell()
        agent.add_tools(run_bash if chosen == "bash" else run_powershell)
        log.info("shell feature installed: %s", chosen)


__all__ = ["Shell", "run_bash", "run_powershell", "bash_binary", "detect_shell"]
