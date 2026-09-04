"""The ``Code`` feature: ``execute_code``, run by an executor the host supplies."""

from __future__ import annotations

import io
import logging
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

from lamssi_agents.features.base import Feature
from lamssi_tools import CapabilityContext, Expose, Str, err, tool

if TYPE_CHECKING:
    from lamssi_agents.agent.base import Agent

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CodeResult:
    """Outcome of executing a code snippet."""

    ok: bool = True
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    failed_at: str = ""
    failed_command: str = ""


@runtime_checkable
class CodeExecutor(Protocol):
    """A host sandbox used by the :class:`Code` feature."""

    def run(self, source: str) -> CodeResult:
        """Execute source synchronously and return its result."""
        ...

    def variables(self) -> Mapping[str, Any]:
        """Return the current sandbox namespace."""
        ...


#: Serializes the process-global stdout/stderr redirect so two concurrent execute_code calls don't capture into each other's buffer; it can't isolate an unrelated thread's print.
_STREAM_LOCK = threading.Lock()


@tool(
    group="code",
    dispatch="worker",
    inject_context=True,
    expose=Expose.ALL,
    description=(
        "Use this to work something out rather than reasoning it out in your head: "
        "arithmetic, counting, parsing, filtering a list, checking a hypothesis "
        "against real data. Anything you would otherwise estimate, compute here "
        "instead: the answer is then a fact rather than a guess. "
        "Runs Python in the host's sandbox and returns what it printed, anything on "
        "stderr, and any variables it defined or changed. Code that neither prints "
        "nor assigns produces no observable result, so print what you want back."
    ),
    approval="always",
    # Without this the tool would be advertised with no possible answer but "not configured".
    requires=CodeExecutor,
    keywords="python calculate compute arithmetic evaluate script math sum average count",
    parameters={"code": Str("Python source to execute.")},
)
def execute_code(
    ctx: CapabilityContext,
    code: str = "",
    **kw,
) -> Dict:
    """Run *code* and report stdout, stderr and the variables it touched."""
    if not code:
        return err("No code provided", retriable=False)

    # Unlike the shell tools' scoped child process, this runs in-process: containment is whatever the host's CodeExecutor enforces, backed by approval.
    executor = ctx.get(CodeExecutor)
    if executor is None:
        return err(
            "No code executor is configured for this host.",
            retriable=False,
            hint="Use the file tools, or ask the user to run the code themselves.",
        )

    before = dict(_variables(executor))

    # Captured around the call, not left to the executor, so a sandbox that prints directly still has its output reported.
    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        with _STREAM_LOCK, redirect_stdout(out_buf), redirect_stderr(err_buf):
            result = executor.run(code.strip())
    except Exception as exc:
        return err(
            f"{type(exc).__name__}: {exc}",
            retriable=False,
            stdout=out_buf.getvalue(),
            stderr=err_buf.getvalue(),
        )

    stdout = out_buf.getvalue() + (getattr(result, "stdout", "") or "")
    stderr = err_buf.getvalue() + (getattr(result, "stderr", "") or "")

    if not getattr(result, "ok", True):
        return err(
            getattr(result, "error", "") or "Execution failed.",
            retriable=False,
            hint=(
                "Read failed_command and stderr, and fix the statement that failed. "
                "Do not resubmit the same code."
            ),
            failed_at=getattr(result, "failed_at", ""),
            failed_command=getattr(result, "failed_command", ""),
            stdout=stdout,
            stderr=stderr,
        )

    after = dict(_variables(executor))
    changed = {
        k: _summarize_value(v)
        for k, v in after.items()
        if k not in before or before.get(k) is not v
    }

    out: Dict[str, Any] = {"executed": True}
    if stdout:
        out["stdout"] = stdout
    if stderr:
        out["stderr"] = stderr
    if changed:
        out["variables"] = changed
    if not stdout and not stderr and not changed:
        out["note"] = (
            "The code ran but produced no output and set no variables. Add a print, "
            "or assign a result, to see anything."
        )
    return out


def _variables(executor: Any) -> Dict[str, Any]:
    try:
        return dict(executor.variables() or {})
    except Exception as exc:
        log.debug("code executor variables() raised: %s", exc)
        return {}


def _summarize_value(v: Any) -> Any:
    """Compact, JSON-safe description of a variable; large values are described rather than serialized."""
    try:
        import numpy as _np

        if isinstance(v, _np.ndarray):
            return {
                "type": "ndarray",
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "min": float(_np.nanmin(v)) if v.size else None,
                "max": float(_np.nanmax(v)) if v.size else None,
            }
    except Exception:
        pass

    if isinstance(v, (int, float, bool, type(None))):
        return v
    if isinstance(v, str):
        return v if len(v) <= 200 else v[:200] + f"… ({len(v)} chars)"
    if isinstance(v, (list, tuple, set)):
        n = len(v)
        head = [_summarize_value(x) for x in list(v)[:5]]
        return head if n <= 5 else {"type": type(v).__name__, "len": n, "head": head}
    if isinstance(v, dict):
        n = len(v)
        head = {str(k): _summarize_value(x) for k, x in list(v.items())[:5]}
        return head if n <= 5 else {"type": "dict", "len": n, "head": head}
    return f"<{type(v).__name__}>"


class Code(Feature):
    """Install ``execute_code``, backed by a :class:`CodeExecutor` the host supplies.

    The kernel has no interpreter of its own: pass one here, or register it with
    ``agent.provide(CodeExecutor, ...)``. Until something does, ``requires=`` keeps
    the tool out of the model's schema rather than advertising one whose only
    possible answer is "not configured".

    Args:
        executor: Application implementation of :class:`CodeExecutor`. Omit it
            when the capability is registered separately with ``agent.provide``.

    Note:
        This feature contributes a gated execution tool; it does not provide a
        Python interpreter or bypass approval.
    """

    name = "code"

    def __init__(self, executor: Optional[Any] = None) -> None:
        self.executor = executor

    def install(self, agent: "Agent") -> None:
        if self.executor is not None:
            agent.provide(CodeExecutor, self.executor)
        agent.add_tools(execute_code)


__all__ = ["Code", "CodeExecutor", "CodeResult", "execute_code"]
