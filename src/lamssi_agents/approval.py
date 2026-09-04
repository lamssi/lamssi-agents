"""Application-owned policy and request values for tool approval."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
import logging
from typing import Any, Optional, get_args

from lamssi_tools.models import ApprovalName

log = logging.getLogger(__name__)

_MISSING = object()
DECLARATIONS = get_args(ApprovalName)

class ToolApproval(str, Enum):
    """Explicit decisions an application may return for a gated tool call.

    Attributes:
        APPROVE: Execute this call.
        REJECT: Refuse this call and continue the conversation.
        ABORT: Refuse this call and cancel the complete run tree.
    """

    APPROVE = "approve"
    REJECT = "reject"
    ABORT = "abort"


@dataclass(frozen=True, slots=True)
class ToolApprovalResult:
    """Approval decision with an optional explanation.

    Attributes:
        decision: Explicit approve, reject, or abort decision.
        reason: Optional text included in the tool result so the model knows why
            the application refused or stopped the call.
    """

    decision: ToolApproval
    reason: str | None = None


class _PolicyMode(str, Enum):
    FOLLOW_TOOL_RULES = "follow-tool-rules"
    ASK_ALL = "ask-all"
    ALLOW_ALL = "allow-all"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Immutable snapshot of one tool call awaiting application consent.

    Attributes:
        tool: Registered tool name.
        arguments: Deeply isolated, read-only normalized arguments.
        reason: Explanation of why this call reached the approval handler.
        declaration: Tool declaration: ``never``, ``conditional``, or ``always``.
    """

    tool: str
    arguments: Mapping[str, Any]
    reason: str
    declaration: str

    @classmethod
    def create(
        cls,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        reason: str,
        declaration: str,
    ) -> ApprovalRequest:
        """Create an isolated request safe to pass across an application boundary.

        Args:
            tool: Registered tool name.
            arguments: Validated arguments to snapshot.
            reason: Explanation shown to the application.
            declaration: Effective tool approval declaration.

        Returns:
            Frozen request whose arguments cannot mutate the pending invocation.
        """
        return cls(
            tool=tool,
            arguments=MappingProxyType(deepcopy(dict(arguments or {}))),
            reason=reason,
            declaration=declaration,
        )


ApprovalHandler = Callable[[ApprovalRequest], ToolApproval | ToolApprovalResult]


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    """How an application handles calls that require consent.

    Use a named constructor; the internal mode values are kept out of the
    public API so a bare ``none`` can't be read as either "ask for none" or
    "approve none".

    Example:
        Ask a GUI only for calls declared risky::

            def approve(request):
                return show_approval_dialog(request)

            policy = ApprovalPolicy.ask_when_required(approve)

    Note:
        The default agent policy is :meth:`reject_when_required`. Approval is
        separate from feature gates and domain safety; :meth:`allow_all` skips
        only this consent layer.
    """

    _mode: _PolicyMode
    handler: ApprovalHandler | None = None

    @classmethod
    def reject_when_required(cls) -> ApprovalPolicy:
        """Build the unattended fail-closed policy.

        Returns:
            Policy that runs declared-safe calls and rejects gated calls without
            invoking a handler.
        """
        return cls(_PolicyMode.FOLLOW_TOOL_RULES)

    @classmethod
    def ask_when_required(cls, handler: ApprovalHandler) -> ApprovalPolicy:
        """Build a policy that follows each tool's approval declaration.

        Args:
            handler: Synchronous application callback returning an explicit
                :class:`ToolApproval` or :class:`ToolApprovalResult`.

        Returns:
            Policy that asks only for calls not declared safe.
        """
        return cls(_PolicyMode.FOLLOW_TOOL_RULES, handler)

    @classmethod
    def ask_for_everything(cls, handler: ApprovalHandler) -> ApprovalPolicy:
        """Build a policy that asks before every tool call.

        Args:
            handler: Synchronous application approval callback.

        Returns:
            Policy that ignores safe-call declarations and always asks.
        """
        return cls(_PolicyMode.ASK_ALL, handler)

    @classmethod
    def allow_all(cls) -> ApprovalPolicy:
        """Build a policy that skips agent-level consent.

        Returns:
            Permissive approval policy. Tool scope, feature gates, loop
            guards, and application safety mechanisms still apply.
        """
        return cls(_PolicyMode.ALLOW_ALL)

    @property
    def asks_for_everything(self) -> bool:
        """Whether this policy sends every call to its handler."""
        return self._mode is _PolicyMode.ASK_ALL

    @property
    def allows_all(self) -> bool:
        """Whether this policy skips the agent-level consent gate."""
        return self._mode is _PolicyMode.ALLOW_ALL

    @property
    def name(self) -> str:
        """Stable display name for logs and status UIs."""
        if self._mode is _PolicyMode.FOLLOW_TOOL_RULES:
            return (
                "ask-when-required"
                if self.handler is not None
                else "reject-when-required"
            )
        return self._mode.value


def _value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (list, tuple, set, frozenset)):
        return any(_value_matches(option, actual) for option in expected)
    if isinstance(expected, bool) or isinstance(actual, bool):
        return (
            isinstance(expected, bool)
            and isinstance(actual, bool)
            and expected is actual
        )
    if expected == actual:
        return True
    if isinstance(expected, (str, int, float)) and isinstance(
        actual, (str, int, float)
    ):
        return str(expected) == str(actual)
    return False


def pattern_check(
    pattern: Mapping[str, Any],
) -> Callable[[Mapping[str, Any]], bool]:
    """Compile a tool's declarative safe-argument pattern."""
    required = dict(pattern or {})

    def check(arguments: Mapping[str, Any]) -> bool:
        if not required:
            return False
        for key, expected in required.items():
            actual = (arguments or {}).get(key, _MISSING)
            if actual is _MISSING or not _value_matches(expected, actual):
                return False
        return True

    check.__name__ = "safe_when_check"
    check.__qualname__ = check.__name__
    return check


def declared_safe(definition: Any, arguments: Mapping[str, Any]) -> bool:
    pattern = getattr(definition, "safe_when", None) if definition is not None else None
    if not pattern:
        return False
    try:
        return pattern_check(pattern)(arguments or {})
    except Exception as exc:
        log.debug("safe_when pattern %r could not be evaluated: %s", pattern, exc)
        return False


def declared_approval(definition: Any) -> str:
    """Read a declaration, failing closed for absent or invalid values."""
    if definition is None:
        return "always"
    value = getattr(definition, "approval", "always")
    if value in DECLARATIONS:
        return str(value)
    log.debug("unrecognised approval value %r; treating as 'always'", value)
    return "always"


def is_safe(
    tool: str,
    arguments: Mapping[str, Any],
    definition: Any = None,
    *,
    rules: Optional[Mapping[str, Any]] = None,
) -> bool:
    check = (rules or {}).get(tool)
    if check is None:
        return declared_safe(definition, arguments)
    try:
        return bool(check(arguments or {}))
    except Exception as exc:
        log.warning(
            "safe_when check for '%s' raised, requiring approval: %s",
            tool,
            exc,
            exc_info=True,
        )
        return False


def needs_approval(
    policy: ApprovalPolicy,
    tool: str,
    arguments: Mapping[str, Any],
    definition: Any = None,
    *,
    rules: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Resolve application policy and the tool declaration in one place."""
    if policy.asks_for_everything:
        return True
    if policy.allows_all:
        return False
    declaration = declared_approval(definition)
    if declaration == "never":
        return False
    if declaration == "conditional":
        return not is_safe(tool, arguments, definition, rules=rules)
    return True


def approval_request(
    policy: ApprovalPolicy,
    tool: str,
    arguments: Mapping[str, Any],
    definition: Any = None,
) -> ApprovalRequest:
    declaration = declared_approval(definition)
    if policy.asks_for_everything:
        reason = "the application requires approval for every tool call"
    elif declaration == "conditional":
        reason = "the call does not match the tool's declared safe conditions"
    else:
        reason = "the tool declares that approval is always required"
    return ApprovalRequest.create(
        tool,
        arguments,
        reason=reason,
        declaration=declaration,
    )


def normalize_approval(value: Any) -> tuple[ToolApproval, Optional[str]]:
    """Normalize an application response, accepting only explicit decisions."""
    reason: Optional[str] = None
    if isinstance(value, ToolApprovalResult):
        value, reason = value.decision, value.reason
    if isinstance(value, ToolApproval):
        return value, reason
    if isinstance(value, str):
        try:
            return ToolApproval(value), reason
        except ValueError:
            pass
    return ToolApproval.REJECT, (
        reason or f"The approval handler returned {value!r}, which is not a decision."
    )


def should_require_approval(
    agent: Any,
    tool: str,
    arguments: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Resolve approval against the Agent's actual visible tool definition."""
    return needs_approval(
        agent.approval,
        tool,
        arguments or {},
        agent._runtime.definition_for(tool),
        rules=agent.safe_when,
    )


def validate_safe_when(runtime: Any) -> None:
    """Check that every safety check can actually find what it reads.

    Fatal when a check names an argument its tool lacks. File checks
    fail closed when the value is missing, but silently prompting for every call
    is still a broken configuration and should be reported at composition time.

    Warns about the two wirings that quietly do nothing: a registered check
    shadowing a declared pattern, and a ``conditional`` tool with neither, which
    is gated exactly like ``always``.
    """
    definitions = {d.name: d for d in runtime._tools.list_tools()}
    problems = []

    for name, check in runtime.safe_when.items():
        definition = definitions.get(name)
        if definition is None:
            continue
        if getattr(definition, "safe_when", None):
            log.warning(
                "'%s' declares safe_when and has a registered check; the "
                "check wins and the declared pattern is never read",
                name,
            )
        key = getattr(check, "argument_key", None)
        if key is None:
            continue
        names = sorted(p.name for p in (definition.parameters or ()))
        if key not in names:
            problems.append(
                f"  {name}: check reads {key!r}, but its arguments are "
                f"{names or ['(none)']}"
            )

    for name, definition in definitions.items():
        if getattr(definition, "approval", "") != "conditional":
            continue
        if name in runtime.safe_when or getattr(definition, "safe_when", None):
            continue
        log.warning(
            "'%s' is declared conditional but has no safe_when pattern or "
            "check, so every call is gated",
            name,
        )

    if problems:
        # ASCII only: a Windows console (cp1252) raises UnicodeEncodeError on an
        # em dash: reporting a disabled gate must not itself crash.
        raise ValueError(
            "safe_when check(s) name an argument the tool does not have:\n"
            + "\n".join(problems)
            + "\n\nThe check cannot inspect the value it was configured for. "
            "Correct the argument name before running the agent."
        )


__all__ = [
    "ApprovalHandler",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ToolApproval",
    "ToolApprovalResult",
    "DECLARATIONS",
    "pattern_check",
    "declared_safe",
    "declared_approval",
    "is_safe",
    "needs_approval",
    "approval_request",
    "normalize_approval",
    "should_require_approval",
    "validate_safe_when",
]
