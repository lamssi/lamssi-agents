"""Where a tool call happens: the surface, the gates, execution, and its result.

`ToolRuntime` owns the agent's tool behavior and receives the run's services
explicitly. One call:

    execute_calls
        -> _run_one_call
            -> reject_out_of_scope   (is this tool in scope this turn)
            -> registry.resolve      (coerce and validate arguments)
            -> _gate                 (dedupe, feature vetoes, guard, approval)
            -> _execute_call         (run, record outcome, format result)
        -> ToolBatch(messages, aborted)

The gate order is fixed. Every announced call gets exactly one result, redacted
before it is formatted. execute_calls returns the messages; the caller appends them.

invoke_unchecked is the raw path: it enforces scope but runs no gates.
invoke_tool_unchecked is its Agent-taking module helper. register_tools,
resolve_scope, and schema_json_len are setup helpers.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from lamssi_agents.approval import (
    ApprovalRequest,
    ToolApproval,
    ToolApprovalResult,
    approval_request,
    needs_approval,
    normalize_approval,
)
from lamssi_agents.events import AgentAborted, AgentEventType
from lamssi_agents.history import clip_result
from lamssi_agents.providers import Message, ToolCall
from lamssi_agents.redaction import redact
from lamssi_agents.tooling import DEFAULT_POLICY, GuardRole, ToolSurface, resolve_surface
from lamssi_agents.tooling.dedupe import DedupeCache
from lamssi_agents.tooling.guard import (
    DEFAULT_GUARD_RULES,
    LoopGuard,
    call_signature,
)
from lamssi_agents.tooling.invocation import ToolInvocation
from lamssi_tools import (
    CallableSource,
    CapabilityContext,
    Expose,
    InstanceSource,
    ModuleSource,
    MountedRegistry,
    ToolDefinition,
    ToolExecutionError,
    err,
    strip_pydantic_urls,
)
from lamssi_tools import (
    tool as tool_decorator,
)
from lamssi_tools.context import capability_context_active

if TYPE_CHECKING:
    from lamssi_agents.agent.base import Agent
    from lamssi_agents.agent.control import RunControl

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolBatch:
    """The result of running one turn's tool calls.

    Attributes:
        messages: One tool message per announced call, in order.
        aborted: Whether the run was cancelled part way through the batch.
    """

    messages: List[Message] = field(default_factory=list)
    aborted: bool = False


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """The agent-owned tool policy the runtime reads through.

    Each field is a container the Agent owns and features mutate in place as they
    install (a dedupe rule, a safe-when check, a guard role, a hook). The
    runtime holds the same objects, so a change after construction is seen.
    """

    dedupe: Dict[str, Any]
    safe_when: Dict[str, Any]
    guard_roles: Dict[str, Any]
    before_tool: List[Any]
    after_tool: List[Any]
    approved_hooks: List[Any]
    max_chars: int


class ToolRuntime:
    """One agent's tool behavior: surface, gates, execution, and result format.

    Owns the loop guard (reset each user request) and the dedupe cache
    (conversation-scoped: it persists across requests and is dropped only when a
    summarisation folds the earlier result away). Also owns the tool scope, the
    truncator, and the guard configuration. Receives the tool registry and
    capabilities, the run authority (approvals, interaction, cancellation, and
    tool access are read from it), an emit callable, and the agent's
    :class:`ToolPolicy`. It holds the same policy containers, so a feature
    installed later is seen.
    """

    def __init__(
        self,
        *,
        registry: MountedRegistry,
        capabilities: CapabilityContext,
        control: RunControl,
        emit: Callable[..., None],
        policy: ToolPolicy,
        guard_rules: Any = None,
    ) -> None:
        self._registry = registry
        self._capabilities = capabilities
        self._approvals = control.approvals
        self._interaction = control.interaction
        self._cancellation = control.cancellation
        self._access = control.tool_access
        self._emit = emit
        self._policies = policy.dedupe
        self._safe_when = policy.safe_when
        self._guard_roles = policy.guard_roles
        self._before_tool = policy.before_tool
        self._after_tool = policy.after_tool
        self._approved_hooks = policy.approved_hooks
        self._max_chars = policy.max_chars

        self._guard_rules = guard_rules
        self.tool_scope: Optional[set] = None
        self.truncator: Callable[..., str] = clip_result

        self.guard = LoopGuard(self.rules)
        self.dedupe = DedupeCache()

    @property
    def rules(self) -> Any:
        """Effective loop-guard rules after applying role overrides."""
        base = self._guard_rules or DEFAULT_GUARD_RULES
        return base.with_roles(self._guard_roles) if self._guard_roles else base

    @property
    def guard_rules(self) -> Any:
        """Base loop-guard rules before role overrides; ``None`` is the default."""
        return self._guard_rules

    @guard_rules.setter
    def guard_rules(self, value: Any) -> None:
        self._guard_rules = value
        self._sync_guard()

    def _sync_guard(self) -> None:
        """Point the live guard at the current effective rules."""
        self.guard.rules = self.rules

    def begin_request(self) -> None:
        """Reset guard state for a new user message; the cache persists."""
        self.guard.on_new_turn()

    def on_cleared(self) -> None:
        """Drop everything derived from a cleared conversation."""
        self.guard.on_cleared()
        self.dedupe.on_cleared()

    def on_history_compacted(self, demoted_only: bool) -> None:
        """Adjust after compaction. A demotion keeps calls visible and leaves the
        cache and guard untouched; a summarisation invalidates them.
        """
        if demoted_only:
            return
        self.guard.on_compacted()
        self.dedupe.on_compacted()

    def surface(self) -> ToolSurface:
        """The tools exposed this turn.

        Narrowed by ``tool_scope``. A tool declaring the ``always_allowed`` guard role
        survives it: the guard's own messages tell the model to call these, so a
        narrowing that removed them would close the only way out of a dead end.
        """
        rules = self.rules
        defs = _servable(self._registry.list_tools(), self._capabilities)
        return resolve_surface(
            all_defs=defs,
            always_available=frozenset(
                d.name for d in defs if rules.role_for(d.name, d) is GuardRole.ALWAYS_ALLOWED
            ),
            disabled=self._access.disabled,
            agent_allow=None if self.tool_scope is None else set(self.tool_scope),
        )

    def all_defs(self) -> List[ToolDefinition]:
        """This turn's definitions, for building the provider schema."""
        return list(self.surface().defs)

    def all_defs_unfiltered(self) -> List[ToolDefinition]:
        """Every agent-exposed tool the host offers, ignoring every narrowing."""
        return [t for t in self._registry.list_tools() if t.expose_to_agent]

    def definition_for(self, name: str) -> Optional[ToolDefinition]:
        """The live registered definition for *name*, regardless of scope."""
        return self._registry.get_tool(name)

    def reject_out_of_scope(
        self, name: str, resolved: ToolSurface
    ) -> Optional[Dict[str, Any]]:
        """Return an error payload if *name* may not be called, else ``None``."""
        if self._access.is_disabled(name):
            return err(f"Tool '{name}' is currently disabled.", retriable=False)
        if name in resolved.names:
            return None
        return err(
            f"Tool '{name}' is not available to this agent.",
            retriable=False,
            available=sorted(resolved.names),
            hint="Use one of the tools listed in 'available'.",
        )

    def execute_calls(self, tool_calls: List[ToolCall], turn: int) -> ToolBatch:
        """Run every announced call, producing exactly one message for each.

        Args:
            tool_calls: Calls the model announced this turn.
            turn: The current run turn, for dedupe bookkeeping.

        Returns:
            A :class:`ToolBatch` with one message per call and whether the run
            was cancelled part way through.
        """
        messages: List[Message] = []
        for index, tc in enumerate(tool_calls):
            if self._cancellation.requested:
                self._answer_remaining(tool_calls[index:], messages)
                return ToolBatch(messages, aborted=True)
            try:
                messages.append(self._run_one_call(tc, turn))
            except AgentAborted:
                # Abort leaves the current and later calls unanswered; pair them.
                self._answer_remaining(tool_calls[index:], messages)
                return ToolBatch(messages, aborted=True)
        return ToolBatch(messages, aborted=False)

    def invoke_unchecked(
        self,
        name: str,
        arguments: Dict[str, Any],
        *,
        resolved: Optional[ToolSurface] = None,
    ) -> Any:
        """Run one tool call directly, enforcing scope but running no gates.

        Unlike :meth:`execute_calls` it consults neither dedupe, the loop
        guard, nor approval.

        Returns:
            The tool result, or an error payload. Never raises.
        """
        try:
            resolved = resolved or self.surface()
            blocked = self.reject_out_of_scope(name, resolved)
            if blocked is not None:
                return blocked
            binding = self._registry.resolve(name, arguments)
            return self._registry.execute_binding(binding)
        except ToolExecutionError as exc:
            return err(strip_pydantic_urls(str(exc)), retriable=False)
        except Exception as exc:
            # Deliberate boundary: neither the loop nor the model can act on a raise.
            log.error("Tool '%s' error: %s", name, exc, exc_info=True)
            return err(strip_pydantic_urls(str(exc)))

    def _run_one_call(self, tc: ToolCall, turn: int) -> Message:
        """Run one announced call through to its single result message."""
        resolved = self.surface()
        raw = ToolInvocation(
            name=tc.name, arguments=tc.arguments or {}, call_id=tc.id, turn=turn
        )

        blocked = self.reject_out_of_scope(raw.name, resolved)
        if blocked is not None:
            return self._format_message(raw, blocked, resolved.by_name.get(tc.name))

        try:
            binding = self._registry.resolve(raw.name, raw.arguments)
        except ToolExecutionError as exc:
            return self._format_message(
                raw,
                err(strip_pydantic_urls(str(exc)), retriable=False),
                resolved.by_name.get(tc.name),
            )
        invocation = replace(raw, arguments=binding.arguments, binding=binding)

        definition = invocation.definition
        self._emit(
            AgentEventType.TOOL_START,
            invocation.name,
            arguments=invocation.arguments,
            tool_call_id=invocation.id,
            dispatch=getattr(definition, "dispatch", None),
        )

        # A gate may need the running agent (e.g. an approved-read hook that
        # records a grant on the agent's conversation), so bind its capability
        # context for the gate phase; the tool body binds its own on execute.
        with capability_context_active(self._capabilities):
            blocked = self._gate(invocation)
        if blocked is not None:
            return self._format_message(invocation, blocked, definition)

        return self._execute_call(invocation)

    def _answer_remaining(
        self, pending: List[ToolCall], messages: List[Message]
    ) -> None:
        """Add an aborted result for any pending call not already answered."""
        answered = {m.tool_call_id for m in messages if m.role == "tool"}
        for tc in pending:
            if tc.id in answered:
                continue
            messages.append(
                self._format_message(
                    tc, err("Aborted before this call ran.", retriable=False), None
                )
            )

    def _gate(self, call: ToolInvocation) -> Optional[Dict[str, Any]]:
        """Run the gates in order and return the first blocking payload."""

        blocked = self._dedupe_gate(call)
        if blocked is not None:
            return blocked
        
        for feature_gate in self._before_tool:
            try:
                blocked = feature_gate(call)
            except AgentAborted:
                raise
            except Exception as exc:
                log.error(
                    "before-tool gate %r failed for %r: %s",
                    getattr(feature_gate, "__qualname__", feature_gate),
                    call.name,
                    exc,
                    exc_info=True,
                )
                return err(
                    f"Tool '{call.name}' was not executed because an application "
                    "safety gate failed.",
                    retriable=False,
                )
            if blocked is not None:
                return blocked
            
        blocked = self._guard_gate(call)
        if blocked is not None:
            return blocked
        return self._approval_gate(call)

    def _policy_for(self, tool_name: str) -> Any:
        return self._policies.get(tool_name, DEFAULT_POLICY)

    def _dedupe_gate(self, call: ToolInvocation) -> Optional[Dict[str, Any]]:
        """Return the cached answer for an identical earlier *call*, or ``None``."""
        definition = call.definition
        if self.guard.rules.skips_repeat_checks(call.name, definition):
            return None
        return self.dedupe.check(
            call.name, call.arguments or {}, self._policy_for(call.name)
        )

    def _confirm_repeat(self, call: ToolInvocation, payload: Dict[str, Any]) -> bool:
        """Request a bounded override for an identical repeated call."""
        rules = self.guard.rules
        sig = call_signature(call.name, call.arguments or {})
        decisions = self.guard.approved_repeats

        remaining = decisions.get(sig)
        if remaining is not None:
            if remaining <= 0:
                return False
            decisions[sig] = remaining - 1
            return True

        reason = payload.get("error", "") if isinstance(payload, dict) else ""
        question = rules.messages.render(rules.messages.repeat_confirm, name=call.name)
        if reason:
            question = f"{question}\n\nGuard note: {reason}"

        from lamssi_agents.interaction import (
            InteractionDecision,
            InteractionKind,
            request_interaction,
        )

        response = request_interaction(
            self._interaction.handler,
            self._emit,
            InteractionKind.GUARD_OVERRIDE,
            question,
            tool=call.name,
            arguments=dict(call.arguments or {}),
            reason=reason,
        )
        if response is None:
            decisions[sig] = 0
            return False
        allowed = response.decision is InteractionDecision.CONTINUE
        decisions[sig] = (rules.max_approved_repeats - 1) if allowed else 0
        log.info(
            "Repeat of '%s' %s by the user.",
            call.name,
            "allowed" if allowed else "blocked",
        )
        return allowed

    def _guard_gate(self, call: ToolInvocation) -> Optional[Dict[str, Any]]:
        """Block *call* when it is looping, unless the user allows a bare repeat."""
        found = self.guard.decide(call.name, call.arguments or {}, call.definition)
        if found is None:
            return None
        payload, kind = found
        if kind == "error" or not self._confirm_repeat(call, payload):
            return payload
        return None

    def _record_dedupe(
        self, call: ToolInvocation, result: Any, is_error: bool
    ) -> None:
        """Record a successful call and invalidate related cached calls."""
        if is_error:
            return
        definition = call.definition
        args = call.arguments or {}

        if not self.guard.rules.skips_repeat_checks(call.name, definition):
            self.dedupe.record(
                call.name, args, self._policy_for(call.name), call.turn
            )

        invalidated = self.dedupe.invalidate(call.name, args, self._policies)
        if invalidated:
            self.guard.forget_tools(invalidated)

    def _record_outcome(
        self, call: ToolInvocation, result: Any, is_error: bool
    ) -> None:
        """Tell the guard the outcome so an error streak can accumulate."""
        err_msg = None
        if is_error and isinstance(result, dict):
            err_msg = result.get("error", "")
            if err_msg and result.get("hint"):
                err_msg = f"{err_msg} (hint: {result['hint']})"
        self.guard.record(
            call.name, call.arguments or {}, is_error=is_error, error_msg=err_msg
        )

    def _approval_gate(self, call: ToolInvocation) -> Optional[Dict[str, Any]]:
        """Block *call* unless policy allows it or a human approves it."""
        policy = self._approvals.policy
        if not needs_approval(
            policy, call.name, call.arguments, call.definition, rules=self._safe_when
        ):
            return None
        if policy.handler is None:
            return err(
                f"'{call.name}' needs approval and this run is unattended, "
                "so it was not executed.",
                retriable=False,
                hint="Use a read-only alternative, or report what needs approving.",
            )
        request = approval_request(
            policy, call.name, call.arguments or {}, call.definition
        )
        return self._handle_approval(call, request)

    def _handle_approval(
        self, tc: ToolInvocation, request: ApprovalRequest
    ) -> Optional[Dict[str, Any]]:
        """Ask the user and return a rejection payload, or ``None`` to proceed.

        Raises:
            AgentAborted: If the user chose ``ToolApproval.ABORT``.
        """
        self._emit(
            AgentEventType.TOOL_APPROVAL,
            tc.name,
            arguments=tc.arguments,
            tool_call_id=tc.id,
            reason=request.reason,
            declaration=request.declaration,
        )
        try:
            handler = self._approvals.policy.handler
            if handler is None:
                raise RuntimeError("no approval handler is configured")
            raw = handler(request)
        except Exception as exc:
            log.warning("Approval handler for '%s' raised (%s): blocking.", tc.name, exc)
            raw = ToolApprovalResult(
                decision=ToolApproval.REJECT,
                reason=f"the approval handler failed: {exc}",
            )
        decision, reason = normalize_approval(raw)

        if decision == ToolApproval.ABORT:
            raise AgentAborted("User aborted at tool approval")
        if decision == ToolApproval.REJECT:
            msg = (
                f"Tool '{tc.name}' was rejected by the user. "
                + (f"Reason: {reason}. " if reason else "")
                + "Do NOT retry it. Ask the user what to do instead."
            )
            self._emit(
                AgentEventType.TOOL_REJECTED,
                msg,
                tool_name=tc.name,
                tool_call_id=tc.id,
                reason=reason,
            )
            return err(msg, retriable=False)

        for hook in self._approved_hooks:
            try:
                hook(tc.name, tc.arguments or {})
            except Exception as exc:
                log.debug("approval hook %r raised: %s", hook, exc)
        return None

    def _execute_call(self, tc: ToolInvocation) -> Message:
        """Run *tc*, tell the remembering gates the outcome, format the result."""
        start = time.monotonic()
        result = self._registry.execute_binding(tc.binding)
        elapsed_ms = (time.monotonic() - start) * 1000

        result_str = json.dumps(result, separators=(",", ":"), default=str)
        is_error = isinstance(result, dict) and "error" in result
        err_msg = result.get("error", "") if is_error else None
        if err_msg and isinstance(result, dict) and result.get("hint"):
            err_msg = f"{err_msg} (hint: {result['hint']})"

        self._fire_after_tool(tc, result, is_error)
        log.log(
            logging.WARNING if is_error else logging.INFO,
            "Tool '%s' %s (%.0fms)",
            tc.name,
            f"ERROR {err_msg}" if is_error else "OK",
            elapsed_ms,
        )
        return self._format_message(tc, result, tc.definition, serialised=result_str)

    def _fire_after_tool(
        self, tc: ToolInvocation, result: Any, is_error: bool
    ) -> None:
        """Record the outcome with the guard, then run feature after-tool hooks."""
        self._record_dedupe(tc, result, is_error)
        self._record_outcome(tc, result, is_error)
        for hook in self._after_tool:
            try:
                hook(tc, result, is_error)
            except Exception as exc:
                log.warning(
                    "after-tool hook %r raised: %s",
                    getattr(hook, "__name__", hook),
                    exc,
                    exc_info=True,
                )

    def _format_message(
        self,
        tc: Any,
        payload: Any,
        definition: Any = None,
        *,
        serialised: Optional[str] = None,
    ) -> Message:
        """Redact, truncate, emit, and return the one result message for *tc*."""
        result_str = (
            serialised
            if serialised is not None
            else json.dumps(payload, separators=(",", ":"), default=str)
        )
        # Redact at the boundary so credentials are masked whichever truncator runs.
        body = self.truncator(
            redact(result_str),
            tool_name=tc.name,
            definition=definition,
            default_max_chars=self._max_chars,
        )
        self._emit(
            AgentEventType.TOOL_RESULT,
            body,
            tool_name=tc.name,
            tool_call_id=tc.id,
            dispatch=getattr(definition, "dispatch", None),
        )
        return Message(role="tool", content=body, tool_call_id=tc.id, name=tc.name)


def register_tools(agent: "Agent", items: Sequence[Any]) -> None:
    """Normalize and register the tool contributions accepted by ``Agent``."""
    loose: list[Callable[..., Any]] = []
    for item in items:
        if isinstance(item, ModuleType):
            agent._registry.add(
                ModuleSource(item, context=agent._capabilities), owner="feature"
            )
        elif _has_tool_methods(item):
            agent._registry.add(
                InstanceSource(item, context=agent._capabilities), owner="feature"
            )
        elif callable(item):
            loose.append(item)
        else:
            raise TypeError(
                "tools must be callables, modules, or objects with @tool methods; "
                f"got {type(item).__name__}"
            )
    if loose:
        decorated = [
            fn
            if getattr(fn, "_tool_definition", None) is not None
            else tool_decorator(expose=Expose.AGENT)(fn)
            for fn in loose
        ]
        agent._registry.add(
            CallableSource(*decorated, context=agent._capabilities),
            owner="feature",
        )


def _has_tool_methods(obj: Any) -> bool:
    """Whether *obj* is an instance carrying decorated tool methods."""
    if getattr(obj, "_tool_definition", None) is not None:
        return False
    return any(
        getattr(getattr(obj, name, None), "_tool_definition", None) is not None
        for name in dir(obj)
        if not name.startswith("__")
    )


def resolve_scope(only: Any, agent: "Agent") -> set:
    """Validate and return the fixed tool-name allow-list from ``only=``."""
    if isinstance(only, str):
        raise TypeError("only= takes a list of names, not a single string")

    available = agent._tools.list_tools()
    names = {t.name for t in available}
    wanted = set(only)
    unknown = sorted(wanted - names)

    if unknown:
        raise ValueError(
            f"only= names {', '.join(repr(u) for u in unknown)}, which this agent "
            f"does not have.\n"
            f"  tools:  {', '.join(sorted(names))}"
        )
    return wanted


def _servable(defs: List[ToolDefinition], capabilities: Any) -> List[ToolDefinition]:
    """Return tools whose required capabilities are currently available."""
    if capabilities is None:
        return list(defs)

    servable, withheld = [], []
    for definition in defs:
        missing = [
            required
            for required in (getattr(definition, "requires", ()) or ())
            if not capabilities.has(required)
        ]
        if missing:
            withheld.append((definition.name, missing))
        else:
            servable.append(definition)

    if withheld:
        log.debug(
            "withheld from the schema, capability not registered: %s",
            ", ".join(
                f"{name} (needs {', '.join(m.__name__ for m in missing)})"
                for name, missing in withheld
            ),
        )
    return servable


def schema_json_len(defs: List[ToolDefinition]) -> int:
    """Byte length of the tool schemas as the provider serialises them (0 if none)."""
    if not defs:
        return 0
    return len(json.dumps([d.input_schema() for d in defs], separators=(",", ":")))


def invoke_tool_unchecked(
    agent: "Agent",
    name: str,
    arguments: Dict[str, Any],
    *,
    resolved: Optional[ToolSurface] = None,
) -> Any:
    """Run one tool call directly, enforcing scope but running no gates."""
    return agent._runtime.invoke_unchecked(name, arguments, resolved=resolved)


__all__ = [
    "ToolBatch",
    "ToolPolicy",
    "ToolRuntime",
    "register_tools",
    "resolve_scope",
    "schema_json_len",
    "invoke_tool_unchecked",
]
