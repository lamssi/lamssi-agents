"""Loop guard behavior and its per-tool policy."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from lamssi_tools import err

log = logging.getLogger(__name__)


class GuardRole(str, Enum):
    """How the loop guard treats a tool's repeats and failures."""

    NORMAL = "normal"  # every repeat and error check applies
    ALWAYS_ALLOWED = "always_allowed"  # never blocked, and kept available even when tools are narrowed (e.g. ask_user)
    RECOVERY = "recovery"  # skips error checks, for a tool used to recover from a failure
    REPEATABLE = "repeatable"  # skips repeat checks; identical calls are legitimate, not a loop


REPEAT_EXEMPT_ROLES = frozenset({GuardRole.ALWAYS_ALLOWED, GuardRole.REPEATABLE})
ERROR_EXEMPT_ROLES = frozenset({GuardRole.ALWAYS_ALLOWED, GuardRole.RECOVERY})


@dataclass(frozen=True, slots=True)
class GuardMessages:
    """Messages returned when a loop check blocks a call."""

    duplicate: str = (
        "You just called {name} with identical arguments. Repeating it will not "
        "change the result. Take a different action, or ask the user for the "
        "information you need: asking is always available and never blocked."
    )
    duplicate_after_error: str = (
        "You just called {name} with identical arguments, and that call FAILED: "
        "{err}. Nothing changed: the operation did NOT take effect. Do not tell "
        "the user it succeeded. Fix the arguments, or ask the user for what you "
        "need: asking is always available and never blocked."
    )
    error_streak: str = (
        "{name} has failed {n} times on the same target. Stop retrying. Ask the "
        "user for the specific input you need, or summarise the blocker for them."
    )
    cycle: str = (
        "You are cycling between tools: {sequence}. The same sequence has repeated "
        "{repeats} times. Stop and ask the user what would unblock you, or summarise "
        "what is missing."
    )
    consecutive_errors: str = (
        "The last {n} tool calls have all failed. Stop trying variations. Ask the "
        "user one specific question, or summarise the blocker in plain text."
    )
    repeat_confirm: str = (
        "The agent wants to run '{name}' again with the same arguments. That is "
        "usually a runaway loop, but it can be deliberate. Allow it? "
        "(yes / continue to allow, anything else to block)"
    )

    def render(self, template: str, **fields: Any) -> str:
        """Format a message, falling back safely when an override is malformed."""
        try:
            return template.format(**fields)
        except (KeyError, IndexError, ValueError):
            name = fields.get("name", "the tool")
            return (
                f"A guard blocked a repeated or failing call to {name}. Take a "
                "different action, or ask the user for what you need."
            )


@dataclass(frozen=True, slots=True)
class GuardRules:
    """Per-tool roles and limits used by :class:`LoopGuard`."""

    roles: Mapping[str, GuardRole] = field(default_factory=dict)
    dup_lookback: int = 4
    cycle_lookback: int = 8
    err_streak_limit: int = 3
    consecutive_error_limit: int = 4
    streak_key_args: Tuple[str, ...] = ("path", "name", "id", "query", "target")
    max_approved_repeats: int = 3
    messages: GuardMessages = field(default_factory=GuardMessages)

    def role_for(self, tool: str, definition: Any = None) -> GuardRole:
        """Resolve an explicit role, then the tool declaration, then NORMAL."""
        explicit = self.roles.get(tool)
        if explicit is not None:
            return explicit if isinstance(explicit, GuardRole) else GuardRole(explicit)
        declared = getattr(definition, "guard_role", None) if definition is not None else None
        try:
            return GuardRole(declared) if declared else GuardRole.NORMAL
        except ValueError:
            return GuardRole.NORMAL

    def skips_repeat_checks(self, tool: str, definition: Any = None) -> bool:
        return self.role_for(tool, definition) in REPEAT_EXEMPT_ROLES

    def skips_error_checks(self, tool: str, definition: Any = None) -> bool:
        return self.role_for(tool, definition) in ERROR_EXEMPT_ROLES

    def with_roles(self, extra: Mapping[str, GuardRole]) -> "GuardRules":
        return replace(self, roles={**self.roles, **extra})


CORE_GUARD_ROLES: Mapping[str, GuardRole] = {
    "ask_user": GuardRole.ALWAYS_ALLOWED,
    "load_skill": GuardRole.ALWAYS_ALLOWED,
    "read_file": GuardRole.RECOVERY,
}
DEFAULT_GUARD_RULES = GuardRules(roles=dict(CORE_GUARD_ROLES))


def _truncate(s: str, limit: int) -> str:
    """Cap a string, marking the cut."""
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def call_signature(name: str, args: Dict[str, Any]) -> str:
    """A stable ``name|canonical-args`` identity for one tool call.

    Sorted keys so argument order doesn't change the identity. Shared with the
    repeat-override cache in ``tool_runtime`` so the two always agree on what
    "the same call" is.
    """
    return f"{name}|{json.dumps(args or {}, sort_keys=True, default=str)}"


class LoopGuard:
    """Short-circuits runaway tool-call loops.

    Args:
        rules: Per-tool roles, limits and message text. Defaults to the kernel's
            neutral configuration.
    """

    def __init__(self, rules: Optional[GuardRules] = None) -> None:
        self.rules = rules or DEFAULT_GUARD_RULES
        self._recent_keys: List[str] = []
        self._error_streak: Dict[str, int] = {}
        self._consecutive_errors: int = 0
        # Per-key memory of the last error, so the duplicate check can give a
        # blunter message when the repeat is of a call that failed.
        self._last_error_by_key: Dict[str, str] = {}
        #: ``"tool|args" -> remaining approved repeats``; a "yes" grants a bounded number, not a loop, and is cleared by the same hooks that clear the evidence it overrode.
        self.approved_repeats: Dict[str, int] = {}

    def check_duplicate(
        self, name: str, args: Dict[str, Any], definition: Any = None
    ) -> Optional[Dict[str, Any]]:
        """A payload if this call duplicates a recent one, else ``None``."""
        if self.rules.skips_repeat_checks(name, definition):
            return None
        key = self._tool_key(name, args)
        if key not in self._recent_keys:
            return None

        msgs = self.rules.messages
        prior_err = self._last_error_by_key.get(key)
        if prior_err:
            return err(
                msgs.render(
                    msgs.duplicate_after_error, name=name, err=_truncate(prior_err, 220)
                ),
                retriable=False,
            )
        return err(
            msgs.render(msgs.duplicate, name=name),
            retriable=False,
        )

    def check_error_streak(
        self, name: str, args: Dict[str, Any], definition: Any = None
    ) -> Optional[Dict[str, Any]]:
        """A payload if this target has failed too many times, else ``None``."""
        if self.rules.skips_error_checks(name, definition):
            return None
        skey = self._streak_key(name, args)
        n = self._error_streak.get(skey, 0)
        if n < self.rules.err_streak_limit:
            return None
        msgs = self.rules.messages
        return err(
            msgs.render(msgs.error_streak, name=name, n=n),
            retriable=False,
        )

    def check_cycle(
        self, name: str, args: Dict[str, Any], definition: Any = None
    ) -> Optional[Dict[str, Any]]:
        """Detect period-2 or period-3 cycles in the recent call sequence.

        Appends the proposed call and checks whether the last ``2*N`` entries
        split into two equal halves.
        """
        if self.rules.skips_repeat_checks(name, definition):
            return None
        seq = self._recent_keys + [self._tool_key(name, args)]
        for period in (2, 3):
            if len(seq) >= 2 * period:
                tail = seq[-2 * period:]
                if tail[:period] == tail[period:]:
                    tool_names = [k.split("|", 1)[0] for k in tail[:period]]
                    msgs = self.rules.messages
                    return err(
                        msgs.render(
                            msgs.cycle, sequence=" -> ".join(tool_names), repeats=2
                        ),
                        retriable=False,
                    )
        return None

    def check_consecutive_errors(
        self, name: str, args: Dict[str, Any], definition: Any = None
    ) -> Optional[Dict[str, Any]]:
        """A payload when too many calls in a row have failed, regardless of target."""
        if self.rules.skips_error_checks(name, definition):
            return None
        if self._consecutive_errors < self.rules.consecutive_error_limit:
            return None
        msgs = self.rules.messages
        return err(
            msgs.render(msgs.consecutive_errors, n=self._consecutive_errors),
            retriable=False,
        )

    def decide(
        self, name: str, args: Dict[str, Any], definition: Any = None
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        """Run the loop checks in order, returning ``(payload, kind)`` or ``None``.

        ``kind`` is ``"error"`` for a failure loop (a hard stop) or ``"repeat"``
        for a duplicate or cycle (which the user may allow). Errors are checked
        first, so a call that both repeats and keeps failing is treated as the
        failure.
        """
        streak = self.check_error_streak(name, args, definition)
        if streak is not None:
            log.warning("Tool '%s' error streak: stopping", name)
            return streak, "error"
        consec = self.check_consecutive_errors(name, args, definition)
        if consec is not None:
            log.warning("Consecutive-error limit reached: stopping")
            return consec, "error"
        dup = self.check_duplicate(name, args, definition)
        if dup is not None:
            log.warning("Tool '%s' duplicated: asking the user", name)
            return dup, "repeat"
        cycle = self.check_cycle(name, args, definition)
        if cycle is not None:
            log.warning("Tool '%s' cycle detected: asking the user", name)
            return cycle, "repeat"
        return None

    def record(
        self,
        name: str,
        args: Dict[str, Any],
        is_error: bool,
        error_msg: Optional[str] = None,
    ) -> None:
        """Update state with the outcome of a completed call."""
        key = self._tool_key(name, args)
        self._recent_keys.append(key)
        max_window = max(self.rules.dup_lookback, self.rules.cycle_lookback)
        if len(self._recent_keys) > max_window:
            evicted = self._recent_keys.pop(0)
            # Drop the error memory once its key leaves the window, so it cannot
            # outlive the check that reads it.
            if evicted not in self._recent_keys:
                self._last_error_by_key.pop(evicted, None)

        skey = self._streak_key(name, args)
        if is_error:
            self._error_streak[skey] = self._error_streak.get(skey, 0) + 1
            self._consecutive_errors += 1
            if error_msg:
                self._last_error_by_key[key] = error_msg
        else:
            self._error_streak.pop(skey, None)
            self._consecutive_errors = 0
            self._last_error_by_key.pop(key, None)

    def clear(self) -> None:
        """Reset all state for a new conversation."""
        self._recent_keys.clear()
        self._error_streak.clear()
        self._consecutive_errors = 0
        self._last_error_by_key.clear()
        self.approved_repeats.clear()

    def reset_for_new_turn(self) -> None:
        """Clear the consecutive-error count when a new user message arrives.

        A new message is new information; the next turn should not start already
        at the limit. Per-target streaks survive.
        """
        self._consecutive_errors = 0

    def reset_repeat_window(self) -> None:
        """Forget the recent-call window used for duplicate and cycle detection.

        Called on a new user message: a repeat the user just asked for is fresh
        intent, not a loop. A loop within a turn re-accumulates immediately.

        The approvals go with it. A "yes, do it again" answers the repeat the
        user was shown; carrying it into the next message would silently spend
        an override on a call nobody was asked about.
        """
        self._recent_keys.clear()
        self.approved_repeats.clear()
        self._last_error_by_key.clear()

    # Named for the events, not what they do here, so the transcript can notify every holder without knowing which is which.
    def on_new_turn(self) -> None:
        self.reset_for_new_turn()
        self.reset_repeat_window()

    def on_compacted(self) -> None:
        self.reset_repeat_window()

    def on_cleared(self) -> None:
        self.clear()

    def forget_tools(self, names: Iterable[str]) -> int:
        """Drop remembered calls to *names*, because their answers are now stale.

        The duplicate check assumes calling again cannot change the result; a
        write breaks that assumption for a subsequent read of the same thing.

        Returns how many entries were dropped.
        """
        wanted = set(names)
        if not wanted:
            return 0
        keep = [k for k in self._recent_keys if k.split("|", 1)[0] not in wanted]
        dropped = len(self._recent_keys) - len(keep)
        if dropped:
            self._recent_keys = keep
            for key in list(self._last_error_by_key):
                if key.split("|", 1)[0] in wanted:
                    del self._last_error_by_key[key]
        return dropped

    @staticmethod
    def _tool_key(name: str, args: Dict[str, Any]) -> str:
        return call_signature(name, args)

    def _streak_key(self, name: str, args: Dict[str, Any]) -> str:
        for k in self.rules.streak_key_args:
            if (args or {}).get(k):
                return f"{name}|{k}={args[k]}"
        return name


__all__ = [
    "CORE_GUARD_ROLES",
    "DEFAULT_GUARD_RULES",
    "LoopGuard",
    "GuardMessages",
    "GuardRole",
    "GuardRules",
    "call_signature",
]
