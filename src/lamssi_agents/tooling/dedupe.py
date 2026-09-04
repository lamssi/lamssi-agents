"""Answers a repeated tool call from a per-run cache instead of re-running it."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

log = logging.getLogger(__name__)

#: Sentinel meaning "clear every entry for this tool" (no invalidation_key to narrow by).
_ALL = object()


def _stable_value(value: Any) -> Any:
    """Return a stable, hashable representation of *value*."""
    try:
        hash(value)
    except TypeError:
        pass
    else:
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        # Unserialisable and unhashable: repr is weaker but stable enough for one run.
        log.debug("dedupe could not serialise an argument value (%s); using repr", exc)
        return repr(value)


def full_arg_signature(args: Mapping[str, Any]) -> Optional[tuple]:
    """Every argument, sorted by name: the default identity for any tool.

    Returns ``None`` for empty arguments: a no-argument call usually reports
    live state, so it should not be answered from the cache. Exhaustive by
    default so a tool gaining an argument can't silently defeat dedupe.
    """
    if not args:
        return None
    pairs = [(str(key), _stable_value(value)) for key, value in args.items()]
    pairs.sort(key=lambda pair: pair[0])
    return tuple(pairs)


def arg_subset_signature(*keys: str) -> Callable[[Mapping[str, Any]], Optional[tuple]]:
    """Build a signature over *keys* only, ignoring every other argument.

    For an argument that can't affect the answer (a label, a formatting
    preference) so varying it doesn't manufacture a fresh identity. The
    first key is the anchor: absent or empty, the call has no identity and
    is never deduplicated.
    """
    if not keys:
        raise ValueError("arg_subset_signature needs at least one key")
    anchor = keys[0]

    def signature(args: Mapping[str, Any]) -> Optional[tuple]:
        if not args or not args.get(anchor):
            return None
        return tuple((key, _stable_value(args.get(key))) for key in keys)

    signature.__name__ = "arg_subset_signature_" + "_".join(keys)
    signature.__qualname__ = signature.__name__
    return signature


def default_repeat_hint(tool: str, sig: tuple, turn: int) -> Mapping[str, Any]:
    """The synthetic result handed back in place of a repeated call.

    States that the call already happened, where its result is, and that
    repeating it won't change the answer. It doesn't claim the earlier
    result was useful, since the cache has no way to know that.
    """
    return {
        "already_called": True,
        "tool": tool,
        "first_called_at_turn": turn,
        "hint": (
            f"You already made this exact '{tool}' call at turn {turn}; its result is "
            "above. If that result did not answer your question, running it again will "
            "not either: the same arguments always give the same answer. Change the "
            "arguments, or try a different tool."
        ),
    }


@dataclass(frozen=True, slots=True)
class DedupePolicy:
    """How one tool's calls are recognised as repeats, and what to say about it.

    Registered by the tool's owner, under the tool's name. A tool with no
    registered policy gets :data:`DEFAULT_POLICY`.
    """

    #: Maps arguments to a hashable identity, or ``None`` to opt out. ``slots=True`` keeps this a plain function rather than a bound method.
    signature: Callable[[Mapping[str, Any]], Optional[tuple]] = full_arg_signature

    #: Builds the synthetic result for a repeat, given ``(tool, signature, turn)``.
    hint: Callable[[str, tuple, int], Mapping[str, Any]] = default_repeat_hint

    #: Names of tools whose successful execution makes entries for *this* tool stale (a read caches; the write that invalidates it clears them).
    invalidated_by: frozenset[str] = frozenset()

    #: What an invalidating call's arguments identify, e.g. ``lambda args: args.get("path")``; ``None`` clears every entry for this tool (the safe default).
    invalidation_key: Optional[Callable[[Mapping[str, Any]], Any]] = None

    #: Set ``False`` to keep a policy registered but inert, e.g. while debugging.
    enabled: bool = True


#: The policy applied to any tool that has not registered one.
DEFAULT_POLICY = DedupePolicy()


def _mentions(sig: tuple, key: Any) -> bool:
    """Return whether a signature contains *key* as a value or bare element."""
    for element in sig:
        if isinstance(element, tuple) and len(element) == 2:
            if element[1] == key:
                return True
        elif element == key:
            return True
    return False


class DedupeCache:
    """Track the turn that answered each tool-call signature."""

    __slots__ = ("_seen",)

    def __init__(self) -> None:
        self._seen: dict[tuple[str, tuple], int] = {}

    def check(
        self, tool: str, args: Mapping[str, Any], policy: DedupePolicy
    ) -> Optional[Mapping[str, Any]]:
        """The synthetic result for a repeat, or ``None`` to let the call run."""
        if not policy.enabled:
            return None
        sig = self._signature(tool, args, policy)
        if sig is None:
            return None
        turn = self._seen.get((tool, sig))
        if turn is None:
            return None
        try:
            hint = policy.hint(tool, sig, turn)
        except Exception as exc:
            log.warning("dedupe hint for '%s' raised: %s", tool, exc, exc_info=True)
            hint = default_repeat_hint(tool, sig, turn)
        log.debug("dedupe hit: '%s' first answered at turn %d", tool, turn)
        return hint

    def record(
        self, tool: str, args: Mapping[str, Any], policy: DedupePolicy, turn: int
    ) -> None:
        """Remember that this call has been answered, at *turn*.

        Only call for a successful result: caching a failure would pin the
        model to the error on retry. First write wins, so the recorded turn
        stays where the result actually is.
        """
        if not policy.enabled:
            return
        sig = self._signature(tool, args, policy)
        if sig is None:
            return
        self._seen.setdefault((tool, sig), turn)

    def invalidate(
        self,
        tool: str,
        args: Mapping[str, Any],
        policies: Mapping[str, DedupePolicy],
    ) -> set[str]:
        """Clear entries made stale by a successful call to *tool*.

        Every policy naming *tool* in ``invalidated_by`` has its entries
        considered: with an ``invalidation_key``, only entries mentioning that
        value are dropped; otherwise all of them are. A false-positive match
        costs one redundant execution; a false negative hands the model stale
        content as trustworthy, so this errs toward dropping too much.

        Returns:
            The owner tool names invalidated by *tool* (whether or not any cache
            entry was actually dropped): the caller forgets the same set from the
            loop guard, so the two are cleared from one traversal.
        """
        invalidated: set[str] = set()
        removed = 0
        for owner, policy in policies.items():
            if tool not in policy.invalidated_by:
                continue
            invalidated.add(owner)
            key = self._invalidation_key(tool, args, policy)
            stale = [
                entry
                for entry in self._seen
                if entry[0] == owner and (key is _ALL or _mentions(entry[1], key))
            ]
            for entry in stale:
                del self._seen[entry]
            removed += len(stale)
        if removed:
            log.debug("dedupe: '%s' invalidated %d entries", tool, removed)
        return invalidated

    def clear(self) -> None:
        """Clear entries after compaction or restart invalidates their results."""
        self._seen.clear()

    # Event-shaped methods let the transcript notify state holders uniformly.

    def on_compacted(self) -> None:
        self.clear()

    def on_cleared(self) -> None:
        self.clear()

    def __len__(self) -> int:
        return len(self._seen)

    def _signature(
        self, tool: str, args: Mapping[str, Any], policy: DedupePolicy
    ) -> Optional[tuple]:
        try:
            sig = policy.signature(args or {})
        except Exception as exc:
            log.warning("dedupe signature for '%s' raised: %s", tool, exc, exc_info=True)
            return None
        if sig is None:
            return None
        try:
            hash(sig)
        except TypeError:
            # An unhashable signature would raise on the dict lookup instead, with no idea the callable is at fault.
            log.warning("dedupe signature for '%s' is not hashable: %r", tool, sig)
            return None
        return sig

    def _invalidation_key(
        self, tool: str, args: Mapping[str, Any], policy: DedupePolicy
    ) -> Any:
        fn = policy.invalidation_key
        if fn is None:
            return _ALL
        try:
            key = fn(args or {})
        except Exception as exc:
            log.warning(
                "dedupe invalidation key for '%s' raised: %s", tool, exc, exc_info=True
            )
            return _ALL
        # An empty key identifies nothing; matching on it would clear whichever entries happen to be empty, so widen instead.
        return _ALL if key is None or key == "" else key


__all__ = [
    "full_arg_signature",
    "arg_subset_signature",
    "default_repeat_hint",
    "DedupePolicy",
    "DEFAULT_POLICY",
    "DedupeCache",
]
