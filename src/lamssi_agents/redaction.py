"""Keeps credentials out of what the model sees and what subprocess tools inherit."""

from __future__ import annotations

import os
import re
import threading
from typing import Dict, Iterable, Mapping, Match, Optional, Set

#: Below this, a value is masked entirely: showing 6 of 14 characters gives away
#: too much of a short token to be worth the debuggability.
_KEEP_THRESHOLD = 18
_KEEP_HEAD = 6
_KEEP_TAIL = 4


def mask(value: str) -> str:
    """Replace *value* with a form that identifies it without disclosing it."""
    if len(value) < _KEEP_THRESHOLD:
        return "[redacted]"
    return f"{value[:_KEEP_HEAD]}[redacted]{value[-_KEEP_TAIL:]}"


#: Known credential formats, matched by fixed prefix + bounded run - linear in
#: the length of the text, nothing to backtrack.
_VENDOR = re.compile(
    r"""(?x)
    \b(?:
        sk-ant-[A-Za-z0-9_\-]{20,}      # Anthropic
      | sk-proj-[A-Za-z0-9_\-]{20,}     # OpenAI project
      | sk-[A-Za-z0-9]{20,}             # OpenAI classic
      | AIza[A-Za-z0-9_\-]{30,}         # Google
      | gh[pousr]_[A-Za-z0-9]{30,}      # GitHub
      | xox[baprs]-[A-Za-z0-9\-]{10,}   # Slack
      | AKIA[0-9A-Z]{16}                # AWS access key id
    )
    """
)

# A bare opaque token has no recognisable prefix and looks like a commit hash or
# UUID; entropy matching would redact those too. So a host registers exact keys.

_known: Set[str] = set()
_known_lock = threading.Lock()

#: Shorter than this and an exact-match rule does more harm than good: a common
#: word registered as a secret would redact it out of ordinary prose.
_MIN_KNOWN = 8

#: Long enough to look like credentials but are local-server / user
#: placeholders, not real secrets. Registering one would mask the phrase
#: everywhere it appears.
_PLACEHOLDERS = frozenset({
    "not-needed", "not_needed", "no-key", "none", "null", "empty", "unset",
    "changeme", "change-me", "placeholder", "your-api-key", "your-api-key-here",
    "sk-xxxxxxxx", "xxxxxxxx", "test-key", "dummy-key", "fake-key",
})


def register_secret(value: object) -> None:
    """Mask *value* wherever it appears, regardless of surrounding shape.

    Covers what the vendor patterns can't: a bare token with no prefix. Ignores
    values too short to be a credential, and known placeholders.
    """
    text = str(value or "")
    if len(text) >= _MIN_KNOWN and text.lower() not in _PLACEHOLDERS:
        with _known_lock:
            _known.add(text)


def forget_secrets() -> None:
    """Drop every registered value. For tests, and for a credential rotation."""
    with _known_lock:
        _known.clear()


def _mask_vendor(m: Match) -> str:
    return mask(m.group(0))


def redact(text: str) -> str:
    """Mask any registered secret or recognised vendor key in *text*.

    Registered values are matched longest-first so one that contains a shorter
    registered value isn't half-masked. Returned unchanged if nothing matches.
    """
    if not text:
        return text
    with _known_lock:
        known = sorted(_known, key=len, reverse=True)
    out = text
    for secret in known:
        if secret in out:
            out = out.replace(secret, mask(secret))
    return _VENDOR.sub(_mask_vendor, out)


# Subprocess environment deny-list.

#: Whole name segments that identify credentials.
_SECRET_WORDS = frozenset({
    "key", "keys", "apikey", "secret", "secrets", "token", "tokens",
    "password", "passwd", "credential", "credentials",
    "auth", "authorization", "bearer", "signature",
})

_NAME_PARTS = re.compile(r"[^A-Za-z0-9]+")

#: Credential-bearing URLs such as ``postgres://user:pw@host/db``.
_CREDENTIALED_URL = re.compile(r"://[^/\s:@]+:[^/\s@]+@")

#: Explicit environment-variable exemptions guarded by ``_known_lock``.
_env_allow: Set[str] = set()


def allow_env(*names: str) -> None:
    """Let *names* through to subprocesses even though they look like secrets.

    Escape hatch for a tool that genuinely needs one, e.g. ``GH_TOKEN`` for
    ``gh``. Case-insensitive and additive.
    """
    with _known_lock:
        _env_allow.update(name.upper() for name in names if name)


def clear_env_allowances() -> None:
    """Forget every :func:`allow_env` name. For tests."""
    with _known_lock:
        _env_allow.clear()


def is_secret_name(name: str) -> bool:
    """Whether *name* reads like the name of a credential variable."""
    parts = {p.lower() for p in _NAME_PARTS.split(name) if p}
    return bool(parts & _SECRET_WORDS)


def is_secret_value(value: str) -> bool:
    """Whether *value* looks like a credential regardless of what it is called.

    Covers the variable an innocuous name hides: a registered key assigned to
    ``HELPER_CONFIG``, a vendor-shaped token, a connection string with a password.
    """
    if not value:
        return False
    with _known_lock:
        known = tuple(_known)
    if any(secret in value for secret in known):
        return True
    return bool(_VENDOR.search(value) or _CREDENTIALED_URL.search(value))


def safe_environ(
    base: Optional[Mapping[str, str]] = None,
    *,
    keep: Iterable[str] = (),
) -> Dict[str, str]:
    """A copy of the environment with credential variables removed.

    Drops a variable when its *name* names a credential or its *value* looks like
    one - the name rule misses ``HELPER_CONFIG=sk-ant-...``, the value rule misses
    unrecognized formats. Never mutates :data:`os.environ` itself.
    """
    source = os.environ if base is None else base
    with _known_lock:
        allowed = set(_env_allow)
    allowed.update(name.upper() for name in keep if name)

    return {
        name: value
        for name, value in source.items()
        if name.upper() in allowed
        or not (is_secret_name(name) or is_secret_value(value))
    }


__all__ = [
    "redact",
    "mask",
    "register_secret",
    "forget_secrets",
    "safe_environ",
    "allow_env",
    "clear_env_allowances",
    "is_secret_name",
    "is_secret_value",
]
