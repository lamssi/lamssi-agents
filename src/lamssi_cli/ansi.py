"""ANSI colour helpers for the CLI (no external deps)."""

from __future__ import annotations

import re


def _c(code: str, s: str) -> str:
    # Leave "" unwrapped: escape codes would still count toward width for anything measuring the string (e.g. readline).
    return f"\033[{code}m{s}\033[0m" if s else ""


_ANSI = re.compile(r"\033\[[0-9;]*m")


def readline_safe(prompt: str) -> str:
    """Wrap ANSI escapes in *prompt* with readline's zero-width markers.

    GNU readline counts escape sequences as printable unless bracketed by
    ``\\001``/``\\002``, corrupting redraws; apply only where readline
    consumes the markers.
    """
    return _ANSI.sub(lambda m: f"\001{m.group(0)}\002", prompt)


def dim(s: str)     -> str: return _c("2", s)
def bold(s: str)    -> str: return _c("1", s)
def red(s: str)     -> str: return _c("31", s)
def green(s: str)   -> str: return _c("32", s)
def yellow(s: str)  -> str: return _c("33", s)
def magenta(s: str) -> str: return _c("35", s)
def cyan(s: str)    -> str: return _c("36", s)
