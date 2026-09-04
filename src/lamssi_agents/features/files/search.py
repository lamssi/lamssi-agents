"""The Files feature's ``fs`` discovery tool."""

from __future__ import annotations

import glob as _glob_mod
import os
import re
import shlex
from collections import deque
from dataclasses import dataclass
from fnmatch import fnmatch, fnmatchcase
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from lamssi_tools import CapabilityContext, Expose, Int, Str, err, tool

from .space import (
    FileSpace,
    _glob_anchor,
    _within,
    display_path,
    has_glob,
    is_denied,
    path_is_hidden_or_pycache,
)


def _glob_candidates(pattern: str, route: Any) -> tuple:
    """Glob *pattern* and return ``(candidates, display base, error)``."""
    base = route.base
    try:
        if Path(_glob_anchor(pattern)).is_absolute():
            candidates = [
                Path(m) for m in sorted(_glob_mod.glob(pattern, recursive=True))
            ]
        else:
            candidates = [
                base / m
                for m in sorted(_glob_mod.glob(pattern, root_dir=base, recursive=True))
            ]
    except Exception as exc:
        return (
            [],
            base,
            err(
                f"Invalid glob: {exc}",
                retriable=True,
                hint=_pattern_hint(pattern),
            ),
        )
    return candidates, base, None


def _pattern_hint(pattern: str) -> str:
    """Return a concrete correction for a malformed glob."""
    stem = pattern.rstrip("/")
    if stem.endswith("/**") or stem == "**":
        return f"Try '{stem}/*': '**' matches directories, '**/*' matches what is in them."
    if "**" in stem:
        head, _, tail = stem.partition("**")
        if tail and not tail.startswith("/"):
            return f"Try '{head}**/*{tail}': '**' has to be a whole path segment."
    return "Use '*' within one segment and '**/*' to recurse into subdirectories."


# ``fs`` implements a small, workspace-scoped subset of POSIX commands.

#: Command names this tool implements; anything else is refused by name.
_FS_SOURCES = ("ls", "find", "grep", "tree")
_FS_FILTERS = ("head", "tail", "wc", "sort", "uniq", "grep")

#: Unsupported shell tokens and their diagnostic labels.
_FS_UNSUPPORTED = {
    ">": "redirection",
    ">>": "redirection",
    "<": "redirection",
    "&&": "command chaining",
    "||": "command chaining",
    ";": "command chaining",
    "&": "backgrounding",
    "$(": "command substitution",
    "`": "command substitution",
}

#: Chat-template tokens rejected before parsing.
_CONTROL_MARKERS = (
    "<|",
    "|>",
    "<tool_call",
    "</tool_call",
    "<channel",
    "<im_start",
    "<im_end",
    "<|endoftext",
    "<start_of_turn",
    "<end_of_turn",
)

#: Maximum command length.
_MAX_COMMAND_CHARS = 2_000

#: Maximum filesystem entries visited by one command.
_SCAN_CEILING = 50_000

#: Largest file read by grep.
_MAX_GREP_BYTES = 5_000_000

#: Default ``tree`` depth.
_TREE_DEFAULT_DEPTH = 2


class _FsError(Exception):
    """Internal exception carrying a tool-result payload."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__(str(payload.get("error", "")))
        self.payload = payload


def _bad(message: str, hint: str, *, retriable: bool = True) -> _FsError:
    """Build an argument error for the tool response."""
    return _FsError(err(message, retriable=retriable, hint=hint))


def _safe(token: str, limit: int = 48) -> str:
    """Sanitize and clip a token before including it in an error."""
    clean = "".join(ch for ch in token if ch.isprintable())
    return clean if len(clean) <= limit else "…" + clean[-limit:]


def _reject_control_tokens(command: str) -> None:
    """Reject oversized, multiline, or chat-template-contaminated commands."""
    if len(command) > _MAX_COMMAND_CHARS:
        raise _bad(
            f"Command is {len(command)} characters; the limit is {_MAX_COMMAND_CHARS}.",
            "Send one short command line, e.g. find src -name '*.py' | head -20",
        )
    lowered = command.lower()
    for marker in _CONTROL_MARKERS:
        if marker in lowered:
            raise _bad(
                "The command contains chat-template control tokens, so the tool call was "
                "not framed correctly: the argument holds part of the reply rather than "
                "a command.",
                "Send only the command line and nothing else, e.g. "
                "grep -rn 'class Agent' src",
                retriable=True,
            )
    if any(ch in command for ch in "\r\n\x00"):
        raise _bad(
            "The command contains a newline; fs takes a single line.",
            "Use one command with pipes, e.g. find . -name '*.py' | head -20",
        )


def _tokenize(command: str) -> List[str]:
    """Tokenize POSIX-style quotes while preserving Windows backslashes."""
    lexer = shlex.shlex(command, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    lexer.commenters = ""
    try:
        return [token for token in lexer if token]
    except ValueError as exc:
        raise _bad(
            f"Could not parse the command: {exc}.",
            "Check for an unclosed quote. Example: grep -rn 'def run' src",
        ) from exc


def _split_pipeline(tokens: List[str]) -> List[List[str]]:
    """Split on ``|`` into stages, refusing shell syntax this tool does not implement."""
    for token in tokens:
        for bad_token, what in _FS_UNSUPPORTED.items():
            if token == bad_token or (
                len(bad_token) > 1 and token.startswith(bad_token)
            ):
                raise _bad(
                    f"fs does not support {what} ({_safe(bad_token)!r}).",
                    "fs runs one pipeline of "
                    f"{', '.join(_FS_SOURCES)} through {', '.join(_FS_FILTERS)}. "
                    "For anything else, use the shell tool.",
                )

    stages: List[List[str]] = []
    current: List[str] = []
    for token in tokens:
        if token == "|":
            stages.append(current)
            current = []
        else:
            current.append(token)
    stages.append(current)

    if any(not stage for stage in stages):
        raise _bad(
            "A pipeline stage is empty.",
            "Every '|' needs a command on both sides, e.g. find . -type f | head -20",
        )
    return stages


def _value_for(argv: List[str], index: int, flag: str, command: str) -> str:
    """The argument following *flag*, or a failure naming what it wanted."""
    if index + 1 >= len(argv):
        raise _bad(
            f"{command}: {flag} needs a value.",
            f"Example: {command} . {flag} "
            + ("'*.py'" if flag in ("-name", "-iname", "-path") else "2"),
        )
    return argv[index + 1]


def _positive_int(raw: str, flag: str, command: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise _bad(
            f"{command}: {flag} expects a number, got {_safe(raw)!r}.",
            f"Example: {command} . {flag} 2",
        ) from None
    if value < 1:
        raise _bad(
            f"{command}: {flag} must be at least 1.", f"Example: {command} . {flag} 2"
        )
    return value


def _expand_short_flags(token: str, known: str, command: str) -> List[str]:
    """Expand clustered short flags and reject unknown options."""
    out: List[str] = []
    for char in token[1:]:
        if char not in known:
            raise _bad(
                f"{command}: unsupported option '-{_safe(char, 4)}'.",
                f"{command} supports: "
                + " ".join(f"-{c}" for c in known)
                + ". Use the shell tool for anything else.",
            )
        out.append(f"-{char}")
    return out


def _resolve_start(space: FileSpace, raw: str) -> Tuple[Path, Optional[Path], str]:
    """Resolve a command's path argument to ``(target, display_base, root)``.

    Every source command uses this resolver and the shared containment check.
    """
    raw = (raw or ".").strip()
    route = space.resolve_glob(raw)
    if route.error is not None:
        raise _FsError(dict(route.error))

    candidate = Path(raw).expanduser()
    try:
        target = (
            candidate.resolve()
            if candidate.is_absolute()
            else (space.workspace() / raw).resolve()
        )
    except (OSError, ValueError):
        raise _bad(
            f"Path {_safe(raw)!r} could not be resolved.",
            "Use a path relative to the workspace, e.g. 'src' or '.'.",
        ) from None

    if is_denied(target):
        raise _FsError(
            err(
                f"Access to {_safe(raw)!r} is refused: it matches the sensitive-path "
                "denylist (credentials, private keys, .env and similar). This boundary "
                "cannot be approved around.",
                retriable=False,
            )
        )

    # An absolute path in a free zone carries that zone as its base; outside every zone it is `external`.
    if route.base is not None and not _within(target, route.base):
        raise _bad(
            f"Path {_safe(raw)!r} resolves outside the workspace.",
            "Use a path relative to the workspace, or an absolute path the user has "
            "approved.",
            retriable=False,
        )

    if not target.exists():
        raise _bad(
            f"Path {_safe(raw)!r} does not exist.",
            "Run 'tree . -L 2' to see what is here.",
        )
    return target, route.base, route.root


def _skip_name(name: str) -> bool:
    return path_is_hidden_or_pycache((name,))


def _tick(meta: Optional[Dict[str, Any]]) -> bool:
    """Count a visited entry and report when the scan limit is exceeded."""
    if meta is None:
        return False
    meta["scanned"] = meta.get("scanned", 0) + 1
    if meta["scanned"] > _SCAN_CEILING:
        meta["hit_ceiling"] = True
        return True
    return False


def _walk(
    start: Path,
    *,
    maxdepth: Optional[int] = None,
    files: bool = True,
    dirs: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> Iterator[Path]:
    """Walk visible, allowed paths under *start* within depth and scan limits."""
    if start.is_file():
        if not is_denied(start) and files:
            if _tick(meta):
                return
            yield start
        return

    for dirpath, dirnames, filenames in os.walk(start, topdown=True):
        here = Path(dirpath)
        try:
            depth = len(here.relative_to(start).parts)
        except ValueError:  # pragma: no cover: os.walk stays under start
            continue

        dirnames[:] = sorted(d for d in dirnames if not _skip_name(d))
        if maxdepth is not None and depth + 1 >= maxdepth:
            dirnames[:] = []

        if dirs:
            for name in dirnames:
                if _tick(meta):
                    return
                yield here / name
        if files:
            for name in sorted(filenames):
                if _skip_name(name):
                    continue
                candidate = here / name
                if is_denied(candidate):
                    continue
                if _tick(meta):
                    return
                yield candidate


@dataclass(frozen=True, slots=True)
class _LsOptions:
    pattern: str = "."
    recursive: bool = False
    show_hidden: bool = False


def _parse_ls(argv: List[str]) -> _LsOptions:
    """Parse ``ls [-R] [-a] [PATH]``."""
    recursive = False
    show_hidden = False
    raw_path: Optional[str] = None

    for token in argv:
        if token.startswith("-") and len(token) > 1:
            for flag in _expand_short_flags(token, "Ra1", "ls"):
                if flag == "-R":
                    recursive = True
                elif flag == "-a":
                    show_hidden = True
            continue
        if raw_path is not None:
            raise _bad("ls takes one path.", "Example: ls src")
        raw_path = token
    return _LsOptions(raw_path or ".", recursive, show_hidden)


def _source_ls(
    space: FileSpace, argv: List[str], meta: Dict[str, Any]
) -> Iterator[str]:
    """List a path; wildcard paths are globbed like ``ls src/*.py``."""
    options = _parse_ls(argv)

    pattern = options.pattern
    if has_glob(pattern):
        route = space.resolve_glob(pattern)
        if route.error is not None:
            raise _FsError(dict(route.error))
        candidates, base, glob_err = _glob_candidates(pattern, route)
        if glob_err is not None:
            raise _FsError(dict(glob_err))
        meta["root"] = route.root
        # Distinguishes "glob matched nothing" from "directory is empty" for the empty-result hint.
        meta["glob"] = pattern
        for candidate in candidates:
            if is_denied(candidate):
                continue
            shown = display_path(candidate, base)
            if not options.show_hidden and path_is_hidden_or_pycache(Path(shown).parts):
                continue
            yield shown + ("/" if candidate.is_dir() else "")
        return

    start, base, root = _resolve_start(space, pattern)
    meta["root"] = root

    if start.is_file():
        yield display_path(start, base)
        return

    if options.recursive:
        for path in _walk(start, files=True, dirs=True, meta=meta):
            shown = display_path(path, base)
            yield shown + ("/" if path.is_dir() else "")
        return

    try:
        entries = sorted(start.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except (OSError, PermissionError) as exc:
        raise _bad(
            f"Could not list the directory: {exc.__class__.__name__}.", ""
        ) from exc
    for entry in entries:
        if not options.show_hidden and _skip_name(entry.name):
            continue
        if is_denied(entry):
            continue
        yield entry.name + ("/" if entry.is_dir() else "")


@dataclass(frozen=True, slots=True)
class _FindOptions:
    path: str = "."
    name: Optional[str] = None
    iname: Optional[str] = None
    path_glob: Optional[str] = None
    path_ignore_case: bool = False
    kind: Optional[str] = None
    maxdepth: Optional[int] = None


def _parse_find(argv: List[str]) -> _FindOptions:
    """Parse find's supported checks without touching the filesystem."""
    raw_path: Optional[str] = None
    name: Optional[str] = None
    iname: Optional[str] = None
    path_glob: Optional[str] = None
    path_ignore_case = False
    kind: Optional[str] = None
    maxdepth: Optional[int] = None

    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("-name", "-iname", "-path", "-ipath", "-type", "-maxdepth"):
            value = _value_for(argv, index, token, "find")
            if token == "-name":
                name = value
            elif token == "-iname":
                iname = value
            elif token in ("-path", "-ipath"):
                path_glob = value
                path_ignore_case = token == "-ipath"
            elif token == "-type":
                if value not in ("f", "d"):
                    raise _bad(
                        f"find: -type takes f or d, got {_safe(value)!r}.",
                        "Example: find . -type f -name '*.py'",
                    )
                kind = value
            else:
                maxdepth = _positive_int(value, "-maxdepth", "find")
            index += 2
            continue
        if token.startswith("-"):
            raise _bad(
                f"find: unsupported option {_safe(token)!r}.",
                "find supports -name, -iname, -path, -type f|d and -maxdepth N.",
            )
        if raw_path is not None:
            raise _bad("find takes one path.", "Example: find src -name '*.py'")
        raw_path = token
        index += 1

    return _FindOptions(
        path=raw_path or ".",
        name=name,
        iname=iname,
        path_glob=path_glob,
        path_ignore_case=path_ignore_case,
        kind=kind,
        maxdepth=maxdepth,
    )


def _source_find(
    space: FileSpace, argv: List[str], meta: Dict[str, Any]
) -> Iterator[str]:
    """Find paths by name, path, kind, or depth."""
    options = _parse_find(argv)

    start, base, root = _resolve_start(space, options.path)
    meta["root"] = root
    meta["filters"] = bool(
        options.name or options.iname or options.path_glob or options.kind
    )

    want_files = options.kind != "d"
    want_dirs = options.kind != "f"
    for path in _walk(
        start,
        maxdepth=options.maxdepth,
        files=want_files,
        dirs=want_dirs,
        meta=meta,
    ):
        if options.name is not None and not fnmatchcase(path.name, options.name):
            continue
        if options.iname is not None and not fnmatch(
            path.name.lower(), options.iname.lower()
        ):
            continue
        shown = display_path(path, base)
        if options.path_glob is not None:
            candidate = shown.lower() if options.path_ignore_case else shown
            pattern = (
                options.path_glob.lower()
                if options.path_ignore_case
                else options.path_glob
            )
            if not fnmatch(candidate, pattern):
                continue
        yield shown + ("/" if want_dirs and path.is_dir() else "")


def _compile_grep(pattern: str, *, regex: bool, ignore_case: bool, word: bool):
    """Compile a literal or explicitly requested regular-expression matcher."""
    body = pattern if regex else re.escape(pattern)
    if word:
        body = rf"\b(?:{body})\b"
    try:
        return re.compile(body, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise _bad(
            f"Invalid regular expression: {exc}.",
            "Drop -E to search for the text literally, or fix the expression.",
        ) from exc


#: The short flags grep accepts, shared by the source and filter forms so the two
#: cannot drift on which flags exist. Behaviour per flag stays in each parser.
_GREP_SHORT_FLAGS = "rRnivwlcEFh"


@dataclass(frozen=True, slots=True)
class _GrepOptions:
    """The validated arguments needed to execute one grep source."""

    pattern: str
    paths: Tuple[str, ...]
    include: Optional[str] = None
    ignore_case: bool = False
    word: bool = False
    regex: bool = False
    files_only: bool = False
    counts: bool = False
    invert: bool = False


def _parse_grep(argv: List[str]) -> _GrepOptions:
    """Parse grep's supported flags without touching the filesystem."""
    flags = {
        "ignore_case": False,
        "word": False,
        "regex": False,
        "files_only": False,
        "counts": False,
        "invert": False,
    }
    flag_names = {
        "-i": "ignore_case",
        "-w": "word",
        "-E": "regex",
        "-l": "files_only",
        "-c": "counts",
        "-v": "invert",
    }
    include: Optional[str] = None
    pattern: Optional[str] = None
    paths: List[str] = []

    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--include="):
            include = token.split("=", 1)[1].strip("'\"")
        elif token == "-e":
            pattern = _value_for(argv, index, "-e", "grep")
            index += 1
        elif token.startswith("-") and len(token) > 1 and not token.startswith("--"):
            for flag in _expand_short_flags(token, _GREP_SHORT_FLAGS, "grep"):
                name = flag_names.get(flag)
                if name is not None:
                    flags[name] = True
                # -r/-R/-n/-F/-h are accepted no-ops: already the defaults.
        elif pattern is None:
            pattern = token
        else:
            paths.append(token)
        index += 1

    if not pattern:
        raise _bad(
            "grep needs a pattern.",
            "Example: grep -rn 'class Agent' src",
        )
    return _GrepOptions(
        pattern=pattern,
        paths=tuple(paths),
        include=include,
        **flags,
    )


def _read_grep_text(path: Path) -> Optional[str]:
    """Read a searchable UTF-8 file, or return ``None`` for data/binary files."""
    try:
        if path.stat().st_size > _MAX_GREP_BYTES:
            return None
        return path.read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _grep_output(
    text: str,
    shown: str,
    matcher: Any,
    options: _GrepOptions,
) -> Iterator[str]:
    """Render matches from one decoded file in the requested grep mode."""
    matches = (
        (number, line)
        for number, line in enumerate(text.splitlines(), 1)
        if bool(matcher.search(line)) != options.invert
    )
    if options.files_only:
        if next(matches, None) is not None:
            yield shown
        return
    if options.counts:
        total = sum(1 for _ in matches)
        if total:
            yield f"{shown}:{total}"
        return
    for number, line in matches:
        yield f"{shown}:{number}:{line.rstrip()[:200]}"


def _source_grep(
    space: FileSpace, argv: List[str], meta: Dict[str, Any]
) -> Iterator[str]:
    """Run the supported grep subset and yield ``path:line:text`` rows."""
    options = _parse_grep(argv)
    matcher = _compile_grep(
        options.pattern,
        regex=options.regex,
        ignore_case=options.ignore_case,
        word=options.word,
    )
    meta["query"] = _safe(options.pattern)
    meta["files_scanned"] = 0
    meta["ignore_case"] = options.ignore_case
    meta["word"] = options.word

    for raw_path in options.paths or (".",):
        start, base, root = _resolve_start(space, raw_path)
        if root != "workspace":
            meta["root"] = root
        for path in _walk(start, files=True, dirs=False, meta=meta):
            shown = display_path(path, base)
            if options.include is not None and not (
                fnmatch(path.name, options.include) or fnmatch(shown, options.include)
            ):
                continue
            text = _read_grep_text(path)
            if text is None:
                continue
            meta["files_scanned"] += 1
            yield from _grep_output(text, shown, matcher, options)


@dataclass(frozen=True, slots=True)
class _TreeOptions:
    """The validated path and depth for one tree command."""

    path: str = "."
    depth: int = _TREE_DEFAULT_DEPTH


def _parse_tree(argv: List[str]) -> _TreeOptions:
    """Parse ``tree [PATH] [-L N]`` without touching the filesystem."""
    raw_path: Optional[str] = None
    depth = _TREE_DEFAULT_DEPTH

    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("-L", "-d", "-a"):
            if token == "-L":
                depth = _positive_int(
                    _value_for(argv, index, "-L", "tree"), "-L", "tree"
                )
                index += 2
                continue
            index += 1
            continue
        if token.startswith("-"):
            raise _bad(
                f"tree: unsupported option {_safe(token)!r}.",
                "tree supports -L N for depth. Example: tree src -L 2",
            )
        if raw_path is not None:
            raise _bad("tree takes one path.", "Example: tree src -L 2")
        raw_path = token
        index += 1

    return _TreeOptions(raw_path or ".", depth)


def _source_tree(
    space: FileSpace, argv: List[str], meta: Dict[str, Any]
) -> Iterator[str]:
    """``tree [PATH] [-L N]``: an indented outline, for orienting in one call."""
    options = _parse_tree(argv)

    start, base, root = _resolve_start(space, options.path)
    meta["root"] = root
    meta["depth"] = options.depth

    def descend(directory: Path, level: int, indent: str) -> Iterator[str]:
        if level > options.depth:
            return
        try:
            entries = sorted(
                directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
            )
        except (OSError, PermissionError):
            return
        for entry in entries:
            if _skip_name(entry.name) or is_denied(entry):
                continue
            if entry.is_dir():
                yield f"{indent}{entry.name}/"
                yield from descend(entry, level + 1, indent + "  ")
            else:
                yield f"{indent}{entry.name}"

    if start.is_file():
        yield display_path(start, base)
        return
    yield from descend(start, 1, "")


_SOURCES: Dict[str, Callable[[FileSpace, List[str], Dict[str, Any]], Iterator[str]]] = {
    "ls": _source_ls,
    "find": _source_find,
    "grep": _source_grep,
    "tree": _source_tree,
}


def _count_arg(argv: List[str], default: int, command: str) -> int:
    """``-n 20`` or the bare ``-20`` both mean twenty."""
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "-n":
            return _positive_int(_value_for(argv, index, "-n", command), "-n", command)
        if token.startswith("-") and token[1:].isdigit():
            return _positive_int(token[1:], "-n", command)
        if token.startswith("-"):
            raise _bad(
                f"{command}: unsupported option {_safe(token)!r}.",
                f"Example: {command} -n 20",
            )
        index += 1
    return default


def _filter_grep(argv: List[str]) -> Callable[[Iterable[str]], Iterator[str]]:
    """``grep`` reading the previous stage's lines instead of files."""
    ignore_case = word = regex = invert = counts = False
    pattern: Optional[str] = None

    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "-e":
            pattern = _value_for(argv, index, "-e", "grep")
            index += 2
            continue
        if token.startswith("-") and len(token) > 1:
            for flag in _expand_short_flags(token, _GREP_SHORT_FLAGS, "grep"):
                if flag == "-i":
                    ignore_case = True
                elif flag == "-w":
                    word = True
                elif flag == "-E":
                    regex = True
                elif flag == "-v":
                    invert = True
                elif flag == "-c":
                    counts = True
            index += 1
            continue
        if pattern is None:
            pattern = token
        index += 1

    if not pattern:
        raise _bad("grep needs a pattern.", "Example: find . -type f | grep test")
    matcher = _compile_grep(pattern, regex=regex, ignore_case=ignore_case, word=word)

    def run(lines: Iterable[str]) -> Iterator[str]:
        matched = (line for line in lines if bool(matcher.search(line)) != invert)
        # `-c` counts rather than lists: ignoring the flag would answer a different question.
        if counts:
            yield str(sum(1 for _ in matched))
            return
        yield from matched

    return run


def _build_filter(argv: List[str]) -> Callable[[Iterable[str]], Iterator[str]]:
    """One pipeline stage after the first."""
    command, rest = argv[0], argv[1:]

    if command == "head":
        count = _count_arg(rest, 10, "head")
        return lambda lines: islice(lines, count)

    if command == "tail":
        count = _count_arg(rest, 10, "tail")
        return lambda lines: iter(deque(lines, maxlen=count))

    if command == "wc":
        if rest and rest != ["-l"]:
            raise _bad(
                f"wc: unsupported option {_safe(rest[0])!r}.",
                "fs deals in lines, so only 'wc -l' is meaningful.",
            )
        return lambda lines: iter([str(sum(1 for _ in lines))])

    if command == "sort":
        unique = "-u" in rest
        reverse = "-r" in rest
        for token in rest:
            if token not in ("-u", "-r"):
                raise _bad(
                    f"sort: unsupported option {_safe(token)!r}.",
                    "sort supports -u (unique) and -r (reverse).",
                )

        def run_sort(lines: Iterable[str]) -> Iterator[str]:
            values = sorted(set(lines) if unique else lines, reverse=reverse)
            return iter(values)

        return run_sort

    if command == "uniq":
        with_counts = "-c" in rest
        for token in rest:
            if token not in ("-c",):
                raise _bad(
                    f"uniq: unsupported option {_safe(token)!r}.",
                    "uniq supports -c (prefix each line with its count).",
                )

        def run_uniq(lines: Iterable[str]) -> Iterator[str]:
            previous: Optional[str] = None
            run_length = 0
            for line in lines:
                if line == previous:
                    run_length += 1
                    continue
                if previous is not None:
                    yield f"{run_length:>7} {previous}" if with_counts else previous
                previous, run_length = line, 1
            if previous is not None:
                yield f"{run_length:>7} {previous}" if with_counts else previous

        return run_uniq

    if command == "grep":
        return _filter_grep(rest)

    raise _bad(
        f"{_safe(command)!r} cannot be used after a pipe.",
        f"Filters available: {', '.join(_FS_FILTERS)}. "
        "For anything else, use the shell tool.",
    )


def _empty_hint(source: str, meta: Dict[str, Any]) -> str:
    """Return a concrete follow-up command for an empty result."""
    if source == "grep":
        scanned = int(meta.get("files_scanned", 0) or 0)
        if not scanned:
            return (
                "No readable files under that path. Run 'tree . -L 2' to see what "
                "is there, then grep a directory that exists."
            )
        extras = []
        if not meta.get("ignore_case"):
            extras.append("-i to ignore case")
        if not meta.get("word"):
            extras.append("-w to match whole words only")
        return (
            f"Read {scanned} file{'' if scanned == 1 else 's'}; none contained "
            f"{meta.get('query', '')!r}. Try a shorter substring"
            + (", or " + " or ".join(extras) if extras else "")
            + "."
        )
    if source == "find":
        return (
            "Nothing matched. 'find . -type f | head -40' shows what is actually here, "
            "and -iname is case-insensitive."
            if meta.get("filters")
            else "The directory is empty. Run 'tree . -L 2' to see the layout."
        )
    if source == "ls":
        glob = meta.get("glob")
        if glob:
            # ls looks one level deep, so name the recursive command rather than "broaden it".
            stem = glob.replace("\\", "/").rsplit("/", 1)
            where, leaf = (stem[0], stem[1]) if len(stem) == 2 else (".", stem[0])
            return (
                f"No entry directly in '{where}' matches {_safe(leaf)!r}. ls looks at "
                f"one level only: use 'find {where} -name {leaf!r}' to search below it."
            )
        return "The directory is empty. Run 'tree . -L 2' to see the layout."
    return "Nothing to show. Run 'tree . -L 2' to see the layout."


@tool(
    group="files",
    expose=Expose.ALL,
    inject_context=True,
    description=(
        "Use this whenever you need to find out WHAT EXISTS or WHERE something is: a "
        "POSIX-style command line over the workspace, where one call can narrow what "
        "another produced. Reach for it before reading files to look for something.\n"
        "Commands: ls, find, grep, tree. After a '|': head, tail, wc, sort, uniq, grep.\n"
        "Examples:\n"
        "  tree src -L 2\n"
        "  find src -name '*.py' | head -20\n"
        "  grep -rn 'class Agent' src\n"
        "  grep -rl --include='*.py' 'def compact' src | head\n"
        "  find . -type d -maxdepth 2\n"
        "grep searches for the text literally; add -E for a regular expression, -i to "
        "ignore case, -w for whole words. Paths are relative to the workspace, or "
        "absolute: an absolute path outside it needs the user's approval. No "
        "redirection, no &&, no subshells and no other programs: use the shell tool "
        "for those."
    ),
    # Safe when every path the command line names is in scope; the check parses it since scope isn't one argument.
    approval="conditional",
    guard_role="recovery",
    truncation_hint="add '| head -20', or narrow with -name / -maxdepth / --include",
    keywords="ls find grep tree bash shell explore browse locate directory folder "
    "listing contents what exists where",
    parameters={
        "command": Str(
            "One command line, e.g. \"find src -name '*.py' | head -20\"."
        ),
        "max_lines": Int("", ge=1, le=500),
    },
)
def fs(
    ctx: CapabilityContext,
    command: str = "tree . -L 2",
    max_lines: int = 100,
    **kw,
) -> Dict:
    space = ctx.require(FileSpace)
    meta: Dict[str, Any] = {"root": "workspace"}

    try:
        _reject_control_tokens(command)
        stages = _split_pipeline(_tokenize(command))

        head, *tail_stages = stages
        source_name = head[0]
        if source_name not in _SOURCES:
            # A filter as the first stage is a different mistake from an unknown program: say which happened.
            if source_name in _FS_FILTERS:
                raise _bad(
                    f"{_safe(source_name)!r} filters what another command produced, so "
                    "it cannot start a pipeline.",
                    f"Put a source first, e.g. find . -type f | {_safe(source_name)}",
                )
            raise _bad(
                f"{_safe(source_name)!r} is not available in fs."
                + (
                    " It prints a file; use read_file for that."
                    if source_name in ("cat", "less", "more", "type")
                    else ""
                ),
                f"A pipeline starts with one of: {', '.join(_FS_SOURCES)}. "
                "For anything else, use the shell tool.",
            )

        lines: Iterable[str] = _SOURCES[source_name](space, head[1:], meta)
        for stage in tail_stages:
            lines = _build_filter(stage)(lines)

        kept: List[str] = []
        produced = 0
        for line in lines:
            produced += 1
            if len(kept) < max_lines:
                kept.append(line)
            if produced >= _SCAN_CEILING:
                break
        # The walk bounds itself against _SCAN_CEILING (entries walked); this also caps emitted lines.
        hit_ceiling = bool(meta.get("hit_ceiling")) or produced >= _SCAN_CEILING
    except _FsError as exc:
        # The parsed form, never the raw input: see :func:`_safe`.
        return exc.payload

    # As understood, not as written: the only echo of model input in this result, rebuilt from validated tokens.
    parsed = " | ".join(" ".join(_safe(token) for token in stage) for stage in stages)

    out: Dict[str, Any] = {
        "command": parsed,
        "lines": kept,
        "count": len(kept),
        "root": meta.get("root", "workspace"),
        "truncated": produced > len(kept),
    }
    if source_name == "grep":
        out["files_scanned"] = meta.get("files_scanned", 0)

    if not kept:
        out["hint"] = _empty_hint(source_name, meta)
    elif hit_ceiling:
        out["total_at_least"] = produced
        out["hint"] = (
            f"Stopped after walking {_SCAN_CEILING:,} entries, so this is a sample and "
            "the counts are not totals. Narrow with -maxdepth, a subdirectory, or "
            "--include."
        )
    elif produced > len(kept):
        out["total"] = produced
        out["hint"] = (
            f"{produced - len(kept):,} more line(s) matched. Add '| head -{max_lines}' "
            "to keep the first, '| tail' for the last, or narrow the command."
        )
    return out


def fs_paths(command: str) -> Optional[List[str]]:
    """Every path :func:`fs` would touch for *command*, or ``None`` if unknowable.

    Separate from execution because the approval gate must answer "is this
    in scope" before the body runs. ``None`` (parse failure, unknown stage)
    must be treated as not-free, so an unanticipated shape prompts rather
    than slipping through.
    """
    try:
        _reject_control_tokens(command)
        stages = _split_pipeline(_tokenize(command))
    except _FsError:
        return None

    source, *_ = stages
    name, argv = source[0], source[1:]
    if name not in _SOURCES:
        return None

    try:
        if name == "ls":
            return [_parse_ls(argv).pattern]
        if name == "find":
            return [_parse_find(argv).path]
        if name == "grep":
            return list(_parse_grep(argv).paths) or ["."]
        return [_parse_tree(argv).path]
    except _FsError:
        return None


def fs_call_is_free(space: FileSpace):
    """Check for ``safe_when``: does this ``fs`` call stay in a free zone?

    Fails closed: anything this cannot parse or model is treated as out of
    scope, so an unrecognised command shape prompts rather than bypassing.
    """

    def check(arguments: Any) -> bool:
        command = str((arguments or {}).get("command") or "").strip()
        if not command:
            return True  # the default is 'tree . -L 2'
        targets = fs_paths(command)
        if targets is None:
            return False
        return all(space.is_free(path, write=False) for path in targets)

    check.__name__ = "fs_is_free"
    # Lets the builder confirm `fs` takes this argument; without it, a missing path would look free.
    check.argument_key = "command"  # type: ignore[attr-defined]
    return check
