"""Parses a source file once and runs an ordered list of rules over it, collecting diagnostics."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable

from lamssi_packages.codecheck.diagnostics import Diagnostics

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuleContext:
    """What a rule is given besides the tree."""

    #: The original source, for rules that need text rather than structure.
    source: str
    #: Path being checked, when there is one. For messages only.
    path: str = ""
    #: Lookup tables the rule set was configured with, so rules stay pure functions.
    tables: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Rule(Protocol):
    """One check over a parsed module."""

    name: str

    def __call__(
        self, tree: ast.Module, ctx: RuleContext, diagnostics: Diagnostics
    ) -> None:
        """Inspect *tree* and record findings. Return value ignored."""
        ...


def rule(name: str) -> Callable[[Callable], Callable]:
    """Name a plain function so it satisfies :class:`Rule`."""

    def deco(fn: Callable) -> Callable:
        fn.name = name  # type: ignore[attr-defined]
        return fn

    return deco


def run_rules(
    source: str,
    rules: Sequence[Rule],
    *,
    path: str = "",
    tables: Optional[Mapping[str, Any]] = None,
) -> Diagnostics:
    """Parse *source* once and run every rule over it."""
    diagnostics = Diagnostics()

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        # Nothing else can run, so the line number is the most useful thing to hand back.
        diagnostics.error(
            f"Syntax error: {exc.msg}",
            line=exc.lineno or 0,
            rule="syntax",
            suggestion="Fix the syntax before anything else can be checked.",
        )
        return diagnostics

    ctx = RuleContext(source=source, path=path, tables=dict(tables or {}))
    for r in rules:
        rule_name = getattr(r, "name", getattr(r, "__name__", "rule"))
        try:
            r(tree, ctx, diagnostics)
        except Exception as exc:
            log.debug("rule %r raised: %s", rule_name, exc, exc_info=True)
            diagnostics.warn(
                f"The {rule_name!r} check could not run: {exc}",
                rule=rule_name,
                suggestion="This is a limitation of the checker, not necessarily of the code.",
            )
    return diagnostics


class RuleSet:
    """A named, ordered collection of rules with its lookup tables.

    Bundling tables with rules keeps rules generic: the same rule serves any
    host that supplies its own table.
    """

    __slots__ = ("name", "rules", "tables")

    def __init__(
        self,
        name: str,
        rules: Sequence[Rule],
        *,
        tables: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.name = name
        self.rules = tuple(rules)
        self.tables: Dict[str, Any] = dict(tables or {})

    def run(self, source: str, *, path: str = "") -> Diagnostics:
        return run_rules(source, self.rules, path=path, tables=self.tables)

    def __len__(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return f"<RuleSet {self.name!r} rules={len(self.rules)}>"


__all__ = ["Rule", "RuleContext", "RuleSet", "run_rules", "rule"]
