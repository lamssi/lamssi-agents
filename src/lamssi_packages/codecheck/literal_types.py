"""Checks literal keyword-argument values in a call against a caller-supplied expected-type table."""

from __future__ import annotations

import ast
from typing import Mapping, Optional, Tuple, Union

from lamssi_packages.codecheck.diagnostics import Diagnostics

#: ``{(callable, argument): (human description, expected python type(s))}``
TypeTable = Mapping[Tuple[str, str], Tuple[str, Union[type, Tuple[type, ...]]]]

#: Literal node kinds worth judging. Anything else is a runtime value.
_LITERAL_NODES = (ast.Constant, ast.Dict, ast.List, ast.Set, ast.Tuple)


def check_literal_kwarg(
    callable_name: str,
    key: str,
    value_node: ast.expr,
    *,
    line: int,
    table: TypeTable,
    diagnostics: Diagnostics,
    rule: str = "literal-types",
) -> None:
    """Report a type mismatch for one keyword argument, if the type is known."""
    spec = table.get((callable_name, key))
    if spec is None:
        return
    if not isinstance(value_node, _LITERAL_NODES):
        return

    expected_desc, expected_type = spec
    actual_desc = _describe(value_node)
    if actual_desc is None:
        return

    if _matches(value_node, expected_type):
        return

    diagnostics.error(
        f"{callable_name}({key}=…) expects {expected_desc}, got {actual_desc}.",
        line=line,
        rule=rule,
        suggestion=f"Pass {expected_desc}.",
    )


def _matches(node: ast.expr, expected: Union[type, Tuple[type, ...]]) -> bool:
    expected_tuple = expected if isinstance(expected, tuple) else (expected,)

    if isinstance(node, ast.Constant):
        value = node.value
        # bool is a subclass of int, so isinstance alone would wrongly accept True as a number.
        if isinstance(value, bool) and bool not in expected_tuple:
            return False
        if isinstance(value, tuple(expected_tuple)):
            return True
        # int is acceptable where float is wanted; not the reverse, since truncation would be silent.
        return float in expected_tuple and isinstance(value, int) and not isinstance(value, bool)

    if isinstance(node, ast.Dict):
        return dict in expected_tuple
    if isinstance(node, ast.List):
        return list in expected_tuple
    if isinstance(node, ast.Set):
        return set in expected_tuple
    if isinstance(node, ast.Tuple):
        return tuple in expected_tuple or list in expected_tuple
    return True


def _describe(node: ast.expr) -> Optional[str]:
    """A short description of a literal, for the message."""
    if isinstance(node, ast.Constant):
        value = node.value
        return f"{type(value).__name__} ({value!r})"
    if isinstance(node, ast.Dict):
        return "a mapping"
    if isinstance(node, ast.List):
        return "a list"
    if isinstance(node, ast.Set):
        return "a set"
    if isinstance(node, ast.Tuple):
        return "a tuple"
    return None


__all__ = ["check_literal_kwarg", "TypeTable"]
