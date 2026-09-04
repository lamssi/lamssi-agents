"""Low-level, pure AST inspection helpers that rule sets build on."""

from __future__ import annotations

import ast
from typing import Iterator, List, Optional, Sequence, Set


def walk_body(body: List[ast.stmt]) -> Iterator[ast.AST]:
    """Walk every node inside a bare statement list, such as a function body.

    Wraps the ``ast.walk(ast.Module(...))`` idiom in one place.
    """
    return ast.walk(ast.Module(body=body, type_ignores=[]))


def string_arg(node: ast.expr) -> Optional[str]:
    """The value of a string literal, or ``None`` if *node* is not one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def int_arg(node: ast.expr) -> Optional[int]:
    """The value of an integer literal, or ``None``.

    Booleans are excluded: ``True`` is an ``int`` to Python, but rarely what's meant.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def receiver_method(node: ast.Call, receiver: str = "self") -> Optional[str]:
    """The method name if *node* is ``<receiver>.<method>(...)``, else ``None``."""
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == receiver
    ):
        return node.func.attr
    return None


def is_receiver_call(node: ast.Call, method: str, receiver: str = "self") -> bool:
    """Whether *node* is a call to one specific method on *receiver*."""
    return receiver_method(node, receiver) == method


def receiver_attr(node: Optional[ast.expr], receiver: str = "self") -> Optional[str]:
    """The attribute name if *node* is a bare ``<receiver>.<attr>`` reference.

    Useful when a callable is passed as a value rather than called: a
    call-shaped matcher would miss it.
    """
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == receiver
    ):
        return node.attr
    return None


def base_names(node: ast.ClassDef) -> Set[str]:
    """Every base class name, whether written bare or dotted.

    ``Foo`` and ``pkg.Foo`` both yield ``"Foo"``: a rule cares which class is being
    subclassed, not how the author chose to import it.
    """
    names: Set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def find_class_with_base(
    tree: ast.Module, bases: Sequence[str]
) -> Optional[ast.ClassDef]:
    """The class in *tree* subclassing any of *bases*.

    With several matches, the last one wins (a helper base is typically defined
    before the real class). Returns ``None`` with no match or an empty *bases*.
    """
    wanted = set(bases)
    if not wanted:
        return None
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and base_names(node) & wanted
    ]
    return candidates[-1] if candidates else None


def method_named(cls: ast.ClassDef, name: str) -> Optional[ast.FunctionDef]:
    """A method of *cls* by name, or ``None``."""
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node  # type: ignore[return-value]
    return None


def decorator_names(node: ast.AST) -> List[str]:
    """Every decorator on *node*, as a dotted string.

    ``@on.clicked("x")`` yields ``"on.clicked"``: the call is unwrapped so a rule can
    match the decorator regardless of its arguments.
    """
    out: List[str] = []
    for dec in getattr(node, "decorator_list", ()) or ():
        target = dec.func if isinstance(dec, ast.Call) else dec
        parts: List[str] = []
        while isinstance(target, ast.Attribute):
            parts.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            parts.append(target.id)
        if parts:
            out.append(".".join(reversed(parts)))
    return out


def call_kwargs(node: ast.Call) -> dict:
    """``{name: value_node}`` for the keyword arguments of *node*.

    ``**kwargs`` entries are skipped: their names aren't known statically.
    """
    return {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}


__all__ = [
    "walk_body",
    "string_arg",
    "int_arg",
    "receiver_method",
    "is_receiver_call",
    "receiver_attr",
    "base_names",
    "find_class_with_base",
    "method_named",
    "decorator_names",
    "call_kwargs",
]
