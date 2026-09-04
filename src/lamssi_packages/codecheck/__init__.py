"""Static checks over generated source: machinery for running rule sets and accumulating diagnostics."""

from __future__ import annotations

from lamssi_packages.codecheck.ast_utils import (
    base_names,
    call_kwargs,
    decorator_names,
    find_class_with_base,
    int_arg,
    is_receiver_call,
    method_named,
    receiver_attr,
    receiver_method,
    string_arg,
    walk_body,
)
from lamssi_packages.codecheck.diagnostics import Diagnostics, Finding, Severity
from lamssi_packages.codecheck.literal_types import TypeTable, check_literal_kwarg
from lamssi_packages.codecheck.rules import Rule, RuleContext, RuleSet, rule, run_rules

__all__ = [
    "Rule", "RuleContext", "RuleSet", "run_rules", "rule",
    "Diagnostics", "Finding", "Severity",
    "walk_body", "string_arg", "int_arg",
    "receiver_method", "is_receiver_call", "receiver_attr",
    "base_names", "find_class_with_base", "method_named",
    "decorator_names", "call_kwargs",
    "check_literal_kwarg", "TypeTable",
]
