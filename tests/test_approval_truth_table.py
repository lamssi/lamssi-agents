"""The whole approval decision as one table, so a change to it has to be deliberate."""

from __future__ import annotations

import pytest

from lamssi_agents import ApprovalPolicy, ToolApproval
from lamssi_agents.tooling import needs_approval

_approve = lambda request: ToolApproval.APPROVE
ASK_ALL = ApprovalPolicy.ask_for_everything(_approve)
FOLLOW = ApprovalPolicy.ask_when_required(_approve)
ALLOW_ALL = ApprovalPolicy.allow_all()


class Definition:
    """The two fields the decision reads, without pulling in pydantic validation."""

    def __init__(self, approval="always", safe_when=None):
        self.name = "t"
        self.approval = approval
        self.safe_when = safe_when


#: ``(mode, declaration, arguments, needs human approval)`` cases.
TABLE = [
    (ASK_ALL, "never", {"action": "query"}, True),
    (ASK_ALL, "conditional", {"action": "query"}, True),
    (ASK_ALL, "conditional", {"action": "write"}, True),
    (ASK_ALL, "always", {"action": "query"}, True),
    (FOLLOW, "never", {"action": "query"}, False),
    (FOLLOW, "conditional", {"action": "query"}, False),
    (FOLLOW, "conditional", {"action": "write"}, True),
    (FOLLOW, "always", {"action": "query"}, True),
    (ALLOW_ALL, "never", {"action": "query"}, False),
    (ALLOW_ALL, "conditional", {"action": "query"}, False),
    (ALLOW_ALL, "conditional", {"action": "write"}, False),
    (ALLOW_ALL, "always", {"action": "query"}, False),
]


@pytest.mark.parametrize("mode, declaration, args, expected", TABLE)
def test_the_table(mode, declaration, args, expected):
    definition = Definition(declaration, safe_when={"action": ["query"]})
    assert needs_approval(mode, "t", args, definition) is expected


def test_the_table_covers_every_combination():
    """A declaration or a mode added without a row here would go untested."""
    from lamssi_agents.tooling import DECLARATIONS

    assert {row[0].name for row in TABLE} == {
        "ask-all",
        "ask-when-required",
        "allow-all",
    }
    assert {row[1] for row in TABLE} == set(DECLARATIONS)


def test_invalid_or_incomplete_safety_declarations_fail_closed():
    """Every malformed route reaches the same gate; report any route that opens."""
    cases = [
        ("undeclared", needs_approval(FOLLOW, "t", {}, Definition())),
        ("missing definition", needs_approval(FOLLOW, "vanished", {}, None)),
        (
            "unknown declaration",
            needs_approval(FOLLOW, "t", {}, Definition("sometimes")),
        ),
        (
            "conditional without a test",
            needs_approval(FOLLOW, "t", {"action": "query"}, Definition("conditional")),
        ),
        (
            "empty pattern",
            needs_approval(
                FOLLOW,
                "t",
                {"action": "query"},
                Definition("conditional", {}),
            ),
        ),
        (
            "missing argument",
            needs_approval(
                FOLLOW,
                "t",
                {},
                Definition("conditional", {"action": ["query"]}),
            ),
        ),
        (
            "raising check",
            needs_approval(
                FOLLOW,
                "t",
                {},
                Definition("conditional"),
                rules={"t": lambda args: 1 / 0},
            ),
        ),
    ]

    assert not [label for label, gated in cases if not gated]


def test_a_registered_check_replaces_the_declared_pattern():
    """Overriding a tool you do not own is the reason to register one."""
    definition = Definition("conditional", safe_when={"action": ["query"]})
    rules = {"t": lambda args: args.get("action") == "write"}

    assert needs_approval(FOLLOW, "t", {"action": "query"}, definition, rules=rules)
    assert not needs_approval(FOLLOW, "t", {"action": "write"}, definition, rules=rules)


def test_a_pattern_accepts_any_of_a_list():
    definition = Definition("conditional", safe_when={"action": ["query", "inspect"]})
    assert not needs_approval(FOLLOW, "t", {"action": "inspect"}, definition)


def test_every_key_in_a_pattern_must_match():
    definition = Definition(
        "conditional", safe_when={"action": "query", "dry_run": True}
    )
    assert needs_approval(FOLLOW, "t", {"action": "query"}, definition)
    assert not needs_approval(
        FOLLOW, "t", {"action": "query", "dry_run": True}, definition
    )


def test_a_truthy_int_does_not_satisfy_a_boolean_pattern():
    """Python's ``True == 1`` would let a pattern meant for one meaning through."""
    definition = Definition("conditional", safe_when={"dry_run": True})
    assert needs_approval(FOLLOW, "t", {"dry_run": 1}, definition)
    assert not needs_approval(FOLLOW, "t", {"dry_run": True}, definition)


def test_a_quoted_scalar_still_matches():
    """Model-supplied JSON is inconsistent about quoting, and 1 means "1"."""
    definition = Definition("conditional", safe_when={"level": 1})
    assert not needs_approval(FOLLOW, "t", {"level": "1"}, definition)
