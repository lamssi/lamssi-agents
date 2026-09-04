"""Diagnostics returned when ``edit_file`` cannot match ``old_string``."""

from __future__ import annotations

import pytest

from lamssi_agents.features.files.write import _no_match_payload

FILE = """class StageAndPowerApp(ApplicationBase):
    @poll(device="Stages", method="get_positions", interval_ms=100)
    def update_pos(self, value):
        self.set("pos", text=str(value))
"""

#: The line in `FILE` every case below is aiming at.
TARGET = '    @poll(device="Stages", method="get_positions", interval_ms=100)'


def hint_for(old_string: str) -> str:
    payload = _no_match_payload(old_string, FILE)
    assert payload["error"] == "old_string not found in file"
    return payload.get("hint", "")


def test_an_escaped_quote_is_named_as_such():
    """Name an escaped-quote mismatch explicitly."""
    hint = hint_for(
        '    @poll(device=\\"Stages\\", method=\\"get_positions\\", interval_ms=100)'
    )

    assert "escaped its quotes" in hint
    assert "matched literally" in hint, "say why, or it reads as a style preference"


def test_indentation_is_named_as_such():
    """Whitespace is the other difference a reader cannot see."""
    hint = hint_for(
        '        @poll(device="Stages", method="get_positions", interval_ms=100)'
    )

    assert "whitespace" in hint
    assert "indentation" in hint


def test_a_real_difference_is_located():
    """A genuine edit mismatch reports its location and both values."""
    hint = hint_for('    @poll(device="Stages", method="get_speed", interval_ms=100)')

    assert "identical up to character" in hint
    assert "speed" in hint and "positions" in hint


@pytest.mark.parametrize(
    "old_string",
    [
        '    @poll(device=\\"Stages\\", method=\\"get_positions\\", interval_ms=100)',
        '        @poll(device="Stages", method="get_positions", interval_ms=100)',
        '    @poll(device="Stages", method="get_speed", interval_ms=100)',
    ],
)
def test_every_hint_still_quotes_both_lines(old_string: str):
    """Edit mismatch hints quote both requested and actual lines."""
    hint = hint_for(old_string)
    assert repr(TARGET) in hint, "the file's line has to be in the hint to be copied"
    assert repr(old_string) in hint, "and so does what was sent, to be compared"


def test_a_line_with_no_near_match_gets_no_invented_hint():
    """Omit a near-match hint when no reliable candidate exists."""
    payload = _no_match_payload("import tensorflow as tf", FILE)
    assert "hint" not in payload
