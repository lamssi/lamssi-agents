"""Token-estimate calibration from provider usage records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lamssi_agents import Agent, ApprovalPolicy
from lamssi_agents.history.compaction import compress_history
from lamssi_agents.history.tokens import estimate_tokens
from lamssi_agents.history.tokens import (
    DEFAULT_CHARS_PER_TOKEN,
    TokenCalibrator,
)
from lamssi_agents.providers import Message
from lamssi_agents.providers.models import StreamDelta, Usage


def make_agent(project: Path) -> Agent:
    return Agent(approval=ApprovalPolicy.allow_all())

# the calibrator itself

def test_it_starts_pessimistic():
    """The initial token estimate favors early compaction."""
    calibrator = TokenCalibrator()
    assert calibrator.ratio == DEFAULT_CHARS_PER_TOKEN
    assert not calibrator.calibrated
    assert DEFAULT_CHARS_PER_TOKEN < 3.5, "the old constant was the optimistic one"

def test_one_sample_replaces_the_guess_outright():
    """Any real reading beats a default. No reason to average with a number nobody
    measured."""
    calibrator = TokenCalibrator()
    calibrator.observe(chars=10_000, prompt_tokens=5_000)
    assert calibrator.ratio == pytest.approx(2.0)
    assert calibrator.calibrated

def test_later_samples_are_blended_not_replaced():
    """One odd turn: a burst of dense JSON: should not swing the whole estimate."""
    calibrator = TokenCalibrator()
    calibrator.observe(10_000, 5_000)          # 2.0
    calibrator.observe(10_000, 2_500)          # 4.0
    assert 2.0 < calibrator.ratio < 3.0, "should move toward 4.0, not jump to it"

def test_it_converges_on_the_true_ratio():
    calibrator = TokenCalibrator()
    for _ in range(12):
        calibrator.observe(chars=23_000, prompt_tokens=10_000)   # 2.3
    assert calibrator.ratio == pytest.approx(2.3, abs=0.05)

@pytest.mark.parametrize("chars, tokens", [
    (10_000, 20_000),      # 0.5: below MIN_RATIO
    (10_000, 500),         # 20.0: above MAX_RATIO
])
def test_an_implausible_reading_is_ignored(chars, tokens):
    """Calibration ignores token ratios outside the plausible range."""
    calibrator = TokenCalibrator()
    calibrator.observe(chars, tokens)
    assert calibrator.ratio == DEFAULT_CHARS_PER_TOKEN
    assert not calibrator.calibrated

def test_a_provider_reporting_nothing_cannot_break_it():
    calibrator = TokenCalibrator()
    calibrator.observe(10_000, 0)
    calibrator.observe(10_000, -5)
    assert calibrator.ratio == DEFAULT_CHARS_PER_TOKEN

def test_a_tiny_request_is_not_a_useful_sample():
    """Mostly per-request overhead: role markers, envelopes: not content."""
    calibrator = TokenCalibrator()
    calibrator.observe(chars=50, prompt_tokens=40)
    assert not calibrator.calibrated

def test_the_last_charge_is_available_for_reporting():
    calibrator = TokenCalibrator()
    assert calibrator.last_reported_tokens is None
    calibrator.observe(10_000, 4_000)
    assert calibrator.last_reported_tokens == 4_000

# it changes what compaction decides

def _history(n: int, chars: int):
    return [Message(role="user", content="x" * chars) for _ in range(n)]

def test_the_estimate_follows_the_calibrated_ratio():
    history = _history(10, 1_000)          # 10,000 chars

    optimistic = TokenCalibrator(initial=4.0)
    pessimistic = TokenCalibrator(initial=2.0)

    assert estimate_tokens(history, optimistic) == 2_500
    assert estimate_tokens(history, pessimistic) == 5_000

def test_a_low_estimate_is_what_makes_compaction_fire_late():
    """An optimistic token estimate delays compaction."""
    history = _history(30, 1_000)          # 30,000 chars

    generous = TokenCalibrator(initial=4.0)     # says 7,500 tokens
    truthful = TokenCalibrator(initial=2.0)     # says 15,000 tokens

    unchanged = compress_history(
        history, model=None, budget_tokens=10_000, keep_recent=6, calibrator=generous)
    compacted = compress_history(
        history, model=None, budget_tokens=10_000, keep_recent=6, calibrator=truthful)

    assert unchanged is history, "an optimistic ratio sails past the budget"
    assert len(compacted) < len(history), "an accurate one acts"

def test_the_tool_schema_counts_toward_the_budget():
    """~3,900 tokens for a default surface. Leaving it out was that much of a lie."""
    history = _history(5, 1_000)
    calibrator = TokenCalibrator(initial=3.0)

    without = compress_history(
        history, model=None, budget_tokens=2_000, keep_recent=3,
        calibrator=calibrator)
    with_schema = compress_history(
        history, model=None, budget_tokens=2_000, keep_recent=3,
        overhead_chars=14_000, calibrator=calibrator)

    assert without is history, "5,000 chars is inside a 2,000-token budget at 3.0"
    assert len(with_schema) < len(history), "adding the schema should tip it over"

# end to end: it learns from a real turn

class Charging:
    """A provider that charges for everything it receives, as real ones do."""

    model = name = "scripted"
    is_local = supports_tools = True
    reasoning_effort = None

    def __init__(self, chars_per_token: float):
        self._ratio = chars_per_token
        self._usage = Usage()

    def stream(self, messages, tools=None, **kw):
        chars = sum(len(m.content or "") for m in messages)
        chars += len(json.dumps(tools, separators=(",", ":"))) if tools else 0
        yield StreamDelta(type="text", text="ok")
        yield StreamDelta(type="usage",
                          usage=Usage(prompt_tokens=int(chars / self._ratio)))
        yield StreamDelta(type="done", finish_reason="stop")

    def check_connectivity(self): return True, "scripted"

    @property
    def cumulative_usage(self): return self._usage

@pytest.fixture()
def project(tmp_path: Path) -> Path:
    return tmp_path

def test_a_conversation_learns_its_models_ratio(project: Path):
    agent = make_agent(project)
    agent.use_model(Charging(2.3))

    assert not agent._conversation.tokens.calibrated
    for _ in range(6):
        agent.chat("hello " * 100)

    assert agent._conversation.tokens.ratio == pytest.approx(2.3, abs=0.1)
    assert agent._conversation.tokens.last_reported_tokens > 0

def test_two_conversations_do_not_share_a_ratio(project: Path):
    """Another agent may be on a different model; one average over two describes neither."""
    a = make_agent(project)
    b = make_agent(project)
    a.use_model(Charging(2.0))
    b.use_model(Charging(4.5))

    for _ in range(6):
        a.chat("x" * 600)
        b.chat("x" * 600)

    assert a._conversation.tokens.ratio == pytest.approx(2.0, abs=0.15)
    assert b._conversation.tokens.ratio == pytest.approx(4.5, abs=0.2)
    assert a._conversation.tokens is not b._conversation.tokens

def test_a_provider_that_reports_no_usage_leaves_the_default(project: Path):
    """Plenty of local servers report nothing. That must not corrupt the estimate."""
    class Silent(Charging):
        def stream(self, messages, tools=None, **kw):
            yield StreamDelta(type="text", text="ok")
            yield StreamDelta(type="done", finish_reason="stop")

    agent = make_agent(project)
    agent.use_model(Silent(2.3))
    agent.chat("hello")

    assert not agent._conversation.tokens.calibrated
    assert agent._conversation.tokens.ratio == DEFAULT_CHARS_PER_TOKEN
