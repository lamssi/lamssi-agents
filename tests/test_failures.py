"""Provider-failure classification and bounded recovery behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from lamssi_agents import Agent, ApprovalPolicy, Files
from lamssi_agents.events import AgentEventType
from lamssi_agents.providers.failures import Recovery, classify
from lamssi_agents.providers.models import StreamDelta, Usage


def make_agent(*, project: Path, approval=None, **config) -> Agent:
    if approval is None:
        approval = ApprovalPolicy.allow_all()
    agent = Agent(approval=approval, features=[Files(project)])
    if config:
        agent._config = agent._config.merged(**config).normalised()
    return agent

class Boom(Exception):
    """A provider error, with a status where a real one would have it."""

    def __init__(self, message: str, status: int | None = None, headers: dict | None = None):
        super().__init__(message)
        if status is not None:
            self.status_code = status
        if headers is not None:
            self.headers = headers

# classification

@pytest.mark.parametrize("exc, reason, recovery", [
    # Transient: the provider already retried, so the loop stops
    (Boom("Rate limit reached for gpt-4o", 429), "rate_limit", Recovery.STOP),
    (Boom("429 Too Many Requests"), "rate_limit", Recovery.STOP),
    (Boom("Service Unavailable", 503), "overloaded", Recovery.STOP),
    (Boom("Internal server error", 500), "transport", Recovery.STOP),
    (Boom("Connection reset by peer"), "transport", Recovery.STOP),
    (Boom("Request timed out"), "transport", Recovery.STOP),

    # Recoverable by shrinking
    (Boom("This model's maximum context length is 8192 tokens", 400),
     "context_overflow", Recovery.COMPACT),
    (Boom("prompt is too long"), "context_overflow", Recovery.COMPACT),
    (Boom(
        'APIConnectionError: Engine protocol predict request returned 400: '
        '{"error":{"message":"request (23923 tokens) exceeds the available '
        'context size (16384 tokens)","type":"exceed_context_size_error",'
        '"n_prompt_tokens":23923,"n_ctx":16384}}'
    ), "context_overflow", Recovery.COMPACT),
    (Boom("Payload too large", 413), "payload_too_large", Recovery.COMPACT),

    # Not recoverable
    (Boom("Incorrect API key provided", 401), "auth", Recovery.STOP),
    (Boom("Forbidden", 403), "auth", Recovery.STOP),
    (Boom("You exceeded your current quota, please check billing", 402),
     "billing", Recovery.STOP),
    (Boom("The model `gpt-5-turbo` does not exist", 404),
     "model_not_found", Recovery.STOP),
    (Boom("Request blocked by content policy"), "content_policy", Recovery.STOP),
    (Boom("tools are not supported by this model"), "tools_unsupported", Recovery.STOP),
    (Boom("something nobody has ever seen"), "unknown", Recovery.STOP),
])
def test_classification(exc, reason, recovery):
    failure = classify(exc)
    assert failure.reason == reason
    assert failure.recovery is recovery

def test_context_overflow_beats_the_status_code():
    """Context-overflow text takes precedence over a generic HTTP 400."""
    failure = classify(Boom("maximum context length exceeded", 400))
    assert failure.recovery is Recovery.COMPACT

def test_a_quota_message_is_billing_not_a_rate_limit():
    """Both say "quota". Only one is fixed by waiting."""
    assert classify(Boom("insufficient_quota", 429)).reason == "billing"

def test_unknown_errors_are_not_optimistically_retried():
    """Guessing "probably transient" spends the user's quota failing identically."""
    failure = classify(Boom("kernel panic in the tokeniser"))
    assert failure.recovery is Recovery.STOP
    assert "not recognised as recoverable" in failure.hint

def test_the_message_is_stripped_of_library_noise():
    assert "litellm." not in classify(Boom("litellm.APIError: upstream said no")).message

def test_every_stop_carries_something_to_do_about_it():
    for exc in (Boom("Incorrect API key", 401), Boom("insufficient credits", 402),
                Boom("model not found", 404), Boom("content policy")):
        failure = classify(exc)
        assert failure.hint, f"{failure.reason} stops with no advice"

def test_classify_never_raises():
    """Failure classification tolerates exceptions with broken string conversion."""
    class Awkward(Exception):
        def __str__(self): raise RuntimeError("even __str__ is broken")

    failure = classify(Awkward())

    assert failure.reason, "an unreadable exception still needs a classification"
    assert "Awkward" in failure.message, (
        "with no readable message, the type name is the last useful identifier"
    )

def test_it_needs_no_provider_library():
    """Duck-typed on purpose: a host supplying its own provider gets this too."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import lamssi_agents.providers.failures as f; "
         "print('litellm' in sys.modules)"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.stdout.strip() == "False", proc.stdout + proc.stderr

# the loop acts on it

class Flaky:
    """Raises *exc* the first *fails* times, then answers."""

    model = name = "scripted"
    is_local = supports_tools = True
    reasoning_effort = None

    def __init__(self, exc, fails=1):
        self._exc, self._left = exc, fails
        self._usage = Usage()
        self.calls = 0

    def stream(self, messages, tools=None, **kw):
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise self._exc
        yield StreamDelta(type="text", text="recovered")
        yield StreamDelta(type="done", finish_reason="stop")

    def check_connectivity(self): return True, "scripted"

    @property
    def cumulative_usage(self): return self._usage

@pytest.fixture()
def project(tmp_path: Path) -> Path:
    return tmp_path

def test_a_transient_provider_error_is_not_retried_by_the_loop(project: Path):
    """Retry lives in the provider now; a provider that raises transient stops the loop."""
    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    agent.use_model(Flaky(Boom("Service Unavailable", 503)))
    answer = agent.chat("hi")
    assert agent.model.calls == 1
    assert "temporarily unavailable" in answer

def test_an_unrecoverable_failure_stops_at_once(project: Path):
    """An unrecoverable failure stops once and returns actionable guidance."""
    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    agent.use_model(Flaky(Boom("Incorrect API key provided", 401)))
    answer = agent.chat("hi")

    assert agent.model.calls == 1
    assert "rejected the credentials" in answer
    assert "ANTHROPIC_API_KEY" in answer

def test_a_safety_refusal_is_not_retried(project: Path):
    """The identical prompt is refused identically. Retrying only costs time."""
    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    agent.use_model(Flaky(Boom("blocked by content policy")))
    agent.chat("hi")
    assert agent.model.calls == 1


def test_a_bug_in_our_delta_handling_is_not_masked_as_a_provider_error(
    project: Path, monkeypatch
):
    """The loop classifies provider failures, not our own bugs: those still raise."""
    from lamssi_agents.history.tokens import TokenCalibrator

    def _boom(self, *args, **kwargs):
        raise RuntimeError("bug in our own delta handling")

    # Observing usage runs in the loop's delta handling, after the provider yields.
    monkeypatch.setattr(TokenCalibrator, "observe", _boom)

    class UsageModel:
        model = name = "scripted"
        is_local = supports_tools = True
        reasoning_effort = None

        def stream(self, messages, tools=None, **kw):
            yield StreamDelta(type="usage", usage=Usage(prompt_tokens=1, total_tokens=1))
            yield StreamDelta(type="done", finish_reason="stop")

        def check_connectivity(self):
            return True, "scripted"

        @property
        def cumulative_usage(self):
            return Usage()

    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    agent.use_model(UsageModel())

    with pytest.raises(RuntimeError, match="bug in our own delta handling"):
        agent.chat("hi")

def test_the_host_is_told_it_is_recovering(project: Path):
    """The COMPACT recovery emits RECOVERING, distinct from ERROR (run ended)."""
    from lamssi_agents.providers import Message

    seen = []
    agent = make_agent(
        project=project,
        approval=ApprovalPolicy.allow_all(),
        history_budget_tokens=80_000,
        max_turns=1,
    )
    for i in range(30):
        agent._conversation.append(Message(role="user", content=f"q{i} " + "x" * 800))
        agent._conversation.append(Message(role="assistant", content=f"a{i} " + "y" * 800))
    agent.use_model(Flaky(Boom("maximum context length is 8192 tokens", 400)))
    agent.add_event_listener(
        lambda e: seen.append((e.type, e.metadata.get("reason")))
        if e.type is AgentEventType.RECOVERING else None
    )
    agent.chat("hi")

    assert (AgentEventType.RECOVERING, "context_overflow") in seen

def test_a_context_overflow_shrinks_before_retrying(project: Path):
    """The recovery has to *do* something, or the retry sends identical bytes."""
    from lamssi_agents.providers import Message

    # Keep the local estimate under budget so provider recovery is exercised.
    agent = make_agent(
        project=project,
        approval=ApprovalPolicy.allow_all(),
        history_budget_tokens=80_000,
        max_turns=1,
    )
    for i in range(30):
        agent._conversation.append(Message(role="user", content=f"q{i} " + "x" * 800))
        agent._conversation.append(Message(role="assistant", content=f"a{i} " + "y" * 800))

    before = len(agent._conversation.history)
    assert before == 60
    agent.use_model(Flaky(Boom("maximum context length is 8192 tokens", 400)))
    answer = agent.chat("hi")

    assert answer == "recovered", "recovery must retry the same logical turn"
    assert agent.model.calls == 3, (
        "one failed request, one summary call, and one retry were expected"
    )
    assert len(agent._conversation.history) < before, "history was not compacted"

def test_lm_studio_overflow_after_a_large_tool_result_recovers(project: Path):
    """The observed LM Studio wording recovers through the selected summary strategy."""
    from lamssi_agents.history import summarise_only
    from lamssi_agents.history.compaction import unframe_summary
    from lamssi_agents.providers import ToolCall

    (project / "large.txt").write_text("z" * 45_000, encoding="utf-8")

    class ToolOverflow(Flaky):
        def __init__(self):
            super().__init__(Boom("unused"), fails=0)

        def stream(self, messages, tools=None, **kw):
            self.calls += 1
            if self.calls == 1:
                yield StreamDelta(
                    type="tool_call",
                    tool_call=ToolCall(
                        id="read-1", name="read_file", arguments={"path": "large.txt"}
                    ),
                )
                yield StreamDelta(type="done", finish_reason="tool_calls")
                return
            if self.calls == 2:
                raise Boom(
                    "APIConnectionError: Engine protocol predict request returned 400: "
                    'request (23923 tokens) exceeds the available context size '
                    '(16384 tokens); type=exceed_context_size_error; n_ctx=16384'
                )
            yield StreamDelta(type="text", text="recovered")
            yield StreamDelta(type="done", finish_reason="stop")

    agent = make_agent(
        project=project,
        approval=ApprovalPolicy.allow_all(),
        history_budget_tokens=80_000,
    )
    agent.compactor = summarise_only
    agent.use_model(ToolOverflow())

    assert agent.chat("read large.txt") == "recovered"
    assert agent.model.calls == 4, (
        "tool turn, overflow, summary call, then the recovered retry"
    )
    assert not any(m.role == "tool" for m in agent._conversation.history)
    assert not any("elided" in (m.content or "") for m in agent._conversation.history)
    assert any(
        unframe_summary(m) for m in agent._conversation.history if m.role == "user"
    )

def test_preflight_refuses_a_request_that_cannot_fit(project: Path):
    """Known impossible requests stop locally; the provider is never invoked."""
    model = Flaky(Boom("unused"), fails=0)
    model.context_window = 100
    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    agent.use_model(model)

    answer = agent.chat("hi")

    assert model.calls == 0
    assert "It was not sent" in answer
    assert "100-token context window" in answer

def test_an_overflow_that_cannot_shrink_is_not_retried(project: Path):
    """Nothing to remove means the retry would send the same request again."""
    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    agent.use_model(Flaky(Boom("context_length_exceeded", 400)))
    agent.chat("hi")
    assert agent.model.calls == 1


def test_a_second_overflow_is_not_compacted_and_retried_again(project: Path):
    """Recovery is one summary pass and one retry, not an unbounded loop."""
    from lamssi_agents.providers import Message

    class AlwaysOverflowsMainRequest(Flaky):
        def __init__(self):
            super().__init__(Boom("unused"), fails=0)

        def stream(self, messages, tools=None, **kwargs):
            self.calls += 1
            if tools is None:  # the separate summary request
                yield StreamDelta(type="text", text="A compact record.")
                yield StreamDelta(type="done", finish_reason="stop")
                return
            raise Boom("maximum context length is 8192 tokens", 400)

    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    for i in range(30):
        agent._conversation.append(Message(role="user", content=f"q{i} " + "x" * 800))
        agent._conversation.append(Message(role="assistant", content=f"a{i} " + "y" * 800))
    agent.use_model(AlwaysOverflowsMainRequest())

    agent.chat("hi")

    assert agent.model.calls == 3, "failed request, one summary, one failed retry"

def test_a_retrying_delta_is_surfaced_as_recovering(project: Path):
    """A provider that retries emits a `retrying` delta; the loop just forwards it."""
    seen = []

    class Retrying(Flaky):
        def stream(self, messages, tools=None, **kw):
            self.calls += 1
            yield StreamDelta(type="retrying", retry={
                "attempt": 1, "max_retries": 5, "delay": 0, "reason": "rate limit",
            })
            yield StreamDelta(type="text", text="recovered")
            yield StreamDelta(type="done", finish_reason="stop")

    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    agent.use_model(Retrying(Boom("unused"), fails=0))
    agent.add_event_listener(
        lambda e: seen.append(e.type) if e.type is AgentEventType.RECOVERING else None
    )
    assert agent.chat("hi") == "recovered"
    assert AgentEventType.RECOVERING in seen

# Length-cut diagnostics

class Cut(Flaky):
    """Finishes on length with the usage that explains why."""

    def __init__(self, *, prompt, completion, reasoning=0, window=262144, text=""):
        super().__init__(Boom("unused"), fails=0)
        self._usage = Usage(
            prompt_tokens=prompt, completion_tokens=completion,
            total_tokens=prompt + completion, reasoning_tokens=reasoning,
        )
        self.context_window = window
        self._text = text

    def stream(self, messages, tools=None, **kw):
        self.calls += 1
        if self.calls > 1:                      # the continuation, if one is asked for
            yield StreamDelta(type="text", text="the rest")
            yield StreamDelta(type="done", finish_reason="stop")
            return
        if self._text:
            yield StreamDelta(type="text", text=self._text)
        yield StreamDelta(type="usage", usage=self._usage)
        yield StreamDelta(type="done", finish_reason="length")

    @property
    def last_usage(self): return self._usage

def test_a_spent_output_budget_is_not_blamed_on_the_context_window(project: Path):
    """The reported session: 8,192 of 8,192 output tokens, all reasoning, 14k in."""
    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    agent.use_model(Cut(prompt=14_086, completion=8_192, reasoning=8_192))

    answer = agent.chat("read them again")
    assert "8,192 on reasoning" in answer
    assert "max_tokens" in answer
    assert "the context window is not what ran out" in answer

def test_an_input_that_really_overran_still_says_so(project: Path):
    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    agent.use_model(Cut(prompt=258_000, completion=0))

    answer = agent.chat("hi")
    assert "leaves no room for a reply" in answer
    assert "258,000 tokens against a 262,144 window" in answer

def test_a_provider_that_reports_nothing_falls_back_to_blaming_the_input(project: Path):
    """The conservative reading: with no figures, assume the older diagnosis."""
    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    agent.use_model(Cut(prompt=0, completion=0, window=0))

    assert "leaves no room for a reply" in agent.chat("hi")

def test_a_truncated_reply_is_continued_and_reassembled(project: Path):
    """A length-cut reply is continued, and the final answer keeps both halves."""
    agent = make_agent(project=project, approval=ApprovalPolicy.allow_all())
    agent.use_model(Cut(prompt=100, completion=8_192, text="half an answ"))

    assert agent.chat("hi") == "half an answthe rest"
    assert agent.model.calls == 2
