"""The provider is the only retry authority: bounded, capped, terminal-aware, visible."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lamssi_agents.providers import Message


class Boom(Exception):
    """A provider error, with a status/headers where a real one would have them."""

    def __init__(self, message, status=None, headers=None):
        super().__init__(message)
        if status is not None:
            self.status_code = status
        if headers is not None:
            self.headers = headers


def _provider():
    """A real LiteLLMModel with backoff zeroed so tests never sleep."""
    litellm = pytest.importorskip("litellm")  # noqa: F841
    from lamssi_agents.providers.litellm_provider import LiteLLMModel

    p = LiteLLMModel("gpt-4o")
    p._RETRY_BASE_DELAY = 0.0
    return p


# checks


def test_transient_and_terminal_checks():
    p = _provider()
    assert p._is_transient(Boom("Service Unavailable", 503))
    assert p._is_transient(Boom("rate_limit hit"))
    assert p._is_transient(ConnectionError("connection refused"))
    assert p._is_transient(Boom("API connection error"))
    assert not p._is_transient(Boom("Incorrect API key", 401))
    assert p._is_terminal(Boom("insufficient_quota", 429))
    # Terminal short-circuits, and a spent budget stops retrying.
    assert p._should_retry(Boom("x", 503), 1)
    assert not p._should_retry(Boom("insufficient_quota", 429), 1)
    assert not p._should_retry(Boom("x", 503), p._MAX_RETRIES)


# backoff honours the server


def test_retry_after_header_is_honoured_and_clamped():
    p = _provider()
    assert p._retry_delay(Boom("x", 429, headers={"retry-after": "12"}), 1) == 12.0
    assert (
        p._retry_delay(Boom("x", 429, headers={"retry-after": "99999"}), 1)
        == p._MAX_RETRY_DELAY
    )
    # No header: the (zeroed) linear backoff.
    assert p._retry_delay(Boom("x", 429), 1) == 0.0


# streaming surfaces each retry


class _FakeDelta:
    content = None
    tool_calls = None


class _FakeChoice:
    def __init__(self, finish_reason=None):
        self.delta = _FakeDelta()
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, finish_reason=None):
        self.choices = [_FakeChoice(finish_reason)]
        self.usage = None


class _FakeLitellm:
    """Raises a transient error *fail_times*, then yields one terminal chunk."""

    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.calls = 0

    def completion(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise Boom("Service Unavailable", 503)
        return iter([_FakeChunk(finish_reason="stop")])


def test_stream_retries_and_emits_a_delta_per_retry():
    p = _provider()
    fake = _FakeLitellm(fail_times=2)
    p._litellm = fake

    deltas = list(p.stream([Message(role="user", content="hi")]))
    types = [d.type for d in deltas]

    assert types.count("retrying") == 2
    assert "done" in types
    assert fake.calls == 3


def test_stream_stops_after_max_retries():
    p = _provider()
    fake = _FakeLitellm(fail_times=99)
    p._litellm = fake

    with pytest.raises(Boom):
        list(p.stream([Message(role="user", content="hi")]))
    assert fake.calls == p._MAX_RETRIES


def test_stream_retries_a_lazy_failure_before_output():
    """Some HTTP clients raise only when their returned iterator is advanced."""
    p = _provider()

    class LazyFailure:
        def __init__(self):
            self.calls = 0

        def completion(self, **kw):
            self.calls += 1
            call = self.calls

            def chunks():
                if call == 1:
                    raise ConnectionError("connection refused")
                yield _FakeChunk(finish_reason="stop")

            return chunks()

    fake = LazyFailure()
    p._litellm = fake

    deltas = list(p.stream([Message(role="user", content="hi")]))

    assert [delta.type for delta in deltas].count("retrying") == 1
    assert deltas[-1].type == "done"
    assert fake.calls == 2


def test_stream_does_not_replay_after_visible_output():
    """Retrying after text was emitted would duplicate part of the answer."""
    p = _provider()

    class MidstreamFailure:
        def __init__(self):
            self.calls = 0

        def completion(self, **kw):
            self.calls += 1

            def chunks():
                delta = SimpleNamespace(
                    content="visible",
                    tool_calls=None,
                    reasoning_content=None,
                    reasoning=None,
                    thinking_blocks=None,
                )
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=delta, finish_reason=None)],
                    usage=None,
                )
                raise ConnectionError("connection reset")

            return chunks()

    fake = MidstreamFailure()
    p._litellm = fake

    with pytest.raises(ConnectionError, match="reset"):
        list(p.stream([Message(role="user", content="hi")]))
    assert fake.calls == 1


def test_malformed_streamed_tool_arguments_are_never_treated_as_empty():
    p = _provider()
    fragment = SimpleNamespace(
        index=0,
        id="call-1",
        function=SimpleNamespace(name="dangerous", arguments="{not-json"),
    )
    delta = SimpleNamespace(
        content=None,
        tool_calls=[fragment],
        reasoning_content=None,
        reasoning=None,
        thinking_blocks=None,
    )
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason="tool_calls")],
        usage=None,
    )
    p._litellm = SimpleNamespace(completion=lambda **kwargs: iter([chunk]))

    with pytest.raises(ValueError, match="invalid JSON arguments"):
        list(p.stream([Message(role="user", content="hi")]))
