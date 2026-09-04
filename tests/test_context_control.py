"""Context admission, tool-result demotion, summarisation, and overflow reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lamssi_agents import Agent
from lamssi_agents import Files, Guidance, SystemTools
from lamssi_agents.history.compaction import compress_history
from lamssi_agents.history.demotion import demote_tool_results, largest_tool_results
from lamssi_agents.history.truncation import (
    DEFAULT_MAX_TOOL_RESULT_CHARS,
    clip_result,
    truncate_tool_result,
)
from lamssi_agents.providers import Message, StreamDelta, ToolCall
from lamssi_agents.runtime import AgentConfig
from _scope import run_scope_active

CHARS_PER_TOKEN = 3.5


def tool_msg(call_id: str, name: str, body: str) -> Message:
    return Message(role="tool", name=name, tool_call_id=call_id, content=body)


def call_msg(call_id: str, name: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments={})],
    )


def conversation(n_results: int, body_chars: int, name: str = "read_file"):
    history = [Message(role="user", content="do the thing")]
    for i in range(n_results):
        history.append(call_msg(str(i), name))
        history.append(
            tool_msg(
                str(i),
                name,
                json.dumps({"path": f"src/mod{i}.py", "content": "x" * body_chars}),
            )
        )
        history.append(Message(role="assistant", content="ok"))
    return history


def total_chars(history) -> int:
    return sum(len(m.content or "") for m in history)


def _pairing(messages) -> tuple:
    """Return the announced call ids and their answered ids, in order."""
    announced = [
        tc.id
        for m in messages
        if m.role == "assistant" and m.tool_calls
        for tc in m.tool_calls
    ]
    answered = [m.tool_call_id for m in messages if m.role == "tool"]
    return announced, answered


# Tool-result admission


def test_one_result_cannot_dominate_the_budget():
    """A single tool result stays below ten percent of the history budget."""
    config = AgentConfig()
    worst_case_tokens = config.max_tool_result_chars / CHARS_PER_TOKEN
    share = worst_case_tokens / config.history_budget_tokens

    assert share < 0.10, (
        f"one tool result can be {share:.0%} of the history budget "
        f"({config.max_tool_result_chars:,} chars vs "
        f"{config.history_budget_tokens:,} tokens): a cap larger than a tenth of the "
        f"budget makes compaction fire on ordinary work"
    )


def test_the_module_default_matches_the_config_default():
    """Two defaults for one setting is how they drift apart."""
    assert DEFAULT_MAX_TOOL_RESULT_CHARS == AgentConfig().max_tool_result_chars


def test_a_tool_may_declare_a_larger_cap(tmp_path: Path):
    """read_file's result is meant to be a document, so it says so."""
    runtime = _runtime(tmp_path)
    read_file = next(t for t in runtime._tools.list_tools() if t.name == "read_file")
    assert read_file.truncation and read_file.truncation > DEFAULT_MAX_TOOL_RESULT_CHARS


def test_truncation_still_clips_at_the_declared_cap():
    body = json.dumps({"data": "y" * 50_000})
    out = truncate_tool_result(body, tool_name="whatever", max_chars=1_000)
    assert len(out) < 1_400  # cap plus the explanatory marker
    assert "truncated" in out


class _Tail:
    truncation = None
    truncation_hint = ""
    truncation_side = "tail"


class _Head:
    truncation = None
    truncation_hint = ""
    truncation_side = "head"


def test_truncation_side_tail_keeps_the_end():
    """Command and log output keep their tail, where the error and exit status live."""
    text = "\n".join(f"line {i}" for i in range(2_000))
    out = clip_result(text, definition=_Tail(), default_max_chars=500)

    assert text.endswith(out[-100:]), "the end of the output must survive"
    assert "line 0\n" not in out, "the start was dropped"
    assert out.startswith("... [truncated to the last")


def test_truncation_side_head_keeps_the_start():
    text = "\n".join(f"line {i}" for i in range(2_000))
    out = clip_result(text, definition=_Head(), default_max_chars=500)

    assert text.startswith(out[:100]), "the start of the output must survive"
    assert "line 1999" not in out, "the end was dropped"
    assert "truncated to the first" in out


class _Declares:
    truncation = 24_000
    truncation_hint = "narrow the range"


def test_a_declared_cap_beats_the_runtime_default():
    """A tool-level result cap overrides the runtime default."""
    out = truncate_tool_result(
        "x" * 30_000,
        tool_name="read_file",
        definition=_Declares(),
        default_max_chars=8_000,
    )
    assert len(out) > 20_000, "the tool's own cap should apply"


def test_the_runtime_default_applies_when_a_tool_declares_nothing():
    out = truncate_tool_result(
        "x" * 30_000,
        tool_name="whatever",
        definition=None,
        default_max_chars=8_000,
    )
    assert 8_000 <= len(out) < 8_500


def test_an_explicit_cap_still_beats_the_declaration():
    """An explicit per-call cap overrides the tool declaration."""
    out = truncate_tool_result(
        "x" * 30_000,
        tool_name="read_file",
        definition=_Declares(),
        default_max_chars=8_000,
        max_chars=1_000,
    )
    assert len(out) < 1_400


def test_a_real_read_keeps_its_declared_cap_through_dispatch(tmp_path: Path):
    """Dispatch preserves ``read_file``'s declared result cap."""
    (tmp_path / "big.py").write_text("# line\n" * 2_500, encoding="utf-8")
    agent = _offline_agent(tmp_path)

    definition = agent._runtime.definition_for("read_file")
    assert definition.truncation > agent._config.max_tool_result_chars, "the premise"

    with run_scope_active(agent):
        batch = agent._runtime.execute_calls(
            [ToolCall(id="c1", name="read_file", arguments={"path": "big.py"})],
            agent._conversation.turn,
        )
    agent._conversation.extend(batch.messages)

    body = agent._conversation.history[-1].content
    assert len(body) > 15_000, (
        f"clipped to {len(body)}: the runtime default overrode the declared cap"
    )


def test_a_truncated_read_is_a_prefix_not_a_splice():
    """A truncated file result contains one contiguous prefix."""
    lines = [f"line {i}" for i in range(2_000)]
    content = "\n".join(lines)
    body = json.dumps({"content": content, "lines": len(lines), "path": "m.py"})

    kept = json.loads(
        truncate_tool_result(body, tool_name="read_file", max_chars=6_000)
    )["content"]

    head = kept.split("\n\n... [")[0]
    assert content.startswith(head), "the kept text is not a prefix of the file"
    assert lines[-1] not in kept, "the file's end was spliced onto its head"


def test_a_truncated_read_says_which_line_to_resume_from():
    content = "\n".join(f"line {i}" for i in range(2_000))
    body = json.dumps({"content": content, "path": "m.py"})

    envelope = json.loads(
        truncate_tool_result(body, tool_name="read_file", max_chars=6_000)
    )
    assert envelope["truncated_chars_remaining"] > 0
    assert (
        f"start_line={envelope['truncated_after_local_line'] + 1}"
        in envelope["truncation_hint"]
    ), "the hint must name the line the model has not read yet"


def test_a_module_sized_file_arrives_whole(tmp_path: Path):
    """A typical module-sized source file fits in one read result."""
    (tmp_path / "module.py").write_text(
        "# a line of source\n" * 1_850, encoding="utf-8"
    )
    agent = _offline_agent(tmp_path)

    with run_scope_active(agent):
        batch = agent._runtime.execute_calls(
            [ToolCall(id="c1", name="read_file", arguments={"path": "module.py"})],
            agent._conversation.turn,
        )
    agent._conversation.extend(batch.messages)
    assert "truncated_chars_remaining" not in (
        agent._conversation.history[-1].content or ""
    )


def test_dispatch_passes_the_config_as_a_default_not_an_override():
    """Dispatch passes the runtime cap through the default-only parameter."""
    import inspect

    from lamssi_agents import tool_runtime as dispatch

    source = inspect.getsource(dispatch)
    assert "default_max_chars=self._max_chars" in source
    assert "max_chars=self._max_chars" not in source.replace(
        "default_max_chars=self._max_chars", ""
    ), "a call site still passes the runtime default as a per-call override"


# Tool-result demotion


def test_demotion_stubs_old_results_and_keeps_recent_ones():
    history = conversation(10, 5_000)
    out, saved = demote_tool_results(history, keep_results=3)

    bodies = [m.content for m in out if m.role == "tool"]
    stubbed = [b for b in bodies if "elided" in b]
    intact = [b for b in bodies if "elided" not in b]

    assert len(stubbed) == 7
    assert len(intact) == 3, "the most recent results keep their bodies"
    assert saved > 30_000


def test_demotion_preserves_the_call_result_pairing():
    """Demotion preserves each tool call and its result message."""
    history = conversation(6, 5_000)
    out, _ = demote_tool_results(history, keep_results=1)

    announced, answered = _pairing(out)

    assert announced == answered, "every call must still have exactly one answer"
    assert len(out) == len(history), "no message may be dropped"


def test_a_stub_says_what_it_replaced():
    """A stub that only said 'elided' would make the model re-run the call blind."""
    history = conversation(2, 5_000)
    out, _ = demote_tool_results(history, keep_results=0)

    stub = json.loads(next(m.content for m in out if m.role == "tool"))
    assert stub["path"] == "src/mod0.py", "the identifier survives"
    assert "chars" in stub["elided"]
    assert "already ran" in stub["note"], "the model is told the call happened"


def test_a_stub_does_not_invite_a_re_read():
    """A demotion stub permits rereading without recommending it."""
    history = conversation(2, 5_000)
    out, _ = demote_tool_results(history, keep_results=0)

    note = json.loads(next(m.content for m in out if m.role == "tool"))["note"]
    assert "Call it again" not in note
    assert "only if" in note, "a re-run is conditional, not the default"


def test_demotion_is_idempotent():
    """A second pass must not stub the stub and lose the record of what went."""
    history = conversation(4, 5_000)
    once, saved_1 = demote_tool_results(history, keep_results=0)
    twice, saved_2 = demote_tool_results(once, keep_results=0)

    assert saved_2 == 0
    assert [m.content for m in once] == [m.content for m in twice]


def test_small_results_are_left_alone():
    """A stub costs ~80 chars; demoting a 100-char body spends more than it saves."""
    history = conversation(5, 20)
    out, saved = demote_tool_results(history, keep_results=0)
    assert saved == 0
    assert out is history, "an unchanged history should not even be rebuilt"


def test_demotion_does_not_mutate_the_input():
    """The caller's history may be the live transcript of a turn in flight."""
    history = conversation(4, 5_000)
    before = [m.content for m in history]
    demote_tool_results(history, keep_results=0)
    assert [m.content for m in history] == before


# Automatic compaction


def test_demotion_alone_avoids_the_summarisation_call():
    """The ordering that matters. Summarising costs a model call and the cache."""
    history = conversation(12, 4_000)
    out = compress_history(history, model=None, budget_tokens=10_000, keep_recent=24)

    assert not any("compacted summary" in (m.content or "") for m in out), (
        "summarisation ran when stubbing would have been enough"
    )
    assert total_chars(out) / CHARS_PER_TOKEN <= 10_000


def test_compaction_escalates_when_the_tail_is_the_weight():
    """Compaction demotes a recent tail that remains over budget."""
    history = conversation(20, 6_000)
    out = compress_history(history, model=None, budget_tokens=8_000, keep_recent=24)

    assert total_chars(out) / CHARS_PER_TOKEN <= 8_000, (
        "compaction returned still over budget"
    )


def test_advanced_compaction_demotes_a_result_larger_than_the_budget():
    """The ladder's final emergency step handles one dominant latest result."""
    history = conversation(1, 200_000)
    out = compress_history(history, model=None, budget_tokens=1_000, keep_recent=24)

    result = next(message for message in out if message.role == "tool")
    assert "elided" in result.content
    assert total_chars(out) / CHARS_PER_TOKEN <= 1_000


def test_an_under_budget_history_is_returned_untouched():
    """The common case must cost nothing: identity, not a rebuild."""
    history = conversation(2, 100)
    assert (
        compress_history(history, model=None, budget_tokens=100_000, keep_recent=24)
        is history
    )


def test_the_anchored_size_spares_a_history_the_ruler_would_compact():
    """Measured provider usage takes precedence over the character estimate."""
    history = conversation(12, 4_000)  # the ruler reads this as well over 8k tokens

    assert (
        compress_history(history, model=None, budget_tokens=8_000, keep_recent=24)
        is not history
    ), "the premise: the ruler compacts this"

    assert (
        compress_history(
            history,
            model=None,
            budget_tokens=8_000,
            keep_recent=24,
            used_estimate=5_000,
        )
        is history
    ), "the anchored size under budget must skip compaction"


def test_largest_tool_results_names_the_culprit():
    history = conversation(3, 1_000, name="search") + conversation(
        1, 40_000, name="read_file"
    )
    heaviest = largest_tool_results(history, 1)
    assert heaviest[0][0] == "read_file"


def test_a_demotion_keeps_the_repeat_guards(tmp_path: Path):
    """Demotion preserves repeat-guard state and valid paginated rereads."""

    agent = _offline_agent(tmp_path)
    agent.compactor = compress_history  # this test is about the ladder's demotion pass
    agent._config = agent._config.merged(history_budget_tokens=2_000).normalised()
    for i in range(8):
        agent._conversation.append(call_msg(str(i), "read_file"))
        agent._conversation.append(
            tool_msg(
                str(i),
                "read_file",
                json.dumps({"path": f"f{i}.py", "content": "z" * 4_000}),
            )
        )
        agent._conversation.append(Message(role="assistant", content="ok"))
        agent._runtime.guard.record(
            "read_file", {"path": f"f{i}.py"}, is_error=False
        )

    before = len(agent._conversation.history)
    agent._conversation.fit_request("system prompt")

    assert len(agent._conversation.history) == before, (
        "the premise: demotion, not summary"
    )
    assert (
        agent._runtime.guard.check_duplicate("read_file", {"path": "f0.py"})
        is not None
    ), "the identical bare re-read must still be recognised as a repeat"
    assert (
        agent._runtime.guard.check_duplicate(
            "read_file", {"path": "f0.py", "start_line": 500}
        )
        is None
    ), "reading the elided middle is a different call and must be allowed"


def test_autocompact_budget_scales_to_the_context_window(tmp_path: Path):
    """The automatic trigger tracks the model's real window, not a fixed number."""

    agent = _offline_agent(tmp_path)

    class _Provider:
        context_window = 200_000

    agent._model = _Provider()
    assert agent._conversation._autocompact_budget() == int(
        agent._config.autocompact_fraction * 200_000
    )

    class _TotalWindowProvider:
        context_window = 200_000
        max_input_tokens = 180_000

    agent._model = _TotalWindowProvider()
    assert agent._conversation._autocompact_budget() == int(
        agent._config.autocompact_fraction * 180_000
    ), "output-reserved tokens are not available to the request"

    # A fixed reserve overrides the fraction: window-minus-headroom model.
    agent._config = agent._config.merged(reserve_tokens=16_384).normalised()
    assert agent._conversation._autocompact_budget() == 180_000 - 16_384

    agent._model = None
    assert agent._conversation._autocompact_budget() == agent._config.history_budget_tokens, (
        "an unknown window falls back to the configured budget"
    )


def test_request_under_budget_does_not_invoke_the_compactor(tmp_path: Path):
    """A custom compactor may call a model, so fitting must skip it when unnecessary."""

    agent = _offline_agent(tmp_path)
    calls = 0

    def compactor(history, **kwargs):
        nonlocal calls
        calls += 1
        return history

    agent.compactor = compactor
    agent._conversation.append(Message(role="user", content="small"))

    agent._conversation.fit_request("system prompt")

    assert calls == 0


def test_request_fitting_does_not_render_dynamic_context_twice() -> None:
    from lamssi_agents.prompt import ContextBlock

    renders = 0

    def live_state() -> str:
        nonlocal renders
        renders += 1
        return f"render {renders}"

    agent = Agent(context=[ContextBlock("live-state", live_state)])
    prompt = agent.assemble_prompt()

    agent._conversation.fit_request(prompt.text)

    assert renders == 1


def test_dynamic_context_cannot_replace_explicit_instructions() -> None:
    from lamssi_agents.prompt import ContextBlock

    with pytest.raises(ValueError, match="reserved"):
        Agent(
            instructions="authoritative",
            context=[ContextBlock("instructions", lambda: "shadow")],
        )


def test_compact_focus_reaches_the_summary_prompt():
    """`compact(focus=…)` steers the recap: the `/compact <instructions>` path."""
    captured: dict = {}

    class _Provider:
        model = "fake"

        def stream(self, messages, tools=None, **kwargs):
            captured["prompt"] = messages[1].content
            yield StreamDelta(type="text", text="## What was asked\n- did the thing")
            yield StreamDelta(type="done", finish_reason="stop")

    history = []
    for i in range(30):
        history.append(Message(role="user", content=f"q{i} " + "x" * 2_000))
        history.append(Message(role="assistant", content=f"a{i} " + "y" * 2_000))

    compress_history(
        history,
        model=_Provider(),
        budget_tokens=5_000,
        keep_recent=6,
        focus="the auth module",
    )

    assert "## Focus" in captured["prompt"]
    assert "the auth module" in captured["prompt"]


def test_an_oversized_summary_request_is_not_sent_to_the_model():
    """The compaction call itself is subject to the same hard-window rule."""

    class _SmallProvider:
        model = "small"
        context_window = 1_000
        calls = 0

        def stream(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("an impossible summary request reached the provider")
            yield  # pragma: no cover

    provider = _SmallProvider()
    history = []
    for i in range(30):
        history.append(Message(role="user", content=f"q{i} " + "x" * 4_000))
        history.append(Message(role="assistant", content=f"a{i} " + "y" * 4_000))

    out = compress_history(
        history,
        model=provider,
        budget_tokens=500,
        keep_recent=2,
    )

    assert provider.calls == 0
    assert len(out[0].content) < 9_000, "the bullet fallback must also stay bounded"


# Artifact-backed results


def _runtime(tmp_path: Path):
    return Agent(
        features=[SystemTools(), Guidance(), Files(tmp_path)],
    )


@pytest.fixture()
def ctx(tmp_path: Path):
    agent = _runtime(tmp_path)
    with run_scope_active(agent):
        yield agent._capabilities, tmp_path


# Tool-level reduction


def test_a_wide_grep_stays_small_in_the_transcript(tmp_path: Path):
    """The property that matters: result size is bounded by the shape, not the corpus."""
    for i in range(40):
        (tmp_path / f"f{i}.py").write_text("needle here\n" * 20, encoding="utf-8")

    runtime = agent = _runtime(tmp_path)
    with run_scope_active(agent):
        result = runtime._tools.execute_binding(
            runtime._tools.resolve("fs", {"command": "grep -rn needle ."})
        )

    serialised = json.dumps(result)
    assert len(serialised) < 8_000, (
        f"a wide grep returned {len(serialised):,} chars into the transcript"
    )
    assert result["truncated"] is True, "it must say the answer was cut short"


def test_a_pipeline_narrows_before_the_result_is_returned(tmp_path: Path):
    """``| head`` is the model's own size control, and it applies in-process."""
    for i in range(40):
        (tmp_path / f"f{i}.py").write_text("needle here\n" * 20, encoding="utf-8")

    runtime = agent = _runtime(tmp_path)
    with run_scope_active(agent):
        result = runtime._tools.execute_binding(
            runtime._tools.resolve("fs", {"command": "grep -rl needle . | head -5"})
        )

    assert result["count"] == 5, result


# Context visibility


def test_the_window_breakdown_says_which_tool_is_heavy():
    """ "The context is large" is not actionable; "read_file is 80% of it" is."""
    from lamssi_agents.agent.turn import _window_breakdown

    messages = [
        Message(role="system", content="s" * 1_000),
        Message(role="user", content="u" * 100),
        tool_msg("1", "read_file", "r" * 40_000),
        tool_msg("2", "search", "s" * 500),
    ]
    window = _window_breakdown(messages, schema_bytes=2_000)

    assert window["by_role"]["tool"] == 40_500
    assert list(window["by_tool"]) == ["read_file", "search"]
    assert window["heaviest_results"][0] == ("read_file", 40_000)
    assert window["est_tokens"] > 0
    assert window["tool_schema_chars"] == 2_000


# Manual compaction


def _offline_agent(tmp_path: Path) -> Agent:
    """Build an agent that cannot make provider calls."""
    agent = Agent(
        features=[SystemTools(), Guidance(), Files(tmp_path)],
    )
    assert agent.model is None, "these tests must not reach a model"
    return agent


def _conversation(agent, pairs: int, chars: int) -> None:
    for i in range(pairs):
        agent._conversation.append(Message(role="user", content=f"q{i} " + "x" * chars))
        agent._conversation.append(
            Message(role="assistant", content=f"a{i} " + "y" * chars)
        )


def test_compact_shrinks_a_history_that_is_nowhere_near_the_budget(tmp_path: Path):
    """Recover immediately after the first context-overflow response."""
    agent = _offline_agent(tmp_path)
    _conversation(agent, 30, 600)
    before = len(agent._conversation.history)

    result = agent.compact()

    assert result, "a deliberate compaction should have done something"
    assert len(agent._conversation.history) < before
    assert result.messages_removed > 0
    assert result.tokens_saved > 0


def test_the_result_reads_as_a_sentence(tmp_path: Path):
    """A compaction result is readable on an ASCII-only console."""
    agent = _offline_agent(tmp_path)
    _conversation(agent, 30, 600)
    text = str(agent.compact())

    assert "->" in text and "messages" in text and "tokens" in text
    assert text.isascii(), f"a host prints this: {text!r}"


def test_manual_compaction_receives_the_run_abort_event(tmp_path: Path):
    """A slow summary started by the host must still obey the shared stop button."""
    agent = _offline_agent(tmp_path)
    agent._conversation.append(Message(role="user", content="remember this"))
    seen = {}

    def capture_abort(history, **kwargs):
        seen["abort_event"] = kwargs.get("abort_event")
        return history

    agent.compactor = capture_abort
    agent.compact()

    assert seen["abort_event"] is agent._control.aborted


def test_a_saving_without_a_dropped_message_still_counts(tmp_path: Path):
    """Token savings are adopted even when message count is unchanged."""
    agent = _offline_agent(tmp_path)
    agent.compactor = compress_history  # demotion is the ladder's job
    for i in range(6):
        agent._conversation.append(call_msg(str(i), "read_file"))
        agent._conversation.append(tool_msg(str(i), "read_file", "z" * 4000))
        agent._conversation.append(Message(role="assistant", content="ok"))

    count_before = len(agent._conversation.history)
    result = agent.compact()

    assert result.messages_removed == 0, "the premise: nothing could be dropped"
    assert result, "but it reclaimed tokens, so it must be adopted"
    assert result.tokens_saved > 1_000
    assert len(agent._conversation.history) == count_before
    assert any("elided" in (m.content or "") for m in agent._conversation.history), (
        "the saving should be visible in the transcript"
    )


def test_the_automatic_path_adopts_the_same_savings(tmp_path: Path):
    """Automatic fitting adopts token savings with no message-count change."""
    agent = _offline_agent(tmp_path)
    agent.compactor = compress_history  # the demotion path under test
    # A budget this history genuinely exceeds: the automatic path only compacts
    # when over, unlike `compact()`, which forces every pass.
    agent._config = agent._config.merged(history_budget_tokens=2_000).normalised()
    for i in range(6):
        agent._conversation.append(call_msg(str(i), "read_file"))
        agent._conversation.append(tool_msg(str(i), "read_file", "z" * 4000))
        agent._conversation.append(Message(role="assistant", content="ok"))

    count_before = len(agent._conversation.history)
    chars_before = sum(len(m.content or "") for m in agent._conversation.history)


    agent._conversation.fit_request("system prompt")

    chars_after = sum(len(m.content or "") for m in agent._conversation.history)
    assert len(agent._conversation.history) == count_before, (
        "the premise: nothing dropped"
    )
    assert chars_after < chars_before / 2, (
        f"the saving was computed and discarded: {chars_before} -> {chars_after}"
    )


def test_both_compaction_paths_share_one_adoption_rule(tmp_path: Path):
    """Automatic and manual compaction call the shared adoption rule."""
    import inspect

    from lamssi_agents.agent.conversation import Conversation, worth_adopting

    for fn in (Conversation.fit_request, Conversation.force_compaction):
        source = inspect.getsource(fn)
        assert "worth_adopting" in source, f"{fn.__name__} decides for itself"
        assert "len(compacted) <" not in source, f"{fn.__name__} still counts messages"

    # And the rule itself: a real saving is kept, a rounding error is not.
    assert worth_adopting(10_000, 2_000)
    assert not worth_adopting(10_000, 9_950)
    assert not worth_adopting(100, 213), (
        "a pass that grew the request is not compaction"
    )


def test_it_stops_instead_of_re_summarising_forever(tmp_path: Path):
    """Compaction declines savings below the adoption threshold."""
    agent = _offline_agent(tmp_path)
    _conversation(agent, 30, 600)

    passes = 0
    while agent.compact():
        passes += 1
        assert passes < 10, "compact() never settled: this is the loop it must not be"
    assert passes >= 1


def test_a_short_history_reports_nothing_rather_than_churning(tmp_path: Path):
    agent = _offline_agent(tmp_path)
    agent._conversation.append(Message(role="user", content="hello"))

    result = agent.compact()
    assert not result
    assert len(agent._conversation.history) == 1


def test_a_budget_can_be_given_instead_of_maximum_reduction(tmp_path: Path):
    """`compact()` means "as far as possible"; a number means "until it fits"."""
    agent = _offline_agent(tmp_path)
    _conversation(agent, 30, 600)

    assert not agent.compact(budget_tokens=1_000_000), "already inside that budget"
    assert len(agent._conversation.history) == 60


def test_keep_recent_can_be_lowered_for_one_call(tmp_path: Path):
    """The only way to reclaim anything when the weight is in the tail."""
    agent = _offline_agent(tmp_path)
    _conversation(agent, 30, 600)

    result = agent.compact(keep_recent=4)
    assert result.messages_after <= 6, f"tail not trimmed: {result}"


def test_the_host_is_told(tmp_path: Path):
    """A UI showing "compacted 60 → 25" needs the event, not a log line."""
    from lamssi_agents.events import AgentEventType

    agent = _offline_agent(tmp_path)
    _conversation(agent, 30, 600)

    seen = []
    agent.add_event_listener(
        lambda e: seen.append(e) if e.type is AgentEventType.HISTORY_COMPACTED else None
    )
    agent.compact()

    assert seen, "no HISTORY_COMPACTED event"
    assert seen[0].metadata["messages_before"] == 60
    assert seen[0].metadata["tokens_saved"] > 0


def test_the_host_is_warned_before_the_pause(tmp_path: Path):
    """The summary pass can take seconds; a host needs the start, not just the end."""
    from lamssi_agents.events import AgentEventType

    agent = _offline_agent(tmp_path)
    _conversation(agent, 30, 600)

    seen = []
    agent.add_event_listener(
        lambda e: (
            seen.append(e)
            if e.type
            in (AgentEventType.HISTORY_COMPACTING, AgentEventType.HISTORY_COMPACTED)
            else None
        )
    )
    agent.compact()

    assert [e.type for e in seen[:2]] == [
        AgentEventType.HISTORY_COMPACTING,
        AgentEventType.HISTORY_COMPACTED,
    ], f"start must precede done: {[e.type for e in seen]}"
    assert seen[0].metadata["messages_before"] == 60
    assert seen[0].metadata["tokens_before"] > 0


def test_a_no_op_compaction_stays_silent(tmp_path: Path):
    """`compact(budget=huge)` does nothing, so it must not flash a start signal."""
    from lamssi_agents.events import AgentEventType

    agent = _offline_agent(tmp_path)
    _conversation(agent, 30, 600)

    seen = []
    agent.add_event_listener(
        lambda e: (
            seen.append(e.type) if e.type is AgentEventType.HISTORY_COMPACTING else None
        )
    )
    assert not agent.compact(budget_tokens=1_000_000), "already inside that budget"
    assert not seen, f"a no-op compaction emitted a start event: {seen}"


def test_the_call_pairing_survives_a_manual_compaction(tmp_path: Path):
    """The invariant every history operation has to keep."""
    agent = _offline_agent(tmp_path)
    for i in range(20):
        agent._conversation.append(call_msg(str(i), "read_file"))
        agent._conversation.append(tool_msg(str(i), "read_file", "z" * 2000))
        agent._conversation.append(Message(role="assistant", content="ok"))

    agent.compact()

    announced, answered = _pairing(agent._conversation.history)
    assert announced == answered


def test_a_pass_that_makes_the_request_bigger_is_refused(tmp_path: Path):
    """A compaction candidate is rejected when it increases request size."""
    agent = _offline_agent(tmp_path)
    for text in ("run some code", "ok", "8", "done"):
        agent._conversation.append(Message(role="user", content=text))

    before = list(agent._conversation.history)
    result = agent.compact()

    assert not result, f"a cost-increasing pass was adopted: {result}"
    assert agent._conversation.history == before, "the transcript must be untouched"
    assert result.tokens_after == result.tokens_before, (
        "a refused pass must report the cost it left in place, not the one it rejected"
    )


def test_the_result_never_claims_a_saving_it_did_not_keep(tmp_path: Path):
    """`tokens_after` describes the transcript now, not a discarded candidate."""
    agent = _offline_agent(tmp_path)
    for text in ("a", "b", "c"):
        agent._conversation.append(Message(role="user", content=text))

    result = agent.compact()
    assert result.tokens_saved == 0
    assert result.messages_after == len(agent._conversation.history)


def test_deliberate_compaction_does_not_warn(tmp_path: Path, caplog):
    """Explicit shrink-as-far-as-possible compaction does not warn."""
    agent = _offline_agent(tmp_path)
    _conversation(agent, 30, 600)

    with caplog.at_level("WARNING"):
        agent.compact()
    assert not caplog.records, f"deliberate compaction warned: {caplog.text}"


# token accounting: what the spend gate counts


class _FakeUsage:
    def __init__(self, total):
        self.total_tokens = total


class _FakeProvider:
    def __init__(self, total):
        self.cumulative_usage = _FakeUsage(total)


class _FakeAgent:
    def __init__(self, main, summary=None):
        self._model = main
        self._summary_model = summary


def test_the_spend_gate_counts_the_summariser_too():
    """Budget accounting includes a distinct summary model."""
    main = _FakeProvider(1_000)
    from lamssi_agents.features.budget import cumulative_tokens, summary_tokens

    assert cumulative_tokens(_FakeAgent(main)) == 1_000
    assert summary_tokens(_FakeAgent(main)) == 0

    agent = _FakeAgent(main, _FakeProvider(250))
    assert cumulative_tokens(agent) == 1_250, "summariser spend went unbilled"
    assert summary_tokens(agent) == 250


def test_one_provider_for_both_roles_is_not_double_counted():
    """A shared main and summary provider is counted once."""
    shared = _FakeProvider(1_000)
    agent = _FakeAgent(shared, shared)

    from lamssi_agents.features.budget import cumulative_tokens, summary_tokens

    assert cumulative_tokens(agent) == 1_000
    assert summary_tokens(agent) == 0, (
        "indistinguishable spend must not be reported as a separate share"
    )


# Repeated compaction


class _Recorder:
    """A provider that returns a fixed summary and remembers what it was asked."""

    model = "recorder"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def stream(self, messages, **kw):
        self.prompts.append(messages[-1].content)
        yield StreamDelta(
            type="text",
            text="## What was asked\n1. the updated record",
        )
        yield StreamDelta(type="done", finish_reason="stop")


def _chatter(n: int, tag: str) -> list:
    out = []
    for i in range(n):
        out.append(Message(role="user", content=f"{tag}{i} " + "x" * 400))
        out.append(Message(role="assistant", content=f"{tag}{i} " + "y" * 400))
    return out


def test_the_first_compaction_summarises_from_scratch():
    from lamssi_agents.history.compaction import compress_history

    provider = _Recorder()
    compress_history(_chatter(30, "A"), provider, budget_tokens=300, keep_recent=4)

    assert provider.prompts, "no summary was requested"
    assert "<earlier-record>" not in provider.prompts[0]


def test_a_later_compaction_updates_the_earlier_record():
    """Later compaction updates the existing summary record."""
    from lamssi_agents.history.compaction import compress_history

    provider = _Recorder()
    first = compress_history(
        _chatter(30, "A"), provider, budget_tokens=300, keep_recent=4
    )
    compress_history(
        first + _chatter(30, "B"), provider, budget_tokens=300, keep_recent=4
    )

    assert len(provider.prompts) == 2
    assert "<earlier-record>" in provider.prompts[1]
    assert "the updated record" in provider.prompts[1], (
        "the earlier record was not carried in"
    )


def test_the_summary_is_never_nested_inside_another():
    from lamssi_agents.history.compaction import (
        _SUMMARY_FRAME_OPEN,
        compress_history,
        unframe_summary,
    )

    provider = _Recorder()
    first = compress_history(
        _chatter(30, "A"), provider, budget_tokens=300, keep_recent=4
    )
    second = compress_history(
        first + _chatter(30, "B"), provider, budget_tokens=300, keep_recent=4
    )

    assert second[0].content.count(_SUMMARY_FRAME_OPEN) == 1
    assert unframe_summary(second[0]) is not None


def test_only_our_own_frame_is_recognised():
    """A user message that merely looks like a recap must not be eaten."""
    from lamssi_agents.history.compaction import split_previous_summary

    lookalike = Message(role="user", content="## What was asked\n1. something")
    previous, rest = split_previous_summary([lookalike])

    assert previous is None
    assert rest == [lookalike]


def test_nothing_new_carries_the_record_forward_without_a_call():
    """Spending a model call to restate an unchanged summary is pure cost."""
    from lamssi_agents.history.compaction import _llm_summarise, frame_summary

    provider = _Recorder()
    only_summary = [
        Message(role="user", content=frame_summary("## What was asked\n1. x"))
    ]

    assert _llm_summarise(provider, only_summary) == "## What was asked\n1. x"
    assert provider.prompts == [], "a call was made for a span with nothing new"


# the tail yields before the budget does


def test_keep_recent_shrinks_when_the_tail_alone_blows_the_budget():
    """The protected tail shrinks when it exceeds the available budget."""
    from lamssi_agents.history.compaction import _keep_recent_that_fits

    history = conversation(20, 4_000)  # every message is large
    fits = _keep_recent_that_fits(
        history, keep_recent=24, room_tokens=2_000, calibrator=None
    )
    assert 2 <= fits < 24, "the tail must give way"


def test_keep_recent_is_untouched_when_the_tail_fits():
    from lamssi_agents.history.compaction import _keep_recent_that_fits

    history = conversation(20, 20)
    assert _keep_recent_that_fits(history, 24, 1_000_000, None) == 24


def test_the_tail_never_shrinks_below_the_last_exchange():
    """A model with no recent turn cannot answer at all."""
    from lamssi_agents.history.compaction import _keep_recent_that_fits

    history = conversation(20, 50_000)
    assert _keep_recent_that_fits(history, 24, room_tokens=1, calibrator=None) == 2


# Compaction strategy selection


def _agent_with(tmp_path: Path, **config) -> Agent:
    agent = Agent(features=[SystemTools(), Guidance(), Files(tmp_path)])
    agent._config = AgentConfig(**config).normalised()
    return agent


def test_default_and_explicit_summarise_select_the_same_compactor(tmp_path: Path):
    from lamssi_agents.history import summarise_only

    assert _agent_with(tmp_path).compactor is summarise_only
    assert _agent_with(tmp_path, compaction="summarise").compactor is summarise_only


def test_an_unknown_strategy_name_is_rejected():
    from lamssi_agents.history import get_compaction_strategy

    with pytest.raises(ValueError, match="unknown compaction strategy 'nope'"):
        get_compaction_strategy("nope")


def test_an_explicit_compactor_overrides_the_config(tmp_path: Path):
    """An explicit compactor beats the config default."""
    from lamssi_agents.history import summarise_only

    agent = _agent_with(tmp_path)
    agent.compactor = summarise_only
    assert agent.compactor is summarise_only


def test_summarise_only_summarises_without_demoting():
    """The pi-style strategy replaces the old span with a summary and never stubs a body."""
    from lamssi_agents.history import summarise_only
    from lamssi_agents.history.compaction import unframe_summary
    from lamssi_agents.history.tokens import estimate_tokens

    out = summarise_only(
        conversation(20, 4_000), model=None, budget_tokens=8_000, keep_recent=6
    )

    assert not any("elided" in (m.content or "") for m in out), (
        "summarise-only must not demote"
    )
    assert any(unframe_summary(m) for m in out if m.role == "user"), (
        "it must leave a summary"
    )
    assert estimate_tokens(out) <= 8_000


def test_maximum_reduction_keeps_a_useful_summary():
    """The budget-one sentinel drops the raw tail, not the summary's substance."""
    from lamssi_agents.history import summarise_only
    from lamssi_agents.history.compaction import unframe_summary

    out = summarise_only(_chatter(10, "A"), model=None, budget_tokens=1)

    summary = unframe_summary(out[0])
    assert len(out) == 1
    assert summary is not None and len(summary) > 100


def test_ordinary_summarisation_keeps_the_latest_tool_result_verbatim():
    """A recent result stays verbatim when it fits beside the summary."""
    from lamssi_agents.history import summarise_only

    latest = json.dumps({"path": "latest.py", "content": "important" * 100})
    history = conversation(20, 2_000)
    history.extend(
        [call_msg("latest", "read_file"), tool_msg("latest", "read_file", latest)]
    )

    out = summarise_only(history, model=None, budget_tokens=4_000, keep_recent=6)

    kept = next(m for m in out if m.role == "tool" and m.tool_call_id == "latest")
    assert kept.content == latest


def test_summarise_only_folds_in_a_latest_result_that_cannot_fit():
    """Simple mode summarises an oversized exchange; it never secretly demotes it."""
    from lamssi_agents.history import summarise_only
    from lamssi_agents.history.compaction import unframe_summary

    latest = json.dumps({"path": "latest.py", "content": "important" * 4_000})
    history = conversation(20, 2_000)
    history.extend(
        [call_msg("latest", "read_file"), tool_msg("latest", "read_file", latest)]
    )

    out = summarise_only(history, model=None, budget_tokens=4_000, keep_recent=6)

    assert not any(m.tool_call_id == "latest" for m in out if m.role == "tool")
    assert not any("elided" in (m.content or "") for m in out)
    assert any(unframe_summary(m) for m in out if m.role == "user")


def test_hard_window_guard_uses_only_the_selected_summary_strategy(tmp_path: Path):
    """Hard fitting may summarise the latest exchange but must not switch strategies."""
    from lamssi_agents.history import summarise_only
    from lamssi_agents.history.compaction import unframe_summary

    class _Provider:
        model = name = "sized"
        context_window = 16_384

        def stream(self, *args, **kwargs):
            yield StreamDelta(type="text", text="A short record of the older work.")
            yield StreamDelta(type="done", finish_reason="stop")

    agent = _offline_agent(tmp_path)
    agent._model = _Provider()
    agent.compactor = summarise_only
    for i in range(12):
        agent._conversation.append(Message(role="user", content=f"q{i} " + "x" * 1_000))
        agent._conversation.append(
            Message(role="assistant", content=f"a{i} " + "y" * 1_000)
        )
    agent._conversation.append(call_msg("latest", "read_file"))
    agent._conversation.append(
        tool_msg(
            "latest",
            "read_file",
            json.dumps({"path": "latest.py", "content": "z" * 50_000}),
        )
    )

    sent = agent._conversation.fit_request("system " + "s" * 1_000)
    from lamssi_agents.history.tokens import estimate_tokens

    estimated = agent._conversation.tokens.estimate(len(sent[0].content))
    estimated += estimate_tokens(sent[1:], agent._conversation.tokens)

    assert estimated <= agent._model.context_window
    assert not any(m.role == "tool" and m.tool_call_id == "latest" for m in sent)
    assert not any("elided" in (m.content or "") for m in sent)
    assert any(unframe_summary(m) for m in sent if m.role == "user")


def test_hard_window_guard_does_not_fall_back_from_a_custom_strategy(tmp_path: Path):
    """The request guard invokes the configured strategy once, then stops honestly."""
    from lamssi_agents.agent.conversation import ContextWindowExceeded

    class _Provider:
        context_window = 1_000

    calls = 0

    def unchanged(history, **kwargs):
        nonlocal calls
        calls += 1
        return history

    agent = _offline_agent(tmp_path)
    agent._model = _Provider()
    agent.compactor = unchanged
    agent._conversation.append(Message(role="user", content="x" * 10_000))

    with pytest.raises(ContextWindowExceeded):
        agent._conversation.fit_request("system")

    assert calls == 1
    assert agent._conversation.history[0].content == "x" * 10_000


def test_advanced_ladder_can_demote_one_oversized_latest_result(tmp_path: Path):
    """Keeping zero results is an explicit ladder step, not request-guard magic."""

    class _Provider:
        context_window = 16_384

    agent = _offline_agent(tmp_path)
    agent._model = _Provider()
    agent.compactor = compress_history
    agent._conversation.append(Message(role="user", content="read latest.py"))
    agent._conversation.append(call_msg("latest", "read_file"))
    agent._conversation.append(
        tool_msg(
            "latest",
            "read_file",
            json.dumps({"path": "latest.py", "content": "z" * 60_000}),
        )
    )

    sent = agent._conversation.fit_request("system")

    latest = next(m for m in sent if m.role == "tool" and m.tool_call_id == "latest")
    assert "elided" in latest.content


def test_summarise_only_keeps_the_call_result_pairing():
    from lamssi_agents.history import summarise_only

    out = summarise_only(
        conversation(20, 4_000), model=None, budget_tokens=8_000, keep_recent=6
    )
    announced, answered = _pairing(out)
    assert announced == answered


def test_the_selected_strategy_drives_the_automatic_path(tmp_path: Path):
    """fit_request compacts through agent.compactor, so selection reaches the loop."""
    from lamssi_agents.history import summarise_only
    from lamssi_agents.history.compaction import unframe_summary

    agent = _offline_agent(tmp_path)
    agent.compactor = summarise_only
    agent._config = agent._config.merged(history_budget_tokens=2_000).normalised()
    for i in range(8):
        agent._conversation.append(call_msg(str(i), "read_file"))
        agent._conversation.append(
            tool_msg(
                str(i),
                "read_file",
                json.dumps({"path": f"f{i}.py", "content": "z" * 4_000}),
            )
        )
        agent._conversation.append(Message(role="assistant", content="ok"))

    agent._conversation.fit_request("system prompt")

    assert not any("elided" in (m.content or "") for m in agent._conversation.history), (
        "the summarise-only strategy must not have demoted"
    )
    assert any(
        unframe_summary(m) for m in agent._conversation.history if m.role == "user"
    ), "the summarise-only strategy should have left a summary"


def test_the_shipped_strategies_match_the_compactor_contract():
    """The Compactor Protocol is the documented shape; keep it in step with the strategies."""
    import inspect
    from lamssi_agents.history import Compactor, compress_history, summarise_only

    contract = list(inspect.signature(Compactor.__call__).parameters)[1:]  # drop self
    for strategy in (compress_history, summarise_only):
        assert list(inspect.signature(strategy).parameters) == contract, (
            strategy.__name__
        )


def test_truncator_default_and_override(tmp_path: Path):
    assert _agent_with(tmp_path).truncator is clip_result

    def keep_head(result_str, **kwargs):
        return result_str[:100]

    agent = _agent_with(tmp_path)
    agent.truncator = keep_head
    assert agent.truncator is keep_head


def test_the_shell_tool_keeps_the_tail(tmp_path: Path):
    """Command output is a tail, so the shell tool declares truncation_side."""
    from lamssi_agents import Shell

    agent = Agent(
        features=[SystemTools(), Guidance(), Shell()],
    )
    shell_tools = [t for t in agent._tools.list_tools() if t.group == "execute"]
    assert shell_tools, "the shell feature should install a command tool"
    assert all(t.truncation_side == "tail" for t in shell_tools)


def test_dispatch_redacts_before_the_truncator(tmp_path: Path):
    """Redact tool output before passing it to a custom truncator."""
    from lamssi_agents.redaction import forget_secrets, register_secret

    seen = {}

    def capture(result_str, **kwargs):
        seen["text"] = result_str
        return result_str

    agent = _offline_agent(tmp_path)
    agent.truncator = capture
    secret = "SEKRET-abcdefghijklmnop-0123456789"
    register_secret(secret)
    try:
        message = agent._runtime._format_message(
            ToolCall(id="c1", name="whatever", arguments={}),
            {"data": secret},
        )
        agent._conversation.append(message)
    finally:
        forget_secrets()

    assert secret not in seen["text"], "the truncator received an unredacted secret"
    assert secret not in agent._conversation.history[-1].content
