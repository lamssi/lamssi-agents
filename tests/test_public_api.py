"""Public API tests for beginner and advanced composition."""

from __future__ import annotations

import pytest

from lamssi_agents import (
    Agent,
    ApprovalPolicy,
    ApprovalRequest,
    Code,
    Feature,
    Files,
    Guidance,
    InteractionRequest,
    InteractionResponse,
    LiteLLMModel,
    Memory,
    PromptPosition,
    RunResult,
    Shell,
    Skills,
    SystemTools,
    ToolApproval,
    ToolApprovalResult,
    tool,
)
from lamssi_agents.ask_model import build_ask_model
from lamssi_agents.prompt import ContextBlock
from lamssi_agents.events import AgentEventType
from lamssi_agents.interaction import InteractionKind, request_interaction
from lamssi_agents.providers import StreamDelta, ToolCall, Usage
from lamssi_tools import Expose, Param


def test_public_configuration_surface_has_entry_point_documentation() -> None:
    """IDE help stays useful as the beginner and advanced surfaces evolve."""
    import inspect

    documented_types = (
        Agent,
        ContextBlock,
        PromptPosition,
        Feature,
        ApprovalPolicy,
        ApprovalRequest,
        ToolApproval,
        ToolApprovalResult,
        InteractionRequest,
        InteractionResponse,
        RunResult,
        LiteLLMModel,
        Files,
        SystemTools,
        Guidance,
        Memory,
        Skills,
        Shell,
        Code,
        Expose,
        Param,
        tool,
    )
    missing_types = [
        obj.__name__ for obj in documented_types if not inspect.getdoc(obj)
    ]
    assert not missing_types, (
        f"public configuration types need docstrings: {missing_types}"
    )

    public_agent_members = {
        name: value
        for name, value in Agent.__dict__.items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or isinstance(value, property))
    }
    missing_members = [
        name
        for name, value in public_agent_members.items()
        if not inspect.getdoc(value)
    ]
    assert not missing_members, (
        f"public Agent members need docstrings: {missing_members}"
    )

    assert "Example:" in inspect.getdoc(Agent)
    assert "Args:" in inspect.getdoc(Agent.__init__)
    context_docs = inspect.getdoc(ContextBlock)
    assert "stable=True" in context_docs
    assert "``position`` selects a named placement" in context_docs


def test_instructions_are_the_complete_base_prompt() -> None:
    agent = Agent(instructions="You operate devices.", features=[SystemTools()])

    assert agent.build_system_prompt() == "You operate devices."


def test_empty_instructions_do_not_load_a_kernel_prompt() -> None:
    assert Agent().build_system_prompt() == ""
    assert Agent().explain_prompt() == "System prompt is empty."


def test_guidance_is_an_explicit_feature() -> None:
    plain = Agent(instructions="Custom", features=[SystemTools()])
    guided = Agent(instructions="Custom", features=[SystemTools(), Guidance()])

    assert plain.build_system_prompt() == "Custom"
    assert "Finish with evidence" in guided.build_system_prompt()


def test_prompt_provenance_names_every_source() -> None:
    def live_state(ctx):
        return f"tools={len(ctx.tools)}"

    agent = Agent(
        instructions="Custom",
        context=[ContextBlock("live-state", live_state)],
    )
    prompt = agent.assemble_prompt()

    assert [part.name for part in prompt.parts] == ["instructions", "live-state"]
    assert prompt.parts[0].source == "Agent.instructions"
    explanation = agent.explain_prompt()
    assert "name | source | position | cacheable | chars" in explanation
    assert "instructions | Agent.instructions | instructions |" in explanation
    assert "live-state |" in explanation
    assert "| context |" in explanation


def test_context_requires_an_instance_instead_of_constructing_classes_magically() -> (
    None
):
    class Section:
        name = "section"

        def render(self, ctx):
            return "text"

    with pytest.raises(TypeError, match="construct it first"):
        Agent(context=[Section])


def test_prompt_composer_owns_callable_errors_and_value_coercion() -> None:
    broken = ContextBlock("broken", lambda ctx: 1 / 0)
    numeric = ContextBlock("numeric", lambda ctx: 42)
    agent = Agent(context=[broken, numeric])

    assert agent.build_system_prompt() == "42"


def test_prompt_callbacks_accept_no_arguments_or_one_context() -> None:
    no_context = ContextBlock("no-context", lambda *, value="ready": value)
    agent = Agent(context=[no_context])

    assert agent.build_system_prompt() == "ready"

    with pytest.raises(TypeError, match="no arguments or one positional"):
        ContextBlock("too-many", lambda first, second: "unused")


def test_explicit_instructions_are_first_at_their_prompt_position() -> None:
    earlier_name = ContextBlock(
        "aaa",
        lambda: "context",
        stable=True,
        position=PromptPosition.INSTRUCTIONS,
    )
    agent = Agent(instructions="base", context=[earlier_name])

    assert agent.build_system_prompt() == "base\n\ncontext"


def test_instructions_is_a_reserved_context_name() -> None:
    with pytest.raises(ValueError, match="reserved"):
        Agent(context=[ContextBlock("instructions", lambda: "shadow")])


def test_approval_policy_has_unambiguous_named_choices() -> None:
    seen: list[ApprovalRequest] = []

    def handler(request: ApprovalRequest):
        seen.append(request)
        return ToolApproval.APPROVE

    gated = Agent(approval=ApprovalPolicy.ask_when_required(handler))
    everything = Agent(approval=ApprovalPolicy.ask_for_everything(handler))
    unattended = Agent(approval=ApprovalPolicy.reject_when_required())
    open_agent = Agent(approval=ApprovalPolicy.allow_all())

    assert gated.approval.name == "ask-when-required"
    assert gated.approval.handler is handler
    assert everything.approval.asks_for_everything
    assert unattended.approval.name == "reject-when-required"
    assert open_agent.approval.allows_all


def test_approval_request_arguments_are_read_only() -> None:
    request = ApprovalRequest.create(
        "move_stage",
        {"position": 3},
        reason="the tool declares that approval is always required",
        declaration="always",
    )

    with pytest.raises(TypeError):
        request.arguments["position"] = 4


class WritingModel:
    model = "scripted"
    name = "scripted"
    is_local = True
    supports_tools = True
    reasoning_effort = None
    context_window = 32_000

    def __init__(self) -> None:
        self.turn = 0
        self.cumulative_usage = Usage()

    def stream(self, messages, tools=None, **kwargs):
        self.turn += 1
        self.cumulative_usage.add(
            Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )
        yield StreamDelta(
            type="usage",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        if self.turn == 1:
            yield StreamDelta(
                type="tool_call",
                tool_call=ToolCall(
                    id="write-1",
                    name="write_file",
                    arguments={"path": "result.txt", "content": "done"},
                ),
            )
            yield StreamDelta(type="done", finish_reason="tool_calls")
            return
        yield StreamDelta(type="text", text="finished")
        yield StreamDelta(type="done", finish_reason="stop")


def test_summary_model_follows_main_until_explicitly_overridden() -> None:
    main = WritingModel()
    summary = WritingModel()
    replacement = WritingModel()
    agent = Agent(model=main)

    assert agent.summary_model is main

    agent.summary_model = summary
    agent.use_model(replacement)
    assert agent.summary_model is summary

    agent.summary_model = None
    assert agent.summary_model is replacement


def test_run_returns_text_files_usage_and_turns(tmp_path) -> None:
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest):
        requests.append(request)
        return ToolApproval.APPROVE

    agent = Agent(
        model=WritingModel(),
        features=[Files(tmp_path)],
        approval=ApprovalPolicy.ask_for_everything(approve),
    )

    result = agent.run("write the result")

    assert isinstance(result, RunResult)
    assert result.text == "finished"
    assert result.files_written == ("result.txt",)
    assert result.usage.total_tokens == 30
    assert result.turns == 2
    assert result.ok
    assert requests[0].tool == "write_file"
    assert requests[0].declaration == "conditional"
    assert "every tool call" in requests[0].reason

    second = agent.run("answer without writing")
    assert second.text == "finished"
    assert second.files_written == ()
    assert second.usage.total_tokens == 15
    assert second.turns == 1
    assert agent.chat("one more answer") == "finished"
    assert agent.usage.total_tokens == 60


def test_ask_model_uses_the_same_explicit_model_seam() -> None:
    class AnswerModel:
        model = "answer-model"

        def stream(self, messages, tools=None, **kwargs):
            yield StreamDelta(type="text", text="4.7")
            yield StreamDelta(type="done", finish_reason="stop")

    ask_model = build_ask_model(AnswerModel())

    assert ask_model("suggest a gain", type=float) == 4.7
    assert ask_model.history[-1]["model"] == "answer-model"


def test_run_without_a_model_reports_an_error() -> None:
    result = Agent().run("hello")

    assert not result.ok
    assert result.error == result.text


def test_image_encoding_failure_stops_before_the_model(monkeypatch) -> None:
    class NeverCalledModel(WritingModel):
        def stream(self, messages, tools=None, **kwargs):
            raise AssertionError("model must not receive a request without its image")
            yield

    def fail(_image):
        raise ValueError("invalid bytes")

    monkeypatch.setattr("lamssi_agents.vision.to_image_urls", fail)
    events = []
    agent = Agent(model=NeverCalledModel())
    agent.add_event_listener(events.append)

    result = agent.run("describe this", image=b"bad")

    assert not result.ok
    assert result.turns == 0
    assert "Could not encode image input" in result.error
    assert agent.history == []
    assert any(event.type is AgentEventType.ERROR for event in events)


def test_run_result_outputs_are_copied_and_read_only() -> None:
    source = {"items": ["first"]}
    result = RunResult("done", outputs=source)
    source["items"].append("second")

    assert result.outputs["items"] == ("first",)
    with pytest.raises(TypeError):
        result.outputs["other"] = ()


def test_typed_interaction_events_can_be_correlated() -> None:
    seen = []
    agent = Agent(
        interaction=lambda request: InteractionResponse.answered("blue"),
    )
    agent.add_event_listener(seen.append)

    response = request_interaction(
        agent._control.interaction.handler,
        agent.emit,
        InteractionKind.QUESTION,
        "Which colour?",
        choices=("blue", "green"),
    )

    assert response == InteractionResponse.answered("blue")
    request_event = next(
        event for event in seen if event.type is AgentEventType.USER_INPUT_REQUEST
    )
    response_event = next(
        event for event in seen if event.type is AgentEventType.USER_INPUT_RESPONSE
    )
    assert request_event.metadata["request_id"] == response_event.metadata["request_id"]
    assert request_event.metadata["interaction_kind"] == "question"
    assert response_event.metadata["handled"] is True


def test_interaction_response_must_match_the_request_kind() -> None:
    agent = Agent(
        interaction=lambda request: InteractionResponse.continue_(),
    )

    assert (
        request_interaction(
            agent._control.interaction.handler,
            agent.emit,
            InteractionKind.QUESTION,
            "Which?",
        )
        is None
    )

    agent.interaction = lambda request: InteractionResponse.answered("yes")
    assert (
        request_interaction(
            agent._control.interaction.handler,
            agent.emit,
            InteractionKind.BUDGET_CHECKPOINT,
            "Continue?",
        )
        is None
    )


def test_interaction_request_metadata_is_read_only() -> None:
    from lamssi_agents import InteractionRequest
    from lamssi_agents.interaction import InteractionKind

    request = InteractionRequest(
        InteractionKind.QUESTION,
        "Which colour?",
        metadata={"choices": ("blue", "green")},
    )

    with pytest.raises(TypeError):
        request.metadata["choices"] = ("red",)

    original = ["blue", "green"]
    isolated = InteractionRequest(
        InteractionKind.QUESTION,
        "Which colour?",
        metadata={"choices": original},
    )
    original.append("red")
    assert isolated.metadata["choices"] == ["blue", "green"]
