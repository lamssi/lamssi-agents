"""Credential defaults for cloud and OpenAI-compatible local models."""

from __future__ import annotations

import pytest

from lamssi_agents import LiteLLMModel
from lamssi_agents.providers import Message, local_models


@pytest.fixture(autouse=True)
def no_local_probe(monkeypatch):
    """Keep model construction independent from running local servers."""
    monkeypatch.setattr(
        local_models,
        "probe",
        lambda api_base, model: local_models.LocalModelInfo(),
    )


def _request_kwargs(model: LiteLLMModel) -> dict:
    return model._common_kwargs(
        [Message(role="user", content="hello")],
        tools=None,
        temperature=0.0,
        max_tokens=10,
    )


def test_cloud_model_leaves_credentials_to_litellm_environment():
    model = LiteLLMModel("openai/gpt-5-mini")

    assert "api_key" not in _request_kwargs(model)


@pytest.mark.parametrize("api_key", [None, ""])
def test_loopback_endpoint_gets_a_local_placeholder(api_key):
    model = LiteLLMModel(
        "openai/local-model",
        api_base="http://127.0.0.1:1234/v1",
        api_key=api_key,
    )

    assert _request_kwargs(model)["api_key"] == "local"


def test_explicit_local_credential_wins_over_the_placeholder():
    model = LiteLLMModel(
        "openai/private-model",
        api_base="http://localhost:1234/v1",
        api_key="actual-local-secret",
    )

    assert _request_kwargs(model)["api_key"] == "actual-local-secret"


def test_remote_custom_endpoint_does_not_receive_a_placeholder():
    model = LiteLLMModel(
        "openai/gateway-model",
        api_base="https://models.example.test/v1",
    )

    assert "api_key" not in _request_kwargs(model)


def test_local_tool_support_is_auto_detected(monkeypatch):
    monkeypatch.setattr(
        local_models,
        "probe",
        lambda api_base, model: local_models.LocalModelInfo(
            context_window=16_000,
            supports_tools=False,
            source="test",
        ),
    )

    model = LiteLLMModel(
        "openai/local-model",
        api_base="http://127.0.0.1:1234/v1",
    )

    assert model.supports_tools is False


def test_explicit_tool_support_overrides_local_detection(monkeypatch):
    monkeypatch.setattr(
        local_models,
        "probe",
        lambda api_base, model: local_models.LocalModelInfo(
            supports_tools=False,
            source="test",
        ),
    )

    model = LiteLLMModel(
        "openai/local-model",
        api_base="http://localhost:1234/v1",
        supports_tools=True,
    )

    assert model.supports_tools is True


@pytest.mark.parametrize("value", [0, -1, "invalid"])
def test_invalid_context_window_overrides_are_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        LiteLLMModel("openai/local-model", context_window=value)


def test_local_total_window_reserves_the_requested_output(monkeypatch):
    monkeypatch.setattr(
        local_models,
        "probe",
        lambda api_base, model: local_models.LocalModelInfo(
            context_window=16_000,
            source="test",
        ),
    )
    model = LiteLLMModel(
        "openai/local-model",
        api_base="http://localhost:1234/v1",
        max_tokens=4_000,
    )

    assert model.context_window == 16_000
    assert model.max_input_tokens == 12_000
