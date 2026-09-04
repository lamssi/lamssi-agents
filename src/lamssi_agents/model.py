"""The one model seam exposed by :class:`lamssi_agents.Agent`."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, Union

from lamssi_agents.providers.litellm_provider import LiteLLMModel
from lamssi_agents.providers.protocol import Model


ModelFactory: TypeAlias = Callable[[], Model]
ModelInput: TypeAlias = Union[str, Model]


def resolve_model(value: ModelInput) -> Model:
    """Turn the public model input into one concrete model adapter.

    Strings have exactly one meaning: a model identifier understood by LiteLLM.
    Endpoint, credentials, and sampling options belong on an explicitly built
    :class:`LiteLLMModel`; the Agent never guesses them from another config object.
    """
    if isinstance(value, str):
        model_id = value.strip()
        if not model_id:
            raise ValueError("model must be a non-empty identifier or a Model object")
        return LiteLLMModel(model_id)
    if not callable(getattr(value, "stream", None)):
        raise TypeError(
            "model must be a LiteLLM model identifier or an object implementing Model"
        )
    return value


def model_id(value: Model | None) -> str:
    """The adapter's stable display/routing identifier, if it publishes one."""
    return str(getattr(value, "model", "") or "") if value is not None else ""


def adapter_name(value: Model | None) -> str:
    """A diagnostic name without making metadata mandatory for custom models."""
    if value is None:
        return ""
    return str(getattr(value, "name", "") or type(value).__name__)


def input_limit(value: Model | None) -> int:
    """Return the model's usable input-token limit when it publishes one."""
    if value is None:
        return 0
    return int(
        getattr(value, "max_input_tokens", 0)
        or getattr(value, "context_window", 0)
        or 0
    )


__all__ = [
    "Model",
    "ModelFactory",
    "ModelInput",
    "LiteLLMModel",
    "adapter_name",
    "input_limit",
    "resolve_model",
    "model_id",
]
