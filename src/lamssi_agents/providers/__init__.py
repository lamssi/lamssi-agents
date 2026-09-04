"""Internal model-transport types and the LiteLLM adapter."""

from __future__ import annotations

from lamssi_agents.providers.errors import clean_model_error
from lamssi_agents.providers.litellm_provider import LiteLLMModel
from lamssi_agents.providers.model_catalog import (
    models_endpoint,
    source_for_model,
)
from lamssi_agents.providers.models import (
    Message,
    ProviderInterrupted,
    StreamDelta,
    ToolCall,
    Usage,
)
from lamssi_agents.providers.protocol import Model

__all__ = [
    "ToolCall",
    "Message",
    "Usage",
    "StreamDelta",
    "ProviderInterrupted",
    "Model",
    "LiteLLMModel",
    "clean_model_error",
    "source_for_model",
    "models_endpoint",
]
