"""A small in-process agent loop with explicit features."""

from lamssi_agents.agent import Agent
from lamssi_agents.approval import (
    ApprovalPolicy,
    ApprovalRequest,
    ToolApproval,
    ToolApprovalResult,
)
from lamssi_agents.features import (
    Code,
    Feature,
    Files,
    Guidance,
    Memory,
    Shell,
    Skills,
    SystemTools,
)
from lamssi_agents.interaction import (
    InteractionRequest,
    InteractionResponse,
)
from lamssi_agents.model import LiteLLMModel
from lamssi_agents.prompt import ContextBlock, PromptPosition
from lamssi_agents.result import RunResult
from lamssi_tools import tool

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "ApprovalPolicy",
    "ApprovalRequest",
    "Code",
    "ContextBlock",
    "Feature",
    "Files",
    "Guidance",
    "InteractionRequest",
    "InteractionResponse",
    "LiteLLMModel",
    "Memory",
    "PromptPosition",
    "RunResult",
    "Shell",
    "Skills",
    "SystemTools",
    "ToolApproval",
    "ToolApprovalResult",
    "__version__",
    "tool",
]
