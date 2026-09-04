"""Optional functionality installed explicitly with ``Agent(features=[...])``."""

from lamssi_agents.features.base import Feature
from lamssi_agents.features.budget import Budget
from lamssi_agents.features.code import Code, CodeExecutor, CodeResult
from lamssi_agents.features.files import Files, WriteEvent, WriteHook, WriteKind
from lamssi_agents.features.guidance import Guidance
from lamssi_agents.features.memory import Memory
from lamssi_agents.features.shell import Shell
from lamssi_agents.features.skills import Skills
from lamssi_agents.features.system import AbortSink, SystemTools

__all__ = [
    "AbortSink", "Budget", "Code", "CodeExecutor", "CodeResult",
    "Feature", "Files", "Guidance",
    "Memory", "Shell", "Skills", "SystemTools",
    "WriteEvent", "WriteHook", "WriteKind",
]
