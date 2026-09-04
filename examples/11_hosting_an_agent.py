# SPDX-License-Identifier: MIT
"""11 - a host composing an Agent directly.

The application passes ordinary values to ``Agent`` and installs optional
features explicitly.
"""

import tempfile
from pathlib import Path

from _support import heading

from lamssi_agents import Agent, Files, Guidance, PromptPosition, SystemTools
from lamssi_agents import ContextBlock
from lamssi_agents.features.files import FileSpace
from lamssi_agents.features.system import AbortSink

project = Path(tempfile.mkdtemp())
reference = Path(tempfile.mkdtemp())
(reference / "api.md").write_text("# The instrument API\n", encoding="utf-8")
stopped = []

heading("One Agent, explicit contributions")

agent = Agent(
    max_turns=12,
    features=[
        SystemTools(), Guidance(),
        Files(project, read_only=[(reference, "The instrument API documentation.")]),
    ],
    capabilities={AbortSink: lambda: stopped.append("stop")},
    context=[
        ContextBlock(
            "live-state",
            lambda ctx: "## Right now\n\n- idle",
            position=PromptPosition.LIVE,
        )
    ],
)

print(f"  readable dirs : {[d.name for d in agent.get(FileSpace).readable_dirs()]}")
print(f"  max turns     : {agent.max_turns}")
print(f"  tools         : {len(agent.available_tool_names())}")
print(f"  prompt blocks : {len(agent.assemble_prompt().parts)}")

heading("Typed capabilities reach application objects")

agent.get(AbortSink).abort_all()
print(f"  abort reached the host: {stopped}")
