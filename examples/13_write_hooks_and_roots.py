# SPDX-License-Identifier: MIT
"""13 - checking what the agent writes, and where it may write.

A write hook runs after a file is written and adds its findings to that tool
result. The model can respond to the feedback on its next turn.

    python examples/13_write_hooks_and_roots.py
"""

import tempfile
from pathlib import Path

from lamssi_agents import Agent
from lamssi_agents import Files, Guidance, SystemTools
from lamssi_agents import tool_runtime as tool_mod
from lamssi_agents.features.files import WriteEvent


from _support import heading
from lamssi_agents.features.files import FileSpace

project = Path(tempfile.mkdtemp())
reference = Path(tempfile.mkdtemp())
(reference / "notes.md").write_text("read only\n", encoding="utf-8")


class NoTabsHook:
    """Claims Python files, and complains about tabs."""

    name = "no-tabs"

    def matches(self, rel_path: str) -> bool:
        return rel_path.endswith(".py")

    def after_write(self, event: WriteEvent):
        if "\t" in (event.content or ""):
            # Whatever is returned is merged into the tool result the model sees.
            return {"style": "failed", "errors": ["tabs found; this project uses spaces"]}
        return None       # silent on a clean pass


agent = Agent(
    features=[
        SystemTools(), Guidance(),
        Files(
            project,
            on_write=[NoTabsHook()],
            read_only=[(reference, "Reference material. Read-only.")],
        ),
    ]
)

heading("A clean write says nothing extra")
print(" ", tool_mod.invoke_tool_unchecked(agent, "write_file",
                                 {"path": "good.py", "content": "x = 1\n"}))

heading("A bad one is reported in the same result")
print(" ", tool_mod.invoke_tool_unchecked(agent, "write_file",
                                 {"path": "bad.py", "content": "if x:\n\treturn\n"}))

heading("Reference dirs: read-free, writes prompt")
# No named roots: a reference dir is addressed by its absolute path, and reading it is free.
print("  readable dirs :", [d.name for d in agent.get(FileSpace).readable_dirs()])
print("  read from reference :",
      "error" not in tool_mod.invoke_tool_unchecked(
          agent, "read_file", {"path": str(reference / "notes.md")}))
# The reference directory is read-only. Writes there require approval.
print("  reference read is free  :", agent.get(FileSpace).is_free(str(reference / "notes.md")))
print("  reference write is free :",
      agent.get(FileSpace).is_free(str(reference / "nope.md"), write=True))
