# SPDX-License-Identifier: MIT
"""10 - notes that outlive one conversation.

Memory is file-backed Markdown with YAML frontmatter, kept in the project's
state directory. An index of what exists is injected into the prompt, so the
agent knows what it could recall without paying for the bodies.

    python examples/10_memory.py
"""

import tempfile
from pathlib import Path

from lamssi_agents import Agent, ApprovalPolicy
from lamssi_agents import Files, Memory, Guidance, SystemTools
from lamssi_agents import tool_runtime as tool_mod

from _support import heading

project = Path(tempfile.mkdtemp())

heading("Writing a note")

agent = Agent(features=[SystemTools(), Guidance(), Files(project), Memory()], approval=ApprovalPolicy.allow_all())
print(tool_mod.invoke_tool_unchecked(agent, "memory", {
    "action": "remember",
    "name": "stage-backlash",
    "content": "The X stage has 12 um of backlash. Always approach from -X.",
    "type": "project",
    "description": "Stage backlash and the approach direction that avoids it.",
}))

heading("Where it went")

# Memory files are UTF-8 Markdown with YAML frontmatter. Print selected fields
# here because a Windows cp1252 console cannot display every character used by
# the generated index.
for path in sorted(project.rglob("*.md")):
    lines = path.read_text(encoding="utf-8").splitlines()
    print(f"  {path.relative_to(project)}  ({len(lines)} lines)")

record = (project / ".lamssi" / "memory" / "stage-backlash.md")
print("\n  frontmatter of the record:")
for line in record.read_text(encoding="utf-8").splitlines():
    if line.strip() == "---" or not line.strip():
        continue
    if ":" in line and line.split(":")[0].isidentifier():
        print(f"      {line}")

heading("A new agent, same project, already knows it exists")

fresh = Agent(features=[SystemTools(), Guidance(), Files(project), Memory()], approval=ApprovalPolicy.allow_all())
print("  index in the prompt:", "stage-backlash" in fresh.assemble_prompt().text)
print("  body in the prompt :", "12 um" in fresh.assemble_prompt().text,
      "(loaded only when recalled)")

print("  recall ->", tool_mod.invoke_tool_unchecked(fresh, "memory",
                                           {"action": "recall", "name": "stage-backlash"}))
