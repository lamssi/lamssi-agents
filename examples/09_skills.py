# SPDX-License-Identifier: MIT
"""09 - skills: procedures the agent loads when they match the job.

A skill is Markdown with YAML frontmatter. The prompt carries a short catalog;
the full procedure is added only when the skill is loaded.

    python examples/09_skills.py
"""

import tempfile
from pathlib import Path

from _support import heading

from lamssi_agents import Agent, Files, Guidance, Skills, SystemTools
from lamssi_agents import tool_runtime as tool_mod
from lamssi_agents.features.skills import SkillRuntime

workspace = Path(tempfile.mkdtemp())
skills = workspace / "skills"
(skills / "calibrate").mkdir(parents=True)
(skills / "calibrate" / "SKILL.md").write_text(
    """---
name: calibrate
description: Calibrate the stage against a reference. Use before any precision run.
allowed-tools: read_file fs
---

1. Read the last calibration report.
2. Compare against the reference block.
3. Report the offset; do not apply it without asking.
""",
    encoding="utf-8",
)

heading("A skill catalog in the prompt")

agent = Agent(
    features=[
        SystemTools(),
        Guidance(),
        Files(workspace),
        Skills(skills, allow_model_loading=True),
    ]
)
runtime = agent.get(SkillRuntime)
print(f"  catalog: {[skill.name for skill in runtime.list()]}")
text = agent.assemble_prompt().text
print("  named in the prompt:", "calibrate" in text)
print("  body in the prompt :", "reference block" in text, "(only after loading)")

heading("Loading one pins its body")

tool_mod.invoke_tool_unchecked(agent, "load_skill", {"name": "calibrate"})

print("  body now in the prompt:", "reference block" in agent.assemble_prompt().text)

print("""
  `allowed-tools` is shown with the procedure as guidance. Skills do not change
  the Agent's authority or tool scope; the application owns those boundaries.
""")
