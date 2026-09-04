# SPDX-License-Identifier: MIT
"""01 - the smallest thing that works.

Needs a real model. Set LAMSSI_MODEL, or run a local OpenAI-compatible server
on http://127.0.0.1:1234 (LM Studio's default port).

    python examples/01_hello.py
"""

from lamssi_agents import Agent
from lamssi_agents import Files, Guidance, SystemTools

from _support import heading, real_model

heading("A whole agent, in two lines")

agent = Agent(real_model(), features=[SystemTools(), Guidance(), Files(".")])
print(agent.chat("List the Python files here and say which is largest."))

# The feature list shows exactly what is installed. `SystemTools` adds user
# interaction, `Guidance` adds operating instructions, and `Files` adds file
# tools rooted at the current directory. Example 11 composes the same `Agent`
# class with application-owned state.
