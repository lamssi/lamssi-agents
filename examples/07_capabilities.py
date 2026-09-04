# SPDX-License-Identifier: MIT
"""07 - providing an application capability to a tool.

The ``Code`` feature needs a host-supplied ``CodeExecutor``. The host chooses
the execution environment and Lamssi advertises the tool only after one is
available.

    python examples/07_capabilities.py
"""

from lamssi_agents import Agent, ApprovalPolicy
from lamssi_agents import Files, Guidance, SystemTools
from lamssi_agents import tool_runtime as tool_mod
from lamssi_agents.features.code import CodeExecutor, CodeResult

from _support import heading

heading("Without the capability, the tool is not even offered")

bare = Agent(features=[SystemTools(), Guidance(), Files(".")], approval=ApprovalPolicy.allow_all())
print("  execute_code in the surface:",
      "execute_code" in {d.name for d in bare.visible_tool_defs()})

# `@tool(requires=CodeExecutor)` keeps the tool out of the schema until the
# application supplies that capability.


class TinyExecutor:
    """The smallest thing satisfying `CodeExecutor`. NOT a sandbox."""

    def __init__(self) -> None:
        self._namespace: dict = {}

    def run(self, source: str) -> CodeResult:
        import io
        from contextlib import redirect_stdout

        out = io.StringIO()
        try:
            with redirect_stdout(out):
                exec(source, self._namespace)
        except Exception as exc:
            return CodeResult(ok=False, error=f"{type(exc).__name__}: {exc}",
                              stdout=out.getvalue())
        return CodeResult(ok=True, stdout=out.getvalue())

    def variables(self) -> dict:
        return {k: v for k, v in self._namespace.items() if not k.startswith("__")}


heading("With it, the tool appears and works")

agent = Agent(
    features=[SystemTools(), Guidance(), Files(".")],
    approval=ApprovalPolicy.allow_all(),
    capabilities={CodeExecutor: TinyExecutor()},
)
print("  execute_code in the surface:",
      "execute_code" in {d.name for d in agent.visible_tool_defs()})

result = tool_mod.invoke_tool_unchecked(agent, "execute_code", {"code": "print(6 * 7)"})
print("  result:", result)

print("""
  A capability is keyed by its protocol type, so a tool asks for a shape rather
  than reaching into a host object. Single-method capabilities take the function
  directly: Agent(capabilities={AbortSink: coordinator.abort}).
""")
