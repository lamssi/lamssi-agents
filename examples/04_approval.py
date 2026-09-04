# SPDX-License-Identifier: MIT
"""04 - deciding what needs your say-so with explicit application policies.

    python examples/04_approval.py
"""

import tempfile
from pathlib import Path

from lamssi_agents import (
    Agent,
    ApprovalPolicy,
    ApprovalRequest,
    ToolApproval,
    ToolApprovalResult,
)
from lamssi_agents import Files, Guidance, SystemTools
from _support import ScriptedModel, calls, heading, says

heading("Named policies")

approve = lambda request: ToolApproval.APPROVE
policies = [
    ApprovalPolicy.reject_when_required(),
    ApprovalPolicy.ask_when_required(approve),
    ApprovalPolicy.ask_for_everything(approve),
    ApprovalPolicy.allow_all(),
]
for policy in policies:
    agent = Agent(features=[SystemTools(), Guidance()], approval=policy)
    print(f"  {agent.approval.name}")

print("""
  reject-when-required blocks gated calls when nobody is present.
  ask-when-required follows each tool's risk declaration.
  ask-for-everything asks before every call.
  allow-all deliberately skips agent-level consent.
""")

heading("Your own rule")

# A callable answers the calls that *are* gated. It is not consulted for a call
# the tool already calls safe, so a read inside the workspace never reaches it.
decisions = []


def only_reads(request: ApprovalRequest):
    allowed = request.tool.startswith(("read_", "fs"))
    decisions.append((request.tool, allowed, request.reason))
    if allowed:
        return ToolApproval.APPROVE
    return ToolApprovalResult(ToolApproval.REJECT, reason="this host only approves reads")


workspace = Path(tempfile.mkdtemp())
(workspace / "notes.txt").write_text("hello\n", encoding="utf-8")
agent = Agent(
    model=ScriptedModel(
        calls("read_file", path="notes.txt"),
        calls("delete_file", path="notes.txt"),
        says("Finished checking the file."),
    ),
    features=[SystemTools(), Guidance(), Files(workspace)],
    approval=ApprovalPolicy.ask_when_required(only_reads),
)

agent.run("Read notes.txt, then delete it.")

for message in agent.history:
    if message.role == "tool":
        verdict = "blocked" if "declined" in (message.content or "") else "ran"
        print(f"  {message.name:<12} -> {verdict}")

print("  asked about  :", decisions)
print("  file on disk :", (workspace / "notes.txt").exists())
