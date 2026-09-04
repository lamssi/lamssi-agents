# SPDX-License-Identifier: MIT
"""14 - the record of a run, and keeping the context window under control.

Each turn can be appended to a JSONL file. When the conversation reaches its
budget, the configured strategy compacts older history.

    python examples/14_logging_and_history.py
"""

import json
import tempfile
from pathlib import Path

from lamssi_agents import Agent, ApprovalPolicy
from lamssi_agents import Files, Guidance, SystemTools
from _support import ScriptedModel, calls, heading, says

project = Path(tempfile.mkdtemp())
logs = project / "conversations"

heading("An append-only JSONL transcript")

agent = Agent(features=[SystemTools(), Guidance(), Files(project)], approval=ApprovalPolicy.allow_all(), log_dir=logs)
agent.use_model(ScriptedModel(
    calls("fs", command="ls"),
    says("Nothing much in here."),
))
agent.chat("what is here?")

for path in sorted(logs.glob("*.jsonl")):
    print(f"  {path.name}")
    for line in path.read_text(encoding="utf-8").splitlines()[:4]:
        record = json.loads(line)
        print(f"      {record.get('type', record.get('role', '?'))}")

heading("Compaction, on demand")

# Fill a conversation with the thing that actually costs: tool results.
agent.clear_history()
agent.use_model(ScriptedModel(*[says("x" * 1200) for _ in range(30)]))
for i in range(30):
    agent.chat(f"question {i}")

print(f"  before: {len(agent.history)} messages")
result = agent.compact()
print(f"  {result}")
print(f"  after : {len(agent.history)} messages")

print("""
  The default strategy is simple: select an old prefix by token budget and
  summarise it once. It never demotes a tool body. The optional "ladder"
  strategy adds progressive tool-result demotion for tool-heavy workloads.

  `agent.compact()` runs the strategy immediately. A host can call it after a
  tool-heavy task finishes and its detailed results are no longer useful.
""")
