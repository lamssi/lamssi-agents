# SPDX-License-Identifier: MIT
"""19 - a compaction strategy, packaged as a feature."""

from _support import heading

from lamssi_agents import Agent, Feature
from lamssi_agents.agent.conversation import strip_orphan_tool_messages
from lamssi_agents.history.tokens import estimate_tokens
from lamssi_agents.providers import Message


# A compactor accepts this signature. ``**kwargs`` leaves room for inputs this
# strategy does not use, such as the model and focus text.
def keep_recent_only(history, *, budget_tokens, keep_recent, calibrator=None, **kwargs):
    if estimate_tokens(history, calibrator) <= budget_tokens:
        return history                                       # idle -> return by identity
    return strip_orphan_tool_messages(history[-keep_recent:])  # keep call/result pairing valid


class KeepRecentOnly(Feature):
    """Compact by keeping only the recent tail."""

    name = "keep-recent-only"

    def install(self, agent):
        agent.compactor = keep_recent_only


heading("The feature overrides the configured strategy")
plain = Agent()
print("  config default compactor:", plain.compactor.__name__)
agent = Agent(features=[KeepRecentOnly()])
print("  with the feature:         ", agent.compactor.__name__)

heading("It compacts an over-budget history")
history = [Message(role="user", content=f"message {i} " + "x" * 500) for i in range(40)]
kept = agent.compactor(history, budget_tokens=2_000, keep_recent=5)
print(f"  before: {len(history):>2} messages, ~{estimate_tokens(history):,} tokens")
print(f"  after:  {len(kept):>2} messages, ~{estimate_tokens(kept):,} tokens")
