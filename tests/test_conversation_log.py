"""Conversation-log records for messages, events, and compaction."""

from __future__ import annotations

import json
from pathlib import Path

from lamssi_agents.conversation_log import DebugConversationLogger
from lamssi_agents.events import AgentEvent, AgentEventType
from lamssi_agents.providers import Message


def _records(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _logger(tmp_path: Path):
    # The debug logger is the one that snapshots messages; the base only
    # records the compaction event.
    return DebugConversationLogger(tmp_path, model="m", adapter="scripted")


def _sent(messages, turn):
    """The event `messages_sent` is logged from."""
    return AgentEvent(
        type=AgentEventType.MESSAGES_SENT,
        data="system prompt",
        metadata={
            "turn": turn,
            "history_msg_count": len(messages),
            "messages": messages,
        },
    )


def _history(n):
    return [
        Message(role="user" if i % 2 == 0 else "assistant", content=f"m{i}" * 200)
        for i in range(n)
    ]


# compaction leaves a trace


def test_a_compaction_is_recorded(tmp_path: Path):
    log = _logger(tmp_path)
    log(
        AgentEvent(
            type=AgentEventType.HISTORY_COMPACTED,
            data="40 -> 12 messages",
            metadata={
                "messages_before": 40,
                "messages_after": 12,
                "tokens_saved": 9_000,
            },
        )
    )
    log.close()

    rec = next(r for r in _records(log.path) if r["kind"] == "compacted")
    assert rec["messages_before"] == 40
    assert rec["messages_after"] == 12
    assert rec["tokens_saved"] == 9_000


# the messages array is appended, not re-sent


def test_only_the_new_messages_are_logged(tmp_path: Path):
    """The whole array every turn is O(turns^2) bytes for O(turns) information."""
    log = _logger(tmp_path)
    history = _history(10)
    log(_sent(history[:4], turn=1))
    log(_sent(history[:10], turn=2))
    log.close()

    sent = [r for r in _records(log.path) if r["kind"] == "messages_sent"]
    assert [len(r["messages"]) for r in sent] == [4, 6]
    assert [r["messages_from"] for r in sent] == [0, 4]


def test_a_rewritten_history_is_logged_in_full(tmp_path: Path):
    """The first snapshot after compaction records the full rewritten history."""
    log = _logger(tmp_path)
    log(_sent(_history(10), turn=1))
    log(
        AgentEvent(
            type=AgentEventType.HISTORY_COMPACTED,
            data="10 -> 3",
            metadata={"messages_before": 10, "messages_after": 3, "tokens_saved": 500},
        )
    )
    log(_sent(_history(3), turn=2))
    log.close()

    sent = [r for r in _records(log.path) if r["kind"] == "messages_sent"]
    assert [len(r["messages"]) for r in sent] == [10, 3]
    assert sent[-1]["messages_from"] == 0


def test_a_shrinking_history_is_not_treated_as_an_append(tmp_path: Path):
    """Belt to the compaction event's braces: any drop in count forces a full log."""
    log = _logger(tmp_path)
    log(_sent(_history(8), turn=1))
    log(_sent(_history(5), turn=2))
    log.close()

    sent = [r for r in _records(log.path) if r["kind"] == "messages_sent"]
    assert [len(r["messages"]) for r in sent] == [8, 5]


def test_the_delta_is_dramatically_smaller(tmp_path: Path):
    """Delta logging is substantially smaller than full snapshots."""
    log = _logger(tmp_path)
    history = _history(40)
    for turn, upto in enumerate(range(4, 41, 4), start=1):
        log(_sent(history[:upto], turn=turn))
    log.close()

    logged = sum(
        len(json.dumps(r.get("messages", [])))
        for r in _records(log.path)
        if r["kind"] == "messages_sent"
    )
    whole_array_once = len(json.dumps([m.content for m in history]))
    assert logged < whole_array_once * 3, "each message should be written about once"
