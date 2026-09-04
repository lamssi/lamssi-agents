"""Credential redaction in history, model input, and conversation logs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lamssi_agents.conversation_log import ConversationLogger
from lamssi_agents.events import AgentEvent, AgentEventType

#: Shaped like the real thing: a recognised vendor prefix and enough entropy to
#: be worth masking. Never a live credential.
ANTHROPIC = "sk-ant-api03-REALLOOKINGSECRET1234567890abcdefGHIJKL"
OPENAI = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH"
GOOGLE = "AIzaSyD-EXAMPLE0123456789abcdefghijklmnopq"
GITHUB = "ghp_0123456789abcdefghijklmnopqrstuvwxyzAB"

OPAQUE = "f3a9c1e07b5d4826ae91c0f5d3b8e274"

def written(tmp_path: Path, *events: AgentEvent) -> str:
    """Feed *events* to a real logger and return what reached disk."""
    logger = ConversationLogger(tmp_path, model="test", adapter="scripted")
    for event in events:
        logger(event)
    logger.close()
    return "\n".join(
        p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.jsonl")
    )

def tool_call(command: str) -> AgentEvent:
    return AgentEvent(
        type=AgentEventType.TOOL_START, data="run_bash",
        metadata={"tool_name": "run_bash", "arguments": {"command": command}},
    )

def tool_result(body: str) -> AgentEvent:
    return AgentEvent(
        type=AgentEventType.TOOL_RESULT, data=json.dumps({"stdout": body}),
        metadata={"tool_name": "run_bash"},
    )

# the three paths into the log

def test_a_key_in_a_tool_argument_never_reaches_the_log(tmp_path: Path):
    """Tool arguments are redacted before logging."""
    text = written(tmp_path, tool_call(f"curl -H 'Authorization: Bearer {ANTHROPIC}'"))

    assert ANTHROPIC not in text
    assert "sk-ant" in text, "masked, not deleted: a vanished key is undebuggable"

def test_a_key_in_a_tool_result_never_reaches_the_log(tmp_path: Path):
    """Tool results are redacted before logging."""
    text = written(tmp_path, tool_result(f"ANTHROPIC_API_KEY={ANTHROPIC}"))

    assert ANTHROPIC not in text

def test_a_key_in_an_error_message_never_reaches_the_log(tmp_path: Path):
    """Error messages are redacted before logging."""
    text = written(
        tmp_path,
        AgentEvent(type=AgentEventType.ERROR, data=f"401 unauthorized (key {ANTHROPIC})"),
    )

    assert ANTHROPIC not in text

# what the detector has to recognise

@pytest.mark.parametrize("secret", [ANTHROPIC, OPENAI, GOOGLE, GITHUB])
def test_every_major_vendor_prefix_is_recognised(tmp_path: Path, secret: str):
    """Known vendor key prefixes are recognized."""
    assert secret not in written(tmp_path, tool_result(f"the key is {secret}"))

# Prefixless values require explicit registration.

def test_a_bare_prefixless_token_is_not_detected(tmp_path: Path):
    """Prefixless opaque values require explicit secret registration."""
    from lamssi_agents.redaction import forget_secrets

    forget_secrets()
    assert OPAQUE in written(tmp_path, tool_result(f"the value is {OPAQUE}"))

def test_a_registered_secret_is_masked_whatever_its_shape(tmp_path: Path):
    """Registered secrets are masked without relying on format heuristics."""
    from lamssi_agents.redaction import forget_secrets, register_secret

    forget_secrets()
    register_secret(OPAQUE)
    try:
        text = written(tmp_path, tool_result(f"the value is {OPAQUE}"))
        assert OPAQUE not in text
    finally:
        forget_secrets()

def test_the_configured_api_key_registers_itself(tmp_path: Path):
    """Model API keys register themselves for redaction."""
    from lamssi_agents import LiteLLMModel
    from lamssi_agents.redaction import forget_secrets, redact

    forget_secrets()
    try:
        LiteLLMModel("claude-sonnet-4-5", api_key=OPAQUE)
        assert OPAQUE not in redact(f"leaked somewhere: {OPAQUE}")
    finally:
        forget_secrets()

def test_a_key_in_a_model_error_is_masked():
    """Model errors are redacted before entering the transcript."""
    from lamssi_agents.providers.errors import clean_model_error

    cleaned = clean_model_error(
        RuntimeError(f"litellm.AuthenticationError: bad key {ANTHROPIC}")
    )
    assert ANTHROPIC not in cleaned
    assert "litellm." not in cleaned, "the original cleaning still applies"

def test_a_key_in_a_tool_result_never_reaches_the_model():
    """Tool-result truncation redacts secrets before model context."""
    from lamssi_agents.history.truncation import truncate_tool_result

    body = json.dumps({"stdout": f"ANTHROPIC_API_KEY={ANTHROPIC}"})
    assert ANTHROPIC not in truncate_tool_result(body, tool_name="run_bash")

# and what it must not do

def test_ordinary_text_is_left_alone(tmp_path: Path):
    """Preserve ordinary hashes, paths, and numeric settings."""
    ordinary = (
        "Read 412 lines from src/app.py. The commit is a1b2c3d4e5f6 and the "
        "checksum is 9f8e7d6c. Set timeout=30 and retries=3."
    )
    text = written(tmp_path, tool_result(ordinary))

    for fragment in ("a1b2c3d4e5f6", "9f8e7d6c", "timeout=30", "src/app.py"):
        assert fragment in text, f"redaction ate ordinary output: {fragment}"

def test_a_masked_key_stays_identifiable(tmp_path: Path):
    """A masked key retains a short non-sensitive identifier."""
    text = written(tmp_path, tool_result(f"key={ANTHROPIC}"))

    assert ANTHROPIC not in text
    assert ANTHROPIC[:6] in text, "keep a prefix so the key can be identified"
    assert ANTHROPIC[-4:] in text, "and a suffix"

def test_a_key_split_across_a_json_boundary_is_still_caught(tmp_path: Path):
    """Nested JSON serialization does not bypass secret redaction."""
    nested = json.dumps({"env": {"vars": [{"name": "KEY", "value": ANTHROPIC}]}})
    text = written(tmp_path, tool_result(nested))

    assert ANTHROPIC not in text
