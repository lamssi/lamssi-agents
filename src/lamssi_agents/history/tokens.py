"""Estimates tokens from character count, calibrating the chars-per-token ratio from what the provider actually charges."""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from lamssi_agents.providers import Message
from lamssi_agents.vision import IMAGE_TOKEN_EST

log = logging.getLogger(__name__)

#: Conservative initial character-to-token ratio.
DEFAULT_CHARS_PER_TOKEN = 3.0

#: Plausible range for measured character-to-token ratios.
MIN_RATIO = 1.5
MAX_RATIO = 8.0

#: EMA weight per calibration sample.
_ALPHA = 0.3

#: Coarse divisor for a human-readable ~token figure in a log/telemetry line only.
#: Kept apart from the calibrating estimator: never use it for a budget decision.
ROUGH_CHARS_PER_TOKEN = 3.5


def rough_tokens(chars: int) -> int:
    """A quick ~token count for a log/telemetry line only; not for budgeting."""
    return int(chars / ROUGH_CHARS_PER_TOKEN)

#: Samples below this are ignored: mostly fixed per-request overhead, not content.
_MIN_SAMPLE_CHARS = 200

_DEFAULT_CALIBRATOR: "TokenCalibrator | None" = None


class TokenCalibrator:
    """Characters-per-token, learned from what the provider charges.

    One per conversation. Not shared across agents: another may be on a different
    model, and averaging two tokenisers together would give a number describing neither.
    """

    __slots__ = ("_ratio", "_samples", "_last_reported", "anchor")

    def __init__(self, initial: float = DEFAULT_CHARS_PER_TOKEN) -> None:
        self._ratio = float(initial)
        self._samples = 0
        self._last_reported: Optional[int] = None

        #: Index of the last message :attr:`last_reported_tokens` covers; everything after must be estimated. ``-1`` = no usable anchor.
        self.anchor: int = -1

    # A rewritten history makes the last reported figure stale; the learned ratio survives since it describes the model, not the conversation.

    def on_compacted(self) -> None:
        self.anchor = -1

    #: A demotion moves the char count without folding a message away, so this anchor is as stale as after a summary.
    on_demoted = on_compacted

    def on_cleared(self) -> None:
        self.anchor = -1

    @property
    def ratio(self) -> float:
        """Characters per token, as currently believed."""
        return self._ratio

    @property
    def calibrated(self) -> bool:
        """Whether any real reading has been folded in yet."""
        return self._samples > 0

    @property
    def last_reported_tokens(self) -> Optional[int]:
        """What the provider charged for the most recent request, if it said."""
        return self._last_reported

    def observe(self, chars: int, prompt_tokens: int) -> None:
        """Fold in one sample: *chars* sent, *prompt_tokens* charged.

        Ignored when either side is missing, zero, or implausible.
        """
        if chars < _MIN_SAMPLE_CHARS or prompt_tokens <= 0:
            return

        self._last_reported = prompt_tokens
        observed = chars / prompt_tokens
        if not (MIN_RATIO <= observed <= MAX_RATIO):
            log.debug(
                "ignoring an implausible token ratio: %d chars / %d tokens = %.2f",
                chars, prompt_tokens, observed,
            )
            return

        before = self._ratio
        self._ratio = (
            observed if self._samples == 0
            else (1 - _ALPHA) * self._ratio + _ALPHA * observed
        )
        self._samples += 1

        if abs(self._ratio - before) / max(before, 0.01) > 0.15:
            log.info(
                "token estimate recalibrated: %.2f -> %.2f chars/token "
                "(%d chars charged as %d tokens)",
                before, self._ratio, chars, prompt_tokens,
            )

    def estimate(self, chars: int) -> int:
        """Tokens for *chars* of text, at the current ratio."""
        return int(chars / self._ratio) if chars > 0 else 0

    def __repr__(self) -> str:
        state = f"{self._ratio:.2f}"
        return (
            f"<TokenCalibrator {state} chars/token, "
            f"{self._samples} sample(s){'' if self._samples else ', uncalibrated'}>"
        )


def estimate_tokens(
    messages: List[Message], calibrator: Optional[TokenCalibrator] = None
) -> int:
    """Estimate the complete token footprint of a list of messages."""
    global _DEFAULT_CALIBRATOR
    if calibrator is None:
        if _DEFAULT_CALIBRATOR is None:
            _DEFAULT_CALIBRATOR = TokenCalibrator()
        calibrator = _DEFAULT_CALIBRATOR

    chars = sum(len(message.content or "") for message in messages)
    chars += sum(
        len(json.dumps(call.arguments, default=str))
        for message in messages
        for call in (message.tool_calls or ())
    )
    images = sum(len(message.images) for message in messages if message.images)
    return calibrator.estimate(chars) + images * IMAGE_TOKEN_EST


__all__ = [
    "TokenCalibrator",
    "estimate_tokens",
    "DEFAULT_CHARS_PER_TOKEN",
    "MIN_RATIO",
    "MAX_RATIO",
    "ROUGH_CHARS_PER_TOKEN",
    "rough_tokens",
]
