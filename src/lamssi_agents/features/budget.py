"""Ask a person whether a run that is costing a lot should keep going."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

from lamssi_agents.events import AgentEventType
from lamssi_agents.interaction import (
    InteractionDecision,
    InteractionKind,
    request_interaction,
)
from lamssi_agents.features.base import Feature
from lamssi_agents.providers import Message

log = logging.getLogger(__name__)


def _fmt(n: int) -> str:
    return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{max(n // 1000, 0)}k"


if TYPE_CHECKING:
    from lamssi_agents.agent.base import Agent


def _model_tokens(model: object) -> int:
    usage = getattr(model, "cumulative_usage", None)
    return int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0


def summary_tokens(agent: "Agent") -> int:
    """Tokens spent by a separate compaction model, if one is configured."""
    summariser = getattr(agent, "_summary_model", None)
    if summariser is None or summariser is agent._model:
        return 0
    return _model_tokens(summariser)


def cumulative_tokens(agent: "Agent") -> int:
    """Total model spend for this Agent, including a separate summariser."""
    return _model_tokens(agent._model) + summary_tokens(agent)


class _BudgetState:
    """Per-conversation token checkpoint state."""

    def __init__(self) -> None:
        self.start = 0
        self.next = 0


class Budget(Feature):
    """Stop and ask once a run crosses *every_tokens*, then again each time.

    Leave the feature out and a run is bounded only by ``max_turns``. With nobody
    reachable the run continues (and logs it) rather than stopping, so an
    unattended job is not stranded.
    """

    name = "budget"

    def __init__(self, every_tokens: int = 0) -> None:
        self.every_tokens = int(every_tokens)

    def _every(self, agent: "Agent") -> int:
        """This feature's interval, read live so applications may retune it."""
        return max(0, int(self.every_tokens))

    def before_turn(self, agent: "Agent", turn: int) -> Optional[str]:
        every = self._every(agent)
        if not every:
            return None

        state = agent.conversation_state(_BudgetState, _BudgetState)
        spent = cumulative_tokens(agent)
        if turn == 1:
            state.start = spent
            state.next = every

        used = spent - state.start
        if used < state.next:
            return None

        answered, keep_going = self._confirm(agent, used, summary_tokens(agent))
        if keep_going:
            state.next = used + every
            return None
        return self._stop(agent, used, answered=answered)

    def _confirm(self, agent: "Agent", used: int, housekeeping: int = 0) -> Tuple[bool, bool]:
        """Returns ``(a person answered, keep going)``: two flags because silence and refusal need different reports."""
        # Shown only when knowable: "half of that was compaction" changes what a person does with the number.
        share = (
            f" ({_fmt(housekeeping)} of that on history compaction)"
            if housekeeping else ""
        )
        # Includes the request count: every turn re-sends the full history, so cumulative spend can be many multiples of a small, steady context.
        window = agent._conversation.context_usage()
        question = (
            f"This run has used about {_fmt(used)} tokens{share} across "
            f"{agent._conversation.turn} request(s); the conversation itself is about "
            f"{_fmt(window.used)}. Continue? "
            "(yes to keep going, anything else to stop here)"
        )

        response = request_interaction(
            agent._control.interaction.handler,
            agent.emit,
            InteractionKind.BUDGET_CHECKPOINT,
            question,
            used_tokens=used,
            compaction_tokens=housekeeping,
            requests=agent._conversation.turn,
            context_tokens=window.used,
        )
        if response is None:
            log.warning("Budget checkpoint at ~%d, nobody answered: continuing.", used)
            return False, True
        return True, response.decision is InteractionDecision.CONTINUE

    def _stop(self, agent: "Agent", used: int, *, answered: bool = True) -> str:
        why = (
            "at your request" if answered
            else "and the confirmation prompt went unanswered"
        )
        msg = (
            f"Stopped at about {_fmt(used)} tokens {why}. "
            "Send another message to continue."
        )
        agent._conversation.append(Message(role="assistant", content=msg))
        agent.emit(AgentEventType.TEXT_DONE, msg)
        agent.emit(AgentEventType.DONE)
        return msg


__all__ = ["Budget"]
