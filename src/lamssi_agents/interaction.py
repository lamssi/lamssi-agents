"""Typed human interaction requests shared by scripts, CLIs, and GUI hosts."""

from __future__ import annotations

from copy import deepcopy
import logging
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional
from uuid import uuid4

from lamssi_agents.events import AgentEventType

log = logging.getLogger(__name__)


class InteractionKind(str, Enum):
    """Reason an agent is synchronously asking its application for input."""

    QUESTION = "question"
    GUARD_OVERRIDE = "guard_override"
    BUDGET_CHECKPOINT = "budget_checkpoint"


class InteractionDecision(str, Enum):
    """Continue or cancel decision for non-text interactions."""

    CONTINUE = "continue"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class InteractionRequest:
    """Immutable application-input request.

    Attributes:
        kind: Question, guard override, or budget checkpoint.
        prompt: Text the application should present to a user or controller.
        request_id: Unique correlation identifier also included in interaction events.
        metadata: Deeply isolated, read-only details such as available choices.
    """

    kind: InteractionKind
    prompt: str
    request_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze a snapshot so the handler cannot rewrite the emitted request."""
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(deepcopy(dict(self.metadata))),
        )


@dataclass(frozen=True, slots=True)
class InteractionResponse:
    """Application response to an :class:`InteractionRequest`.

    Use the named constructors rather than building contradictory combinations
    of answer and decision directly.

    Attributes:
        answer: Text supplied for a question.
        decision: Continue or cancel decision for a checkpoint.
    """

    answer: str = ""
    decision: Optional[InteractionDecision] = None

    @classmethod
    def answered(cls, text: str) -> "InteractionResponse":
        """Return a text answer.

        Args:
            text: Answer supplied to the waiting agent.

        Returns:
            Response containing ``answer`` and no checkpoint decision.
        """
        return cls(answer=str(text))

    @classmethod
    def continue_(cls) -> "InteractionResponse":
        """Return a decision to continue the current run."""
        return cls(decision=InteractionDecision.CONTINUE)

    @classmethod
    def cancel(cls) -> "InteractionResponse":
        """Return a decision to cancel the current run."""
        return cls(decision=InteractionDecision.CANCEL)


InteractionHandler = Callable[[InteractionRequest], InteractionResponse]


def request_interaction(
    handler: Optional[InteractionHandler],
    emit: Callable[..., Any],
    kind: InteractionKind,
    prompt: str,
    **metadata: Any,
) -> Optional[InteractionResponse]:
    """Emit and synchronously resolve one typed interaction, if a handler exists.

    Args:
        handler: The run's interaction handler, or ``None`` when unattended.
        emit: The run's event emit callable.
    """
    if handler is None:
        return None
    request = InteractionRequest(kind=kind, prompt=prompt, metadata=metadata)
    emit(
        AgentEventType.USER_INPUT_REQUEST,
        prompt,
        request_id=request.request_id,
        interaction_kind=kind.value,
        interaction_metadata=dict(metadata),
    )
    try:
        response = handler(request)
    except Exception as exc:
        log.warning("%s interaction failed: %s", kind.value, exc)
        emit(
            AgentEventType.USER_INPUT_RESPONSE,
            None,
            request_id=request.request_id,
            interaction_kind=kind.value,
            handled=False,
            error=str(exc),
        )
        return None
    if not isinstance(response, InteractionResponse):
        log.warning(
            "%s interaction returned %s, expected InteractionResponse",
            kind.value,
            type(response).__name__,
        )
        emit(
            AgentEventType.USER_INPUT_RESPONSE,
            None,
            request_id=request.request_id,
            interaction_kind=kind.value,
            handled=False,
            error="invalid interaction response",
        )
        return None
    valid = (
        response.decision is None
        if kind is InteractionKind.QUESTION
        else response.decision is not None and not response.answer
    )
    if not valid:
        log.warning("%s interaction returned an incompatible response", kind.value)
        emit(
            AgentEventType.USER_INPUT_RESPONSE,
            None,
            request_id=request.request_id,
            interaction_kind=kind.value,
            handled=False,
            error="incompatible interaction response",
        )
        return None
    emit(
        AgentEventType.USER_INPUT_RESPONSE,
        response,
        request_id=request.request_id,
        interaction_kind=kind.value,
        handled=True,
    )
    return response


__all__ = [
    "InteractionDecision",
    "InteractionHandler",
    "InteractionKind",
    "InteractionRequest",
    "InteractionResponse",
]
