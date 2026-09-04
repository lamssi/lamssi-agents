"""Structured output from one agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from lamssi_agents.providers import Usage

_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Structured outcome of one :meth:`Agent.run` request.

    Attributes:
        text: Final response text, including a readable failure message when the
            run could not complete.
        outputs: Named immutable output groups contributed by features.
        usage: Tokens consumed during this request, not cumulative model usage.
        turns: Number of model turns used by this request.
        aborted: Whether the application or user cancelled the run.
        error: Last reported run error, or ``None`` on success.
    """

    text: str
    outputs: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    aborted: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        """Copy and freeze feature outputs at the end of the run."""
        object.__setattr__(
            self,
            "outputs",
            MappingProxyType(
                {name: tuple(values) for name, values in self.outputs.items()}
            ),
        )

    @property
    def ok(self) -> bool:
        """Whether the run completed without cancellation or a reported error."""
        return not self.aborted and self.error is None

    @property
    def files_written(self) -> tuple[str, ...]:
        """Files changed by the Files feature during this run."""
        return tuple(str(path) for path in self.outputs.get("files_written", ()))

    def __str__(self) -> str:
        return self.text


def usage_snapshot(model: Any) -> Usage:
    """Copy a model adapter's cumulative counters, tolerating partial adapters."""
    source = getattr(model, "cumulative_usage", None)
    return Usage(
        **{field: int(getattr(source, field, 0) or 0) for field in _USAGE_FIELDS}
    )


def usage_since(before: Usage, model: Any) -> Usage:
    """Return non-negative usage accumulated after *before*."""
    after = usage_snapshot(model)
    return Usage(
        **{
            field: max(0, getattr(after, field) - getattr(before, field))
            for field in _USAGE_FIELDS
        }
    )


__all__ = ["RunResult"]
