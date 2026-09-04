"""``Diagnostics`` accumulates findings (errors and warnings) from a rule pass."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Finding:
    """One observation about a source file."""

    message: str
    severity: Severity = Severity.ERROR
    line: int = 0
    #: The rule that produced this, so a host can suppress it by name.
    rule: str = ""
    #: What to do instead: worth supplying, or a retry repeats the mistake.
    suggestion: str = ""

    def render(self) -> str:
        where = f"Line {self.line}: " if self.line else ""
        tail = f" {self.suggestion}" if self.suggestion else ""
        return f"{where}{self.message}{tail}"


@dataclass
class Diagnostics:
    """Findings from one pass, plus whatever the rules extracted along the way."""

    findings: List[Finding] = field(default_factory=list)
    #: Facts a rule learned that later rules or the caller may want, shared without a second parse.
    facts: Dict[str, Any] = field(default_factory=dict)

    def error(
        self, message: str, *, line: int = 0, rule: str = "", suggestion: str = ""
    ) -> None:
        self.findings.append(
            Finding(message, Severity.ERROR, line, rule, suggestion)
        )

    def warn(
        self, message: str, *, line: int = 0, rule: str = "", suggestion: str = ""
    ) -> None:
        self.findings.append(
            Finding(message, Severity.WARNING, line, rule, suggestion)
        )

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def valid(self) -> bool:
        """True when nothing was reported as an error."""
        return not self.errors

    def extend(self, other: "Diagnostics") -> None:
        self.findings.extend(other.findings)
        self.facts.update(other.facts)

    def to_result(self) -> Dict[str, Any]:
        """A JSON-safe summary, shaped for returning from a tool.

        Errors and warnings are kept separate rather than interleaved with a
        severity field, since a flat list reads as if the first entry mattered most.
        """
        out: Dict[str, Any] = {"valid": self.valid}
        if self.errors:
            out["errors"] = [f.render() for f in self.errors]
        if self.warnings:
            out["warnings"] = [f.render() for f in self.warnings]
        if self.facts:
            out["extracted"] = self.facts
        return out

    def __len__(self) -> int:
        return len(self.findings)

    def __bool__(self) -> bool:
        # Means "did this pass", not "has findings": a warning-only result passes.
        return self.valid


__all__ = ["Diagnostics", "Finding", "Severity"]
