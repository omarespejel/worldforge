"""Typed models for TheWorldHarness flow orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from worldforge.models import JSONDict


@dataclass(frozen=True, slots=True)
class HarnessFlow:
    """A runnable WorldForge demonstration flow exposed through TheWorldHarness."""

    id: str
    title: str
    short_title: str
    focus: str
    provider: str
    capability: str
    command: str
    accent: str
    summary: str

    def to_dict(self) -> JSONDict:
        return {
            "id": self.id,
            "title": self.title,
            "short_title": self.short_title,
            "focus": self.focus,
            "provider": self.provider,
            "capability": self.capability,
            "command": self.command,
            "accent": self.accent,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class HarnessStep:
    """One visible step in a harness run."""

    title: str
    detail: str
    result: str
    artifact: str = ""

    def to_dict(self) -> JSONDict:
        return {
            "title": self.title,
            "detail": self.detail,
            "result": self.result,
            "artifact": self.artifact,
        }


@dataclass(frozen=True, slots=True)
class HarnessMetric:
    """A compact metric shown in the harness inspector."""

    label: str
    value: str
    detail: str = ""

    def to_dict(self) -> JSONDict:
        return {
            "label": self.label,
            "value": self.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class HarnessRun:
    """Captured output from a completed harness flow."""

    flow: HarnessFlow
    state_dir: Path
    summary: JSONDict
    steps: tuple[HarnessStep, ...]
    metrics: tuple[HarnessMetric, ...]
    transcript: tuple[str, ...]

    def to_dict(self) -> JSONDict:
        return {
            "flow": self.flow.to_dict(),
            "state_dir": str(self.state_dir),
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
            "metrics": [metric.to_dict() for metric in self.metrics],
            "transcript": list(self.transcript),
        }
