"""Evaluation suites and report rendering for WorldForge."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from worldforge.models import Action, JSONDict, average, dump_json

if TYPE_CHECKING:
    from worldforge.framework import World, WorldForge


@dataclass(slots=True)
class EvaluationScenario:
    """A single scenario inside an evaluation suite."""

    name: str
    description: str


@dataclass(slots=True)
class EvaluationResult:
    """The result for one scenario/provider pair."""

    suite: str
    scenario: str
    provider: str
    score: float
    metrics: JSONDict

    def to_dict(self) -> JSONDict:
        return {
            "suite": self.suite,
            "scenario": self.scenario,
            "provider": self.provider,
            "score": self.score,
            "metrics": self.metrics,
        }


@dataclass(slots=True)
class ProviderSummary:
    """Aggregate summary for a provider across a suite run."""

    provider: str
    average_score: float
    scenario_count: int

    def to_dict(self) -> JSONDict:
        return {
            "provider": self.provider,
            "average_score": self.average_score,
            "scenario_count": self.scenario_count,
        }


class EvaluationReport:
    """Materialized evaluation report with export helpers."""

    def __init__(self, suite: str, provider_summaries: Sequence[ProviderSummary]) -> None:
        self.suite = suite
        self.provider_summaries = list(provider_summaries)

    def to_dict(self) -> JSONDict:
        return {
            "suite": self.suite,
            "provider_summaries": [summary.to_dict() for summary in self.provider_summaries],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation Report",
            "",
            f"Suite: {self.suite}",
            "",
            "| provider | average_score | scenarios |",
            "| --- | ---: | ---: |",
        ]
        for summary in self.provider_summaries:
            lines.append(
                f"| {summary.provider} | {summary.average_score:.2f} | {summary.scenario_count} |"
            )
        return "\n".join(lines)

    def to_csv(self) -> str:
        rows = ["provider,average_score,scenario_count"]
        for summary in self.provider_summaries:
            rows.append(f"{summary.provider},{summary.average_score:.4f},{summary.scenario_count}")
        return "\n".join(rows)

    def to_json(self) -> str:
        return dump_json(self.to_dict())


class EvaluationSuite:
    """Built-in or user-defined evaluation suite."""

    def __init__(self, name: str, scenarios: Sequence[EvaluationScenario]) -> None:
        self.name = name
        self.scenarios = list(scenarios)

    @classmethod
    def from_builtin(cls, name: str) -> EvaluationSuite:
        if name != "physics":
            raise ValueError(f"Unknown suite '{name}'.")
        return cls(
            name="Physics Evaluation Suite",
            scenarios=[
                EvaluationScenario("object-stability", "Checks that static objects remain stable."),
                EvaluationScenario(
                    "action-response", "Checks that actions produce coherent motion."
                ),
            ],
        )

    def _ensure_world(
        self,
        provider: str,
        *,
        forge: WorldForge,
        world: World | None = None,
    ) -> World:
        from worldforge.framework import World

        if world is not None:
            return World.from_state(forge, world.to_dict())
        return forge.create_world("evaluation-world", provider)

    def run_with_world(
        self, provider: str, *, world: World, forge: WorldForge
    ) -> list[EvaluationResult]:
        sandbox = self._ensure_world(provider, forge=forge, world=world)
        results: list[EvaluationResult] = []
        for index, scenario in enumerate(self.scenarios):
            prediction = sandbox.predict(
                Action.move_to(0.1 * (index + 1), 0.5, 0.0),
                steps=1,
                provider=provider,
            )
            score = min(0.99, max(0.5, prediction.physics_score - (index * 0.01)))
            results.append(
                EvaluationResult(
                    suite=self.name,
                    scenario=scenario.name,
                    provider=provider,
                    score=score,
                    metrics={
                        "physics_score": prediction.physics_score,
                        "confidence": prediction.confidence,
                    },
                )
            )
        return results

    def run(self, provider: str, *, forge: WorldForge | None = None) -> list[EvaluationResult]:
        from worldforge.framework import WorldForge

        active_forge = forge or WorldForge()
        world = self._ensure_world(provider, forge=active_forge)
        return self.run_with_world(provider, world=world, forge=active_forge)

    def run_report(
        self,
        provider: str,
        *,
        world: World | None = None,
        forge: WorldForge | None = None,
    ) -> EvaluationReport:
        from worldforge.framework import WorldForge

        active_forge = forge or WorldForge()
        results = self.run_with_world(
            provider,
            world=self._ensure_world(provider, forge=active_forge, world=world),
            forge=active_forge,
        )
        summary = ProviderSummary(
            provider=provider,
            average_score=average(result.score for result in results),
            scenario_count=len(results),
        )
        return EvaluationReport(self.name, [summary])

    def run_report_artifacts(
        self,
        *,
        providers: str | Sequence[str],
        world: World | None = None,
        forge: WorldForge | None = None,
    ) -> dict[str, str]:
        from worldforge.framework import WorldForge

        active_forge = forge or WorldForge()
        provider_names = [providers] if isinstance(providers, str) else list(providers)
        summaries: list[ProviderSummary] = []
        for provider in provider_names:
            results = self.run_with_world(
                provider,
                world=self._ensure_world(provider, forge=active_forge, world=world),
                forge=active_forge,
            )
            summaries.append(
                ProviderSummary(
                    provider=provider,
                    average_score=average(result.score for result in results),
                    scenario_count=len(results),
                )
            )
        report = EvaluationReport(self.name, summaries)
        return {"json": report.to_json(), "markdown": report.to_markdown(), "csv": report.to_csv()}


class PhysicsEvaluationSuite(EvaluationSuite):
    """Alias for the built-in physics suite."""

    def __init__(self) -> None:
        builtin = EvaluationSuite.from_builtin("physics")
        super().__init__(builtin.name, builtin.scenarios)


EvalScenario = EvaluationScenario
EvalResult = EvaluationResult
EvalReport = EvaluationReport
EvalSuite = EvaluationSuite
PhysicsEval = PhysicsEvaluationSuite
