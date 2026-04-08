"""Evaluation suites and report rendering for WorldForge."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from worldforge.models import (
    Action,
    BBox,
    JSONDict,
    Position,
    SceneObject,
    WorldForgeError,
    average,
    dump_json,
)

if TYPE_CHECKING:
    from worldforge.framework import World, WorldForge


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _distance(a: Position, b: Position) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def _seed_object(world: World, name: str, position: Position) -> SceneObject:
    existing = next((obj for obj in world.objects() if obj.name == name), None)
    if existing is not None:
        return existing
    obj = SceneObject(
        name,
        position,
        BBox(
            Position(position.x - 0.05, position.y - 0.05, position.z - 0.05),
            Position(position.x + 0.05, position.y + 0.05, position.z + 0.05),
        ),
        is_graspable=True,
    )
    world.add_object(obj)
    return obj


@dataclass(slots=True)
class EvaluationScenario:
    """A single scenario inside an evaluation suite."""

    name: str
    description: str
    required_capabilities: tuple[str, ...] = ()


@dataclass(slots=True)
class EvaluationResult:
    """The result for one scenario/provider pair."""

    suite_id: str
    suite: str
    scenario: str
    provider: str
    score: float
    passed: bool
    metrics: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = _clamp_score(self.score)

    def to_dict(self) -> JSONDict:
        return {
            "suite_id": self.suite_id,
            "suite": self.suite,
            "scenario": self.scenario,
            "provider": self.provider,
            "score": self.score,
            "passed": self.passed,
            "metrics": self.metrics,
        }


@dataclass(slots=True)
class ProviderSummary:
    """Aggregate summary for a provider across a suite run."""

    provider: str
    average_score: float
    scenario_count: int
    passed_scenario_count: int
    failed_scenario_count: int

    @property
    def pass_rate(self) -> float:
        if self.scenario_count == 0:
            return 0.0
        return self.passed_scenario_count / self.scenario_count

    def to_dict(self) -> JSONDict:
        return {
            "provider": self.provider,
            "average_score": self.average_score,
            "scenario_count": self.scenario_count,
            "passed_scenario_count": self.passed_scenario_count,
            "failed_scenario_count": self.failed_scenario_count,
            "pass_rate": self.pass_rate,
        }


class EvaluationReport:
    """Materialized evaluation report with export helpers."""

    def __init__(
        self,
        suite_id: str,
        suite: str,
        results: Sequence[EvaluationResult],
    ) -> None:
        self.suite_id = suite_id
        self.suite = suite
        self.results = list(results)
        self.provider_summaries = self._build_provider_summaries()

    def _build_provider_summaries(self) -> list[ProviderSummary]:
        provider_names = sorted({result.provider for result in self.results})
        summaries: list[ProviderSummary] = []
        for provider in provider_names:
            provider_results = [result for result in self.results if result.provider == provider]
            passed_count = sum(1 for result in provider_results if result.passed)
            summaries.append(
                ProviderSummary(
                    provider=provider,
                    average_score=average(result.score for result in provider_results),
                    scenario_count=len(provider_results),
                    passed_scenario_count=passed_count,
                    failed_scenario_count=len(provider_results) - passed_count,
                )
            )
        return summaries

    def to_dict(self) -> JSONDict:
        return {
            "suite_id": self.suite_id,
            "suite": self.suite,
            "provider_summaries": [summary.to_dict() for summary in self.provider_summaries],
            "results": [result.to_dict() for result in self.results],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation Report",
            "",
            f"Suite: {self.suite} ({self.suite_id})",
            "",
            "| provider | average_score | passed | scenarios |",
            "| --- | ---: | ---: | ---: |",
        ]
        for summary in self.provider_summaries:
            lines.append(
                f"| {summary.provider} | {summary.average_score:.2f} | "
                f"{summary.passed_scenario_count}/{summary.scenario_count} | "
                f"{summary.scenario_count} |"
            )

        lines.extend(
            [
                "",
                "| provider | scenario | score | passed |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for result in self.results:
            lines.append(
                f"| {result.provider} | {result.scenario} | {result.score:.2f} | "
                f"{'yes' if result.passed else 'no'} |"
            )
        return "\n".join(lines)

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "suite_id",
                "suite",
                "provider",
                "scenario",
                "score",
                "passed",
                "metrics_json",
            ],
        )
        writer.writeheader()
        for result in self.results:
            writer.writerow(
                {
                    "suite_id": self.suite_id,
                    "suite": self.suite,
                    "provider": result.provider,
                    "scenario": result.scenario,
                    "score": f"{result.score:.4f}",
                    "passed": str(result.passed).lower(),
                    "metrics_json": json.dumps(result.metrics, sort_keys=True),
                }
            )
        return buffer.getvalue().strip()

    def to_json(self) -> str:
        return dump_json(self.to_dict())

    def artifacts(self) -> dict[str, str]:
        return {
            "json": self.to_json(),
            "markdown": self.to_markdown(),
            "csv": self.to_csv(),
        }


class EvaluationSuite:
    """Built-in or user-defined evaluation suite."""

    def __init__(
        self,
        name: str,
        scenarios: Sequence[EvaluationScenario],
        *,
        suite_id: str | None = None,
    ) -> None:
        self.name = name
        self.scenarios = list(scenarios)
        self.suite_id = suite_id or name.lower().replace(" ", "-")

    @classmethod
    def _builtin_registry(cls) -> dict[str, Callable[[], EvaluationSuite]]:
        return {
            "physics": PhysicsEvaluationSuite,
            "planning": PlanningEvaluationSuite,
            "reasoning": ReasoningEvaluationSuite,
        }

    @classmethod
    def builtin_names(cls) -> list[str]:
        return sorted(cls._builtin_registry())

    @classmethod
    def from_builtin(cls, name: str) -> EvaluationSuite:
        registry = cls._builtin_registry()
        try:
            factory = registry[name]
        except KeyError as exc:
            known = ", ".join(sorted(registry))
            raise WorldForgeError(
                f"Unknown evaluation suite '{name}'. Known suites: {known}."
            ) from exc
        return factory()

    def _required_capabilities(self) -> tuple[str, ...]:
        names = {
            capability
            for scenario in self.scenarios
            for capability in scenario.required_capabilities
        }
        return tuple(sorted(names))

    def _require_provider_capabilities(self, provider: str, *, forge: WorldForge) -> None:
        profile = forge.provider_profile(provider)
        missing = [
            capability
            for capability in self._required_capabilities()
            if not profile.capabilities.supports(capability)
        ]
        if missing:
            joined = ", ".join(missing)
            raise WorldForgeError(
                f"Provider '{provider}' cannot run evaluation suite '{self.suite_id}': "
                f"missing required capabilities: {joined}."
            )

    def _build_world(self, provider: str, *, forge: WorldForge) -> World:
        return forge.create_world(f"{self.suite_id}-evaluation-world", provider)

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
        return self._build_world(provider, forge=forge)

    def evaluate_scenario(
        self,
        scenario: EvaluationScenario,
        provider: str,
        *,
        world: World,
        forge: WorldForge,
        index: int,
    ) -> EvaluationResult:
        prediction = world.predict(
            Action.move_to(0.1 * (index + 1), 0.5, 0.0),
            steps=1,
            provider=provider,
        )
        score = _clamp_score((prediction.physics_score + prediction.confidence) / 2)
        return EvaluationResult(
            suite_id=self.suite_id,
            suite=self.name,
            scenario=scenario.name,
            provider=provider,
            score=score,
            passed=score >= 0.7,
            metrics={
                "physics_score": prediction.physics_score,
                "confidence": prediction.confidence,
            },
        )

    def run_with_world(
        self,
        provider: str,
        *,
        world: World,
        forge: WorldForge,
    ) -> list[EvaluationResult]:
        self._require_provider_capabilities(provider, forge=forge)
        base_world = self._ensure_world(provider, forge=forge, world=world)
        results: list[EvaluationResult] = []
        for index, scenario in enumerate(self.scenarios):
            sandbox = self._ensure_world(provider, forge=forge, world=base_world)
            results.append(
                self.evaluate_scenario(
                    scenario,
                    provider,
                    world=sandbox,
                    forge=forge,
                    index=index,
                )
            )
        return results

    def run(self, provider: str, *, forge: WorldForge | None = None) -> list[EvaluationResult]:
        from worldforge.framework import WorldForge

        active_forge = forge or WorldForge()
        self._require_provider_capabilities(provider, forge=active_forge)
        world = self._ensure_world(provider, forge=active_forge)
        return self.run_with_world(provider, world=world, forge=active_forge)

    def run_report(
        self,
        providers: str | Sequence[str],
        *,
        world: World | None = None,
        forge: WorldForge | None = None,
    ) -> EvaluationReport:
        from worldforge.framework import WorldForge

        active_forge = forge or WorldForge()
        provider_names = [providers] if isinstance(providers, str) else list(providers)
        if not provider_names:
            raise WorldForgeError("run_report() requires at least one provider.")

        results: list[EvaluationResult] = []
        for provider in provider_names:
            self._require_provider_capabilities(provider, forge=active_forge)
            results.extend(
                self.run_with_world(
                    provider,
                    world=self._ensure_world(provider, forge=active_forge, world=world),
                    forge=active_forge,
                )
            )
        return EvaluationReport(self.suite_id, self.name, results)

    def run_report_artifacts(
        self,
        *,
        providers: str | Sequence[str],
        world: World | None = None,
        forge: WorldForge | None = None,
    ) -> dict[str, str]:
        report = self.run_report(providers=providers, world=world, forge=forge)
        return report.artifacts()


class PhysicsEvaluationSuite(EvaluationSuite):
    """Built-in suite for deterministic physics-style checks."""

    def __init__(self) -> None:
        super().__init__(
            "Physics Evaluation Suite",
            scenarios=[
                EvaluationScenario(
                    "object-stability",
                    "Checks that an object remains stable under a no-op move.",
                    required_capabilities=("predict",),
                ),
                EvaluationScenario(
                    "action-response",
                    "Checks that a move action reaches the target pose.",
                    required_capabilities=("predict",),
                ),
            ],
            suite_id="physics",
        )

    def _build_world(self, provider: str, *, forge: WorldForge) -> World:
        world = super()._build_world(provider, forge=forge)
        _seed_object(world, "cube", Position(0.0, 0.5, 0.0))
        return world

    def evaluate_scenario(
        self,
        scenario: EvaluationScenario,
        provider: str,
        *,
        world: World,
        forge: WorldForge,
        index: int,
    ) -> EvaluationResult:
        primary = _seed_object(world, "cube", Position(0.0, 0.5, 0.0))
        start = primary.position

        if scenario.name == "object-stability":
            prediction = world.predict(
                Action.move_to(start.x, start.y, start.z),
                steps=1,
                provider=provider,
            )
            current = world.get_object_by_id(primary.id)
            if current is None:  # pragma: no cover - world state corruption guard
                raise WorldForgeError("Evaluation lost the primary object during physics run.")
            displacement = _distance(start, current.position)
            passed = displacement <= 0.01 and prediction.physics_score >= 0.7
            score = _clamp_score(
                ((prediction.physics_score + prediction.confidence) / 2) - min(0.25, displacement)
            )
            return EvaluationResult(
                suite_id=self.suite_id,
                suite=self.name,
                scenario=scenario.name,
                provider=provider,
                score=score,
                passed=passed,
                metrics={
                    "physics_score": prediction.physics_score,
                    "confidence": prediction.confidence,
                    "displacement": displacement,
                    "step": world.step,
                },
            )

        if scenario.name == "action-response":
            target = Position(start.x + 0.35, start.y, start.z)
            prediction = world.predict(
                Action.move_to(target.x, target.y, target.z),
                steps=2,
                provider=provider,
            )
            current = world.get_object_by_id(primary.id)
            if current is None:  # pragma: no cover - world state corruption guard
                raise WorldForgeError("Evaluation lost the primary object during physics run.")
            target_error = _distance(target, current.position)
            moved_distance = _distance(start, current.position)
            passed = target_error <= 0.05 and moved_distance >= 0.3
            score = _clamp_score(
                ((prediction.physics_score + prediction.confidence) / 2)
                + min(0.2, moved_distance / 2)
                - min(0.3, target_error)
            )
            return EvaluationResult(
                suite_id=self.suite_id,
                suite=self.name,
                scenario=scenario.name,
                provider=provider,
                score=score,
                passed=passed,
                metrics={
                    "physics_score": prediction.physics_score,
                    "confidence": prediction.confidence,
                    "moved_distance": moved_distance,
                    "target_error": target_error,
                    "step": world.step,
                },
            )

        return super().evaluate_scenario(
            scenario,
            provider,
            world=world,
            forge=forge,
            index=index,
        )


class PlanningEvaluationSuite(EvaluationSuite):
    """Built-in suite for heuristic planning and execution checks."""

    def __init__(self) -> None:
        super().__init__(
            "Planning Evaluation Suite",
            scenarios=[
                EvaluationScenario(
                    "object-relocation",
                    "Plans and executes a relocation objective for a seeded object.",
                    required_capabilities=("predict",),
                ),
                EvaluationScenario(
                    "object-spawn",
                    "Plans and executes a simple spawn goal.",
                    required_capabilities=("predict",),
                ),
            ],
            suite_id="planning",
        )

    def _build_world(self, provider: str, *, forge: WorldForge) -> World:
        world = super()._build_world(provider, forge=forge)
        _seed_object(world, "cube", Position(0.0, 0.5, 0.0))
        return world

    def evaluate_scenario(
        self,
        scenario: EvaluationScenario,
        provider: str,
        *,
        world: World,
        forge: WorldForge,
        index: int,
    ) -> EvaluationResult:
        primary = _seed_object(world, "cube", Position(0.0, 0.5, 0.0))

        if scenario.name == "object-relocation":
            plan = world.plan(goal="move the cube to the right", max_steps=4, provider=provider)
            execution = world.execute_plan(plan, provider)
            final_world = execution.final_world()
            final_object = final_world.get_object_by_id(primary.id)
            if final_object is None:  # pragma: no cover - world state corruption guard
                raise WorldForgeError("Evaluation lost the primary object during plan execution.")
            moved_distance = final_object.position.x - primary.position.x
            passed = plan.action_count >= 1 and moved_distance >= 0.25
            score = _clamp_score((plan.success_probability + min(1.0, moved_distance)) / 2)
            return EvaluationResult(
                suite_id=self.suite_id,
                suite=self.name,
                scenario=scenario.name,
                provider=provider,
                score=score,
                passed=passed,
                metrics={
                    "action_count": plan.action_count,
                    "success_probability": plan.success_probability,
                    "moved_distance": moved_distance,
                    "final_step": final_world.step,
                },
            )

        if scenario.name == "object-spawn":
            initial_count = world.object_count
            plan = world.plan(goal="spawn cube", max_steps=3, provider=provider)
            execution = world.execute_plan(plan, provider)
            final_world = execution.final_world()
            final_count = final_world.object_count
            spawned = final_count > initial_count
            score = _clamp_score((plan.success_probability + (1.0 if spawned else 0.0)) / 2)
            return EvaluationResult(
                suite_id=self.suite_id,
                suite=self.name,
                scenario=scenario.name,
                provider=provider,
                score=score,
                passed=spawned,
                metrics={
                    "action_count": plan.action_count,
                    "success_probability": plan.success_probability,
                    "initial_object_count": initial_count,
                    "final_object_count": final_count,
                },
            )

        return super().evaluate_scenario(
            scenario,
            provider,
            world=world,
            forge=forge,
            index=index,
        )


class ReasoningEvaluationSuite(EvaluationSuite):
    """Built-in suite for scene reasoning quality checks."""

    def __init__(self) -> None:
        super().__init__(
            "Reasoning Evaluation Suite",
            scenarios=[
                EvaluationScenario(
                    "scene-count",
                    "Checks whether the provider reports the tracked object count.",
                    required_capabilities=("reason",),
                ),
                EvaluationScenario(
                    "scene-identity",
                    "Checks whether provider evidence references tracked object identifiers.",
                    required_capabilities=("reason",),
                ),
            ],
            suite_id="reasoning",
        )

    def _build_world(self, provider: str, *, forge: WorldForge) -> World:
        world = super()._build_world(provider, forge=forge)
        _seed_object(world, "cube", Position(0.0, 0.5, 0.0))
        _seed_object(world, "mug", Position(0.3, 0.8, 0.0))
        return world

    def evaluate_scenario(
        self,
        scenario: EvaluationScenario,
        provider: str,
        *,
        world: World,
        forge: WorldForge,
        index: int,
    ) -> EvaluationResult:
        _seed_object(world, "cube", Position(0.0, 0.5, 0.0))
        _seed_object(world, "mug", Position(0.3, 0.8, 0.0))
        object_ids = sorted(obj.id for obj in world.objects())

        if scenario.name == "scene-count":
            expected_count = world.object_count
            reasoning = forge.reason(provider, "How many objects are tracked?", world=world)
            answer = reasoning.answer.lower()
            mentions_count = str(expected_count) in answer
            has_evidence = bool(reasoning.evidence)
            score = _clamp_score(
                (
                    reasoning.confidence
                    + (1.0 if mentions_count else 0.0)
                    + (1.0 if has_evidence else 0.0)
                )
                / 3
            )
            return EvaluationResult(
                suite_id=self.suite_id,
                suite=self.name,
                scenario=scenario.name,
                provider=provider,
                score=score,
                passed=mentions_count and has_evidence,
                metrics={
                    "confidence": reasoning.confidence,
                    "expected_count": expected_count,
                    "evidence_count": len(reasoning.evidence),
                    "mentions_count": mentions_count,
                },
            )

        if scenario.name == "scene-identity":
            reasoning = forge.reason(provider, "Which object ids are tracked?", world=world)
            haystack = " ".join([reasoning.answer, *reasoning.evidence]).lower()
            matched_ids = [object_id for object_id in object_ids if object_id.lower() in haystack]
            coverage = len(matched_ids) / len(object_ids) if object_ids else 1.0
            score = _clamp_score((reasoning.confidence + coverage) / 2)
            return EvaluationResult(
                suite_id=self.suite_id,
                suite=self.name,
                scenario=scenario.name,
                provider=provider,
                score=score,
                passed=coverage == 1.0,
                metrics={
                    "confidence": reasoning.confidence,
                    "tracked_object_count": len(object_ids),
                    "matched_object_count": len(matched_ids),
                    "coverage": coverage,
                },
            )

        return super().evaluate_scenario(
            scenario,
            provider,
            world=world,
            forge=forge,
            index=index,
        )


EvalScenario = EvaluationScenario
EvalResult = EvaluationResult
EvalReport = EvaluationReport
EvalSuite = EvaluationSuite
PhysicsEval = PhysicsEvaluationSuite
PlanningEval = PlanningEvaluationSuite
ReasoningEval = ReasoningEvaluationSuite
