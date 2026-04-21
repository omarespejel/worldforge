"""Capability-aware provider benchmark harness."""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter

from worldforge.framework import WorldForge
from worldforge.models import (
    Action,
    BBox,
    JSONDict,
    Position,
    SceneObject,
    VideoClip,
    WorldForgeError,
    dump_json,
    require_finite_number,
    require_non_negative_int,
    require_positive_int,
    require_probability,
)
from worldforge.observability import ProviderMetricsSink, compose_event_handlers
from worldforge.providers.base import ProviderError

BENCHMARKABLE_OPERATIONS = (
    "predict",
    "reason",
    "generate",
    "transfer",
    "embed",
    "score",
    "policy",
)


def _sample_transfer_clip() -> VideoClip:
    return VideoClip(
        frames=[b"worldforge-benchmark-transfer-seed"],
        fps=8.0,
        resolution=(160, 90),
        duration_seconds=1.0,
        metadata={
            "provider": "worldforge",
            "content_type": "video/mp4",
            "mode": "benchmark-seed",
        },
    )


def _sample_score_info() -> JSONDict:
    return {
        "pixels": [[[[0.0], [0.1]], [[0.2], [0.3]]]],
        "goal": [[[0.3, 0.5, 0.0]]],
        "action": [[[0.0, 0.5, 0.0]]],
        "metadata": {"mode": "benchmark-score"},
    }


def _sample_score_action_candidates() -> list[list[list[list[float]]]]:
    return [
        [
            [[0.0, 0.5, 0.0], [0.1, 0.5, 0.0]],
            [[0.0, 0.5, 0.0], [0.3, 0.5, 0.0]],
        ]
    ]


def _sample_policy_info() -> JSONDict:
    return {
        "observation": {
            "state": {
                "cube": [0.0, 0.5, 0.0],
                "mug": [0.25, 0.8, 0.0],
            },
            "language": "move the cube toward the target",
        },
        "options": {"temperature": 0.0},
        "mode": "select_action",
        "action_horizon": 2,
        "embodiment_tag": "benchmark",
    }


def _json_input_preview(value: object) -> object:
    try:
        dump_json(value)
    except WorldForgeError:
        payload: JSONDict = {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "json_serializable": False,
        }
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                payload["shape"] = [int(dimension) for dimension in shape]
            except (TypeError, ValueError):
                payload["shape"] = [str(dimension) for dimension in shape]
        return payload
    return value


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = quantile * (len(ordered) - 1)
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    weight = index - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _non_empty_optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise WorldForgeError(f"{name} must be a non-empty string when provided.")
    return value.strip()


def _optional_non_negative_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return require_non_negative_int(value, name=name)


def _optional_non_negative_number(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    number = require_finite_number(value, name=name)
    if number < 0.0:
        raise WorldForgeError(f"{name} must be greater than or equal to 0.")
    return number


def _format_optional_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


@dataclass(slots=True, frozen=True)
class BenchmarkBudget:
    """Thresholds for release or claim-oriented benchmark gates.

    ``provider`` and ``operation`` are optional selectors. When either selector is omitted, the
    budget applies to every matching result on that dimension.
    """

    provider: str | None = None
    operation: str | None = None
    min_success_rate: float | None = None
    max_error_count: int | None = None
    max_retry_count: int | None = None
    max_average_latency_ms: float | None = None
    max_p95_latency_ms: float | None = None
    min_throughput_per_second: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider",
            _non_empty_optional_text(self.provider, name="BenchmarkBudget provider"),
        )
        operation = _non_empty_optional_text(
            self.operation,
            name="BenchmarkBudget operation",
        )
        if operation is not None and operation not in BENCHMARKABLE_OPERATIONS:
            known = ", ".join(BENCHMARKABLE_OPERATIONS)
            raise WorldForgeError(f"BenchmarkBudget operation must be one of: {known}.")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(
            self,
            "min_success_rate",
            (
                require_probability(
                    self.min_success_rate,
                    name="BenchmarkBudget min_success_rate",
                )
                if self.min_success_rate is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "max_error_count",
            _optional_non_negative_int(
                self.max_error_count,
                name="BenchmarkBudget max_error_count",
            ),
        )
        object.__setattr__(
            self,
            "max_retry_count",
            _optional_non_negative_int(
                self.max_retry_count,
                name="BenchmarkBudget max_retry_count",
            ),
        )
        object.__setattr__(
            self,
            "max_average_latency_ms",
            _optional_non_negative_number(
                self.max_average_latency_ms,
                name="BenchmarkBudget max_average_latency_ms",
            ),
        )
        object.__setattr__(
            self,
            "max_p95_latency_ms",
            _optional_non_negative_number(
                self.max_p95_latency_ms,
                name="BenchmarkBudget max_p95_latency_ms",
            ),
        )
        object.__setattr__(
            self,
            "min_throughput_per_second",
            _optional_non_negative_number(
                self.min_throughput_per_second,
                name="BenchmarkBudget min_throughput_per_second",
            ),
        )
        if not any(
            value is not None
            for value in (
                self.min_success_rate,
                self.max_error_count,
                self.max_retry_count,
                self.max_average_latency_ms,
                self.max_p95_latency_ms,
                self.min_throughput_per_second,
            )
        ):
            raise WorldForgeError("BenchmarkBudget requires at least one threshold.")

    @classmethod
    def from_dict(cls, payload: JSONDict) -> BenchmarkBudget:
        if not isinstance(payload, dict):
            raise WorldForgeError("Benchmark budget entries must be JSON objects.")
        return cls(
            provider=payload.get("provider"),
            operation=payload.get("operation"),
            min_success_rate=payload.get("min_success_rate"),
            max_error_count=payload.get("max_error_count"),
            max_retry_count=payload.get("max_retry_count"),
            max_average_latency_ms=payload.get("max_average_latency_ms"),
            max_p95_latency_ms=payload.get("max_p95_latency_ms"),
            min_throughput_per_second=payload.get("min_throughput_per_second"),
        )

    def matches(self, result: BenchmarkResult) -> bool:
        provider_matches = self.provider is None or self.provider == result.provider
        operation_matches = self.operation is None or self.operation == result.operation
        return provider_matches and operation_matches

    def selector_label(self) -> str:
        provider = self.provider or "*"
        operation = self.operation or "*"
        return f"{provider}/{operation}"

    def to_dict(self) -> JSONDict:
        return {
            "provider": self.provider,
            "operation": self.operation,
            "min_success_rate": self.min_success_rate,
            "max_error_count": self.max_error_count,
            "max_retry_count": self.max_retry_count,
            "max_average_latency_ms": self.max_average_latency_ms,
            "max_p95_latency_ms": self.max_p95_latency_ms,
            "min_throughput_per_second": self.min_throughput_per_second,
        }


@dataclass(slots=True, frozen=True)
class BenchmarkGateViolation:
    """One failed benchmark budget check."""

    provider: str
    operation: str
    metric: str
    observed: float | int | None
    threshold: float | int
    condition: str
    budget_selector: str

    def to_dict(self) -> JSONDict:
        return {
            "provider": self.provider,
            "operation": self.operation,
            "metric": self.metric,
            "observed": self.observed,
            "threshold": self.threshold,
            "condition": self.condition,
            "budget_selector": self.budget_selector,
        }


@dataclass(slots=True)
class BenchmarkGateReport:
    """Budget evaluation report for a benchmark run."""

    budgets: list[BenchmarkBudget]
    checked_result_count: int
    violations: list[BenchmarkGateViolation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def to_dict(self) -> JSONDict:
        return {
            "passed": self.passed,
            "budget_count": len(self.budgets),
            "checked_result_count": self.checked_result_count,
            "violation_count": self.violation_count,
            "budgets": [budget.to_dict() for budget in self.budgets],
            "violations": [violation.to_dict() for violation in self.violations],
        }

    def to_json(self) -> str:
        return dump_json(self.to_dict())

    def to_markdown(self) -> str:
        status = "passed" if self.passed else "failed"
        lines = [
            "# Benchmark Gate Report",
            "",
            f"Status: {status}",
            f"Budgets: {len(self.budgets)}",
            f"Checked results: {self.checked_result_count}",
            f"Violations: {self.violation_count}",
        ]
        if self.violations:
            lines.extend(
                [
                    "",
                    "| provider | operation | metric | observed | threshold | condition | budget |",
                    "| --- | --- | --- | ---: | ---: | --- | --- |",
                ]
            )
            for violation in self.violations:
                lines.append(
                    "| "
                    f"{violation.provider} | "
                    f"{violation.operation} | "
                    f"{violation.metric} | "
                    f"{_format_optional_number(violation.observed)} | "
                    f"{_format_optional_number(violation.threshold)} | "
                    f"{violation.condition} | "
                    f"{violation.budget_selector} |"
                )
        return "\n".join(lines)

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "provider",
                "operation",
                "metric",
                "observed",
                "threshold",
                "condition",
                "budget_selector",
            ],
        )
        writer.writeheader()
        for violation in self.violations:
            writer.writerow(
                {
                    "provider": violation.provider,
                    "operation": violation.operation,
                    "metric": violation.metric,
                    "observed": _format_optional_number(violation.observed),
                    "threshold": _format_optional_number(violation.threshold),
                    "condition": violation.condition,
                    "budget_selector": violation.budget_selector,
                }
            )
        return buffer.getvalue().strip()


def load_benchmark_budgets(payload: object) -> list[BenchmarkBudget]:
    """Parse benchmark budget JSON from a list or ``{"budgets": [...]}`` object."""

    budget_entries = payload
    if isinstance(payload, dict):
        budget_entries = payload.get("budgets")
    if not isinstance(budget_entries, list) or not budget_entries:
        raise WorldForgeError(
            "Benchmark budget payload must be a non-empty list or an object with a non-empty "
            "'budgets' list."
        )
    return [BenchmarkBudget.from_dict(entry) for entry in budget_entries]


@dataclass(slots=True)
class BenchmarkInputs:
    """Default inputs used by the provider benchmark harness."""

    prediction_action: Action = field(default_factory=lambda: Action.move_to(0.25, 0.5, 0.0))
    prediction_steps: int = 2
    reason_query: str = "How many objects are tracked?"
    generation_prompt: str = "benchmark orbiting cube"
    generation_duration_seconds: float = 1.0
    transfer_prompt: str = "benchmark transfer rerender"
    transfer_width: int = 320
    transfer_height: int = 180
    transfer_fps: float = 12.0
    transfer_clip: VideoClip = field(default_factory=_sample_transfer_clip)
    embedding_text: str = "benchmark cube state"
    score_info: JSONDict = field(default_factory=_sample_score_info)
    score_action_candidates: object = field(default_factory=_sample_score_action_candidates)
    policy_info: JSONDict = field(default_factory=_sample_policy_info)

    def __post_init__(self) -> None:
        require_positive_int(self.prediction_steps, name="prediction_steps")
        if self.generation_duration_seconds <= 0.0:
            raise WorldForgeError("generation_duration_seconds must be greater than 0.")
        if self.transfer_width <= 0 or self.transfer_height <= 0:
            raise WorldForgeError("transfer_width and transfer_height must be greater than 0.")
        if self.transfer_fps <= 0.0:
            raise WorldForgeError("transfer_fps must be greater than 0.")
        if not isinstance(self.embedding_text, str) or not self.embedding_text.strip():
            raise WorldForgeError("embedding_text must be a non-empty string.")
        if not isinstance(self.score_info, dict) or not self.score_info:
            raise WorldForgeError("score_info must be a non-empty JSON object.")
        dump_json(self.score_info)
        if self.score_action_candidates is None:
            raise WorldForgeError("score_action_candidates must not be None.")
        if not isinstance(self.policy_info, dict) or not self.policy_info:
            raise WorldForgeError("policy_info must be a non-empty JSON object.")
        dump_json(self.policy_info)

    def to_dict(self) -> JSONDict:
        return {
            "prediction_action": self.prediction_action.to_dict(),
            "prediction_steps": self.prediction_steps,
            "reason_query": self.reason_query,
            "generation_prompt": self.generation_prompt,
            "generation_duration_seconds": self.generation_duration_seconds,
            "transfer_prompt": self.transfer_prompt,
            "transfer_width": self.transfer_width,
            "transfer_height": self.transfer_height,
            "transfer_fps": self.transfer_fps,
            "transfer_clip": self.transfer_clip.to_dict(),
            "embedding_text": self.embedding_text,
            "score_info": dict(self.score_info),
            "score_action_candidates": _json_input_preview(self.score_action_candidates),
            "policy_info": dict(self.policy_info),
        }


@dataclass(slots=True)
class BenchmarkResult:
    """Aggregate result for one provider/operation benchmark case."""

    provider: str
    operation: str
    iterations: int
    concurrency: int
    success_count: int
    error_count: int
    retry_count: int
    total_time_ms: float
    average_latency_ms: float | None
    min_latency_ms: float | None
    max_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    throughput_per_second: float
    operation_metrics: JSONDict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return {
            "provider": self.provider,
            "operation": self.operation,
            "iterations": self.iterations,
            "concurrency": self.concurrency,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "total_time_ms": self.total_time_ms,
            "average_latency_ms": self.average_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "throughput_per_second": self.throughput_per_second,
            "operation_metrics": dict(self.operation_metrics),
            "errors": list(self.errors),
        }


def _add_max_violation(
    violations: list[BenchmarkGateViolation],
    *,
    budget: BenchmarkBudget,
    result: BenchmarkResult,
    metric: str,
    observed: float | int | None,
    threshold: float | int | None,
) -> None:
    if threshold is None:
        return
    if observed is None or observed > threshold:
        violations.append(
            BenchmarkGateViolation(
                provider=result.provider,
                operation=result.operation,
                metric=metric,
                observed=observed,
                threshold=threshold,
                condition=f"<= {_format_optional_number(threshold)}",
                budget_selector=budget.selector_label(),
            )
        )


def _add_min_violation(
    violations: list[BenchmarkGateViolation],
    *,
    budget: BenchmarkBudget,
    result: BenchmarkResult,
    metric: str,
    observed: float | int | None,
    threshold: float | int | None,
) -> None:
    if threshold is None:
        return
    if observed is None or observed < threshold:
        violations.append(
            BenchmarkGateViolation(
                provider=result.provider,
                operation=result.operation,
                metric=metric,
                observed=observed,
                threshold=threshold,
                condition=f">= {_format_optional_number(threshold)}",
                budget_selector=budget.selector_label(),
            )
        )


def _evaluate_budget_for_result(
    budget: BenchmarkBudget,
    result: BenchmarkResult,
) -> list[BenchmarkGateViolation]:
    violations: list[BenchmarkGateViolation] = []
    success_rate = result.success_count / result.iterations if result.iterations else None
    _add_min_violation(
        violations,
        budget=budget,
        result=result,
        metric="success_rate",
        observed=success_rate,
        threshold=budget.min_success_rate,
    )
    _add_max_violation(
        violations,
        budget=budget,
        result=result,
        metric="error_count",
        observed=result.error_count,
        threshold=budget.max_error_count,
    )
    _add_max_violation(
        violations,
        budget=budget,
        result=result,
        metric="retry_count",
        observed=result.retry_count,
        threshold=budget.max_retry_count,
    )
    _add_max_violation(
        violations,
        budget=budget,
        result=result,
        metric="average_latency_ms",
        observed=result.average_latency_ms,
        threshold=budget.max_average_latency_ms,
    )
    _add_max_violation(
        violations,
        budget=budget,
        result=result,
        metric="p95_latency_ms",
        observed=result.p95_latency_ms,
        threshold=budget.max_p95_latency_ms,
    )
    _add_min_violation(
        violations,
        budget=budget,
        result=result,
        metric="throughput_per_second",
        observed=result.throughput_per_second,
        threshold=budget.min_throughput_per_second,
    )
    return violations


@dataclass(slots=True)
class BenchmarkReport:
    """Materialized benchmark report with export helpers."""

    results: list[BenchmarkResult]

    def to_dict(self) -> JSONDict:
        return {"results": [result.to_dict() for result in self.results]}

    def to_json(self) -> str:
        return dump_json(self.to_dict())

    def to_markdown(self) -> str:
        lines = [
            "# Benchmark Report",
            "",
            "| provider | operation | ok | retries | avg_ms | p95_ms | throughput/s |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for result in self.results:
            lines.append(
                f"| {result.provider} | {result.operation} | "
                f"{result.success_count}/{result.iterations} | {result.retry_count} | "
                f"{(result.average_latency_ms or 0.0):.2f} | "
                f"{(result.p95_latency_ms or 0.0):.2f} | "
                f"{result.throughput_per_second:.2f} |"
            )
        return "\n".join(lines)

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "provider",
                "operation",
                "iterations",
                "concurrency",
                "success_count",
                "error_count",
                "retry_count",
                "total_time_ms",
                "average_latency_ms",
                "min_latency_ms",
                "max_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "throughput_per_second",
                "operation_metrics_json",
                "errors_json",
            ],
        )
        writer.writeheader()
        for result in self.results:
            writer.writerow(
                {
                    "provider": result.provider,
                    "operation": result.operation,
                    "iterations": result.iterations,
                    "concurrency": result.concurrency,
                    "success_count": result.success_count,
                    "error_count": result.error_count,
                    "retry_count": result.retry_count,
                    "total_time_ms": f"{result.total_time_ms:.4f}",
                    "average_latency_ms": (
                        f"{result.average_latency_ms:.4f}"
                        if result.average_latency_ms is not None
                        else ""
                    ),
                    "min_latency_ms": (
                        f"{result.min_latency_ms:.4f}" if result.min_latency_ms is not None else ""
                    ),
                    "max_latency_ms": (
                        f"{result.max_latency_ms:.4f}" if result.max_latency_ms is not None else ""
                    ),
                    "p50_latency_ms": (
                        f"{result.p50_latency_ms:.4f}" if result.p50_latency_ms is not None else ""
                    ),
                    "p95_latency_ms": (
                        f"{result.p95_latency_ms:.4f}" if result.p95_latency_ms is not None else ""
                    ),
                    "throughput_per_second": f"{result.throughput_per_second:.4f}",
                    "operation_metrics_json": dump_json(result.operation_metrics),
                    "errors_json": dump_json(result.errors),
                }
            )
        return buffer.getvalue().strip()

    def artifacts(self) -> dict[str, str]:
        return {
            "json": self.to_json(),
            "markdown": self.to_markdown(),
            "csv": self.to_csv(),
        }

    def evaluate_budgets(
        self,
        budgets: Sequence[BenchmarkBudget],
    ) -> BenchmarkGateReport:
        """Evaluate release or claim budgets against materialized benchmark results."""

        if not budgets:
            raise WorldForgeError("evaluate_budgets() requires at least one BenchmarkBudget.")

        violations: list[BenchmarkGateViolation] = []
        checked_result_count = 0
        for budget in budgets:
            if not isinstance(budget, BenchmarkBudget):
                raise WorldForgeError("evaluate_budgets() accepts only BenchmarkBudget entries.")
            matched_results = [result for result in self.results if budget.matches(result)]
            if not matched_results:
                violations.append(
                    BenchmarkGateViolation(
                        provider=budget.provider or "*",
                        operation=budget.operation or "*",
                        metric="matching_results",
                        observed=0,
                        threshold=1,
                        condition=">= 1 matching result",
                        budget_selector=budget.selector_label(),
                    )
                )
                continue

            checked_result_count += len(matched_results)
            for result in matched_results:
                violations.extend(_evaluate_budget_for_result(budget, result))

        return BenchmarkGateReport(
            budgets=list(budgets),
            checked_result_count=checked_result_count,
            violations=violations,
        )


@dataclass(slots=True)
class _BenchmarkSample:
    latency_ms: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> JSONDict:
        return {
            "latency_ms": self.latency_ms,
            "error": self.error,
            "succeeded": self.succeeded,
        }


class ProviderBenchmarkHarness:
    """Run latency, retry, and throughput benchmarks across registered providers."""

    benchmarkable_operations = BENCHMARKABLE_OPERATIONS

    def __init__(self, forge: WorldForge | None = None) -> None:
        self._forge = forge or WorldForge()
        self._operation_handlers: dict[str, Callable[[str, BenchmarkInputs], None]] = {
            "predict": self._op_predict,
            "reason": self._op_reason,
            "generate": self._op_generate,
            "transfer": self._op_transfer,
            "embed": self._op_embed,
            "score": self._op_score,
            "policy": self._op_policy,
        }

    def supported_operations(self, provider: str) -> list[str]:
        profile = self._forge.provider_profile(provider)
        return [
            operation
            for operation in self.benchmarkable_operations
            if profile.capabilities.supports(operation)
        ]

    def _seed_world(self, provider: str) -> tuple[object, object]:
        world = self._forge.create_world("benchmark-world", provider)
        cube = world.add_object(
            SceneObject(
                "cube",
                Position(0.0, 0.5, 0.0),
                BBox(Position(-0.05, 0.45, -0.05), Position(0.05, 0.55, 0.05)),
                is_graspable=True,
            )
        )
        mug = world.add_object(
            SceneObject(
                "mug",
                Position(0.25, 0.8, 0.0),
                BBox(Position(0.2, 0.75, -0.05), Position(0.3, 0.85, 0.05)),
                is_graspable=True,
            )
        )
        return world, (cube, mug)

    def _op_predict(self, provider: str, inputs: BenchmarkInputs) -> None:
        world, _ = self._seed_world(provider)
        world.predict(
            inputs.prediction_action,
            steps=inputs.prediction_steps,
            provider=provider,
        )

    def _op_reason(self, provider: str, inputs: BenchmarkInputs) -> None:
        world, _ = self._seed_world(provider)
        self._forge.reason(provider, inputs.reason_query, world=world)

    def _op_generate(self, provider: str, inputs: BenchmarkInputs) -> None:
        self._forge.generate(
            inputs.generation_prompt,
            provider,
            duration_seconds=inputs.generation_duration_seconds,
        )

    def _op_transfer(self, provider: str, inputs: BenchmarkInputs) -> None:
        self._forge.transfer(
            inputs.transfer_clip,
            provider,
            width=inputs.transfer_width,
            height=inputs.transfer_height,
            fps=inputs.transfer_fps,
            prompt=inputs.transfer_prompt,
        )

    def _op_embed(self, provider: str, inputs: BenchmarkInputs) -> None:
        self._forge.embed(provider, text=inputs.embedding_text)

    def _op_score(self, provider: str, inputs: BenchmarkInputs) -> None:
        self._forge.score_actions(
            provider,
            info=inputs.score_info,
            action_candidates=inputs.score_action_candidates,
        )

    def _op_policy(self, provider: str, inputs: BenchmarkInputs) -> None:
        self._forge.select_actions(provider, info=inputs.policy_info)

    def _invoke_operation(
        self,
        provider: str,
        operation: str,
        inputs: BenchmarkInputs,
    ) -> None:
        handler = self._operation_handlers.get(operation)
        if handler is None:
            raise WorldForgeError(
                f"Unknown benchmark operation '{operation}'. "
                f"Known operations: {', '.join(self.benchmarkable_operations)}."
            )
        handler(provider, inputs)

    def _sample_once(
        self,
        provider: str,
        operation: str,
        inputs: BenchmarkInputs,
    ) -> _BenchmarkSample:
        started = perf_counter()
        try:
            self._invoke_operation(provider, operation, inputs)
        except (ProviderError, WorldForgeError, TimeoutError) as exc:
            return _BenchmarkSample(
                latency_ms=max(0.1, (perf_counter() - started) * 1000),
                error=str(exc),
            )
        return _BenchmarkSample(latency_ms=max(0.1, (perf_counter() - started) * 1000))

    @contextmanager
    def _capture_metrics(self, provider: str):
        metrics = ProviderMetricsSink()
        provider_instance = self._forge._require_provider(provider)
        original_handler = provider_instance.event_handler
        provider_instance.event_handler = compose_event_handlers(original_handler, metrics)
        try:
            yield metrics
        finally:
            provider_instance.event_handler = original_handler

    def _run_operation(
        self,
        provider: str,
        operation: str,
        *,
        iterations: int,
        concurrency: int,
        inputs: BenchmarkInputs,
        on_sample: Callable[[JSONDict], None] | None = None,
    ) -> BenchmarkResult:
        started = perf_counter()
        samples: list[_BenchmarkSample] = []
        with self._capture_metrics(provider) as metrics:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(self._sample_once, provider, operation, inputs)
                    for _ in range(iterations)
                ]
                for future in as_completed(futures):
                    sample = future.result()
                    samples.append(sample)
                    if on_sample is not None:
                        on_sample(
                            {
                                "provider": provider,
                                "operation": operation,
                                "iteration": len(samples),
                                **sample.to_dict(),
                            }
                        )

            provider_metrics = [
                metric.to_dict() for metric in metrics.snapshot() if metric.provider == provider
            ]

        total_time_ms = max(0.1, (perf_counter() - started) * 1000)
        latencies = [sample.latency_ms for sample in samples]
        successful_latencies = [sample.latency_ms for sample in samples if sample.succeeded]
        errors = [sample.error for sample in samples if sample.error is not None]
        retry_count = sum(metric["retry_count"] for metric in provider_metrics)
        total_seconds = total_time_ms / 1000
        throughput = len(successful_latencies) / total_seconds if total_seconds > 0 else 0.0

        average_latency_ms = sum(latencies) / len(latencies) if latencies else None
        min_latency_ms = min(latencies) if latencies else None
        max_latency_ms = max(latencies) if latencies else None

        return BenchmarkResult(
            provider=provider,
            operation=operation,
            iterations=iterations,
            concurrency=concurrency,
            success_count=len(successful_latencies),
            error_count=len(errors),
            retry_count=retry_count,
            total_time_ms=total_time_ms,
            average_latency_ms=average_latency_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
            throughput_per_second=throughput,
            operation_metrics={
                "provider": provider,
                "operation": operation,
                "events": provider_metrics,
            },
            errors=[error for error in errors if error is not None],
        )

    def run(
        self,
        providers: str | Sequence[str],
        *,
        operations: Sequence[str] | None = None,
        iterations: int = 5,
        concurrency: int = 1,
        inputs: BenchmarkInputs | None = None,
        on_sample: Callable[[JSONDict], None] | None = None,
    ) -> BenchmarkReport:
        provider_names = [providers] if isinstance(providers, str) else list(providers)
        if not provider_names:
            raise WorldForgeError("Benchmark run requires at least one provider.")
        require_positive_int(iterations, name="iterations")
        require_positive_int(concurrency, name="concurrency")
        benchmark_inputs = inputs or BenchmarkInputs()

        requested_operations = list(dict.fromkeys(operations or self.benchmarkable_operations))
        unknown_operations = [
            operation
            for operation in requested_operations
            if operation not in self.benchmarkable_operations
        ]
        if unknown_operations:
            joined = ", ".join(unknown_operations)
            raise WorldForgeError(
                f"Unknown benchmark operations: {joined}. "
                f"Known operations: {', '.join(self.benchmarkable_operations)}."
            )

        results: list[BenchmarkResult] = []
        for provider in provider_names:
            self._forge._require_provider(provider)
            supported = self.supported_operations(provider)
            selected_operations = supported if operations is None else list(requested_operations)
            unsupported = [
                operation for operation in selected_operations if operation not in supported
            ]
            if unsupported:
                joined = ", ".join(unsupported)
                raise WorldForgeError(
                    f"Provider '{provider}' cannot benchmark unsupported operations: {joined}."
                )
            if not selected_operations:
                raise WorldForgeError(
                    f"Provider '{provider}' does not expose benchmarkable operations."
                )

            for operation in selected_operations:
                results.append(
                    self._run_operation(
                        provider,
                        operation,
                        iterations=iterations,
                        concurrency=concurrency,
                        inputs=benchmark_inputs,
                        on_sample=on_sample,
                    )
                )

        return BenchmarkReport(results)


def run_benchmark(
    providers: str | Sequence[str],
    *,
    forge: WorldForge | None = None,
    operations: Sequence[str] | None = None,
    iterations: int = 5,
    concurrency: int = 1,
    inputs: BenchmarkInputs | None = None,
    on_sample: Callable[[JSONDict], None] | None = None,
) -> BenchmarkReport:
    """Convenience wrapper around ProviderBenchmarkHarness.run()."""

    return ProviderBenchmarkHarness(forge=forge).run(
        providers,
        operations=operations,
        iterations=iterations,
        concurrency=concurrency,
        inputs=inputs,
        on_sample=on_sample,
    )


__all__ = [
    "BenchmarkInputs",
    "BenchmarkBudget",
    "BenchmarkGateReport",
    "BenchmarkGateViolation",
    "BenchmarkReport",
    "BenchmarkResult",
    "BENCHMARKABLE_OPERATIONS",
    "ProviderBenchmarkHarness",
    "load_benchmark_budgets",
    "run_benchmark",
]
