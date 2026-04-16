"""Capability-aware provider benchmark harness."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
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
    require_positive_int,
)
from worldforge.observability import ProviderMetricsSink, compose_event_handlers


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

    def __post_init__(self) -> None:
        require_positive_int(self.prediction_steps, name="prediction_steps")
        if self.generation_duration_seconds <= 0.0:
            raise WorldForgeError("generation_duration_seconds must be greater than 0.")
        if self.transfer_width <= 0 or self.transfer_height <= 0:
            raise WorldForgeError("transfer_width and transfer_height must be greater than 0.")
        if self.transfer_fps <= 0.0:
            raise WorldForgeError("transfer_fps must be greater than 0.")

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


@dataclass(slots=True)
class _BenchmarkSample:
    latency_ms: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class ProviderBenchmarkHarness:
    """Run latency, retry, and throughput benchmarks across registered providers."""

    benchmarkable_operations = ("predict", "reason", "generate", "transfer")

    def __init__(self, forge: WorldForge | None = None) -> None:
        self._forge = forge or WorldForge()

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

    def _invoke_operation(
        self,
        provider: str,
        operation: str,
        inputs: BenchmarkInputs,
    ) -> None:
        if operation == "predict":
            world, _ = self._seed_world(provider)
            world.predict(
                inputs.prediction_action,
                steps=inputs.prediction_steps,
                provider=provider,
            )
            return

        if operation == "reason":
            world, _ = self._seed_world(provider)
            self._forge.reason(provider, inputs.reason_query, world=world)
            return

        if operation == "generate":
            self._forge.generate(
                inputs.generation_prompt,
                provider,
                duration_seconds=inputs.generation_duration_seconds,
            )
            return

        if operation == "transfer":
            self._forge.transfer(
                inputs.transfer_clip,
                provider,
                width=inputs.transfer_width,
                height=inputs.transfer_height,
                fps=inputs.transfer_fps,
                prompt=inputs.transfer_prompt,
            )
            return

        raise WorldForgeError(
            f"Unknown benchmark operation '{operation}'. "
            f"Known operations: {', '.join(self.benchmarkable_operations)}."
        )

    def _sample_once(
        self,
        provider: str,
        operation: str,
        inputs: BenchmarkInputs,
    ) -> _BenchmarkSample:
        started = perf_counter()
        try:
            self._invoke_operation(provider, operation, inputs)
        except Exception as exc:
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
                    samples.append(future.result())

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
) -> BenchmarkReport:
    """Convenience wrapper around ProviderBenchmarkHarness.run()."""

    return ProviderBenchmarkHarness(forge=forge).run(
        providers,
        operations=operations,
        iterations=iterations,
        concurrency=concurrency,
        inputs=inputs,
    )


__all__ = [
    "BenchmarkInputs",
    "BenchmarkReport",
    "BenchmarkResult",
    "ProviderBenchmarkHarness",
    "run_benchmark",
]
