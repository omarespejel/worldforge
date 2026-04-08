from __future__ import annotations

import json

import httpx
import pytest

from worldforge import ProviderBenchmarkHarness, ProviderRequestPolicy, WorldForge, WorldForgeError
from worldforge.providers import CosmosProvider, RunwayProvider


def test_provider_benchmark_harness_reports_mock_operations(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    harness = ProviderBenchmarkHarness(forge=forge)

    report = harness.run(
        "mock",
        operations=["predict", "reason", "generate", "transfer"],
        iterations=2,
        concurrency=2,
    )

    assert len(report.results) == 4
    assert json.loads(report.to_json())["results"][0]["provider"] == "mock"
    assert report.to_csv().startswith("provider,operation,iterations")
    assert report.to_markdown().startswith("# Benchmark Report")
    for result in report.results:
        assert result.provider == "mock"
        assert result.success_count == 2
        assert result.error_count == 0
        assert result.retry_count == 0
        assert result.average_latency_ms is not None
        assert result.operation_metrics["provider"] == "mock"
        assert result.operation_metrics["events"]


def test_provider_benchmark_harness_captures_retry_metrics(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RUNWAYML_API_SECRET", "runway-test-key")
    attempts = {"poll": 0, "download": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/image_to_video":
            return httpx.Response(200, json={"id": "task_generate"})

        if request.method == "GET" and request.url.path == "/v1/tasks/task_generate":
            attempts["poll"] += 1
            if attempts["poll"] == 1:
                return httpx.Response(503, text="retry poll")
            return httpx.Response(
                200,
                json={
                    "id": "task_generate",
                    "status": "SUCCEEDED",
                    "output": ["https://downloads.example.com/generated.mp4"],
                },
            )

        if request.method == "GET" and request.url.host == "downloads.example.com":
            attempts["download"] += 1
            if attempts["download"] == 1:
                return httpx.Response(503, text="retry download")
            return httpx.Response(200, content=b"benchmark-generated")

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    forge = WorldForge(state_dir=tmp_path, auto_register_remote=False)
    forge.register_provider(
        RunwayProvider(
            request_policy=ProviderRequestPolicy.remote_defaults(
                request_timeout_seconds=30.0,
                read_retry_attempts=2,
                read_backoff_seconds=0.0,
            ),
            transport=httpx.MockTransport(handler),
            poll_interval_seconds=0.0,
            max_polls=1,
        )
    )

    report = ProviderBenchmarkHarness(forge=forge).run(
        "runway",
        operations=["generate"],
        iterations=1,
    )

    result = report.results[0]
    assert result.provider == "runway"
    assert result.operation == "generate"
    assert result.success_count == 1
    assert result.error_count == 0
    assert result.retry_count == 2
    emitted_operations = {event["operation"] for event in result.operation_metrics["events"]}
    assert emitted_operations == {
        "generation request",
        "task poll",
        "artifact download",
    }


def test_provider_benchmark_harness_rejects_unsupported_operations(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path, auto_register_remote=False)
    forge.register_provider(CosmosProvider(base_url="http://cosmos.test"))
    harness = ProviderBenchmarkHarness(forge=forge)

    with pytest.raises(WorldForgeError, match="unsupported operations: transfer"):
        harness.run("cosmos", operations=["transfer"], iterations=1)
