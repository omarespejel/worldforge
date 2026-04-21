from __future__ import annotations

import json

import httpx
import pytest

from worldforge import (
    Action,
    ActionPolicyResult,
    ActionScoreResult,
    ProviderBenchmarkHarness,
    ProviderCapabilities,
    ProviderEvent,
    ProviderRequestPolicy,
    WorldForge,
    WorldForgeError,
)
from worldforge.benchmark import BenchmarkBudget, BenchmarkInputs, load_benchmark_budgets
from worldforge.models import JSONDict
from worldforge.providers import CosmosProvider, RunwayProvider
from worldforge.providers.base import BaseProvider, ProviderError


class _ScoreBenchmarkProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(
            name="scorebench",
            capabilities=ProviderCapabilities(score=True),
            description="Injected score provider for benchmark tests.",
        )
        self.calls: list[dict[str, object]] = []

    def score_actions(self, *, info: JSONDict, action_candidates: object) -> ActionScoreResult:
        self.calls.append({"info": info, "action_candidates": action_candidates})
        result = ActionScoreResult(
            provider=self.name,
            scores=[0.4, 0.1],
            best_index=1,
            lower_is_better=True,
            metadata={"candidate_count": 2},
        )
        self._emit_event(
            ProviderEvent(
                provider=self.name,
                operation="score",
                phase="success",
                duration_ms=0.1,
                metadata={"candidate_count": 2},
            )
        )
        return result


class _PolicyBenchmarkProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(
            name="policybench",
            capabilities=ProviderCapabilities(policy=True),
            description="Injected policy provider for benchmark tests.",
        )
        self.calls: list[JSONDict] = []

    def select_actions(self, *, info: JSONDict) -> ActionPolicyResult:
        self.calls.append(info)
        candidate_plans = [
            [Action.move_to(0.1, 0.5, 0.0)],
            [Action.move_to(0.3, 0.5, 0.0)],
        ]
        result = ActionPolicyResult(
            provider=self.name,
            actions=list(candidate_plans[0]),
            raw_actions={"candidate_count": len(candidate_plans)},
            action_horizon=1,
            embodiment_tag="benchmark",
            metadata={"candidate_count": len(candidate_plans)},
            action_candidates=candidate_plans,
        )
        self._emit_event(
            ProviderEvent(
                provider=self.name,
                operation="policy",
                phase="success",
                duration_ms=0.1,
                metadata={"candidate_count": len(candidate_plans)},
            )
        )
        return result


def test_provider_benchmark_harness_reports_mock_operations(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    harness = ProviderBenchmarkHarness(forge=forge)

    report = harness.run(
        "mock",
        operations=["predict", "reason", "generate", "transfer", "embed"],
        iterations=2,
        concurrency=2,
    )

    assert len(report.results) == 5
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


def test_benchmark_report_evaluates_budget_gates(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    report = ProviderBenchmarkHarness(forge=forge).run(
        "mock",
        operations=["generate"],
        iterations=2,
    )

    passing = report.evaluate_budgets(
        [
            BenchmarkBudget(
                provider="mock",
                operation="generate",
                min_success_rate=1.0,
                max_error_count=0,
                max_retry_count=0,
                max_average_latency_ms=10_000.0,
                max_p95_latency_ms=10_000.0,
                min_throughput_per_second=0.0,
            )
        ]
    )

    assert passing.passed is True
    assert passing.checked_result_count == 1
    assert passing.to_markdown().startswith("# Benchmark Gate Report")
    assert passing.to_csv().startswith("provider,operation,metric")
    assert json.loads(passing.to_json())["passed"] is True

    failing = report.evaluate_budgets(
        [
            BenchmarkBudget(
                provider="mock",
                operation="generate",
                max_average_latency_ms=0.0,
            ),
            BenchmarkBudget(provider="mock", operation="policy", max_error_count=0),
        ]
    )

    assert failing.passed is False
    assert {violation.metric for violation in failing.violations} == {
        "average_latency_ms",
        "matching_results",
    }
    assert json.loads(failing.to_json())["violation_count"] == 2


def test_load_benchmark_budgets_accepts_list_or_object_payload() -> None:
    loaded = load_benchmark_budgets(
        {
            "budgets": [
                {
                    "provider": "mock",
                    "operation": "generate",
                    "min_success_rate": 1.0,
                    "max_error_count": 0,
                }
            ]
        }
    )

    assert loaded == [
        BenchmarkBudget(
            provider="mock",
            operation="generate",
            min_success_rate=1.0,
            max_error_count=0,
        )
    ]
    assert load_benchmark_budgets([{"max_retry_count": 0}]) == [BenchmarkBudget(max_retry_count=0)]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "non-empty 'budgets' list"),
        ([], "non-empty list"),
        ([{"operation": "not-real", "max_error_count": 0}], "operation must be one of"),
        ([{"provider": "", "max_error_count": 0}], "provider must be a non-empty string"),
        ([{"provider": "mock"}], "requires at least one threshold"),
        ([{"max_error_count": -1}], "max_error_count must be an integer"),
    ],
)
def test_load_benchmark_budgets_rejects_invalid_payloads(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(WorldForgeError, match=message):
        load_benchmark_budgets(payload)


def test_provider_benchmark_harness_reports_score_and_policy_operations(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path, auto_register_remote=False)
    score_provider = _ScoreBenchmarkProvider()
    policy_provider = _PolicyBenchmarkProvider()
    forge.register_provider(score_provider)
    forge.register_provider(policy_provider)
    harness = ProviderBenchmarkHarness(forge=forge)

    report = harness.run(["scorebench", "policybench"], iterations=2)

    assert [(result.provider, result.operation) for result in report.results] == [
        ("scorebench", "score"),
        ("policybench", "policy"),
    ]
    for result in report.results:
        assert result.success_count == 2
        assert result.error_count == 0
        assert result.operation_metrics["events"][0]["operation"] == result.operation

    assert score_provider.calls[0]["info"]["metadata"]["mode"] == "benchmark-score"
    assert policy_provider.calls[0]["observation"]["language"] == "move the cube toward the target"


class _TensorLikeCandidates:
    shape = (1, 2, 2, 3)


def test_benchmark_inputs_preview_provider_native_score_candidates() -> None:
    inputs = BenchmarkInputs(score_action_candidates=_TensorLikeCandidates())

    payload = inputs.to_dict()

    assert payload["score_action_candidates"] == {
        "type": "test_benchmark._TensorLikeCandidates",
        "json_serializable": False,
        "shape": [1, 2, 2, 3],
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"embedding_text": ""}, "embedding_text must be a non-empty string"),
        ({"score_info": {}}, "score_info must be a non-empty JSON object"),
        ({"score_action_candidates": None}, "score_action_candidates must not be None"),
        ({"policy_info": []}, "policy_info must be a non-empty JSON object"),
    ],
)
def test_benchmark_inputs_validate_planning_surface_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(WorldForgeError, match=message):
        BenchmarkInputs(**kwargs)


def test_provider_benchmark_harness_uses_custom_score_and_policy_inputs(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path, auto_register_remote=False)
    score_provider = _ScoreBenchmarkProvider()
    policy_provider = _PolicyBenchmarkProvider()
    forge.register_provider(score_provider)
    forge.register_provider(policy_provider)
    inputs = BenchmarkInputs(
        score_info={
            "pixels": [[[[1.0]]]],
            "goal": [[[0.1, 0.2, 0.3]]],
            "action": [[[0.0, 0.0, 0.0]]],
            "metadata": {"fixture": "custom"},
        },
        score_action_candidates=[[[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]]],
        policy_info={
            "observation": {"state": {"object": [1.0, 2.0, 3.0]}, "language": "custom"},
            "mode": "select_action",
        },
    )

    ProviderBenchmarkHarness(forge=forge).run(
        ["scorebench", "policybench"],
        iterations=1,
        inputs=inputs,
    )

    assert score_provider.calls == [
        {
            "info": inputs.score_info,
            "action_candidates": inputs.score_action_candidates,
        }
    ]
    assert policy_provider.calls == [inputs.policy_info]


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


def test_provider_benchmark_harness_rejects_unknown_invoke_operation(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    harness = ProviderBenchmarkHarness(forge=forge)

    with pytest.raises(WorldForgeError, match="Unknown benchmark operation"):
        harness._invoke_operation("mock", "not-a-real-op", BenchmarkInputs())


def test_provider_benchmark_harness_records_provider_error_samples(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    harness = ProviderBenchmarkHarness(forge=forge)

    def _boom(provider: str, inputs: BenchmarkInputs) -> None:
        raise ProviderError("simulated provider outage")

    # Swap the predict handler so the narrowed except branch fires without an outbound call.
    harness._operation_handlers["predict"] = _boom

    report = harness.run("mock", operations=["predict"], iterations=2, concurrency=1)
    result = report.results[0]
    assert result.success_count == 0
    assert result.error_count == 2
    assert all("simulated provider outage" in message for message in result.errors)


def test_provider_benchmark_harness_propagates_unexpected_exceptions(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    harness = ProviderBenchmarkHarness(forge=forge)

    class _UnexpectedError(Exception):
        pass

    def _boom(provider: str, inputs: BenchmarkInputs) -> None:
        raise _UnexpectedError("this must propagate")

    harness._operation_handlers["predict"] = _boom

    with pytest.raises(_UnexpectedError, match="this must propagate"):
        harness.run("mock", operations=["predict"], iterations=1, concurrency=1)
