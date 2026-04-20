from __future__ import annotations

from worldforge.harness import available_flows, flow_index, run_flow
from worldforge.harness.flows import flow_to_dicts


def test_harness_flow_metadata_is_available_without_textual() -> None:
    flows = available_flows()
    assert [flow.id for flow in flows] == ["leworldmodel", "lerobot", "diagnostics"]
    assert flow_index()["leworldmodel"].provider == "LeWorldModelProvider"

    payload = flow_to_dicts()
    assert payload[0]["command"] == "uv run worldforge-demo-leworldmodel"
    assert payload[1]["focus"] == "policy plus score planning"
    assert payload[2]["command"] == "uv run worldforge harness --flow diagnostics"


def test_harness_runs_leworldmodel_flow(tmp_path) -> None:
    run = run_flow("leworldmodel", state_dir=tmp_path)

    assert run.flow.id == "leworldmodel"
    assert len(run.steps) == 6
    assert len(run.metrics) == 6
    assert run.summary["selected_candidate_index"] == 1
    assert run.summary["saved_worlds"] == [run.summary["saved_world_id"]]
    assert run.summary["event_phases"] == ["success", "success"]
    assert "final_position: (0.55, 0.50, 0.00)" in run.transcript


def test_harness_runs_lerobot_flow(tmp_path) -> None:
    run = run_flow("lerobot", state_dir=tmp_path)

    assert run.flow.id == "lerobot"
    assert len(run.steps) == 6
    assert run.summary["policy_candidate_count"] == 3
    assert run.summary["selected_candidate_index"] == 1
    assert run.summary["policy_select_calls"] == 2
    assert "policy_select_calls: 2" in run.transcript


def test_harness_runs_diagnostics_flow(tmp_path) -> None:
    run = run_flow("diagnostics", state_dir=tmp_path)

    assert run.flow.id == "diagnostics"
    assert len(run.steps) == 6
    assert len(run.metrics) == 6
    assert run.summary["registered_providers"] == ["mock"]
    assert run.summary["benchmark_operation_count"] == 5
    assert run.summary["mock_supported_operations"] == [
        "predict",
        "reason",
        "generate",
        "transfer",
        "embed",
    ]
    assert run.summary["benchmark_event_count"] >= 10
    assert "benchmark_operations: predict, reason, generate, transfer, embed" in run.transcript
