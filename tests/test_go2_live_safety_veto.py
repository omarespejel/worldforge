from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldforge.decision_trace import validate_decision_trace
from worldforge.demos.go2_live_safety_veto import (
    ObstacleEvidence,
    main,
    run_go2_live_safety_veto,
    run_go2_live_safety_veto_demo_pair,
    run_go2_live_safety_veto_workflow,
)


def test_go2_live_safety_veto_rejects_forward_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")

    result = run_go2_live_safety_veto(dataset_csv=csv_path, output_dir=tmp_path / "out")

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    bridge = json.loads(result.bridge_path.read_text(encoding="utf-8"))
    trace = validate_decision_trace(
        json.loads(result.decision_trace_path.read_text(encoding="utf-8"))
    )

    assert summary["artifact_kind"] == "worldforge.go2_live_safety_veto_summary"
    assert summary["decision"]["proposed_candidate_id"] == "forward_50cm"
    assert summary["decision"]["selected_candidate_id"] == "stop_hold"
    assert summary["decision"]["authorization"] == "rejected_forward_selected_stop"
    assert summary["decision"]["unsafe_forward_rejected"] is True
    assert summary["decision"]["primary_reason"] == (
        "predicted_swept_footprint_intersects_obstacle_zone"
    )
    assert summary["runtime"]["mode"] == "shadow_no_execution"
    assert summary["runtime"]["imports_dimos"] is False
    assert summary["runtime"]["hardware_commands_sent"] is False
    assert summary["claim_boundary"]["live_dimos_execution"] is False
    assert summary["claim_boundary"]["cosmos3_in_control_path"] is False
    assert summary["claim_boundary"]["relative_move_used"] is False

    assert bridge["artifact_kind"] == "worldforge.dimos_go2_live_safety_bridge_plan"
    assert bridge["required_bridge_methods"]["stop_move"]["preferred"].startswith(
        "UnitreeGo2TwistAdapter.write_stop"
    )
    assert bridge["selected_shadow_action"]["candidate_id"] == "stop_hold"
    assert bridge["selected_shadow_action"]["execution_authorized"] is False
    assert bridge["hard_gates"]["operator_approval_required"] is True
    assert bridge["hard_gates"]["stopmove_required_before_motion"] is True

    assert trace["host_runtime"]["mode"] == "shadow_no_execution"
    assert trace["selected_action"]["candidate_id"] == "stop_hold"
    assert trace["outcome"]["status"] == "unsafe_action_rejected_in_shadow"
    assert trace["outcome"]["metrics"]["hardware_commands_sent"] is False
    assert trace["outcome"]["metrics"]["unsafe_forward_rejected"] is True
    assert trace["claim_boundary"]["outcome_kind"] == "analytic"
    assert trace["claim_boundary"]["hardware_executed"] is False
    assert trace["claim_boundary"]["learned_model_used"] is False
    assert trace["interop"]["dimos"]["mode"] == "shadow_no_execution"

    payload_text = json.dumps(
        {"summary": summary, "bridge": bridge, "trace": trace},
        sort_keys=True,
    )
    assert str(tmp_path) not in payload_text
    assert "/Users/" not in payload_text
    assert "192.168." not in payload_text


def test_go2_live_safety_veto_can_authorize_forward_in_open_space_shadow(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    evidence = ObstacleEvidence(
        source_kind="fixture_open_space_lidar_costmap",
        forward_clearance_m=2.0,
        lidar_freshness_ms=80.0,
        costmap_freshness_ms=90.0,
        odom_freshness_ms=70.0,
        unknown_cells_in_forward_corridor=False,
        stopmove_verified=True,
    )

    result = run_go2_live_safety_veto(
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        obstacle_evidence=evidence,
    )

    summary = result.summary
    trace = validate_decision_trace(result.decision_trace)

    assert summary["decision"]["selected_candidate_id"] == "forward_50cm"
    assert summary["decision"]["authorization"] == "authorized_forward_shadow"
    assert summary["decision"]["unsafe_forward_rejected"] is False
    assert trace["selected_action"]["candidate_id"] == "forward_50cm"
    assert trace["outcome"]["status"] == "forward_action_authorized_in_shadow"
    assert trace["outcome"]["metrics"]["hardware_commands_sent"] is False


def test_go2_live_safety_veto_fails_closed_on_stale_odom(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    evidence = ObstacleEvidence(
        source_kind="fixture_stale_odom",
        forward_clearance_m=2.0,
        lidar_freshness_ms=80.0,
        costmap_freshness_ms=90.0,
        odom_freshness_ms=300.0,
        unknown_cells_in_forward_corridor=False,
        stopmove_verified=True,
    )

    result = run_go2_live_safety_veto(
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        obstacle_evidence=evidence,
    )

    forward = next(
        candidate
        for candidate in result.summary["candidates"]
        if candidate["candidate_id"] == "forward_50cm"
    )
    assert result.summary["decision"]["selected_candidate_id"] == "stop_hold"
    assert result.summary["decision"]["unsafe_forward_rejected"] is True
    assert "stale_odom" in forward["rejection_reasons"]


def test_go2_live_safety_veto_fails_closed_on_zero_freshness(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    evidence = ObstacleEvidence(
        source_kind="fixture_zero_freshness",
        forward_clearance_m=2.0,
        lidar_freshness_ms=0.0,
        costmap_freshness_ms=0.0,
        odom_freshness_ms=0.0,
        unknown_cells_in_forward_corridor=False,
        stopmove_verified=True,
    )

    result = run_go2_live_safety_veto(
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        obstacle_evidence=evidence,
    )

    forward = next(
        candidate
        for candidate in result.summary["candidates"]
        if candidate["candidate_id"] == "forward_50cm"
    )
    assert result.summary["decision"]["selected_candidate_id"] == "stop_hold"
    assert "stale_odom" in forward["rejection_reasons"]
    assert "stale_lidar_and_costmap" in forward["rejection_reasons"]
    assert result.summary["obstacle_evidence"]["freshness_policy"]["odom_present"] is False
    assert result.summary["obstacle_evidence"]["freshness_policy"]["lidar_present"] is False
    assert result.summary["obstacle_evidence"]["freshness_policy"]["costmap_present"] is False


def test_go2_live_safety_veto_fails_closed_on_missing_freshness(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    evidence = ObstacleEvidence(
        source_kind="fixture_missing_freshness",
        forward_clearance_m=2.0,
        lidar_freshness_ms=None,
        costmap_freshness_ms=None,
        odom_freshness_ms=None,
        unknown_cells_in_forward_corridor=False,
        stopmove_verified=True,
    )

    result = run_go2_live_safety_veto(
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        obstacle_evidence=evidence,
    )

    forward = next(
        candidate
        for candidate in result.summary["candidates"]
        if candidate["candidate_id"] == "forward_50cm"
    )
    assert result.summary["decision"]["selected_candidate_id"] == "stop_hold"
    assert "stale_odom" in forward["rejection_reasons"]
    assert "stale_lidar_and_costmap" in forward["rejection_reasons"]


def test_go2_live_safety_veto_rejects_raw_hardware_command_self_report() -> None:
    kwargs = {
        "source_kind": "fixture_raw_hardware_claim",
        "forward_clearance_m": 2.0,
        "lidar_freshness_ms": 80.0,
        "costmap_freshness_ms": 90.0,
        "odom_freshness_ms": 70.0,
        "unknown_cells_in_forward_corridor": False,
        "stopmove_verified": True,
        "hardware_commands_sent": True,
    }

    with pytest.raises(TypeError):
        ObstacleEvidence(**kwargs)


def test_go2_live_safety_veto_workflow_returns_portable_paths(tmp_path: Path) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")

    workflow = run_go2_live_safety_veto_workflow(
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
    )

    assert workflow["mode"] == "shadow_no_execution"
    assert workflow["decision"] == "rejected_forward_selected_stop"
    assert workflow["hardware_commands_sent"] is False
    assert workflow["summary_path"] == "live-safety-veto-summary.json"
    assert workflow["bridge_path"] == "dimos-live-safety-bridge-plan.json"
    assert workflow["decision_trace_path"] == "decision-trace-go2-live-safety-veto.json"
    assert str(tmp_path) not in json.dumps(workflow, sort_keys=True)


def test_go2_live_safety_veto_demo_pair_runs_both_cases(tmp_path: Path) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")

    result = run_go2_live_safety_veto_demo_pair(
        dataset_csv=csv_path,
        output_dir=tmp_path / "demo",
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["artifact_kind"] == "worldforge.go2_live_safety_veto_demo_pair_summary"
    assert summary["runtime"]["hardware_commands_sent"] is False
    assert summary["cases"][0]["case_id"] == "obstacle"
    assert summary["cases"][0]["decision"] == "rejected_forward_selected_stop"
    assert summary["cases"][0]["selected_candidate_id"] == "stop_hold"
    assert summary["cases"][0]["unsafe_forward_rejected"] is True
    assert summary["cases"][0]["decision_trace_digest"].startswith("sha256:")
    assert len(summary["cases"][0]["decision_trace_digest_short"]) == 12
    assert summary["cases"][1]["case_id"] == "open_space"
    assert summary["cases"][1]["decision"] == "authorized_forward_shadow"
    assert summary["cases"][1]["selected_candidate_id"] == "forward_50cm"
    assert summary["cases"][1]["unsafe_forward_rejected"] is False
    assert summary["cases"][1]["decision_trace_digest"].startswith("sha256:")
    assert len(summary["cases"][1]["decision_trace_digest_short"]) == 12
    assert result.obstacle_result.decision_trace_path.is_file()
    assert result.open_space_result.decision_trace_path.is_file()
    assert result.report_path.name == "demo-pair-report.md"


def test_go2_live_safety_veto_cli_help_has_description(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Go2 live safety-veto shadow demo" in output
    assert "--forward-clearance-m" in output
    assert "--demo-pair" in output


def _write_trials_csv(path: Path) -> Path:
    rows = [
        ("release03", "hold_noop", 1, 0.0, 0.0, 0.0, 1.0, 0.000, 0.000, 0.000),
        ("release03", "hold_noop", 2, 0.0, 0.0, 0.0, 1.0, 0.001, 0.000, 0.000),
        ("release03", "hold_noop", 3, 0.0, 0.0, 0.0, 1.0, -0.001, 0.000, 0.000),
        ("release03", "forward_025", 1, 0.25, 0.0, 0.0, 0.6, 0.120, 0.010, 0.010),
        ("release03", "forward_025", 2, 0.25, 0.0, 0.0, 0.6, 0.125, 0.010, 0.008),
        ("release03", "forward_025", 3, 0.25, 0.0, 0.0, 0.6, 0.118, 0.011, 0.009),
        ("release04", "forward_040", 1, 0.40, 0.0, 0.0, 0.75, 0.205, 0.018, 0.020),
        ("release04", "forward_040", 2, 0.40, 0.0, 0.0, 0.75, 0.210, 0.017, 0.022),
        ("release04", "forward_040", 3, 0.40, 0.0, 0.0, 0.75, 0.207, 0.019, 0.021),
        ("release04", "yaw_left_035", 1, 0.0, 0.0, 0.35, 0.9, 0.010, 0.002, 0.090),
        ("release04", "yaw_left_035", 2, 0.0, 0.0, 0.35, 0.9, 0.009, 0.002, 0.095),
        ("release04", "yaw_left_035", 3, 0.0, 0.0, 0.35, 0.9, 0.011, 0.003, 0.092),
    ]
    header = [
        "release",
        "trial_index",
        "suite",
        "name",
        "rep",
        "cmd_x",
        "cmd_y",
        "cmd_z",
        "duration_s",
        "cmd_integral_x_m",
        "cmd_integral_y_m",
        "cmd_integral_yaw_rad",
        "odom_dx_m",
        "odom_dy_m",
        "odom_dyaw_rad",
        "hardware_executed",
        "outcome_kind",
    ]
    lines = [",".join(header)]
    for index, row in enumerate(rows):
        release, name, rep, x, y, z, duration, dx, dy, dyaw = row
        lines.append(
            ",".join(
                [
                    release,
                    str(index),
                    "unit_test",
                    name,
                    str(rep),
                    str(x),
                    str(y),
                    str(z),
                    str(duration),
                    str(x * duration),
                    str(y * duration),
                    str(z * duration),
                    str(dx),
                    str(dy),
                    str(dyaw),
                    "true",
                    "real_measured_native_odom",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
