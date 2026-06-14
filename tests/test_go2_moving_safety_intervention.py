from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from worldforge.demos.go2_moving_safety_intervention import (
    MovingObservation,
    MovingSafetyConfig,
    StopProofMeasurement,
    build_dimos_execution_receipt,
    main,
    run_go2_moving_safety_intervention,
    run_go2_moving_safety_intervention_workflow,
    validate_decision_trace,
)
from worldforge.models import WorldStateError

_RECEIPT_HMAC_KEY = "test-only-dimos-receipt-key"


def test_go2_moving_safety_intervention_emits_chained_veto_resume_traces(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")

    result = run_go2_moving_safety_intervention(
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    bridge = json.loads(result.bridge_path.read_text(encoding="utf-8"))
    traces = [
        validate_decision_trace(json.loads(path.read_text(encoding="utf-8")))
        for path in result.trace_paths
    ]

    assert summary["artifact_kind"] == "worldforge.go2_moving_safety_intervention_summary"
    assert summary["mode"] == "moving-intervention"
    assert summary["runtime"]["imports_dimos"] is False
    assert summary["runtime"]["hardware_commands_sent"] is False
    assert summary["claim_boundary"]["cosmos3_in_control_path"] is False
    assert summary["claim_boundary"]["relative_move_used"] is False
    assert summary["control_envelope"]["fail_safe_default"] == "stopped_between_chunks"
    assert summary["control_envelope"]["rage_mode_allowed"] is False
    assert summary["world_model"]["prediction_scope"] == "command_displacement_only"
    assert summary["world_model"]["calibration_source"] == "native_go2_odom_public_preview"
    assert summary["world_model"]["external_gt"] == "pending"
    assert summary["stop_proof_protocol"]["required_before_live_moving_intervention"] is True
    assert [step["step_id"] for step in summary["steps"]] == [
        "moving_clear",
        "obstacle_veto",
        "hold_blocked",
        "resume_clear",
    ]
    assert summary["steps"][0]["decision"]["selected_candidate_id"] == "continue_forward_chunk"
    assert summary["steps"][1]["decision"]["selected_candidate_id"] == "stop_move"
    assert summary["steps"][2]["decision"]["selected_candidate_id"] == "stop_move"
    assert summary["steps"][3]["decision"]["selected_candidate_id"] == "continue_forward_chunk"
    assert (
        "clearance_budget_intersects_obstacle_zone"
        in (summary["steps"][1]["decision"]["forward_rejection_reasons"])
    )

    assert bridge["required_bridge_methods"]["write_stop"].endswith("SportClient.StopMove")
    assert (
        bridge["live_perception_contract"]["forward_clearance_wired_in_this_checkout_demo"] is False
    )
    assert bridge["live_perception_contract"]["host_must_supply_forward_clearance"] is True
    assert bridge["bounded_motion_contract"]["stop_between_chunks"] is True
    assert bridge["bounded_motion_contract"]["rage_mode_must_be_disabled"] is True
    assert bridge["hard_gates"]["stop_proof_required_before_live_intervention"] is True
    assert bridge["hard_gates"]["stopping_distance_measurement_required"] is True
    assert bridge["hard_gates"]["live_forward_clearance_wiring_required"] is True
    assert bridge["hard_gates"]["rage_mode_disabled_required"] is True

    assert traces[0]["prev_trace_id"] is None
    assert traces[1]["prev_trace_id"] == traces[0]["trace_id"]
    assert traces[2]["prev_trace_id"] == traces[1]["trace_id"]
    assert traces[3]["prev_trace_id"] == traces[2]["trace_id"]
    assert traces[1]["selected_action"]["candidate_id"] == "stop_move"
    assert traces[1]["observation"]["external_gt"] == "pending"
    assert traces[1]["outcome"]["metrics"]["outcome_source"] == "analytic_shadow"
    assert traces[1]["outcome"]["metrics"]["builtin_obstacle_avoidance_enabled"] is True
    assert traces[1]["outcome"]["metrics"]["stop_attribution"] == "worldforge_shadow_selected_stop"
    assert traces[1]["planner_diagnostics"]["world_model"]["prediction_scope"] == (
        "command_displacement_only"
    )
    assert traces[1]["interop"]["dimos"]["live_forward_clearance"] == (
        "host_supplied_not_wired_in_checkout_demo"
    )
    assert traces[1]["interop"]["dimos"]["rage_mode"] == "must_remain_disabled"
    assert traces[1]["claim_boundary"]["outcome_kind"] == "analytic"
    assert traces[1]["claim_boundary"]["hardware_executed"] is False

    payload_text = json.dumps(
        {"summary": summary, "bridge": bridge, "traces": traces},
        sort_keys=True,
    )
    assert str(tmp_path) not in payload_text
    assert "/Users/" not in payload_text
    assert "192.168." not in payload_text


def test_go2_moving_safety_intervention_fails_closed_on_stale_live_inputs(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    observation = MovingObservation(
        step_id="stale_inputs",
        label="Stale sensor probe",
        forward_clearance_m=2.0,
        odom_freshness_ms=math.nan,
        lidar_freshness_ms=math.inf,
        costmap_freshness_ms=None,
        heartbeat_freshness_ms=0.0,
        measured_vx_mps=0.08,
        stopmove_verified=True,
        operator_resume_authorized=True,
        clear_frames_seen=5,
    )

    result = run_go2_moving_safety_intervention(
        mode="dry-run",
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        observations=[observation],
    )

    step = result.summary["steps"][0]
    assert step["decision"]["selected_candidate_id"] == "stop_move"
    reasons = step["decision"]["forward_rejection_reasons"]
    assert "stale_odom" in reasons
    assert "stale_lidar_and_costmap" in reasons
    assert "stale_authorization_heartbeat" in reasons
    json.dumps(result.summary, allow_nan=False)


def test_go2_moving_safety_intervention_rejects_copyable_receipt_string(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    observation = MovingObservation(
        step_id="copyable_receipt",
        label="Copyable receipt string must not claim hardware.",
        forward_clearance_m=2.0,
        odom_freshness_ms=80.0,
        lidar_freshness_ms=120.0,
        costmap_freshness_ms=160.0,
        heartbeat_freshness_ms=80.0,
        measured_vx_mps=0.08,
        stopmove_verified=True,
        operator_resume_authorized=True,
        clear_frames_seen=5,
        hardware_execution_receipt="dimos_bounded_motion_receipt_v1",
    )

    result = run_go2_moving_safety_intervention(
        mode="open-space-bounded-motion",
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        observations=[observation],
    )

    trace = validate_decision_trace(result.traces[0])
    assert result.summary["runtime"]["hardware_commands_sent"] is False
    assert result.summary["claim_boundary"]["outcome_kind"] == "analytic"
    assert result.summary["steps"][0]["decision"]["selected_candidate_id"] == (
        "continue_forward_chunk"
    )
    assert trace["claim_boundary"]["hardware_executed"] is False
    assert trace["outcome"]["kind"] == "analytic"
    assert trace["planner_diagnostics"]["execution_receipt_verification"]["provided"] is True
    assert trace["planner_diagnostics"]["execution_receipt_verification"]["verified"] is False
    assert (
        trace["planner_diagnostics"]["execution_receipt_verification"]["failure_reason"]
        == "receipt_not_json_object"
    )


def test_go2_moving_safety_intervention_rejects_receipt_without_measured_outcome(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    observation = MovingObservation(
        step_id="missing_measurement_receipt",
        label="Receipt without measured outcome must not claim hardware.",
        forward_clearance_m=2.0,
        odom_freshness_ms=80.0,
        lidar_freshness_ms=120.0,
        costmap_freshness_ms=160.0,
        heartbeat_freshness_ms=80.0,
        measured_vx_mps=0.08,
        stopmove_verified=True,
        operator_resume_authorized=True,
        clear_frames_seen=5,
        hardware_execution_receipt={
            "receipt_kind": "dimos_bounded_motion_receipt_v1",
            "step_id": "missing_measurement_receipt",
            "command": {"vx": 0.10, "vy": 0.0, "wz": 0.0, "duration_s": 0.15},
            "issued_at_unix_s": 1_800_000_000.0,
            "stopmove_ack": True,
            "outcome_source": "native_odom",
            "hmac_sha256": "not-checked-because-measurement-is-missing",
        },
    )

    result = run_go2_moving_safety_intervention(
        mode="open-space-bounded-motion",
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        observations=[observation],
        receipt_hmac_key=_RECEIPT_HMAC_KEY,
    )

    trace = validate_decision_trace(result.traces[0])
    assert result.summary["runtime"]["hardware_commands_sent"] is False
    assert trace["claim_boundary"]["hardware_executed"] is False
    assert trace["outcome"]["kind"] == "analytic"
    assert (
        trace["planner_diagnostics"]["execution_receipt_verification"]["failure_reason"]
        == "missing_measured_stop_outcome"
    )


def test_go2_moving_safety_intervention_accepts_live_receipt_after_stop_proof(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    step_id = "live_after_stop_proof"
    observation = MovingObservation(
        step_id=step_id,
        label="Host receipt after stop proof",
        forward_clearance_m=2.0,
        odom_freshness_ms=80.0,
        lidar_freshness_ms=120.0,
        costmap_freshness_ms=160.0,
        heartbeat_freshness_ms=80.0,
        measured_vx_mps=0.08,
        stopmove_verified=True,
        operator_resume_authorized=True,
        clear_frames_seen=5,
        hardware_execution_receipt=build_dimos_execution_receipt(
            step_id=step_id,
            command=(0.10, 0.0, 0.0, 0.15),
            issued_at_unix_s=1_800_000_000.0,
            stopmove_ack=True,
            measured_stop_time_s=0.22,
            measured_stop_distance_m=0.03,
            outcome_source="native_odom",
            receipt_hmac_key=_RECEIPT_HMAC_KEY,
        ),
    )

    result = run_go2_moving_safety_intervention(
        mode="open-space-bounded-motion",
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        observations=[observation],
        receipt_hmac_key=_RECEIPT_HMAC_KEY,
        stop_proof=StopProofMeasurement(
            available=True,
            stop_time_s=0.22,
            stop_distance_m=0.03,
            measured_speed_mps=0.10,
            command_rtt_ms=80.0,
            velocity_before_stop_mps=0.08,
            velocity_after_stop_mps=0.0,
        ),
    )

    summary = result.summary
    trace = validate_decision_trace(result.traces[0])
    assert summary["runtime"]["hardware_commands_sent"] is True
    assert summary["steps"][0]["decision"]["selected_candidate_id"] == "continue_forward_chunk"
    assert trace["outcome"]["status"] == "forward_chunk_executed_with_host_receipt"
    assert trace["claim_boundary"]["outcome_kind"] == "real_measured"
    assert trace["planner_diagnostics"]["execution_receipt_verification"]["verified"] is True
    assert trace["outcome"]["metrics"]["measured_stop_time_s"] == 0.22
    assert trace["outcome"]["metrics"]["measured_stop_distance_m"] == 0.03
    assert trace["outcome"]["metrics"]["outcome_source"] == "native_odom"
    assert trace["claim_boundary"]["hardware_executed"] is True
    assert (
        trace["planner_diagnostics"]["execution_receipt_verification"][
            "receipt_command_matches_selected_action"
        ]
        is True
    )


def test_go2_moving_safety_intervention_receipt_must_match_selected_action(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    step_id = "receipt_for_rejected_forward"
    observation = MovingObservation(
        step_id=step_id,
        label="Receipt for forward must not claim hardware when stop is selected.",
        forward_clearance_m=0.20,
        odom_freshness_ms=80.0,
        lidar_freshness_ms=120.0,
        costmap_freshness_ms=160.0,
        heartbeat_freshness_ms=80.0,
        measured_vx_mps=0.08,
        stopmove_verified=True,
        hardware_execution_receipt=build_dimos_execution_receipt(
            step_id=step_id,
            command=(0.10, 0.0, 0.0, 0.15),
            issued_at_unix_s=1_800_000_000.0,
            stopmove_ack=True,
            measured_stop_time_s=0.22,
            measured_stop_distance_m=0.03,
            outcome_source="native_odom",
            receipt_hmac_key=_RECEIPT_HMAC_KEY,
        ),
    )

    result = run_go2_moving_safety_intervention(
        mode="moving-intervention",
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        observations=[observation],
        receipt_hmac_key=_RECEIPT_HMAC_KEY,
        stop_proof=StopProofMeasurement(
            available=True,
            stop_time_s=0.22,
            stop_distance_m=0.03,
            measured_speed_mps=0.10,
        ),
    )

    trace = validate_decision_trace(result.traces[0])
    assert result.summary["steps"][0]["decision"]["selected_candidate_id"] == "stop_move"
    assert result.summary["runtime"]["hardware_commands_sent"] is False
    assert trace["claim_boundary"]["hardware_executed"] is False
    assert trace["outcome"]["kind"] == "analytic"
    receipt = trace["planner_diagnostics"]["execution_receipt_verification"]
    assert receipt["raw_receipt_verified"] is True
    assert receipt["verified"] is False
    assert receipt["failure_reason"] == "receipt_command_does_not_match_selected_action"


def test_go2_moving_safety_intervention_rejects_unsafe_step_id_filename(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    observation = MovingObservation(
        step_id="../escape",
        label="Unsafe trace id probe",
        forward_clearance_m=2.0,
        odom_freshness_ms=80.0,
        lidar_freshness_ms=120.0,
        costmap_freshness_ms=160.0,
        heartbeat_freshness_ms=80.0,
        measured_vx_mps=0.08,
        stopmove_verified=True,
        operator_resume_authorized=True,
        clear_frames_seen=5,
    )

    with pytest.raises(WorldStateError, match="trace_id"):
        run_go2_moving_safety_intervention(
            mode="open-space-bounded-motion",
            dataset_csv=csv_path,
            output_dir=tmp_path / "out",
            observations=[observation],
        )

    assert not (tmp_path / "escape").exists()


def test_go2_moving_safety_intervention_ignores_slower_stop_proof(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    observation = MovingObservation(
        step_id="slow_stop_proof",
        label="Stop proof measured below configured demo speed",
        forward_clearance_m=2.0,
        odom_freshness_ms=80.0,
        lidar_freshness_ms=120.0,
        costmap_freshness_ms=160.0,
        heartbeat_freshness_ms=80.0,
        measured_vx_mps=0.08,
        stopmove_verified=True,
        operator_resume_authorized=True,
        clear_frames_seen=5,
    )

    result = run_go2_moving_safety_intervention(
        mode="open-space-bounded-motion",
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        config=MovingSafetyConfig(vx_mps=0.10, fallback_decel_distance_m=0.08),
        observations=[observation],
        stop_proof=StopProofMeasurement(
            available=True,
            stop_time_s=0.10,
            stop_distance_m=0.01,
            measured_speed_mps=0.05,
        ),
    )

    budget = result.summary["steps"][0]["safety_budget"]
    assert budget["stop_proof_covers_config_speed"] is False
    assert budget["stop_distance_m"] == 0.08
    assert budget["stop_budget_source"] == "conservative_fallback_unmeasured"


def test_go2_moving_safety_intervention_reports_unusable_stop_proof_unavailable(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")

    result = run_go2_moving_safety_intervention(
        mode="open-space-bounded-motion",
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        stop_proof=StopProofMeasurement(
            available=True,
            stop_time_s=math.nan,
            stop_distance_m=0.01,
            measured_speed_mps=0.10,
        ),
    )

    assert result.summary["stop_proof"]["provided"] is True
    assert result.summary["stop_proof"]["available"] is False
    assert result.bridge_plan["hard_gates"]["stop_proof_available"] is False
    assert (
        result.summary["steps"][0]["safety_budget"]["stop_budget_source"]
        == "conservative_fallback_unmeasured"
    )
    json.dumps(result.summary, allow_nan=False)


def test_go2_moving_safety_intervention_holds_resume_hysteresis_band(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    observation = MovingObservation(
        step_id="hysteresis_band",
        label="Clearance above required budget but below resume hysteresis.",
        forward_clearance_m=0.72,
        odom_freshness_ms=80.0,
        lidar_freshness_ms=120.0,
        costmap_freshness_ms=160.0,
        heartbeat_freshness_ms=80.0,
        measured_vx_mps=0.08,
        stopmove_verified=True,
        operator_resume_authorized=True,
        clear_frames_seen=5,
    )

    result = run_go2_moving_safety_intervention(
        mode="open-space-bounded-motion",
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
        observations=[observation],
    )

    step = result.summary["steps"][0]
    assert step["decision"]["selected_candidate_id"] == "stop_move"
    assert "clearance_hysteresis_band_hold" in step["decision"]["forward_rejection_reasons"]


def test_go2_moving_safety_intervention_rejects_unsafe_config(tmp_path: Path) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")

    with pytest.raises(WorldStateError, match="vx_mps"):
        run_go2_moving_safety_intervention(
            dataset_csv=csv_path,
            output_dir=tmp_path / "out",
            config=MovingSafetyConfig(vx_mps=0.2),
        )
    with pytest.raises(WorldStateError, match="decision_budget_ms"):
        run_go2_moving_safety_intervention(
            dataset_csv=csv_path,
            output_dir=tmp_path / "bad-decision-budget",
            config=MovingSafetyConfig(decision_budget_ms=-1.0),
        )
    with pytest.raises(WorldStateError, match="fallback_decel_distance_m"):
        run_go2_moving_safety_intervention(
            dataset_csv=csv_path,
            output_dir=tmp_path / "bad-decel-distance",
            config=MovingSafetyConfig(fallback_decel_distance_m=math.nan),
        )
    with pytest.raises(WorldStateError, match="resume_clear_frames_required"):
        run_go2_moving_safety_intervention(
            dataset_csv=csv_path,
            output_dir=tmp_path / "bad-resume-frames",
            config=MovingSafetyConfig(resume_clear_frames_required=0),
        )


def test_go2_moving_safety_intervention_workflow_returns_portable_paths(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")

    workflow = run_go2_moving_safety_intervention_workflow(
        mode="dry-run",
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
    )

    assert workflow["mode"] == "dry-run"
    assert workflow["hardware_commands_sent"] is False
    assert workflow["decision_sequence"] == ["stop_move"]
    assert workflow["summary_path"].endswith("moving-safety-intervention-summary.json")
    assert len(workflow["trace_paths"]) == 1


def test_go2_moving_safety_intervention_cli_help_has_description(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Go2 moving safety-intervention plan" in output
    assert "--mode" in output
    assert "--stop-proof-distance-m" in output


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
