from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldforge.decision_trace import validate_decision_trace
from worldforge.demos.go2_controlbench import (
    main,
    render_go2_controlbench_report,
    run_go2_controlbench,
    run_go2_controlbench_workflow,
)
from worldforge.models import WorldForgeError

_FAKE_SERIAL = "B99D" + "FAKEGO2BOT00"


def test_go2_controlbench_writes_summary_report_and_decision_trace(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    result = run_go2_controlbench(capture_dir, tmp_path / "out")

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    trace = validate_decision_trace(
        json.loads(result.decision_trace_path.read_text(encoding="utf-8"))
    )

    assert result.report_path.is_file()
    assert summary["artifact_kind"] == "worldforge.go2_controlbench_summary"
    assert summary["dataset"]["trial_count"] == 6
    assert summary["dataset"]["command_cell_count"] == 3
    assert summary["dataset"]["rgb_frames_embedded"] is False
    assert set(summary["tasks"]) == {
        "command_outcome_prediction",
        "inverse_control",
        "odom_vs_external_ground_truth",
    }
    assert set(summary["tasks"]["command_outcome_prediction"]["baselines"]) == {
        "command_integral",
        "affine_linear_system_id",
        "deadband_affine_system_id",
    }
    assert trace["claim_boundary"]["outcome_kind"] == "real_measured"
    assert trace["claim_boundary"]["score_kind"] == "hand_cost"
    assert trace["claim_boundary"]["hardware_executed"] is True
    assert trace["claim_boundary"]["learned_model_used"] is False
    assert trace["outcome"]["metrics"]["inverse_control_regret"] >= 0.0

    payload_text = json.dumps({"summary": summary, "trace": trace}, sort_keys=True)
    assert str(capture_dir) not in payload_text
    assert "/Users/" not in payload_text
    assert "192.168." not in payload_text
    assert _FAKE_SERIAL not in payload_text


def test_go2_controlbench_report_states_benchmark_boundary(tmp_path: Path) -> None:
    result = run_go2_controlbench(_write_capture(tmp_path / "capture"), tmp_path / "out")

    report = render_go2_controlbench_report(result.summary)

    assert "Task A - Command Outcome Prediction" in report
    assert "Task B - Inverse Control Ranking" in report
    assert "WMCP" in report
    assert "DecisionTrace" in report
    assert "not an autonomous planner win" in report


def test_go2_controlbench_workflow_returns_portable_paths(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    out_dir = tmp_path / "absolute-output"

    summary = run_go2_controlbench_workflow(capture_dir, out_dir)

    assert summary["decision_trace_path"] == "decision-trace-go2-controlbench.json"
    assert summary["summary_path"] == "controlbench-summary.json"
    assert summary["report_path"] == "controlbench-report.md"
    assert str(out_dir) not in json.dumps(summary, sort_keys=True)


def test_go2_controlbench_rejects_missing_trial_outcomes(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / _FAKE_SERIAL)
    (capture_dir / "analysis" / "trial_outcomes.csv").unlink()

    with pytest.raises(WorldForgeError, match="trial_outcomes"):
        run_go2_controlbench(capture_dir, tmp_path / "out")


def test_go2_controlbench_cli_help_has_description(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Run Go2 ControlBench over a host-owned native-rate system-ID capture" in output


def _write_capture(capture_dir: Path) -> Path:
    analysis_dir = capture_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    (capture_dir / "streams" / "topics").mkdir(parents=True)
    (capture_dir / "streams" / "topics" / "robotodom.jsonl").write_text("", encoding="utf-8")
    (capture_dir / "streams" / "topics" / "ulidar_array.jsonl").write_text("", encoding="utf-8")
    _write_text(
        analysis_dir / "trial_outcomes.csv",
        "\n".join(
            [
                "trial_index,label,cmd_x_mps,cmd_y_mps,cmd_z_radps,duration_s,"
                "cmd_forward_m,cmd_lateral_m,cmd_yaw_rad,measured_dx_world_m,"
                "measured_dy_world_m,measured_forward_body_m,measured_left_body_m,"
                "measured_dyaw_rad,measured_planar_m,odom_samples,ulidar_messages,ulidar_npz",
                "0,slow-r01,0.10,0.0,0.0,0.6,0.06,0.0,0.0,0.001,0.001,0.002,0.0,0.001,0.002,10,4,2",
                "1,slow-r02,0.10,0.0,0.0,0.6,0.06,0.0,0.0,0.001,0.001,0.004,0.0,0.001,0.004,11,5,2",
                "2,fast-r01,0.25,0.0,0.0,1.2,0.3,0.0,0.0,0.19,0.01,0.210,0.01,0.02,0.210,12,6,3",
                "3,fast-r02,0.25,0.0,0.0,1.2,0.3,0.0,0.0,0.20,0.01,0.220,0.01,0.02,0.220,13,7,3",
                "4,yaw-r01,0.0,0.0,0.35,1.5,0.0,0.0,0.525,0.0,0.0,0.0,0.0,0.07,0.0,14,8,4",
                "5,yaw-r02,0.0,0.0,0.35,1.5,0.0,0.0,0.525,0.0,0.0,0.0,0.0,0.08,0.0,15,9,4",
            ]
        ),
    )
    _write_text(
        analysis_dir / "command_group_stats.csv",
        "\n".join(
            [
                "group,n,cmd_forward_m,cmd_lateral_m,cmd_yaw_rad,"
                "measured_forward_body_m_mean,measured_forward_body_m_sd,"
                "measured_forward_body_m_min,measured_forward_body_m_max,"
                "measured_left_body_m_mean,measured_left_body_m_sd,"
                "measured_left_body_m_min,measured_left_body_m_max,"
                "measured_dyaw_rad_mean,measured_dyaw_rad_sd,"
                "measured_dyaw_rad_min,measured_dyaw_rad_max,"
                "measured_planar_m_mean,measured_planar_m_sd,"
                "measured_planar_m_min,measured_planar_m_max",
                "x=0.10 y=0.0 z=0.0 d=0.6,2,0.06,0.0,0.0,0.003,0.001,0.002,0.004,"
                "0.0,0.0,0.0,0.0,0.001,0.0,0.001,0.001,0.003,0.001,0.002,0.004",
                "x=0.25 y=0.0 z=0.0 d=1.2,2,0.3,0.0,0.0,0.215,0.005,0.210,0.220,"
                "0.01,0.0,0.01,0.01,0.02,0.0,0.02,0.02,0.215,0.005,0.210,0.220",
                "x=0.0 y=0.0 z=0.35 d=1.5,2,0.0,0.0,0.525,0.0,0.0,0.0,0.0,"
                "0.0,0.0,0.0,0.0,0.075,0.005,0.07,0.08,0.0,0.0,0.0,0.0",
            ]
        ),
    )
    (analysis_dir / "system_id_fit.json").write_text(
        json.dumps(
            {
                "model": "measured_body_outcome = intercept + A @ command",
                "features": ["intercept", "cmd_forward_m", "cmd_yaw_rad"],
                "targets": ["measured_forward_body_m", "measured_dyaw_rad"],
                "r2": {"measured_forward_body_m": 0.8, "measured_dyaw_rad": 0.5},
                "rmse": {"measured_forward_body_m": 0.04, "measured_dyaw_rad": 0.03},
            }
        ),
        encoding="utf-8",
    )
    (capture_dir / "go2-native-system-id-run.json").write_text(
        json.dumps(
            {
                "artifact_kind": "worldforge.go2_air_native_rate_system_id_capture",
                "capture_design": {"native_rate_streams": True, "rgb_frames_recorded": False},
                "driver": "unitree_webrtc_connect",
                "errors": [],
                "robot": {
                    "model": "Unitree Go2 Air",
                    "serial": _FAKE_SERIAL,
                    "firmware_version": "V1.1.12",
                    "hardware_version": "V2.0",
                    "ip_redacted": True,
                    "serial_redacted": True,
                },
                "scene": {"surface": "office indoor floor"},
                "saved_frame_count": 0,
                "saved_lidar_count": 16,
                "topic_counts": {"ROBOTODOM": 60, "ULIDAR_ARRAY": 32, "LOW_STATE": 8},
            }
        ),
        encoding="utf-8",
    )
    return capture_dir


def _write_text(path: Path, text: str) -> None:
    path.write_text(f"{text}\n", encoding="utf-8")
