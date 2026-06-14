from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldforge.decision_trace import SCORE_KINDS, validate_decision_trace
from worldforge.demos.go2_measured_outcomes import (
    build_go2_measured_outcome_trace,
    main,
    render_go2_measured_outcomes_report,
    run_go2_measured_outcomes,
    run_go2_measured_outcomes_workflow,
)
from worldforge.models import WorldForgeError, WorldStateError

_FAKE_SERIAL = "B99D" + "FAKEROBOT000"


def test_go2_measured_outcomes_writes_valid_real_measured_trace(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    result = run_go2_measured_outcomes(capture_dir, tmp_path / "out")

    trace = validate_decision_trace(
        json.loads(result.decision_trace_path.read_text(encoding="utf-8"))
    )

    assert result.report_path.is_file()
    assert result.summary_path.is_file()
    assert trace["outcome"]["kind"] == "real_measured"
    assert trace["claim_boundary"]["outcome_kind"] == "real_measured"
    assert trace["claim_boundary"]["score_kind"] == "measured_metric"
    assert trace["claim_boundary"]["hardware_executed"] is True
    assert trace["claim_boundary"]["learned_model_used"] is False
    assert trace["claim_boundary"]["redaction_state"]["paths"]["host_paths_embedded"] is False
    assert (
        trace["claim_boundary"]["redaction_state"]["capture_labels"]["source_labels_embedded"]
        is False
    )
    assert {row["kind"] for row in trace["candidate_outcomes"]} == {"real_measured"}
    assert {row["action_executed"] for row in trace["candidate_outcomes"]} == {True}
    assert trace["selected_action"]["candidate_id"] == "cmd-01-forward-0p300"
    assert trace["outcome"]["metrics"]["completed_movement_trials"] == 4
    assert trace["outcome"]["metrics"]["native_odom_messages"] == 42

    trace_text = json.dumps(trace, sort_keys=True)
    assert str(capture_dir) not in trace_text
    assert "/Users/" not in trace_text
    assert "192.168." not in trace_text
    assert _FAKE_SERIAL not in trace_text
    assert "streams/topics/robotodom.jsonl" in trace_text


def test_go2_measured_outcomes_report_states_boundary(tmp_path: Path) -> None:
    trace = build_go2_measured_outcome_trace(_write_capture(tmp_path / "capture"))

    report = render_go2_measured_outcomes_report(trace)

    assert "not an autonomous planner win" in report
    assert "`outcome_kind=real_measured`" in report
    assert "`score_kind=measured_metric`" in report
    assert "does not by itself satisfy the planner kill criterion" in report


def test_go2_measured_outcomes_sanitizes_private_capture_metadata(tmp_path: Path) -> None:
    capture_dir = _write_capture(
        tmp_path / "capture",
        robot_overrides={
            "serial": _FAKE_SERIAL,
            "robot_locator": "redaction-sentinel-network-handle",
            "operator_note": "redaction-sentinel-operator-note",
        },
    )

    trace = build_go2_measured_outcome_trace(capture_dir)

    trace_text = json.dumps(trace, sort_keys=True)
    assert _FAKE_SERIAL not in trace_text
    assert "redaction-sentinel-network-handle" not in trace_text
    assert "redaction-sentinel-operator-note" not in trace_text


def test_go2_measured_outcomes_replaces_capture_labels_with_synthetic_ids(
    tmp_path: Path,
) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    trial_path = capture_dir / "analysis" / "trial_outcomes.csv"
    trial_path.write_text(
        trial_path.read_text(encoding="utf-8").replace(
            "forward-slow-r01",
            "redaction-sentinel-trial-hostname",
        ),
        encoding="utf-8",
    )
    stats_path = capture_dir / "analysis" / "command_group_stats.csv"
    stats_path.write_text(
        stats_path.read_text(encoding="utf-8").replace(
            "x=0.18 y=0.0 z=0.0 d=1.0",
            "redaction-sentinel-command-group",
        ),
        encoding="utf-8",
    )

    trace = build_go2_measured_outcome_trace(capture_dir)

    trace_text = json.dumps(trace, sort_keys=True)
    assert "redaction-sentinel-trial-hostname" not in trace_text
    assert "redaction-sentinel-command-group" not in trace_text
    assert {row["label"] for row in trace["candidate_actions"]} == {
        "command-group-0",
        "command-group-1",
    }
    for outcome in trace["candidate_outcomes"]:
        assert "trial_labels" not in outcome["metrics"]
        assert all(
            str(trial_id).startswith("trial-") for trial_id in outcome["metrics"]["trial_ids"]
        )


def test_go2_measured_outcomes_summary_paths_are_portable(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    output_dir = tmp_path / "absolute-output"

    result = run_go2_measured_outcomes(capture_dir, output_dir)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert summary["decision_trace_path"] == "decision-trace-go2-real-measured.json"
    assert summary["report_path"] == "go2-measured-outcomes-report.md"
    assert str(output_dir) not in json.dumps(summary, sort_keys=True)

    workflow = run_go2_measured_outcomes_workflow(capture_dir, output_dir)
    assert workflow["decision_trace_path"] == "decision-trace-go2-real-measured.json"
    assert workflow["summary_path"] == "summary.json"
    assert workflow["report_path"] == "go2-measured-outcomes-report.md"
    assert str(output_dir) not in json.dumps(workflow, sort_keys=True)


def test_go2_measured_outcomes_drops_unsafe_metric_keys(tmp_path: Path) -> None:
    unsafe_key = "/Use" + "rs/alice/private"
    capture_dir = _write_capture(
        tmp_path / "capture",
        topic_counts={unsafe_key: 9, "ROBOTODOM": 42, "ULIDAR_ARRAY": 22, "unsafe.key": 7},
    )

    trace = build_go2_measured_outcome_trace(capture_dir)

    topic_counts = trace["observation"]["summary"]["topic_counts"]
    assert topic_counts == {"ROBOTODOM": 42, "ULIDAR_ARRAY": 22}
    assert unsafe_key not in json.dumps(trace, sort_keys=True)


def test_go2_measured_outcomes_rejects_fractional_counts(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    stats_path = capture_dir / "analysis" / "command_group_stats.csv"
    stats_path.write_text(
        stats_path.read_text(encoding="utf-8").replace(
            "x=0.18 y=0.0 z=0.0 d=1.0,2,0.18",
            "x=0.18 y=0.0 z=0.0 d=1.0,1.5,0.18",
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorldStateError, match="must be an integer"):
        build_go2_measured_outcome_trace(capture_dir)


def test_go2_measured_outcomes_rejects_whitespace_numeric_cells(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    stats_path = capture_dir / "analysis" / "command_group_stats.csv"
    stats_path.write_text(
        stats_path.read_text(encoding="utf-8").replace(
            "x=0.18 y=0.0 z=0.0 d=1.0,2,0.18",
            "x=0.18 y=0.0 z=0.0 d=1.0,2,   ",
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorldStateError, match="missing numeric field 'cmd_forward_m'"):
        build_go2_measured_outcome_trace(capture_dir)


def test_go2_measured_outcomes_rejects_zero_trial_command_groups(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    stats_path = capture_dir / "analysis" / "command_group_stats.csv"
    stats_path.write_text(
        stats_path.read_text(encoding="utf-8").replace(
            "x=0.18 y=0.0 z=0.0 d=1.0,2,0.18",
            "x=0.18 y=0.0 z=0.0 d=1.0,0,0.18",
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorldStateError, match="must be greater than 0"):
        build_go2_measured_outcome_trace(capture_dir)


def test_go2_measured_outcomes_rejects_unmatched_command_groups(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    stats_path = capture_dir / "analysis" / "command_group_stats.csv"
    stats_path.write_text(
        stats_path.read_text(encoding="utf-8").replace(
            "x=0.18 y=0.0 z=0.0 d=1.0,2,0.18",
            "x=0.18 y=0.0 z=0.0 d=1.0,2,0.181",
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorldStateError, match="no matching trial rows"):
        build_go2_measured_outcome_trace(capture_dir)


def test_go2_measured_outcomes_rejects_missing_group_stats(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    (capture_dir / "analysis" / "command_group_stats.csv").write_text(
        "group,n,cmd_forward_m,cmd_lateral_m,cmd_yaw_rad\n",
        encoding="utf-8",
    )

    with pytest.raises(WorldForgeError, match="command_group_stats"):
        build_go2_measured_outcome_trace(capture_dir)


def test_go2_measured_outcomes_wraps_invalid_capture_json(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / _FAKE_SERIAL)
    (capture_dir / "go2-native-system-id-run.json").write_text("{", encoding="utf-8")

    with pytest.raises(WorldStateError, match="JSON is invalid") as exc_info:
        build_go2_measured_outcome_trace(capture_dir)
    error_text = str(exc_info.value)
    assert "<capture-dir>/go2-native-system-id-run.json" in error_text
    assert _FAKE_SERIAL not in error_text


def test_go2_measured_outcomes_rejects_non_object_capture_json(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / _FAKE_SERIAL)
    (capture_dir / "go2-native-system-id-run.json").write_text("[]", encoding="utf-8")

    with pytest.raises(WorldStateError, match="JSON must be an object") as exc_info:
        build_go2_measured_outcome_trace(capture_dir)
    error_text = str(exc_info.value)
    assert "<capture-dir>/go2-native-system-id-run.json" in error_text
    assert _FAKE_SERIAL not in error_text


def test_go2_measured_outcomes_wraps_unreadable_capture_csv(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    (capture_dir / "analysis" / "trial_outcomes.csv").write_bytes(b"\xff")

    with pytest.raises(WorldStateError, match="CSV could not be read"):
        build_go2_measured_outcome_trace(capture_dir)


def test_go2_measured_outcomes_rejects_missing_stream_reference(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    (capture_dir / "streams" / "topics" / "robotodom.jsonl").unlink()

    with pytest.raises(WorldForgeError, match="ROBOTODOM stream reference"):
        build_go2_measured_outcome_trace(capture_dir)


def test_go2_measured_outcomes_rejects_zero_stream_counts(tmp_path: Path) -> None:
    capture_dir = _write_capture(
        tmp_path / "capture",
        topic_counts={"ROBOTODOM": 0, "ULIDAR_ARRAY": 22, "LOW_STATE": 4},
    )

    with pytest.raises(WorldForgeError, match="ROBOTODOM messages"):
        build_go2_measured_outcome_trace(capture_dir)


def test_go2_measured_outcomes_rejects_non_boolean_capture_flags(tmp_path: Path) -> None:
    capture_dir = _write_capture(
        tmp_path / "capture",
        robot_overrides={"ip_redacted": "false"},
    )

    with pytest.raises(WorldStateError, match="ip_redacted"):
        build_go2_measured_outcome_trace(capture_dir)


def test_go2_measured_outcomes_wraps_output_directory_errors(tmp_path: Path) -> None:
    capture_dir = _write_capture(tmp_path / "capture")
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("existing file", encoding="utf-8")

    with pytest.raises(WorldStateError, match=r"output_dir\.mkdir"):
        run_go2_measured_outcomes(capture_dir, output_path)


def test_go2_measured_outcomes_report_rejects_malformed_trace(tmp_path: Path) -> None:
    trace = build_go2_measured_outcome_trace(_write_capture(tmp_path / "capture"))
    del trace["candidate_outcomes"][0]["commanded"]

    with pytest.raises(WorldForgeError, match="report requires candidate_outcomes"):
        render_go2_measured_outcomes_report(trace)


def test_measured_metric_score_kind_is_registered() -> None:
    assert "measured_metric" in SCORE_KINDS


def test_go2_measured_outcomes_cli_help_has_description(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Sanitize a host-owned Go2 native-rate system-ID capture" in output


def _write_capture(
    capture_dir: Path,
    *,
    robot_overrides: dict[str, object] | None = None,
    topic_counts: dict[str, int] | None = None,
) -> Path:
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
                "0,forward-slow-r01,0.18,0.0,0.0,1.0,0.18,0.0,0.0,0.01,0.01,"
                "0.02,0.0,0.01,0.02,10,4,2",
                "1,forward-slow-r02,0.18,0.0,0.0,1.0,0.18,0.0,0.0,0.01,0.01,"
                "0.03,0.0,0.01,0.03,11,5,2",
                "2,forward-fast-r01,0.25,0.0,0.0,1.2,0.3,0.0,0.0,0.20,0.01,"
                "0.22,0.01,0.02,0.22,12,6,3",
                "3,forward-fast-r02,0.25,0.0,0.0,1.2,0.3,0.0,0.0,0.19,0.01,"
                "0.21,0.01,0.02,0.21,13,7,3",
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
                "x=0.18 y=0.0 z=0.0 d=1.0,2,0.18,0.0,0.0,0.025,0.005,0.02,0.03,"
                "0.0,0.0,0.0,0.0,0.01,0.0,0.01,0.01,0.025,0.005,0.02,0.03",
                "x=0.25 y=0.0 z=0.0 d=1.2,2,0.3,0.0,0.0,0.215,0.005,0.21,0.22,"
                "0.01,0.0,0.01,0.01,0.02,0.0,0.02,0.02,0.215,0.005,0.21,0.22",
            ]
        ),
    )
    (analysis_dir / "system_id_fit.json").write_text(
        json.dumps(
            {
                "model": "measured_body_outcome = intercept + A @ command",
                "features": ["intercept", "cmd_forward_m"],
                "targets": ["measured_forward_body_m"],
                "r2": {"measured_forward_body_m": 0.8},
                "rmse": {"measured_forward_body_m": 0.04},
            }
        ),
        encoding="utf-8",
    )
    robot = {
        "model": "Unitree Go2 Air",
        "firmware_version": "V1.1.12",
        "hardware_version": "V2.0",
        "ip_redacted": True,
        "serial_redacted": True,
    }
    if robot_overrides:
        robot.update(robot_overrides)
    (capture_dir / "go2-native-system-id-run.json").write_text(
        json.dumps(
            {
                "artifact_kind": "worldforge.go2_air_native_rate_system_id_capture",
                "capture_design": {"native_rate_streams": True, "rgb_frames_recorded": False},
                "driver": "unitree_webrtc_connect",
                "errors": [],
                "robot": robot,
                "scene": {"surface": "office indoor floor"},
                "saved_frame_count": 0,
                "saved_lidar_count": 10,
                "topic_counts": topic_counts
                or {"ROBOTODOM": 42, "ULIDAR_ARRAY": 22, "LOW_STATE": 4},
            }
        ),
        encoding="utf-8",
    )
    return capture_dir


def _write_text(path: Path, text: str) -> None:
    path.write_text(f"{text}\n", encoding="utf-8")
