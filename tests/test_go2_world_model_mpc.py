from __future__ import annotations

import json
from pathlib import Path

import pytest

import worldforge.demos.go2_world_model_mpc as go2_mpc
from worldforge.decision_trace import validate_decision_trace
from worldforge.demos.go2_world_model_mpc import (
    main,
    run_go2_world_model_mpc,
    run_go2_world_model_mpc_workflow,
)
from worldforge.models import WorldForgeError, WorldStateError


def test_go2_world_model_mpc_writes_dimos_shadow_bridge_and_traces(
    tmp_path: Path,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")

    result = run_go2_world_model_mpc(dataset_csv=csv_path, output_dir=tmp_path / "out")

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    bridge = json.loads(result.dimos_bridge_path.read_text(encoding="utf-8"))
    traces = [
        validate_decision_trace(json.loads(path.read_text(encoding="utf-8")))
        for path in result.decision_trace_paths
    ]

    assert summary["artifact_kind"] == "worldforge.go2_world_model_mpc_summary"
    assert summary["dataset"]["trial_count"] == 12
    assert summary["dataset"]["command_cell_count"] == 4
    assert summary["dimos"]["role"] == "host_owned_runtime_and_memory_bridge"
    assert summary["dimos"]["mode"] == "shadow_replay_no_execution"
    assert summary["claim_boundary"]["live_dimos_execution"] is False

    assert bridge["artifact_kind"] == "worldforge.dimos_go2_world_model_mpc_shadow_bridge"
    assert bridge["runtime"]["name"] == "DimOS"
    assert bridge["runtime"]["mode"] == "shadow_replay_no_execution"
    assert bridge["runtime"]["imports_dimos"] is False
    assert bridge["runtime"]["hardware_commands_sent"] is False
    assert bridge["safety_gate"]["operator_approval_required"] is True
    assert bridge["decisions"][0]["dimos_skill_contract"]["method"] == (
        "bounded_sport_move_then_stop"
    )

    assert len(traces) == 3
    for trace in traces:
        assert trace["host_runtime"]["name"] == "DimOS shadow bridge"
        assert trace["host_runtime"]["mode"] == "shadow_replay_no_execution"
        assert trace["claim_boundary"]["outcome_kind"] == "real_measured"
        assert trace["claim_boundary"]["hardware_executed"] is True
        assert trace["claim_boundary"]["learned_model_used"] is False
        assert trace["outcome"]["metrics"]["live_dimos_execution"] is False
        assert trace["interop"]["dimos"]["required_live_skill"] == ("bounded_sport_move_then_stop")

    payload_text = json.dumps(
        {"summary": summary, "bridge": bridge, "traces": traces},
        sort_keys=True,
    )
    assert str(tmp_path) not in payload_text
    assert "/Users/" not in payload_text
    assert "192.168." not in payload_text


def test_go2_world_model_mpc_rejects_local_trial_table_url(tmp_path: Path) -> None:
    with pytest.raises(WorldForgeError):
        run_go2_world_model_mpc(
            dataset_csv=None,
            trial_table_url="http://127.0.0.1/private/all_trials_normalized.csv",
            output_dir=tmp_path / "out",
        )


def test_go2_world_model_mpc_rejects_non_huggingface_trial_table_url(
    tmp_path: Path,
) -> None:
    with pytest.raises(WorldForgeError):
        run_go2_world_model_mpc(
            dataset_csv=None,
            trial_table_url="https://93.184.216.34/all_trials_normalized.csv",
            output_dir=tmp_path / "out",
        )


def test_go2_world_model_mpc_rejects_oversized_remote_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(go2_mpc, "validate_remote_url", lambda url, **_: url)

    def fake_urlopen(url: str, *, timeout: int) -> _FakeResponse:
        assert url == go2_mpc.DEFAULT_TRIAL_TABLE_URL
        assert timeout == 20
        return _FakeResponse(b"x" * (go2_mpc._MAX_TRIAL_TABLE_BYTES + 1))

    monkeypatch.setattr(go2_mpc, "urlopen", fake_urlopen)

    with pytest.raises(WorldForgeError, match="exceeds the download limit"):
        run_go2_world_model_mpc(dataset_csv=None, output_dir=tmp_path / "out")


def test_go2_world_model_mpc_does_not_persist_raw_remote_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")
    payload = csv_path.read_bytes()
    trial_table_url = go2_mpc.DEFAULT_TRIAL_TABLE_URL

    monkeypatch.setattr(go2_mpc, "validate_remote_url", lambda url, **_: url)
    monkeypatch.setattr(go2_mpc, "urlopen", lambda *_args, **_kwargs: _FakeResponse(payload))

    result = run_go2_world_model_mpc(
        dataset_csv=None,
        trial_table_url=trial_table_url,
        output_dir=tmp_path / "out",
    )

    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            result.summary_path,
            result.report_path,
            result.dimos_bridge_path,
            *result.decision_trace_paths,
        ]
    )
    assert trial_table_url not in artifact_text
    assert "huggingface.co" not in artifact_text
    assert "remote_hf_pinned_csv" in artifact_text
    assert go2_mpc.DEFAULT_DATASET_REVISION in artifact_text


def test_go2_world_model_mpc_rejects_unsafe_target_id(tmp_path: Path) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")

    with pytest.raises(WorldStateError):
        run_go2_world_model_mpc(
            dataset_csv=csv_path,
            targets=[("../outside", (0.2, 0.0, 0.0))],
            output_dir=tmp_path / "out",
        )


def test_go2_world_model_mpc_slugs_safe_target_id(tmp_path: Path) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")

    result = run_go2_world_model_mpc(
        dataset_csv=csv_path,
        targets=[("Forward Target 20cm", (0.2, 0.0, 0.0))],
        output_dir=tmp_path / "out",
    )

    assert len(result.decision_trace_paths) == 1
    assert result.decision_trace_paths[0].name == (
        "decision-trace-go2-world-model-mpc-forward-target-20cm.json"
    )


def test_go2_world_model_mpc_workflow_returns_portable_paths(tmp_path: Path) -> None:
    csv_path = _write_trials_csv(tmp_path / "all_trials_normalized.csv")

    summary = run_go2_world_model_mpc_workflow(
        dataset_csv=csv_path,
        output_dir=tmp_path / "out",
    )

    assert summary["trial_count"] == 12
    assert summary["dimos_mode"] == "shadow_replay_no_execution"
    assert summary["summary_path"] == "world-model-mpc-summary.json"
    assert summary["report_path"] == "world-model-mpc-report.md"
    assert summary["dimos_bridge_path"] == "dimos-shadow-bridge-plan.json"
    assert all(
        path.startswith("decision-trace-go2-world-model-mpc-")
        for path in summary["decision_trace_paths"]
    )
    assert str(tmp_path) not in json.dumps(summary, sort_keys=True)


def test_go2_world_model_mpc_cli_help_has_description(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "offline Go2 world-model MPC demo" in output
    assert "DimOS shadow-bridge" in output


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int = -1) -> bytes:
        return self._payload


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
