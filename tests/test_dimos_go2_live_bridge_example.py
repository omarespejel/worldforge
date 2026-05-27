from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from worldforge import WorldForgeError

ROOT = Path(__file__).resolve().parents[1]
LIVE_BRIDGE = ROOT / "examples" / "dimos-go2-trace-judge" / "live_dimos_bridge.py"


def _load_bridge():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "worldforge_dimos_go2_live_bridge_example",
        LIVE_BRIDGE,
    )
    if spec is None:
        raise AssertionError(f"Failed to load module spec for {LIVE_BRIDGE}.")
    if spec.loader is None:
        raise AssertionError(f"No loader found on module spec for {LIVE_BRIDGE}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRunner:
    def __init__(self, bridge, *, exit_code: int = 0) -> None:
        self.bridge = bridge
        self.exit_code = exit_code
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, argv: list[str], timeout_seconds: float):
        self.calls.append((argv, timeout_seconds))
        return self.bridge.CommandResult(
            argv=argv,
            exit_code=self.exit_code,
            stdout='{"ok": true, "robot_ip": "192.168.12.1"}',
            stderr="token=secret-value",
        )


def test_probe_writes_sanitized_safe_artifacts(tmp_path) -> None:
    bridge = _load_bridge()
    runner = FakeRunner(bridge)

    result = bridge.run_probe(
        output_dir=tmp_path / "probe",
        run_id="probe-1",
        dimos_bin="dimos",
        runner=runner,
    )

    output_dir = Path(result["output_dir"])
    probe = json.loads((output_dir / "probe.json").read_text())
    manifest = json.loads((output_dir / "bridge_manifest.json").read_text())

    assert result["status"] == "probe_completed"
    assert probe["safe_probe_only"] is True
    assert probe["worldforge_executes_robot"] is False
    assert probe["summary"]["mcp_reachable"] is True
    assert len(runner.calls) == 3
    assert "<redacted-ipv4>" in probe["commands"]["status"]["stdout"]
    assert "secret-value" not in probe["commands"]["status"]["stderr"]
    assert manifest["operation"] == "probe"
    assert manifest["safety_boundary"]["worldforge_executes_robot"] is False


def test_probe_handles_missing_dimos_binary(tmp_path) -> None:
    bridge = _load_bridge()

    result = bridge.run_probe(
        output_dir=tmp_path / "probe",
        run_id="probe-missing",
        dimos_bin="/no/such/dimos",
    )
    probe = json.loads((Path(result["output_dir"]) / "probe.json").read_text())

    assert result["status"] == "probe_completed"
    assert probe["summary"]["mcp_reachable"] is False
    assert probe["commands"]["status"]["error"].startswith("command not found")


def test_sanitize_capture_redacts_json_secrets_and_signed_urls() -> None:
    bridge = _load_bridge()

    sanitized = bridge._sanitize_capture(
        '{"token": "json-token", "api_key": "json-key", '
        '"authorization": "Bearer json-bearer", '
        '"signed_url": '
        '"https://assets.example/video.mp4?X-Amz-Signature=sig-secret&token=url-secret"} '
        "download=https://assets.example/clip.mp4?token=query-secret robot=192.168.12.1"
    )

    for secret in (
        "json-token",
        "json-key",
        "json-bearer",
        "sig-secret",
        "url-secret",
        "query-secret",
        "192.168.12.1",
    ):
        assert secret not in sanitized
    assert "[redacted]" in sanitized
    assert "<redacted-ipv4>" in sanitized


def test_dry_run_selected_builds_gated_mcp_command(tmp_path) -> None:
    bridge = _load_bridge()

    result = bridge.dry_run_selected(
        output_dir=tmp_path / "dry",
        run_id="dry-1",
        goal="follow the indicated safe target",
        dimos_bin="dimos",
    )
    output_dir = Path(result["output_dir"])
    selected = json.loads((output_dir / "selected_mcp_command.json").read_text())
    manifest = json.loads((output_dir / "bridge_manifest.json").read_text())

    assert result["status"] == "dry_run_completed"
    assert selected["action"] == "relative_move"
    assert selected["params"] == {"forward": 0.25, "left": -0.2}
    assert selected["will_execute"] is False
    assert selected["argv"][:4] == ["dimos", "mcp", "call", "relative_move"]
    assert selected["execution_gates"]["required_env_var"] == bridge.EXECUTE_ENV_VAR
    assert manifest["execution_status"] == "not_executed"
    assert manifest["upstream_ready"] is False


def test_dry_run_selected_can_include_probe(tmp_path) -> None:
    bridge = _load_bridge()
    runner = FakeRunner(bridge)

    result = bridge.dry_run_selected(
        output_dir=tmp_path / "dry",
        run_id="dry-with-probe",
        goal="inspect safely",
        with_probe=True,
        runner=runner,
    )
    manifest = json.loads((Path(result["output_dir"]) / "bridge_manifest.json").read_text())

    assert manifest["probe_summary"]["can_list_tools"] is True
    assert (Path(result["output_dir"]) / "probe" / "probe.json").exists()


def test_execute_selected_requires_env_gate(tmp_path) -> None:
    bridge = _load_bridge()

    with pytest.raises(WorldForgeError, match=bridge.EXECUTE_ENV_VAR):
        bridge.execute_selected(
            output_dir=tmp_path / "execute",
            run_id="execute-1",
            goal="inspect safely",
            confirm=bridge.EXECUTE_CONFIRMATION,
            environ={},
        )


def test_execute_selected_requires_confirmation(tmp_path) -> None:
    bridge = _load_bridge()

    with pytest.raises(WorldForgeError, match=bridge.EXECUTE_CONFIRMATION):
        bridge.execute_selected(
            output_dir=tmp_path / "execute",
            run_id="execute-1",
            goal="inspect safely",
            confirm="wrong",
            environ={bridge.EXECUTE_ENV_VAR: "1"},
        )


def test_execute_selected_runs_selected_command_when_gated(tmp_path) -> None:
    bridge = _load_bridge()
    runner = FakeRunner(bridge)

    result = bridge.execute_selected(
        output_dir=tmp_path / "execute",
        run_id="execute-1",
        goal="inspect safely",
        confirm=bridge.EXECUTE_CONFIRMATION,
        with_probe=False,
        runner=runner,
        environ={bridge.EXECUTE_ENV_VAR: "1"},
    )
    output_dir = Path(result["output_dir"])
    execution = json.loads((output_dir / "execution_result.json").read_text())

    assert result["status"] == "execute_completed"
    assert len(runner.calls) == 1
    assert runner.calls[0][0][:4] == ["dimos", "mcp", "call", "relative_move"]
    assert execution["executed_by"] == "host_runtime:dimos-cli"
    assert execution["worldforge_executes_robot"] is False


def test_main_returns_nonzero_for_execute_failure(monkeypatch, tmp_path, capsys) -> None:
    bridge = _load_bridge()

    def fake_execute_selected(**_kwargs):
        return {
            "status": "execute_failed",
            "run_id": "failed-run",
            "output_dir": str(tmp_path),
            "artifact_paths": {"execution_result": "execution_result.json"},
        }

    monkeypatch.setattr(bridge, "execute_selected", fake_execute_selected)

    exit_code = bridge.main(
        [
            "execute-selected",
            "--output-dir",
            str(tmp_path),
            "--confirm",
            bridge.EXECUTE_CONFIRMATION,
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "execute_failed"


def test_execute_selected_ignores_stale_saved_argv_when_gated(tmp_path) -> None:
    bridge = _load_bridge()
    runner = FakeRunner(bridge)
    output_dir = tmp_path / "execute"
    output_dir.mkdir()
    (output_dir / "selected_mcp_command.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": "tampered",
                "action": "relative_move",
                "params": {"forward": 99},
                "argv": ["/tmp/not-dimos", "unsafe"],
            }
        ),
        encoding="utf-8",
    )

    result = bridge.execute_selected(
        output_dir=output_dir,
        run_id="execute-ignore-stale",
        goal="inspect safely",
        confirm=bridge.EXECUTE_CONFIRMATION,
        with_probe=False,
        runner=runner,
        environ={bridge.EXECUTE_ENV_VAR: "1"},
    )
    selected = json.loads((Path(result["output_dir"]) / "selected_mcp_command.json").read_text())

    assert runner.calls[0][0][:4] == ["dimos", "mcp", "call", "relative_move"]
    assert "/tmp/not-dimos" not in runner.calls[0][0]
    assert selected["candidate_id"] == "detour_right"
    assert selected["params"] == {"forward": 0.25, "left": -0.2}


def test_build_selected_mcp_command_rejects_unsafe_motion() -> None:
    bridge = _load_bridge()

    with pytest.raises(WorldForgeError, match="outside safe range"):
        bridge.build_selected_mcp_command(
            selected_action={
                "selected_candidate_id": "bad",
                "action": "relative_move",
                "params": {"forward": 5.0},
            },
        )


def test_build_selected_mcp_command_rejects_unknown_action() -> None:
    bridge = _load_bridge()

    with pytest.raises(WorldForgeError, match="unsupported live DimOS action"):
        bridge.build_selected_mcp_command(
            selected_action={
                "selected_candidate_id": "bad",
                "action": "flip",
                "params": {"seconds": 1.0},
            },
        )


@pytest.mark.parametrize(
    ("selected_action", "match"),
    (
        (
            {"action": "relative_move", "params": {"forward": 0.1}},
            "selected_action.selected_candidate_id",
        ),
        (
            {
                "selected_candidate_id": "bad",
                "action": "relative_move",
                "params": ["forward", 0.1],
            },
            "selected_action.params",
        ),
        (
            {
                "selected_candidate_id": "bad",
                "action": "relative_move",
                "params": {"forward": math.nan},
            },
            "finite",
        ),
        (
            {
                "selected_candidate_id": "bad",
                "action": "relative_move",
                "params": {"forward": math.inf},
            },
            "finite",
        ),
        (
            {
                "selected_candidate_id": "bad",
                "action": "relative_move",
                "params": {"forward": b"0.1"},
            },
            "numeric",
        ),
        (
            {
                "selected_candidate_id": "bad",
                "action": "relative_move",
                "params": {1: 0.1},
            },
            "keys must be strings",
        ),
    ),
)
def test_build_selected_mcp_command_rejects_malformed_payloads(selected_action, match) -> None:
    bridge = _load_bridge()

    with pytest.raises(WorldForgeError, match=match):
        bridge.build_selected_mcp_command(selected_action=selected_action)


def test_read_json_translates_missing_and_malformed_artifacts(tmp_path) -> None:
    bridge = _load_bridge()

    with pytest.raises(WorldForgeError, match="missing required artifact"):
        bridge._read_json(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(WorldForgeError, match="valid JSON"):
        bridge._read_json(malformed)

    array_payload = tmp_path / "array.json"
    array_payload.write_text("[]", encoding="utf-8")
    with pytest.raises(WorldForgeError, match="JSON object"):
        bridge._read_json(array_payload)
