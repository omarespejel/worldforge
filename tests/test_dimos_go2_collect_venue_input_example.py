from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from worldforge import WorldForgeError

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "examples" / "dimos-go2-trace-judge" / "collect_venue_input.py"


def _load_collector():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "worldforge_dimos_go2_collect_venue_input_example",
        COLLECTOR,
    )
    if spec is None:
        raise AssertionError(f"Failed to load module spec for {COLLECTOR}.")
    if spec.loader is None:
        raise AssertionError(f"No loader found on module spec for {COLLECTOR}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRunner:
    def __init__(self, bridge) -> None:
        self.bridge = bridge
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, argv: list[str], timeout_seconds: float):
        self.calls.append((argv, timeout_seconds))
        return self.bridge.CommandResult(
            argv=argv,
            exit_code=0,
            stdout='{"tools": ["relative_move", "wait"], "robot_ip": "192.168.12.1"}',
            stderr="",
        )


def test_collect_venue_input_writes_safe_probe_and_starter_input(tmp_path) -> None:
    collector = _load_collector()
    bridge = collector._load_module(
        collector.LIVE_BRIDGE_PATH,
        "worldforge_dimos_go2_live_bridge_for_collect_test",
    )
    runner = FakeRunner(bridge)

    result = collector.collect_venue_input(
        output_dir=tmp_path / "venue",
        run_id="venue-1",
        runner=runner,
    )

    output_dir = Path(result["output_dir"])
    venue_probe = json.loads((output_dir / "venue_probe.json").read_text())
    venue_input = json.loads((output_dir / "venue_input.json").read_text())
    manifest = json.loads((output_dir / "run_manifest.json").read_text())

    assert result["status"] == "venue_input_collected"
    assert len(runner.calls) == 3
    assert venue_probe["safe_probe_only"] is True
    assert venue_probe["worldforge_executes_robot"] is False
    assert "<redacted-ipv4>" in venue_probe["dimos_probe"]["commands"]["status"]["stdout"]
    assert venue_input["host_runtime"] == "dimos"
    assert venue_input["candidates"][2]["id"] == "detour_right"
    assert manifest["artifact_paths"]["venue_input"] == "venue_input.json"


def test_collect_venue_input_read_json_rejects_directory_without_path_leak(tmp_path) -> None:
    collector = _load_collector()

    with pytest.raises(WorldForgeError, match="missing a required probe artifact") as exc_info:
        collector._read_json(tmp_path)

    assert str(tmp_path) not in str(exc_info.value)


def test_collect_venue_input_read_json_rejects_non_finite_json(tmp_path) -> None:
    collector = _load_collector()
    bad_json = tmp_path / "probe.json"
    bad_json.write_text('{"elapsed_seconds": NaN}', encoding="utf-8")

    with pytest.raises(WorldForgeError, match="must not contain NaN or Infinity"):
        collector._read_json(bad_json)


def test_collect_venue_input_read_json_rejects_overflow_json(tmp_path) -> None:
    collector = _load_collector()
    bad_json = tmp_path / "probe.json"
    bad_json.write_text('{"elapsed_seconds": 1e309}', encoding="utf-8")

    with pytest.raises(WorldForgeError, match="must contain only finite numbers"):
        collector._read_json(bad_json)
