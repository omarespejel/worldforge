from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from worldforge import WorldForgeError

ROOT = Path(__file__).resolve().parents[1]
TRACE_JUDGE_APP = ROOT / "examples" / "dimos-go2-trace-judge" / "app.py"


def _load_app():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "worldforge_dimos_go2_trace_judge_example",
        TRACE_JUDGE_APP,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dimos_go2_trace_judge_writes_offline_artifacts(tmp_path) -> None:
    app = _load_app()

    result = app.run_trace_judge(output_dir=tmp_path / "run", run_id="test-go2")

    assert result["status"] == "passed"
    assert result["selected_candidate_id"] == "detour_right"
    output_dir = Path(result["output_dir"])
    observation = json.loads((output_dir / "observation_summary.json").read_text())
    scores = json.loads((output_dir / "candidate_scores.json").read_text())
    selected = json.loads((output_dir / "selected_action.json").read_text())
    outcome = json.loads((output_dir / "outcome_after_execution.json").read_text())
    manifest = json.loads((output_dir / "run_manifest.json").read_text())

    assert observation["costmap_summary"]["frontier_count"] == 4
    assert scores["schema_version"] == 1
    assert scores["worldforge_score_result"]["provider"] == "transparent-go2-score"
    assert scores["worldforge_score_result"]["lower_is_better"] is False
    assert scores["selected_candidate_id"] == "detour_right"
    assert selected["worldforge_executes_robot"] is False
    assert selected["execute_with"] == "host_runtime:mock-dimos"
    assert selected["live_execute_with"] == "host_runtime:dimos"
    assert outcome["executed_by"] == "mock-dimos"
    assert manifest["kind"] == "dimos_go2_trace_judge"
    assert manifest["artifact_paths"]["run_manifest"] == "run_manifest.json"
    assert manifest["safety_boundary"]["host_owns_emergency_stop"] is True
    assert manifest["safety_boundary"]["worldforge_certifies_robot_safety"] is False
    assert (output_dir / "report.md").read_text().startswith("# DimOS Go2 Trace Judge")


def test_transparent_go2_scorer_rejects_missing_candidate_features() -> None:
    app = _load_app()
    provider = app.TransparentGo2ScoreProvider()
    bad_candidate = {
        "id": "bad",
        "action": "relative_move",
        "params": {"forward": 0.2},
        "features": {"goal_alignment": 0.5},
        "reason_hint": "incomplete candidate",
    }

    with pytest.raises(WorldForgeError, match="information_gain"):
        provider.score_actions(info={"task": {}}, action_candidates=[bad_candidate])


def test_transparent_go2_scorer_rejects_missing_reason_hint() -> None:
    app = _load_app()
    provider = app.TransparentGo2ScoreProvider()
    bad_candidate = app.sample_candidates()[0]
    bad_candidate.pop("reason_hint")

    with pytest.raises(WorldForgeError, match="reason_hint"):
        provider.score_actions(info={"task": {}}, action_candidates=[bad_candidate])


def test_transparent_go2_scorer_validates_score_weights(monkeypatch) -> None:
    app = _load_app()
    monkeypatch.setitem(app.SCORE_WEIGHTS, "progress", None)

    with pytest.raises(WorldForgeError, match="score weight progress"):
        app.TransparentGo2ScoreProvider()


def test_dimos_go2_trace_judge_cli_defaults_to_offline_run(tmp_path, capsys) -> None:
    app = _load_app()

    exit_code = app.main(
        [
            "run",
            "--output-dir",
            str(tmp_path / "run"),
            "--run-id",
            "cli-go2",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_id"] == "cli-go2"
    assert payload["selected_candidate_id"] == "detour_right"
