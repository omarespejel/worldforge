from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldforge.decision_trace import DECISION_TRACE_SCHEMA_VERSION, validate_decision_trace
from worldforge.demos.cross_embodiment_decision_evidence import (
    _round_json_floats,
    normalize_go2_decision_trace,
    normalize_so101_decision_trace,
    run_cross_embodiment_decision_evidence,
)
from worldforge.models import WorldForgeError

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_BUNDLE = ROOT / "examples" / "cross-embodiment-decision-evidence"


def test_cross_embodiment_evidence_bundle_writes_valid_decision_traces(tmp_path: Path) -> None:
    summary = run_cross_embodiment_decision_evidence(output_dir=tmp_path)

    assert summary["trace_count"] == 3
    assert (tmp_path / "cross-embodiment-report.md").is_file()
    for trace_name in ("go2", "pimsim", "so101"):
        path = Path(summary["traces"][trace_name]["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        trace = validate_decision_trace(payload)

        assert trace["schema_version"] == DECISION_TRACE_SCHEMA_VERSION
        assert trace["claim_boundary"]["hardware_executed"] is False
        assert trace["claim_boundary"]["score_kind"] == "hand_cost"
        assert trace["candidate_actions"]
        assert trace["scores"]
        selected_score = next(
            score
            for score in trace["scores"]
            if score["candidate_id"] == trace["selected_action"]["candidate_id"]
        )
        assert 0.0 <= selected_score["normalized"]["value_signal"] <= 1.0

    so101_payload = json.loads(Path(summary["traces"]["so101"]["path"]).read_text(encoding="utf-8"))
    assert so101_payload["outcome"]["metrics"]["outcome_source"] == "mock_replay_execution"


def test_cross_embodiment_report_contains_kill_criterion(tmp_path: Path) -> None:
    summary = run_cross_embodiment_decision_evidence(output_dir=tmp_path)
    report = Path(summary["report_path"]).read_text(encoding="utf-8")

    assert "DecisionTrace v1" in report
    assert "Value Signal" in report
    assert "Local Margin" in report
    assert "Kill Criterion" in report
    assert "stop pushing this integration and pivot" in report
    assert "mock_replay_execution" in report


def test_go2_normalization_wraps_parse_errors() -> None:
    with pytest.raises(WorldForgeError, match="Failed to normalize Go2 decision trace"):
        normalize_go2_decision_trace(
            {},
            trace_id="bad-go2",
            step_index=0,
            host_runtime_name="bad fixture",
        )


def test_so101_normalization_wraps_parse_errors() -> None:
    with pytest.raises(WorldForgeError, match="Failed to normalize SO-101 decision trace"):
        normalize_so101_decision_trace({}, trace_id="bad-so101", step_index=0)


def test_round_json_floats_rejects_non_json_values() -> None:
    assert _round_json_floats({"score": 1.23456789}) == {"score": 1.234568}

    with pytest.raises(TypeError, match="keys must be strings"):
        _round_json_floats({1: "bad"})
    with pytest.raises(TypeError, match="finite"):
        _round_json_floats({"score": float("nan")})
    with pytest.raises(TypeError, match="Unsupported JSON value type"):
        _round_json_floats({"tuple": (1.0,)})


def test_checked_in_cross_embodiment_evidence_bundle_matches_generator(
    tmp_path: Path,
) -> None:
    run_cross_embodiment_decision_evidence(output_dir=tmp_path)

    for filename in (
        "decision-trace-go2.json",
        "decision-trace-pimsim.json",
        "decision-trace-so101.json",
    ):
        generated = json.loads((tmp_path / filename).read_text(encoding="utf-8"))
        committed = json.loads((COMMITTED_BUNDLE / filename).read_text(encoding="utf-8"))
        assert generated == committed

    assert (tmp_path / "cross-embodiment-report.md").read_text(encoding="utf-8") == (
        COMMITTED_BUNDLE / "cross-embodiment-report.md"
    ).read_text(encoding="utf-8")

    generated_summary = _portable_summary(
        json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    )
    committed_summary = _portable_summary(
        json.loads((COMMITTED_BUNDLE / "summary.json").read_text(encoding="utf-8"))
    )
    assert generated_summary == committed_summary


def _portable_summary(summary: dict[str, object]) -> dict[str, object]:
    portable = dict(summary)
    traces = {}
    raw_traces = portable["traces"]
    assert isinstance(raw_traces, dict)
    for name, raw_row in raw_traces.items():
        assert isinstance(raw_row, dict)
        row = dict(raw_row)
        row["path"] = Path(str(row["path"])).name
        traces[name] = row
    portable["traces"] = traces
    portable["report_path"] = Path(str(portable["report_path"])).name
    return portable
