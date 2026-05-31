from __future__ import annotations

import json
from pathlib import Path

from worldforge.decision_trace import DECISION_TRACE_SCHEMA_VERSION, validate_decision_trace
from worldforge.demos.cross_embodiment_decision_evidence import (
    run_cross_embodiment_decision_evidence,
)


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


def test_cross_embodiment_report_contains_kill_criterion(tmp_path: Path) -> None:
    summary = run_cross_embodiment_decision_evidence(output_dir=tmp_path)
    report = Path(summary["report_path"]).read_text(encoding="utf-8")

    assert "DecisionTrace v1" in report
    assert "Kill Criterion" in report
    assert "stop pushing this integration and pivot" in report
