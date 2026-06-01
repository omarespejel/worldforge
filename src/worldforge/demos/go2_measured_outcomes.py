from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldforge.artifact_io import write_json_artifact
from worldforge.decision_trace import (
    DECISION_TRACE_ARTIFACT_KIND,
    DECISION_TRACE_SCHEMA_VERSION,
    decision_trace_digest,
    validate_decision_trace,
)
from worldforge.models import JSONDict, WorldForgeError, WorldStateError
from worldforge.provider_redaction import _SENSITIVE_FIELD_PATTERN

# Sanitize Go2 native-rate system-ID captures into DecisionTrace evidence. This module consumes
# host-owned Go2 Air capture summaries and emits small DecisionTrace-compatible artifacts. Raw
# telemetry, LiDAR sidecars, local paths, IP addresses, serial numbers, and office media stay
# outside the repository.

DEFAULT_OUTPUT_DIR = Path(".worldforge/go2-measured-outcomes")
RUN_ID = "go2-native-system-id-measured-outcomes"
CODE_REF = "feat/go2-measured-outcomes-decisiontrace"
CLI_DESCRIPTION = (
    "Sanitize a host-owned Go2 native-rate system-ID capture into DecisionTrace "
    "measured-outcome artifacts."
)
_YAW_WEIGHT_M_PER_RAD = 0.25
_SAFE_PUBLIC_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_:-]{0,80}$")
_SAFE_ARTIFACT_LABELS = {
    "command_group_stats.csv": "<analysis-dir>/command_group_stats.csv",
    "decision-trace-go2-real-measured.json": "<output-dir>/decision-trace-go2-real-measured.json",
    "go2-measured-outcomes-report.md": "<output-dir>/go2-measured-outcomes-report.md",
    "go2-native-system-id-run.json": "<capture-dir>/go2-native-system-id-run.json",
    "summary.json": "<output-dir>/summary.json",
    "system_id_fit.json": "<analysis-dir>/system_id_fit.json",
    "trial_outcomes.csv": "<analysis-dir>/trial_outcomes.csv",
}


@dataclass(frozen=True, slots=True)
class Go2MeasuredOutcomeResult:
    """Paths and payloads emitted for one sanitized Go2 measured-outcome run."""

    trace: JSONDict
    summary: JSONDict
    report_markdown: str
    decision_trace_path: Path
    summary_path: Path
    report_path: Path


def run_go2_measured_outcomes(
    capture_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Go2MeasuredOutcomeResult:
    """Build sanitized measured-outcome artifacts from a host-owned Go2 capture directory."""

    trace = build_go2_measured_outcome_trace(capture_dir)
    report = render_go2_measured_outcomes_report(trace)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorldStateError(
            f"Go2 measured-outcome output_dir.mkdir failed: {_safe_artifact_path(output_dir)}"
        ) from exc

    try:
        trace_path = write_json_artifact(
            output_dir / "decision-trace-go2-real-measured.json", trace
        )
    except OSError as exc:
        raise WorldStateError(
            "Go2 measured-outcome write_json_artifact failed for decision trace: "
            f"{_safe_artifact_path(output_dir / 'decision-trace-go2-real-measured.json')}"
        ) from exc
    report_path = output_dir / "go2-measured-outcomes-report.md"
    try:
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        raise WorldStateError(
            "Go2 measured-outcome report_path.write_text failed: "
            f"{_safe_artifact_path(report_path)}"
        ) from exc
    summary: JSONDict = {
        "schema_version": 1,
        "artifact_kind": "worldforge.go2_measured_outcomes_summary",
        "run_id": trace["run_id"],
        "trace_id": trace["trace_id"],
        "decision_trace_path": trace_path.name,
        "report_path": report_path.name,
        "outcome_kind": trace["outcome"]["kind"],
        "score_kind": trace["claim_boundary"]["score_kind"],
        "hardware_executed": trace["claim_boundary"]["hardware_executed"],
        "selected_action_id": trace["selected_action"]["candidate_id"],
        "selected_tracking_score": trace["selected_action"]["score"],
        "candidate_count": len(trace["candidate_actions"]),
        "completed_movement_trials": trace["outcome"]["metrics"]["completed_movement_trials"],
        "raw_capture_policy": "local_private_not_committed",
    }
    try:
        summary_path = write_json_artifact(output_dir / "summary.json", summary)
    except OSError as exc:
        raise WorldStateError(
            "Go2 measured-outcome write_json_artifact failed for summary: "
            f"{_safe_artifact_path(output_dir / 'summary.json')}"
        ) from exc
    return Go2MeasuredOutcomeResult(
        trace=trace,
        summary=summary,
        report_markdown=report,
        decision_trace_path=trace_path,
        summary_path=summary_path,
        report_path=report_path,
    )


def build_go2_measured_outcome_trace(capture_dir: Path) -> JSONDict:
    """Return a sanitized DecisionTrace v1 payload for the Go2 native-rate capture."""

    capture = _load_capture(capture_dir)
    candidates = _measured_command_candidates(capture)
    if not candidates:
        raise WorldForgeError("Go2 measured-outcome capture has no command groups to rank.")
    ranked = sorted(candidates, key=lambda row: (row["tracking_score"], row["candidate_id"]))
    selected = ranked[0]
    baseline = candidates[0]
    score_span = max(candidate["tracking_score"] for candidate in candidates) - min(
        candidate["tracking_score"] for candidate in candidates
    )
    trace: JSONDict = {
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "artifact_kind": DECISION_TRACE_ARTIFACT_KIND,
        "trace_id": "go2-native-system-id-real-measured",
        "run_id": RUN_ID,
        "step_index": 0,
        "prev_trace_id": None,
        "embodiment": {
            "kind": "quadruped_navigation",
            "platform": "unitree_go2_air",
            "embodiment_id": "go2-air-redacted",
            "action_space": "planar_body_velocity_system_id",
        },
        "host_runtime": {
            "name": "Unitree WebRTC native-rate capture",
            "mode": "host_owned_private_capture",
            "version": str(capture.run.get("driver", "unitree_webrtc_connect")),
        },
        "task": {
            "task_id": "go2-air-native-rate-system-id",
            "description": (
                "Summarize operator-scripted Go2 command groups with native odometry and LiDAR "
                "measurements for future measured-outcome scoring."
            ),
        },
        "observation": _observation(capture),
        "goal": _goal(),
        "candidate_actions": [_candidate_action(candidate) for candidate in ranked],
        "scores": _score_records(ranked, baseline_score=float(baseline["tracking_score"])),
        "selected_action": {
            "candidate_id": str(selected["candidate_id"]),
            "score": float(selected["tracking_score"]),
            "score_margin": _score_margin(ranked),
            "why_selected": (
                "Lowest post-hoc commanded-vs-measured tracking error across executed "
                "system-ID command groups. This is a measurement ranking, not an autonomous "
                "planner decision."
            ),
        },
        "counterfactuals": [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "score": float(candidate["tracking_score"]),
                "delta_vs_selected": round(
                    float(candidate["tracking_score"]) - float(selected["tracking_score"]),
                    6,
                ),
                "why_rejected": (
                    "Higher measured command-tracking error in the native-rate system-ID run."
                ),
            }
            for candidate in ranked
            if candidate["candidate_id"] != selected["candidate_id"]
        ],
        "candidate_outcomes": [_candidate_outcome(candidate) for candidate in ranked],
        "baseline": {
            "candidate_id": str(baseline["candidate_id"]),
            "score": float(baseline["tracking_score"]),
            "regret_vs_selected": round(
                float(baseline["tracking_score"]) - float(selected["tracking_score"]),
                6,
            ),
            "policy": "first_operator_scripted_command_group",
        },
        "outcome": _outcome(capture, selected, score_span),
        "planner_diagnostics": {
            "planner": "post_hoc_system_identification_ranker",
            "score_provider": "measured_command_tracking_error",
            "candidate_count": len(ranked),
            "operator_scripted_capture": True,
            "not_autonomous_decision": True,
            "source_artifact_kind": str(
                capture.run.get("artifact_kind", "worldforge.go2_native_capture")
            ),
        },
        "reproducibility": {
            "provider_version": "Go2 native-rate measured-outcome sanitizer",
            "checkpoint_hash": None,
            "model_card_ref": None,
            "input_digest": f"sha256:{_capture_digest(capture)}",
            "seed": None,
            "code_ref": CODE_REF,
        },
        "claim_boundary": {
            "score_kind": "measured_metric",
            "outcome_kind": "real_measured",
            "hardware_executed": True,
            "learned_model_used": False,
            "safety_controller": "operator_stop_and_unitree_balance_stand",
            "redaction_state": _redaction_state(capture),
            "limitations": [
                "This is an operator-scripted system-identification capture, not autonomous "
                "WorldForge control.",
                "Scores are post-hoc tracking-error metrics over measured odometry.",
                "Raw telemetry, LiDAR sidecars, capture labels, LAN identifiers, and robot "
                "serials remain local and are not embedded.",
                "No RGB frames are included because video destabilized the data channel during "
                "stable capture.",
                "External ground truth was not used; measurements are native odometry and LiDAR "
                "only.",
            ],
        },
        "interop": _interop_block(),
    }
    return validate_decision_trace(
        _round_json_floats(trace),
        name="Go2 native measured-outcome DecisionTrace",
    )


def render_go2_measured_outcomes_report(trace: JSONDict) -> str:
    """Render a compact Markdown report for the sanitized Go2 measured-outcome trace."""

    trace = validate_decision_trace(trace, name="Go2 measured-outcome report trace")
    try:
        metrics = trace["outcome"]["metrics"]
        selected_action = trace["selected_action"]
        scores_by_id = {str(row["candidate_id"]): row for row in trace["scores"]}
        candidate_outcomes = trace["candidate_outcomes"]
    except (KeyError, TypeError) as exc:
        raise WorldForgeError(
            "Go2 measured-outcome report requires outcome metrics, selected_action, "
            "scores, and candidate_outcomes."
        ) from exc
    lines = [
        "# Go2 Native Measured Outcomes",
        "",
        "This artifact summarizes a private native-rate Unitree Go2 Air system-identification "
        "run as DecisionTrace-compatible `real_measured` evidence.",
        "",
        "Boundary: this is not an autonomous planner win. The operator/script executed all "
        "movement groups; WorldForge ranks the measured command-tracking outcomes after the fact "
        "so future planners and scorers can compare against real hardware response.",
        "",
        "Raw telemetry and LiDAR sidecars stay local/private. The trace contains only sanitized "
        "counts, relative stream references, command parameters, and measured odometry summaries.",
        "",
        "## Capture Summary",
        "",
        f"- Completed movement trials: `{metrics['completed_movement_trials']}`",
        f"- Native odometry messages: `{metrics['native_odom_messages']}`",
        f"- ULIDAR_ARRAY messages: `{metrics['ulidar_array_messages']}`",
        f"- Saved LiDAR sidecars: `{metrics['saved_lidar_sidecars']}`",
        f"- Saved RGB frames: `{metrics['saved_rgb_frames']}`",
        f"- Selected measurement group: `{selected_action['candidate_id']}`",
        f"- Tracking score: `{selected_action['score']:.6f}`",
        "",
        "## Command Tracking",
        "",
        "| Rank | Candidate | Trials | Cmd Fwd m | Meas Fwd m | Cmd Lat m | Meas Lat m | "
        "Cmd Yaw rad | Meas Yaw rad | Score |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    try:
        for outcome in candidate_outcomes:
            candidate_id = str(outcome["candidate_id"])
            score = scores_by_id[candidate_id]
            commanded = outcome["commanded"]
            measured = outcome["measured"]
            lines.append(
                "| "
                f"{score['rank']} | "
                f"`{candidate_id}` | "
                f"{outcome['metrics']['trial_count']} | "
                f"{float(commanded['cmd_forward_m']):.3f} | "
                f"{float(measured['forward_body_m_mean']):.3f} | "
                f"{float(commanded['cmd_lateral_m']):.3f} | "
                f"{float(measured['left_body_m_mean']):.3f} | "
                f"{float(commanded['cmd_yaw_rad']):.3f} | "
                f"{float(measured['dyaw_rad_mean']):.3f} | "
                f"{float(score['score']):.6f} |"
            )
        fit = metrics.get("system_id_fit", {})
        if fit:
            lines.extend(
                [
                    "",
                    "## System-ID Fit",
                    "",
                    f"- Forward R2: `{fit['r2'].get('measured_forward_body_m', 0.0):.3f}`",
                    f"- Lateral R2: `{fit['r2'].get('measured_left_body_m', 0.0):.3f}`",
                    f"- Yaw R2: `{fit['r2'].get('measured_dyaw_rad', 0.0):.3f}`",
                    f"- Forward RMSE: `{fit['rmse'].get('measured_forward_body_m', 0.0):.4f} m`",
                    f"- Lateral RMSE: `{fit['rmse'].get('measured_left_body_m', 0.0):.4f} m`",
                    f"- Yaw RMSE: `{fit['rmse'].get('measured_dyaw_rad', 0.0):.4f} rad`",
                ]
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorldForgeError(
            "Go2 measured-outcome report requires candidate_outcomes with commanded, "
            "measured, metrics, and score fields."
        ) from exc
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- `outcome_kind=real_measured` because every command group was executed on hardware.",
            "- `score_kind=measured_metric` because ranking is post-hoc command-tracking error.",
            "- `hardware_executed=true`; `learned_model_used=false`.",
            "- This advances the measured-outcome axis for #45. It does not by itself satisfy "
            "the planner kill criterion.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_capture(capture_dir: Path) -> _Capture:
    if not capture_dir.is_dir():
        raise WorldForgeError(
            f"Go2 native capture directory not found: {_safe_artifact_path(capture_dir)}"
        )
    analysis_dir = capture_dir / "analysis"
    trials = _read_csv(analysis_dir / "trial_outcomes.csv")
    groups = _read_csv(analysis_dir / "command_group_stats.csv")
    if not trials:
        raise WorldForgeError(
            "Go2 native capture trial_outcomes.csv must contain at least one row."
        )
    if not groups:
        raise WorldForgeError(
            "Go2 native capture command_group_stats.csv must contain at least one row."
        )
    run = _read_json(capture_dir / "go2-native-system-id-run.json")
    _validate_stream_references(capture_dir, run)
    fit_path = analysis_dir / "system_id_fit.json"
    fit = _read_json(fit_path) if fit_path.is_file() else {}
    return _Capture(capture_dir=capture_dir, trials=trials, groups=groups, run=run, fit=fit)


@dataclass(frozen=True, slots=True)
class _Capture:
    capture_dir: Path
    trials: list[dict[str, str]]
    groups: list[dict[str, str]]
    run: JSONDict
    fit: JSONDict


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except FileNotFoundError as exc:
        raise WorldForgeError(
            f"Go2 native capture CSV not found: {_safe_artifact_path(path)}"
        ) from exc
    except (csv.Error, UnicodeDecodeError, OSError) as exc:
        raise WorldStateError(
            f"Go2 native capture CSV could not be read: {_safe_artifact_path(path)}"
        ) from exc


def _read_json(path: Path) -> JSONDict:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise WorldForgeError(
            f"Go2 native capture JSON not found: {_safe_artifact_path(path)}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise WorldStateError(
            f"Go2 native capture JSON is invalid: {_safe_artifact_path(path)}"
        ) from exc
    except OSError as exc:
        raise WorldStateError(
            f"Go2 native capture JSON could not be read: {_safe_artifact_path(path)}"
        ) from exc
    if not isinstance(payload, dict):
        raise WorldStateError(
            f"Go2 native capture JSON must be an object: {_safe_artifact_path(path)}"
        )
    return payload


def _validate_stream_references(capture_dir: Path, run: JSONDict) -> None:
    topic_counts = _safe_count_mapping(run.get("topic_counts"))
    required_streams = {
        "ROBOTODOM": capture_dir / "streams" / "topics" / "robotodom.jsonl",
        "ULIDAR_ARRAY": capture_dir / "streams" / "topics" / "ulidar_array.jsonl",
    }
    for topic, path in required_streams.items():
        if not path.is_file():
            raise WorldForgeError(f"Go2 native capture is missing {topic} stream reference.")
        if _int_mapping(topic_counts, topic) <= 0:
            raise WorldForgeError(f"Go2 native capture is missing {topic} messages.")


def _measured_command_candidates(capture: _Capture) -> list[JSONDict]:
    trials_by_command = _trials_by_command(capture.trials)
    candidates: list[JSONDict] = []
    for index, group in enumerate(capture.groups):
        cmd_forward = _float(group, "cmd_forward_m")
        cmd_lateral = _float(group, "cmd_lateral_m")
        cmd_yaw = _float(group, "cmd_yaw_rad")
        expected_trial_count = _int(group, "n")
        if expected_trial_count <= 0:
            raise WorldStateError(
                "Go2 measured-outcome command group trial count must be greater than 0."
            )
        matching_trials = trials_by_command.get(_command_key(cmd_forward, cmd_lateral, cmd_yaw), [])
        if expected_trial_count > 0 and not matching_trials:
            raise WorldStateError("Go2 measured-outcome command group has no matching trial rows.")
        if len(matching_trials) != expected_trial_count:
            raise WorldStateError(
                "Go2 measured-outcome command group trial count does not match trial_outcomes.csv."
            )
        measured_forward = _float(group, "measured_forward_body_m_mean")
        measured_left = _float(group, "measured_left_body_m_mean")
        measured_yaw = _float(group, "measured_dyaw_rad_mean")
        forward_error = abs(measured_forward - cmd_forward)
        lateral_error = abs(measured_left - cmd_lateral)
        yaw_error = abs(measured_yaw - cmd_yaw)
        planar_error = math.hypot(forward_error, lateral_error)
        tracking_score = planar_error + (_YAW_WEIGHT_M_PER_RAD * yaw_error)
        candidates.append(
            {
                "candidate_id": _candidate_id(index, cmd_forward, cmd_lateral, cmd_yaw),
                "label": f"command-group-{index}",
                "cmd_forward_m": cmd_forward,
                "cmd_lateral_m": cmd_lateral,
                "cmd_yaw_rad": cmd_yaw,
                "duration_s": _duration_from_trials(matching_trials),
                "trial_count": expected_trial_count,
                "trial_ids": [f"trial-{_int(row, 'trial_index'):04d}" for row in matching_trials],
                "odom_samples": sum(_int(row, "odom_samples") for row in matching_trials),
                "ulidar_messages": sum(_int(row, "ulidar_messages") for row in matching_trials),
                "ulidar_npz": sum(_int(row, "ulidar_npz") for row in matching_trials),
                "measured_forward_body_m_mean": measured_forward,
                "measured_forward_body_m_sd": _float(
                    group,
                    "measured_forward_body_m_sd",
                    default=0.0,
                ),
                "measured_forward_body_m_min": _float(group, "measured_forward_body_m_min"),
                "measured_forward_body_m_max": _float(group, "measured_forward_body_m_max"),
                "measured_left_body_m_mean": measured_left,
                "measured_left_body_m_sd": _float(group, "measured_left_body_m_sd", default=0.0),
                "measured_left_body_m_min": _float(group, "measured_left_body_m_min"),
                "measured_left_body_m_max": _float(group, "measured_left_body_m_max"),
                "measured_dyaw_rad_mean": measured_yaw,
                "measured_dyaw_rad_sd": _float(group, "measured_dyaw_rad_sd", default=0.0),
                "measured_dyaw_rad_min": _float(group, "measured_dyaw_rad_min"),
                "measured_dyaw_rad_max": _float(group, "measured_dyaw_rad_max"),
                "measured_planar_m_mean": _float(group, "measured_planar_m_mean"),
                "measured_planar_m_sd": _float(group, "measured_planar_m_sd", default=0.0),
                "tracking_score": tracking_score,
                "forward_error_m": forward_error,
                "lateral_error_m": lateral_error,
                "yaw_error_rad": yaw_error,
                "planar_tracking_error_m": planar_error,
            }
        )
    return candidates


def _candidate_action(candidate: JSONDict) -> JSONDict:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "label": str(candidate["label"]),
        "action": {
            "type": "go2_body_velocity_system_id",
            "params": {
                "x_mps": _velocity_from_integral(candidate["cmd_forward_m"], candidate),
                "y_mps": _velocity_from_integral(candidate["cmd_lateral_m"], candidate),
                "z_radps": _velocity_from_integral(candidate["cmd_yaw_rad"], candidate),
                "duration_s": candidate["duration_s"],
                "cmd_forward_m": candidate["cmd_forward_m"],
                "cmd_lateral_m": candidate["cmd_lateral_m"],
                "cmd_yaw_rad": candidate["cmd_yaw_rad"],
            },
            "units": {
                "x_mps": "m/s",
                "y_mps": "m/s",
                "z_radps": "rad/s",
                "duration_s": "s",
                "cmd_forward_m": "m",
                "cmd_lateral_m": "m",
                "cmd_yaw_rad": "rad",
            },
        },
        "measurement_context": {
            "trial_count": candidate["trial_count"],
            "outcome_source": "native_robotodom_and_ulidar_summary",
        },
    }


def _candidate_outcome(candidate: JSONDict) -> JSONDict:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "kind": "real_measured",
        "status": "executed_system_id_group",
        "outcome_source": "unitree_go2_native_odom_summary",
        "action_executed": True,
        "commanded": {
            "cmd_forward_m": candidate["cmd_forward_m"],
            "cmd_lateral_m": candidate["cmd_lateral_m"],
            "cmd_yaw_rad": candidate["cmd_yaw_rad"],
            "duration_s": candidate["duration_s"],
        },
        "measured": {
            "forward_body_m_mean": candidate["measured_forward_body_m_mean"],
            "forward_body_m_sd": candidate["measured_forward_body_m_sd"],
            "forward_body_m_min": candidate["measured_forward_body_m_min"],
            "forward_body_m_max": candidate["measured_forward_body_m_max"],
            "left_body_m_mean": candidate["measured_left_body_m_mean"],
            "left_body_m_sd": candidate["measured_left_body_m_sd"],
            "left_body_m_min": candidate["measured_left_body_m_min"],
            "left_body_m_max": candidate["measured_left_body_m_max"],
            "dyaw_rad_mean": candidate["measured_dyaw_rad_mean"],
            "dyaw_rad_sd": candidate["measured_dyaw_rad_sd"],
            "dyaw_rad_min": candidate["measured_dyaw_rad_min"],
            "dyaw_rad_max": candidate["measured_dyaw_rad_max"],
            "planar_m_mean": candidate["measured_planar_m_mean"],
            "planar_m_sd": candidate["measured_planar_m_sd"],
        },
        "metrics": {
            "tracking_score": candidate["tracking_score"],
            "planar_tracking_error_m": candidate["planar_tracking_error_m"],
            "forward_error_m": candidate["forward_error_m"],
            "lateral_error_m": candidate["lateral_error_m"],
            "yaw_error_rad": candidate["yaw_error_rad"],
            "yaw_weight_m_per_rad": _YAW_WEIGHT_M_PER_RAD,
            "trial_count": candidate["trial_count"],
            "odom_samples": candidate["odom_samples"],
            "ulidar_messages": candidate["ulidar_messages"],
            "ulidar_sidecar_count": candidate["ulidar_npz"],
            "trial_ids": list(candidate["trial_ids"]),
            "local_capture_refs": {
                "odom_timeseries": "streams/topics/robotodom.jsonl",
                "ulidar_stream": "streams/topics/ulidar_array.jsonl",
                "trial_outcomes": "analysis/trial_outcomes.csv",
                "lidar_sidecar_naming": "local-sidecar-files-redacted",
                "local_only": True,
            },
        },
    }


def _score_records(candidates: list[JSONDict], *, baseline_score: float) -> list[JSONDict]:
    best = min(float(candidate["tracking_score"]) for candidate in candidates)
    worst = max(float(candidate["tracking_score"]) for candidate in candidates)
    span = max(worst - best, 0.0)
    score_margin = _score_margin(candidates)
    records: list[JSONDict] = []
    for rank, candidate in enumerate(candidates, start=1):
        score = float(candidate["tracking_score"])
        desirability = 1.0 if span == 0.0 else (worst - score) / span
        regret = max(0.0, baseline_score - score)
        separability = 0.0 if span == 0.0 else max(0.0, score_margin) / span
        value_signal = _clamp01(
            (0.65 * desirability)
            + (0.2 * min(regret / max(span, 1e-9), 1.0))
            + (0.15 * separability)
        )
        records.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "rank": rank,
                "score": score,
                "lower_is_better": True,
                "components": {
                    "planar_tracking_error_m": candidate["planar_tracking_error_m"],
                    "forward_error_m": candidate["forward_error_m"],
                    "lateral_error_m": candidate["lateral_error_m"],
                    "yaw_error_rad": candidate["yaw_error_rad"],
                    "yaw_weight_m_per_rad": _YAW_WEIGHT_M_PER_RAD,
                },
                "normalized": {
                    "value_signal": value_signal,
                    "desirability": _clamp01(desirability),
                    "regret_vs_baseline": regret,
                    "separability": _clamp01(separability),
                    "calibration_target": "measured_command_tracking_error",
                },
            }
        )
    return records


def _outcome(capture: _Capture, selected: JSONDict, score_span: float) -> JSONDict:
    topic_counts = capture.run.get("topic_counts", {})
    if not isinstance(topic_counts, dict):
        topic_counts = {}
    return {
        "kind": "real_measured",
        "status": "executed_system_id_summary",
        "metrics": {
            "outcome_source": "unitree_go2_native_system_id_run",
            "completed_movement_trials": len(capture.trials),
            "command_group_count": len(capture.groups),
            "native_odom_messages": _int_mapping(topic_counts, "ROBOTODOM"),
            "ulidar_array_messages": _int_mapping(topic_counts, "ULIDAR_ARRAY"),
            "low_state_messages": _int_mapping(topic_counts, "LOW_STATE"),
            "saved_lidar_sidecars": _int_mapping(capture.run, "saved_lidar_count"),
            "saved_rgb_frames": _int_mapping(capture.run, "saved_frame_count"),
            "script_recorded_errors": len(capture.run.get("errors") or []),
            "selected_tracking_score": selected["tracking_score"],
            "score_span": score_span,
            "system_id_fit": _sanitized_fit(capture.fit),
            "raw_capture_policy": "local_private_not_committed",
        },
    }


def _observation(capture: _Capture) -> JSONDict:
    topic_counts = capture.run.get("topic_counts", {})
    if not isinstance(topic_counts, dict):
        topic_counts = {}
    robot = capture.run.get("robot", {})
    if not isinstance(robot, dict):
        robot = {}
    capture_design = capture.run.get("capture_design", {})
    if not isinstance(capture_design, dict):
        capture_design = {}
    scene = capture.run.get("scene", {})
    if not isinstance(scene, dict):
        scene = {}
    return {
        "ref": {
            "source": "host_owned_go2_native_capture",
            "run_artifact": "go2-native-system-id-run.json",
            "trial_outcomes": "analysis/trial_outcomes.csv",
            "command_group_stats": "analysis/command_group_stats.csv",
            "odom_timeseries": "streams/topics/robotodom.jsonl",
            "ulidar_stream": "streams/topics/ulidar_array.jsonl",
            "local_only": True,
        },
        "summary": {
            "robot_model": str(robot.get("model", "Unitree Go2 Air")),
            "firmware_version": str(robot.get("firmware_version", "redacted_or_unknown")),
            "hardware_version": str(robot.get("hardware_version", "redacted_or_unknown")),
            "ip_redacted": _strict_bool(robot, "ip_redacted", default=True),
            "serial_redacted": _strict_bool(robot, "serial_redacted", default=True),
            "redaction_state": _redaction_state(capture),
            "surface": str(scene.get("surface", "indoor floor")),
            "native_rate_streams": _strict_bool(
                capture_design, "native_rate_streams", default=True
            ),
            "rgb_frames_recorded": _strict_bool(
                capture_design, "rgb_frames_recorded", default=False
            ),
            "topic_counts": _safe_count_mapping(topic_counts),
        },
    }


def _redaction_state(capture: _Capture) -> JSONDict:
    robot = capture.run.get("robot", {})
    if not isinstance(robot, dict):
        robot = {}
    return {
        "media": {
            "rgb_frames_embedded": False,
            "raw_media_committed": False,
            "saved_rgb_frames": _int_mapping(capture.run, "saved_frame_count"),
        },
        "paths": {
            "host_paths_embedded": False,
            "local_capture_refs_only": True,
            "artifact_paths_portable": True,
        },
        "network": {
            "ip_addresses_embedded": False,
            "ip_redacted": _strict_bool(robot, "ip_redacted", default=True),
        },
        "robot_identity": {
            "serials_embedded": False,
            "serial_redacted": _strict_bool(robot, "serial_redacted", default=True),
        },
        "access_material": {
            "embedded": False,
            "sensitive_fields_dropped": True,
        },
        "raw_sensor_sidecars": {
            "lidar_sidecars_embedded": False,
            "raw_lidar_sidecars_committed": False,
            "saved_lidar_sidecars": _int_mapping(capture.run, "saved_lidar_count"),
        },
        "capture_labels": {
            "source_labels_embedded": False,
            "synthetic_labels_only": True,
        },
    }


def _goal() -> JSONDict:
    return {
        "type": "system_identification",
        "description": (
            "Measure how scripted Go2 body-velocity commands map to native odometry outcomes."
        ),
        "sub_goals": [
            {
                "id": "record_native_odom",
                "description": "Capture native odometry timeseries for every command group.",
                "required": True,
                "weight": 0.35,
            },
            {
                "id": "record_lidar_context",
                "description": "Pair command outcomes with LiDAR stream and sidecar references.",
                "required": True,
                "weight": 0.25,
            },
            {
                "id": "estimate_command_tracking",
                "description": "Quantify commanded-vs-measured body-frame tracking error.",
                "required": True,
                "weight": 0.4,
            },
        ],
        "success_criteria": {
            "metric": "measured_command_tracking_summary",
            "partial_credit": True,
            "protocol": "operator-scripted native-rate Go2 system identification",
        },
    }


def _interop_block() -> JSONDict:
    return {
        "trajectory_ref": {
            "standard": "host-owned native Unitree Go2 capture summary",
            "odom_timeseries": "streams/topics/robotodom.jsonl",
            "ulidar_stream": "streams/topics/ulidar_array.jsonl",
            "local_only": True,
        },
        "prov": {
            "entities": ["scripted_commands", "native_odometry", "ulidar_stream", "scores"],
            "activities": ["execute_system_id_commands", "summarize_measured_outcomes"],
            "agents": ["Unitree Go2 Air", "operator", "WorldForge"],
            "selected_action_was_derived_from": ["native_odometry", "scripted_commands"],
        },
        "otel_span": {
            "name": "worldforge.decision_trace.measured_outcome_summary",
            "attributes": {
                "worldforge.schema_version": DECISION_TRACE_SCHEMA_VERSION,
                "worldforge.outcome_kind": "real_measured",
                "worldforge.score_kind": "measured_metric",
            },
        },
    }


def _capture_digest(capture: _Capture) -> str:
    payload = {
        "artifact_kind": capture.run.get("artifact_kind"),
        "capture_design": _safe_mapping(capture.run.get("capture_design", {})),
        "topic_counts": _safe_mapping(capture.run.get("topic_counts", {})),
        "saved_lidar_count": capture.run.get("saved_lidar_count"),
        "saved_frame_count": capture.run.get("saved_frame_count"),
        "trial_outcomes": capture.trials,
        "command_group_stats": capture.groups,
        "system_id_fit": capture.fit,
    }
    return decision_trace_digest(_round_json_floats(payload))


def _trials_by_command(
    trials: Iterable[dict[str, str]],
) -> dict[tuple[float, float, float], list[dict[str, str]]]:
    grouped: dict[tuple[float, float, float], list[dict[str, str]]] = {}
    for row in trials:
        key = _command_key(
            _float(row, "cmd_forward_m"),
            _float(row, "cmd_lateral_m"),
            _float(row, "cmd_yaw_rad"),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def _command_key(forward_m: float, lateral_m: float, yaw_rad: float) -> tuple[float, float, float]:
    return (round(forward_m, 6), round(lateral_m, 6), round(yaw_rad, 6))


def _candidate_id(index: int, forward_m: float, lateral_m: float, yaw_rad: float) -> str:
    if abs(forward_m) >= abs(lateral_m) and abs(forward_m) >= abs(yaw_rad):
        direction = "forward" if forward_m >= 0 else "backward"
        magnitude = abs(forward_m)
    elif abs(lateral_m) >= abs(yaw_rad):
        direction = "left" if lateral_m >= 0 else "right"
        magnitude = abs(lateral_m)
    else:
        direction = "yaw-left" if yaw_rad >= 0 else "yaw-right"
        magnitude = abs(yaw_rad)
    return f"cmd-{index:02d}-{direction}-{_slug_number(magnitude)}"


def _slug_number(value: float) -> str:
    return f"{value:.3f}".replace("-", "neg").replace(".", "p")


def _duration_from_trials(trials: list[dict[str, str]]) -> float:
    if not trials:
        return 0.0
    durations = [_float(row, "duration_s") for row in trials]
    return sum(durations) / len(durations)


def _velocity_from_integral(command_integral: object, candidate: JSONDict) -> float:
    duration = float(candidate["duration_s"])
    if duration <= 0.0:
        return 0.0
    return float(command_integral) / duration


def _score_margin(ranked: list[JSONDict]) -> float:
    if len(ranked) < 2:
        return 0.0
    return round(float(ranked[1]["tracking_score"]) - float(ranked[0]["tracking_score"]), 6)


def _sanitized_fit(fit: JSONDict) -> JSONDict:
    if not fit:
        return {}
    return {
        "model": str(fit.get("model", "measured_body_outcome affine fit")),
        "features": [str(item) for item in fit.get("features", [])],
        "targets": [str(item) for item in fit.get("targets", [])],
        "r2": _safe_numeric_mapping(fit.get("r2", {})),
        "rmse": _safe_numeric_mapping(fit.get("rmse", {})),
    }


def _safe_mapping(value: object) -> JSONDict:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _json_safe_scalar(item) for key, item in value.items()}


def _safe_numeric_mapping(value: object) -> JSONDict:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if _is_safe_public_key(str(key)) and _is_finite_number(item)
    }


def _safe_count_mapping(value: object) -> JSONDict:
    if not isinstance(value, Mapping):
        return {}
    counts: JSONDict = {}
    for key in value:
        key_text = str(key)
        if _is_safe_public_key(key_text):
            counts[key_text] = _int_mapping(value, key_text)
    return counts


def _json_safe_scalar(value: object) -> object:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return _safe_mapping(value)
    if isinstance(value, list):
        return [_json_safe_scalar(item) for item in value]
    return str(value)


def _strict_bool(mapping: Mapping[str, object], key: str, *, default: bool) -> bool:
    if key not in mapping:
        return default
    value = mapping[key]
    if isinstance(value, bool):
        return value
    raise WorldStateError(f"Go2 native capture field {key!r} must be a boolean.")


def _float(row: Mapping[str, object], key: str, *, default: float | None = None) -> float:
    raw = row.get(key)
    if isinstance(raw, str):
        raw = raw.strip()
    if raw in (None, ""):
        if default is not None:
            return default
        raise WorldStateError(f"Go2 measured-outcome row is missing numeric field {key!r}.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise WorldStateError(f"Go2 measured-outcome field {key!r} must be numeric.") from exc
    if not math.isfinite(value):
        raise WorldStateError(f"Go2 measured-outcome field {key!r} must be finite.")
    return value


def _int(row: Mapping[str, object], key: str, *, default: int = 0) -> int:
    raw = row.get(key)
    if raw in (None, ""):
        return default
    if isinstance(raw, bool):
        raise WorldStateError(f"Go2 measured-outcome field {key!r} must be an integer.")
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float):
        if not math.isfinite(raw) or not raw.is_integer():
            raise WorldStateError(f"Go2 measured-outcome field {key!r} must be an integer.")
        value = int(raw)
    else:
        raw_text = str(raw).strip()
        if not re.fullmatch(r"[0-9]+", raw_text):
            raise WorldStateError(f"Go2 measured-outcome field {key!r} must be an integer.")
        value = int(raw_text)
    if value < 0:
        raise WorldStateError(f"Go2 measured-outcome field {key!r} must be non-negative.")
    return value


def _int_mapping(row: Mapping[str, object], key: str, *, default: int = 0) -> int:
    return _int(row, key, default=default)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_safe_public_key(key: str) -> bool:
    return bool(_SAFE_PUBLIC_KEY_PATTERN.fullmatch(key)) and not _SENSITIVE_FIELD_PATTERN.search(
        key
    )


def _round_json_floats(value: object) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("JSON float values must be finite.")
        return round(value, 6)
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return [_round_json_floats(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _round_json_floats(item) for key, item in value.items()}
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}.")


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _safe_artifact_path(path: Path) -> str:
    return _SAFE_ARTIFACT_LABELS.get(path.name, "<host-local-path>")


def run_go2_measured_outcomes_workflow(capture_dir: Path, output_dir: Path) -> JSONDict:
    """Workflow wrapper used by the checkout example script."""

    result = run_go2_measured_outcomes(capture_dir=capture_dir, output_dir=output_dir)
    return {
        "trace_id": result.trace["trace_id"],
        "selected_action_id": result.trace["selected_action"]["candidate_id"],
        "selected_tracking_score": result.trace["selected_action"]["score"],
        "candidate_count": len(result.trace["candidate_actions"]),
        "completed_movement_trials": result.trace["outcome"]["metrics"][
            "completed_movement_trials"
        ],
        "decision_trace_path": result.decision_trace_path.name,
        "summary_path": result.summary_path.name,
        "report_path": result.report_path.name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        required=True,
        help=(
            "Host-owned Go2 native-rate capture directory. Raw contents stay local; only "
            "sanitized summaries are emitted."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for sanitized DecisionTrace artifacts and report.",
    )
    args = parser.parse_args(argv)
    summary = run_go2_measured_outcomes_workflow(args.capture_dir, args.out)
    print(f"trace_id={summary['trace_id']}")
    print(f"selected_action={summary['selected_action_id']}")
    print(f"selected_tracking_score={summary['selected_tracking_score']:.6f}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"completed_movement_trials={summary['completed_movement_trials']}")
    print(f"trace={summary['decision_trace_path']}")
    print(f"report={summary['report_path']}")
    return 0


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "Go2MeasuredOutcomeResult",
    "build_go2_measured_outcome_trace",
    "render_go2_measured_outcomes_report",
    "run_go2_measured_outcomes",
    "run_go2_measured_outcomes_workflow",
]
