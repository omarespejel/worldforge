from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Callable, Iterable, Mapping
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
from worldforge.demos.go2_measured_outcomes import build_go2_measured_outcome_trace
from worldforge.models import JSONDict, WorldForgeError, WorldStateError

DEFAULT_OUTPUT_DIR = Path(".worldforge/go2-controlbench")
RUN_ID = "go2-controlbench-native-system-id"
CODE_REF = "feat/go2-controlbench-benchmark"
CLI_DESCRIPTION = (
    "Run Go2 ControlBench over a host-owned native-rate system-ID capture and emit "
    "public-safe benchmark artifacts."
)

_YAW_WEIGHT_M_PER_RAD = 0.25
_MOTION_THRESHOLD = 0.02
_RIDGE = 1e-9
_LINEAR_BASELINE = "affine_linear_system_id"
_DEADBAND_BASELINE = "deadband_affine_system_id"
_COMMAND_BASELINE = "command_integral"
_BASELINE_ORDER = (_COMMAND_BASELINE, _LINEAR_BASELINE, _DEADBAND_BASELINE)
_SAFE_ARTIFACT_LABELS = {
    "controlbench-summary.json": "<output-dir>/controlbench-summary.json",
    "controlbench-report.md": "<output-dir>/controlbench-report.md",
    "decision-trace-go2-controlbench.json": "<output-dir>/decision-trace-go2-controlbench.json",
    "trial_outcomes.csv": "<analysis-dir>/trial_outcomes.csv",
}


@dataclass(frozen=True, slots=True)
class Go2ControlBenchResult:
    """Artifacts emitted by one Go2 ControlBench run."""

    summary: JSONDict
    report_markdown: str
    decision_trace: JSONDict
    summary_path: Path
    report_path: Path
    decision_trace_path: Path


@dataclass(frozen=True, slots=True)
class _Trial:
    trial_id: str
    command_cell_id: str
    command: tuple[float, float, float]
    measured: tuple[float, float, float]
    measured_planar_m: float
    odom_samples: int
    ulidar_messages: int
    ulidar_sidecars: int


@dataclass(frozen=True, slots=True)
class _Group:
    command_cell_id: str
    command: tuple[float, float, float]
    measured: tuple[float, float, float]
    trial_count: int


def run_go2_controlbench(
    capture_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Go2ControlBenchResult:
    """Run checkout-safe Go2 ControlBench baselines over a private capture directory."""

    measured_trace = build_go2_measured_outcome_trace(capture_dir)
    trials = _load_trials(capture_dir)
    groups = _groups_from_trials(trials)
    predictions = _baseline_predictions(trials)
    summary = _build_summary(
        measured_trace=measured_trace,
        trials=trials,
        groups=groups,
        predictions=predictions,
    )
    decision_trace = _build_controlbench_decision_trace(
        measured_trace=measured_trace,
        summary=summary,
        trials=trials,
        groups=groups,
        predictions=predictions,
    )
    report = render_go2_controlbench_report(summary)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorldStateError(
            f"Go2 ControlBench output_dir.mkdir failed: {_safe_artifact_path(output_dir)}"
        ) from exc

    try:
        summary_path = write_json_artifact(output_dir / "controlbench-summary.json", summary)
        decision_trace_path = write_json_artifact(
            output_dir / "decision-trace-go2-controlbench.json",
            decision_trace,
        )
        report_path = output_dir / "controlbench-report.md"
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        raise WorldStateError(
            f"Go2 ControlBench artifact write failed under {_safe_artifact_path(output_dir)}."
        ) from exc

    return Go2ControlBenchResult(
        summary=summary,
        report_markdown=report,
        decision_trace=decision_trace,
        summary_path=summary_path,
        report_path=report_path,
        decision_trace_path=decision_trace_path,
    )


def run_go2_controlbench_workflow(capture_dir: Path, output_dir: Path) -> JSONDict:
    """CLI-friendly wrapper that returns portable output paths."""

    result = run_go2_controlbench(capture_dir=capture_dir, output_dir=output_dir)
    best = result.summary["best_baseline"]
    return {
        "run_id": result.summary["run_id"],
        "task_a_best_baseline": best["task_a_command_outcome_prediction"],
        "task_b_best_baseline": best["task_b_inverse_control"],
        "trial_count": result.summary["dataset"]["trial_count"],
        "command_cell_count": result.summary["dataset"]["command_cell_count"],
        "decision_trace_path": result.decision_trace_path.name,
        "summary_path": result.summary_path.name,
        "report_path": result.report_path.name,
    }


def render_go2_controlbench_report(summary: JSONDict) -> str:
    """Render a compact research note plus benchmark report for Go2 ControlBench."""

    try:
        dataset = summary["dataset"]
        task_a = summary["tasks"]["command_outcome_prediction"]
        task_b = summary["tasks"]["inverse_control"]
        best = summary["best_baseline"]
    except (KeyError, TypeError) as exc:
        raise WorldForgeError(
            "Go2 ControlBench report requires dataset, tasks, and best_baseline."
        ) from exc

    lines = [
        "# Go2 ControlBench",
        "",
        "Go2 ControlBench turns a stock Unitree Go2 Air system-identification capture into a "
        "small real-hardware benchmark: command candidates are known, outcomes are measured, "
        "and baselines are judged on whether they predict and rank those outcomes.",
        "",
        "Boundary: this is not an autonomous planner win. The robot was operator/script driven. "
        "WorldForge evaluates command-to-outcome evidence so future planners, WMCP world models, "
        "and DecisionTrace scorers can be compared against real measurements.",
        "",
        "## Dataset",
        "",
        f"- Trial count: `{dataset['trial_count']}`",
        f"- Command cells: `{dataset['command_cell_count']}`",
        f"- Native odometry messages: `{dataset['native_odom_messages']}`",
        f"- ULIDAR_ARRAY messages: `{dataset['ulidar_array_messages']}`",
        f"- Saved LiDAR sidecars: `{dataset['saved_lidar_sidecars']}`",
        f"- RGB frames embedded: `{dataset['rgb_frames_embedded']}`",
        "",
        "## Task A - Command Outcome Prediction",
        "",
        "Predict measured body-frame `(forward_m, left_m, dyaw_rad)` from the commanded "
        "integral. Metrics are leave-one-command-cell-out.",
        "",
        "| Baseline | RMSE fwd m | RMSE left m | RMSE yaw rad | Mean RMSE | Deadband F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in _BASELINE_ORDER:
        metrics = task_a["baselines"][name]
        rmse = metrics["per_axis_rmse"]
        lines.append(
            "| "
            f"`{name}` | "
            f"{rmse['forward_m']:.6f} | "
            f"{rmse['left_m']:.6f} | "
            f"{rmse['dyaw_rad']:.6f} | "
            f"{metrics['mean_rmse']:.6f} | "
            f"{metrics['deadband_classification_f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Task B - Inverse Control Ranking",
            "",
            "For each measured target outcome, rank all command cells by predicted distance to "
            "that target. This is the DecisionTrace-relevant task: a planner needs the right "
            "relative ranking, not just a low average reconstruction error.",
            "",
            "| Baseline | Top-1 | Mean regret | Mean Spearman |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in _BASELINE_ORDER:
        metrics = task_b["baselines"][name]
        lines.append(
            "| "
            f"`{name}` | "
            f"{metrics['top1_accuracy']:.3f} | "
            f"{metrics['mean_inverse_control_regret']:.6f} | "
            f"{metrics['mean_rank_correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Current Read",
            "",
            f"- Best Task A baseline: `{best['task_a_command_outcome_prediction']}`.",
            f"- Best Task B baseline: `{best['task_b_inverse_control']}`.",
            "- Treat deadband and command asymmetry as hypotheses to validate with external "
            "ground truth, not as settled science.",
            "- WMCP maps to the future `rollout/evaluate` interface for predicted outcomes; "
            "DecisionTrace records candidates, predicted scores, measured outcomes, and regret.",
            "",
            "## Claim Boundary",
            "",
            "- `outcome_kind=real_measured` for command outcomes because the command cells were "
            "executed on hardware.",
            "- Baseline scores are deterministic analytic/system-ID estimates, not learned "
            "world-model scores.",
            "- Raw telemetry, LiDAR sidecars, RGB frames, IPs, serials, host paths, and access "
            "material stay outside the public artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_summary(
    *,
    measured_trace: JSONDict,
    trials: list[_Trial],
    groups: list[_Group],
    predictions: dict[str, dict[str, tuple[float, float, float]]],
) -> JSONDict:
    task_a = {name: _task_a_metrics(trials, predictions[name]) for name in _BASELINE_ORDER}
    task_b = {name: _task_b_metrics(groups, trials, predictions[name]) for name in _BASELINE_ORDER}
    best_task_a = min(
        _BASELINE_ORDER,
        key=lambda name: (task_a[name]["mean_rmse"], name),
    )
    best_task_b = min(
        _BASELINE_ORDER,
        key=lambda name: (task_b[name]["mean_inverse_control_regret"], name),
    )
    metrics = measured_trace["outcome"]["metrics"]
    summary: JSONDict = {
        "schema_version": 1,
        "artifact_kind": "worldforge.go2_controlbench_summary",
        "run_id": RUN_ID,
        "source_trace_id": measured_trace["trace_id"],
        "dataset": {
            "trial_count": len(trials),
            "command_cell_count": len(groups),
            "native_odom_messages": metrics["native_odom_messages"],
            "ulidar_array_messages": metrics["ulidar_array_messages"],
            "saved_lidar_sidecars": metrics["saved_lidar_sidecars"],
            "rgb_frames_embedded": False,
            "raw_capture_policy": "local_private_not_committed",
            "external_ground_truth": "pending",
        },
        "tasks": {
            "command_outcome_prediction": {
                "id": "task_a_command_outcome_prediction",
                "split": "leave_one_command_cell_out",
                "baselines": task_a,
            },
            "inverse_control": {
                "id": "task_b_choose_command_for_target_motion",
                "split": "leave_one_command_cell_out_predictions",
                "baselines": task_b,
            },
            "odom_vs_external_ground_truth": {
                "id": "task_c_odom_vs_external_ground_truth",
                "status": "pending_external_gt_capture",
            },
        },
        "best_baseline": {
            "task_a_command_outcome_prediction": best_task_a,
            "task_b_inverse_control": best_task_b,
        },
        "claim_boundary": {
            "hardware_executed": True,
            "operator_scripted_capture": True,
            "autonomous_planner_claim": False,
            "learned_model_claim": False,
            "privacy": {
                "raw_rgb_published": False,
                "raw_lidar_sidecars_embedded": False,
                "host_paths_embedded": False,
                "robot_identity_embedded": False,
            },
        },
        "wmcp_alignment": {
            "wmcp_role": "future runtime rollout/evaluate interface for predicted outcomes",
            "decisiontrace_role": "durable evidence of candidates, scores, outcomes, and regret",
        },
    }
    return _round_json_floats(summary)


def _build_controlbench_decision_trace(
    *,
    measured_trace: JSONDict,
    summary: JSONDict,
    trials: list[_Trial],
    groups: list[_Group],
    predictions: dict[str, dict[str, tuple[float, float, float]]],
) -> JSONDict:
    baseline_name = str(summary["best_baseline"]["task_b_inverse_control"])
    group_predictions_by_baseline = {
        name: _group_predictions_from_trials(trials, baseline_predictions)
        for name, baseline_predictions in predictions.items()
    }
    target = _representative_target(groups, group_predictions_by_baseline[baseline_name])
    scores = _inverse_control_scores(
        groups,
        group_predictions_by_baseline[baseline_name],
        target.measured,
    )
    ranked_scores = sorted(scores, key=lambda row: (row["score"], row["candidate_id"]))
    selected = ranked_scores[0]
    command_integral_scores = _inverse_control_scores(
        groups,
        group_predictions_by_baseline[_COMMAND_BASELINE],
        target.measured,
    )
    command_integral_selected = min(
        command_integral_scores,
        key=lambda row: (row["score"], row["candidate_id"]),
    )
    measured_by_id = {group.command_cell_id: group.measured for group in groups}
    true_errors = {
        group.command_cell_id: _outcome_distance(group.measured, target.measured)
        for group in groups
    }
    true_best_id = min(
        true_errors,
        key=lambda candidate_id: (true_errors[candidate_id], candidate_id),
    )
    selected_true_error = true_errors[str(selected["candidate_id"])]
    true_best_error = true_errors[true_best_id]
    baseline_score = next(
        row["score"]
        for row in ranked_scores
        if row["candidate_id"] == command_integral_selected["candidate_id"]
    )
    candidate_actions = [_candidate_action(group) for group in groups]
    score_records = [
        _score_record(
            row,
            rank=index + 1,
            predicted=group_predictions_by_baseline[baseline_name][str(row["candidate_id"])],
        )
        for index, row in enumerate(ranked_scores)
    ]
    trace: JSONDict = {
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "artifact_kind": DECISION_TRACE_ARTIFACT_KIND,
        "trace_id": "go2-controlbench-inverse-control",
        "run_id": RUN_ID,
        "step_index": 0,
        "prev_trace_id": measured_trace["trace_id"],
        "embodiment": dict(measured_trace["embodiment"]),
        "host_runtime": {
            "name": "Go2 ControlBench",
            "mode": "offline_benchmark_over_real_measured_capture",
            "version": CODE_REF,
        },
        "task": {
            "task_id": "go2-controlbench-inverse-control",
            "description": (
                "Choose the command cell predicted to best reach a measured target motion."
            ),
        },
        "observation": {
            "source_trace_id": measured_trace["trace_id"],
            "benchmark_summary": {
                "trial_count": summary["dataset"]["trial_count"],
                "command_cell_count": summary["dataset"]["command_cell_count"],
                "external_ground_truth": summary["dataset"]["external_ground_truth"],
            },
        },
        "goal": {
            "type": "target_body_motion",
            "description": (
                "Select the command cell whose predicted outcome is closest to target motion."
            ),
            "target": {
                "forward_m": target.measured[0],
                "left_m": target.measured[1],
                "dyaw_rad": target.measured[2],
                "source": "held_out_measured_command_cell",
            },
            "sub_goals": [
                {
                    "id": "rank_candidates",
                    "description": "Rank command cells by predicted distance to target motion.",
                    "required": True,
                    "weight": 0.55,
                },
                {
                    "id": "measure_regret",
                    "description": "Compare selected command against real measured outcomes.",
                    "required": True,
                    "weight": 0.45,
                },
            ],
            "success_criteria": {
                "metric": "inverse_control_regret",
                "partial_credit": True,
                "protocol": "ControlBench Task B",
            },
        },
        "candidate_actions": candidate_actions,
        "scores": score_records,
        "selected_action": {
            "candidate_id": str(selected["candidate_id"]),
            "score": selected["score"],
            "score_margin": _score_margin(ranked_scores),
            "why_selected": (
                f"Lowest predicted distance to target motion under `{baseline_name}`. "
                "Measured outcomes are used only for benchmark regret."
            ),
        },
        "counterfactuals": [
            {
                "candidate_id": str(row["candidate_id"]),
                "score": row["score"],
                "delta_vs_selected": round(float(row["score"] - selected["score"]), 6),
                "why_rejected": "Higher predicted distance to the target motion.",
            }
            for row in ranked_scores
            if row["candidate_id"] != selected["candidate_id"]
        ],
        "candidate_outcomes": [
            _candidate_outcome(group, measured_by_id=measured_by_id, target=target.measured)
            for group in groups
        ],
        "baseline": {
            "candidate_id": str(command_integral_selected["candidate_id"]),
            "score": baseline_score,
            "regret_vs_selected": round(float(baseline_score - selected["score"]), 6),
            "policy": "command_integral_inverse_control",
        },
        "outcome": {
            "kind": "real_measured",
            "status": "benchmark_regret_computed",
            "metrics": {
                "outcome_source": "go2_controlbench_real_measured_command_cells",
                "selected_true_error": selected_true_error,
                "true_best_error": true_best_error,
                "inverse_control_regret": max(0.0, selected_true_error - true_best_error),
                "true_best_candidate_id": true_best_id,
                "target_command_cell_id": target.command_cell_id,
                "baseline_name": baseline_name,
            },
        },
        "planner_diagnostics": {
            "planner": "controlbench_inverse_control_ranker",
            "score_provider": baseline_name,
            "candidate_count": len(groups),
            "operator_scripted_capture": True,
            "not_autonomous_decision": True,
        },
        "reproducibility": {
            "provider_version": "Go2 ControlBench deterministic baselines",
            "checkpoint_hash": None,
            "model_card_ref": None,
            "input_digest": f"sha256:{decision_trace_digest(summary)}",
            "seed": None,
            "code_ref": CODE_REF,
        },
        "claim_boundary": {
            "score_kind": "hand_cost",
            "outcome_kind": "real_measured",
            "hardware_executed": True,
            "learned_model_used": False,
            "safety_controller": "offline_benchmark_only",
            "limitations": [
                "The selected action is a benchmark ranking, not a live robot command.",
                "Baseline scores are deterministic command-to-outcome estimates.",
                "Candidate outcomes are real measured command-cell summaries from an "
                "operator-scripted capture.",
                "External ground truth is pending; current outcomes use native Go2 odometry.",
            ],
        },
        "interop": {
            "wmcp": {
                "role": "future rollout/evaluate provider for predicted outcomes",
                "methods": ["rollout", "evaluate"],
                "durable_handles_embedded": False,
            },
            "decision_trace": {
                "role": (
                    "evidence artifact for candidates, predicted scores, measured outcomes, "
                    "and regret"
                ),
            },
        },
    }
    return validate_decision_trace(_round_json_floats(trace), name="Go2 ControlBench DecisionTrace")


def _load_trials(capture_dir: Path) -> list[_Trial]:
    rows = _read_csv(capture_dir / "analysis" / "trial_outcomes.csv")
    if not rows:
        raise WorldForgeError("Go2 ControlBench trial_outcomes.csv must contain at least one row.")
    trials: list[_Trial] = []
    for index, row in enumerate(rows):
        command = (
            _float(row, "cmd_forward_m"),
            _float(row, "cmd_lateral_m"),
            _float(row, "cmd_yaw_rad"),
        )
        measured = (
            _float(row, "measured_forward_body_m"),
            _float(row, "measured_left_body_m"),
            _float(row, "measured_dyaw_rad"),
        )
        trial_index = _int(row, "trial_index", default=index)
        trials.append(
            _Trial(
                trial_id=f"trial-{trial_index:04d}",
                command_cell_id=_command_cell_id(command),
                command=command,
                measured=measured,
                measured_planar_m=_float(row, "measured_planar_m"),
                odom_samples=_int(row, "odom_samples"),
                ulidar_messages=_int(row, "ulidar_messages"),
                ulidar_sidecars=_int(row, "ulidar_npz"),
            )
        )
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise WorldStateError("Go2 ControlBench trial ids must be unique after sanitization.")
    return trials


def _groups_from_trials(trials: list[_Trial]) -> list[_Group]:
    grouped: dict[str, list[_Trial]] = {}
    for trial in trials:
        grouped.setdefault(trial.command_cell_id, []).append(trial)
    groups: list[_Group] = []
    for command_cell_id, members in sorted(grouped.items()):
        command = members[0].command
        if any(member.command != command for member in members):
            raise WorldStateError("Go2 ControlBench command cell contains mixed commands.")
        measured = tuple(
            sum(member.measured[axis] for member in members) / len(members) for axis in range(3)
        )
        groups.append(
            _Group(
                command_cell_id=command_cell_id,
                command=command,
                measured=measured,  # type: ignore[arg-type]
                trial_count=len(members),
            )
        )
    if len(groups) < 2:
        raise WorldForgeError("Go2 ControlBench requires at least two command cells.")
    return groups


def _baseline_predictions(trials: list[_Trial]) -> dict[str, dict[str, tuple[float, float, float]]]:
    return {
        _COMMAND_BASELINE: {trial.trial_id: trial.command for trial in trials},
        _LINEAR_BASELINE: _leave_one_group_out_predictions(trials, _linear_features),
        _DEADBAND_BASELINE: _deadband_predictions(trials),
    }


def _leave_one_group_out_predictions(
    trials: list[_Trial],
    feature_fn: Callable[[tuple[float, float, float]], list[float]],
) -> dict[str, tuple[float, float, float]]:
    predictions: dict[str, tuple[float, float, float]] = {}
    groups = sorted({trial.command_cell_id for trial in trials})
    for group in groups:
        train = [trial for trial in trials if trial.command_cell_id != group]
        test = [trial for trial in trials if trial.command_cell_id == group]
        model = _fit_model(
            [feature_fn(trial.command) for trial in train],
            [trial.measured for trial in train],
        )
        for trial in test:
            predictions[trial.trial_id] = _predict(model, feature_fn(trial.command))
    return predictions


def _deadband_predictions(trials: list[_Trial]) -> dict[str, tuple[float, float, float]]:
    predictions: dict[str, tuple[float, float, float]] = {}
    groups = sorted({trial.command_cell_id for trial in trials})
    for group in groups:
        train = [trial for trial in trials if trial.command_cell_id != group]
        test = [trial for trial in trials if trial.command_cell_id == group]
        thresholds = _select_deadband_thresholds(train)

        def feature_fn(
            command: tuple[float, float, float],
            thresholds: tuple[float, float, float] = thresholds,
        ) -> list[float]:
            return _deadband_features(command, thresholds)

        model = _fit_model(
            [feature_fn(trial.command) for trial in train],
            [trial.measured for trial in train],
        )
        for trial in test:
            predictions[trial.trial_id] = _predict(model, feature_fn(trial.command))
    return predictions


def _select_deadband_thresholds(trials: list[_Trial]) -> tuple[float, float, float]:
    if len({trial.command_cell_id for trial in trials}) < 2:
        return (0.0, 0.0, 0.0)
    x_thresholds = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25)
    y_thresholds = (0.0, 0.05, 0.1, 0.12, 0.14)
    yaw_thresholds = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    best_thresholds = (0.0, 0.0, 0.0)
    best_error: float | None = None
    for x_threshold in x_thresholds:
        for y_threshold in y_thresholds:
            for yaw_threshold in yaw_thresholds:
                thresholds = (x_threshold, y_threshold, yaw_threshold)

                def feature_fn(
                    command: tuple[float, float, float],
                    thresholds: tuple[float, float, float] = thresholds,
                ) -> list[float]:
                    return _deadband_features(command, thresholds)

                model = _fit_model(
                    [feature_fn(trial.command) for trial in trials],
                    [trial.measured for trial in trials],
                )
                predictions = [_predict(model, feature_fn(trial.command)) for trial in trials]
                error = _mean_axis_rmse(predictions, [trial.measured for trial in trials])
                if best_error is None or (error, thresholds) < (best_error, best_thresholds):
                    best_error = error
                    best_thresholds = thresholds
    return best_thresholds


def _task_a_metrics(
    trials: list[_Trial],
    predictions: Mapping[str, tuple[float, float, float]],
) -> JSONDict:
    predicted = [predictions[trial.trial_id] for trial in trials]
    actual = [trial.measured for trial in trials]
    rmse = _per_axis_rmse(predicted, actual)
    return {
        "sample_count": len(trials),
        "per_axis_rmse": {
            "forward_m": rmse[0],
            "left_m": rmse[1],
            "dyaw_rad": rmse[2],
        },
        "mean_rmse": sum(rmse) / len(rmse),
        "deadband_classification_f1": _deadband_f1(predicted, actual),
    }


def _task_b_metrics(
    groups: list[_Group],
    trials: list[_Trial],
    trial_predictions: Mapping[str, tuple[float, float, float]],
) -> JSONDict:
    group_predictions = _group_predictions_from_trials(trials, trial_predictions)
    target_results = []
    for target in groups:
        scores = _inverse_control_scores(groups, group_predictions, target.measured)
        selected = min(scores, key=lambda row: (row["score"], row["candidate_id"]))
        true_errors = {
            group.command_cell_id: _outcome_distance(group.measured, target.measured)
            for group in groups
        }
        true_best_id = min(
            true_errors,
            key=lambda candidate_id: (true_errors[candidate_id], candidate_id),
        )
        regret = max(0.0, true_errors[str(selected["candidate_id"])] - true_errors[true_best_id])
        target_results.append(
            {
                "target_command_cell_id": target.command_cell_id,
                "selected_candidate_id": selected["candidate_id"],
                "true_best_candidate_id": true_best_id,
                "inverse_control_regret": regret,
                "rank_correlation": _spearman(
                    [row["score"] for row in scores],
                    [true_errors[str(row["candidate_id"])] for row in scores],
                ),
                "top1": selected["candidate_id"] == true_best_id,
            }
        )
    return {
        "target_count": len(target_results),
        "top1_accuracy": sum(1 for row in target_results if row["top1"]) / len(target_results),
        "mean_inverse_control_regret": sum(
            float(row["inverse_control_regret"]) for row in target_results
        )
        / len(target_results),
        "mean_rank_correlation": sum(float(row["rank_correlation"]) for row in target_results)
        / len(target_results),
        "targets": target_results,
    }


def _group_predictions_from_trials(
    trials: list[_Trial],
    trial_predictions: Mapping[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float]]:
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for trial in trials:
        grouped.setdefault(trial.command_cell_id, []).append(trial_predictions[trial.trial_id])
    return {
        group_id: tuple(sum(pred[axis] for pred in values) / len(values) for axis in range(3))  # type: ignore[arg-type]
        for group_id, values in grouped.items()
    }


def _representative_target(
    groups: list[_Group],
    group_predictions: Mapping[str, tuple[float, float, float]],
) -> _Group:
    if group_predictions:
        return max(
            groups,
            key=lambda group: (
                _outcome_distance(group_predictions[group.command_cell_id], group.measured),
                group.command_cell_id,
            ),
        )
    return groups[0]


def _inverse_control_scores(
    groups: list[_Group],
    group_predictions: Mapping[str, tuple[float, float, float]],
    target: tuple[float, float, float],
) -> list[JSONDict]:
    return [
        {
            "candidate_id": group.command_cell_id,
            "score": _outcome_distance(group_predictions[group.command_cell_id], target),
        }
        for group in groups
    ]


def _candidate_action(group: _Group) -> JSONDict:
    return {
        "candidate_id": group.command_cell_id,
        "label": group.command_cell_id,
        "action": {
            "type": "go2_body_velocity_system_id",
            "params": {
                "cmd_forward_m": group.command[0],
                "cmd_lateral_m": group.command[1],
                "cmd_yaw_rad": group.command[2],
            },
            "units": {
                "cmd_forward_m": "m",
                "cmd_lateral_m": "m",
                "cmd_yaw_rad": "rad",
            },
        },
        "measurement_context": {
            "trial_count": group.trial_count,
            "outcome_source": "go2_controlbench_command_cell_summary",
        },
    }


def _score_record(
    score: JSONDict,
    *,
    rank: int,
    predicted: tuple[float, float, float],
) -> JSONDict:
    return {
        "candidate_id": str(score["candidate_id"]),
        "rank": rank,
        "score": float(score["score"]),
        "lower_is_better": True,
        "components": {
            "predicted_forward_m": predicted[0],
            "predicted_left_m": predicted[1],
            "predicted_dyaw_rad": predicted[2],
            "predicted_distance_to_target": float(score["score"]),
            "yaw_weight_m_per_rad": _YAW_WEIGHT_M_PER_RAD,
        },
        "normalized": {
            "value_signal": 1.0 / (1.0 + float(score["score"])),
            "calibration_target": "real_measured_command_cell_target",
        },
    }


def _candidate_outcome(
    group: _Group,
    *,
    measured_by_id: Mapping[str, tuple[float, float, float]],
    target: tuple[float, float, float],
) -> JSONDict:
    measured = measured_by_id[group.command_cell_id]
    return {
        "candidate_id": group.command_cell_id,
        "kind": "real_measured",
        "status": "executed_system_id_group",
        "outcome_source": "go2_controlbench_command_cell_summary",
        "action_executed": True,
        "commanded": {
            "cmd_forward_m": group.command[0],
            "cmd_lateral_m": group.command[1],
            "cmd_yaw_rad": group.command[2],
        },
        "measured": {
            "forward_body_m_mean": measured[0],
            "left_body_m_mean": measured[1],
            "dyaw_rad_mean": measured[2],
        },
        "metrics": {
            "trial_count": group.trial_count,
            "true_distance_to_target": _outcome_distance(measured, target),
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except FileNotFoundError as exc:
        raise WorldForgeError(
            f"Go2 ControlBench CSV not found: {_safe_artifact_path(path)}"
        ) from exc
    except (csv.Error, UnicodeDecodeError, OSError) as exc:
        raise WorldStateError(
            f"Go2 ControlBench CSV could not be read: {_safe_artifact_path(path)}"
        ) from exc


def _linear_features(command: tuple[float, float, float]) -> list[float]:
    return [command[0], command[1], command[2]]


def _deadband_features(
    command: tuple[float, float, float],
    thresholds: tuple[float, float, float],
) -> list[float]:
    return [
        math.copysign(max(abs(value) - threshold, 0.0), value)
        for value, threshold in zip(command, thresholds, strict=True)
    ]


def _fit_model(
    features: list[list[float]],
    targets: list[tuple[float, float, float]],
) -> list[list[float]]:
    if not features or not targets or len(features) != len(targets):
        raise WorldStateError("Go2 ControlBench model fit requires aligned features and targets.")
    design = [[1.0, *row] for row in features]
    width = len(design[0])
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [[0.0 for _ in range(3)] for _ in range(width)]
    for row, target in zip(design, targets, strict=True):
        for i in range(width):
            for j in range(width):
                xtx[i][j] += row[i] * row[j]
            for axis in range(3):
                xty[i][axis] += row[i] * target[axis]
    for i in range(width):
        xtx[i][i] += _RIDGE
    return [
        _solve_linear_system(
            [list(row) for row in xtx],
            [xty_row[axis] for xty_row in xty],
        )
        for axis in range(3)
    ]


def _predict(model: list[list[float]], features: list[float]) -> tuple[float, float, float]:
    row = [1.0, *features]
    return tuple(
        sum(weight * value for weight, value in zip(axis_model, row, strict=True))
        for axis_model in model
    )  # type: ignore[return-value]


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    for pivot in range(n):
        pivot_row = max(range(pivot, n), key=lambda row: abs(matrix[row][pivot]))
        if abs(matrix[pivot_row][pivot]) < 1e-12:
            matrix[pivot][pivot] += 1e-6
            pivot_row = pivot
        if pivot_row != pivot:
            matrix[pivot], matrix[pivot_row] = matrix[pivot_row], matrix[pivot]
            rhs[pivot], rhs[pivot_row] = rhs[pivot_row], rhs[pivot]
        scale = matrix[pivot][pivot]
        for col in range(pivot, n):
            matrix[pivot][col] /= scale
        rhs[pivot] /= scale
        for row in range(n):
            if row == pivot:
                continue
            factor = matrix[row][pivot]
            if factor == 0.0:
                continue
            for col in range(pivot, n):
                matrix[row][col] -= factor * matrix[pivot][col]
            rhs[row] -= factor * rhs[pivot]
    return rhs


def _per_axis_rmse(
    predicted: Iterable[tuple[float, float, float]],
    actual: Iterable[tuple[float, float, float]],
) -> tuple[float, float, float]:
    predicted_rows = list(predicted)
    actual_rows = list(actual)
    if not predicted_rows or len(predicted_rows) != len(actual_rows):
        raise WorldStateError("Go2 ControlBench RMSE requires aligned non-empty rows.")
    return tuple(
        math.sqrt(
            sum(
                (prediction[axis] - target[axis]) ** 2
                for prediction, target in zip(predicted_rows, actual_rows, strict=True)
            )
            / len(predicted_rows)
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def _mean_axis_rmse(
    predicted: Iterable[tuple[float, float, float]],
    actual: Iterable[tuple[float, float, float]],
) -> float:
    rmse = _per_axis_rmse(predicted, actual)
    return sum(rmse) / len(rmse)


def _deadband_f1(
    predicted: list[tuple[float, float, float]],
    actual: list[tuple[float, float, float]],
) -> float:
    tp = fp = fn = 0
    for prediction, target in zip(predicted, actual, strict=True):
        predicted_active = _outcome_norm(prediction) >= _MOTION_THRESHOLD
        actual_active = _outcome_norm(target) >= _MOTION_THRESHOLD
        if predicted_active and actual_active:
            tp += 1
        elif predicted_active and not actual_active:
            fp += 1
        elif not predicted_active and actual_active:
            fn += 1
    denominator = (2 * tp) + fp + fn
    return 1.0 if denominator == 0 else (2 * tp) / denominator


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    return _pearson(left_ranks, right_ranks)


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0 for _ in values]
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0
        for ranked_index in range(index, end):
            ranks[indexed[ranked_index][0]] = rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_den = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_den = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_den == 0.0 or right_den == 0.0:
        return 0.0
    return numerator / (left_den * right_den)


def _outcome_distance(
    value: tuple[float, float, float],
    target: tuple[float, float, float],
) -> float:
    return math.sqrt(
        (value[0] - target[0]) ** 2
        + (value[1] - target[1]) ** 2
        + ((_YAW_WEIGHT_M_PER_RAD * (value[2] - target[2])) ** 2)
    )


def _outcome_norm(value: tuple[float, float, float]) -> float:
    return _outcome_distance(value, (0.0, 0.0, 0.0))


def _score_margin(ranked_scores: list[JSONDict]) -> float:
    if len(ranked_scores) < 2:
        return 0.0
    return round(float(ranked_scores[1]["score"]) - float(ranked_scores[0]["score"]), 6)


def _command_cell_id(command: tuple[float, float, float]) -> str:
    axis = "forward"
    value = command[0]
    if abs(command[1]) > abs(value):
        axis = "left"
        value = command[1]
    if abs(command[2]) > abs(value):
        axis = "yaw"
        value = command[2]
    sign = "pos" if value >= 0.0 else "neg"
    return f"cmdcell-{axis}-{sign}-{_slug_number(abs(value))}"


def _slug_number(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _float(row: Mapping[str, object], key: str) -> float:
    raw = row.get(key)
    if isinstance(raw, str):
        raw = raw.strip()
    if raw in (None, ""):
        raise WorldStateError(f"Go2 ControlBench row is missing numeric field {key!r}.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise WorldStateError(f"Go2 ControlBench field {key!r} must be numeric.") from exc
    if not math.isfinite(value):
        raise WorldStateError(f"Go2 ControlBench field {key!r} must be finite.")
    return value


def _int(row: Mapping[str, object], key: str, *, default: int = 0) -> int:
    raw = row.get(key)
    if raw in (None, ""):
        return default
    if isinstance(raw, bool):
        raise WorldStateError(f"Go2 ControlBench field {key!r} must be an integer.")
    raw_text = str(raw).strip()
    if not raw_text.isdigit():
        raise WorldStateError(f"Go2 ControlBench field {key!r} must be an integer.")
    return int(raw_text)


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


def _safe_artifact_path(path: Path) -> str:
    return _SAFE_ARTIFACT_LABELS.get(path.name, "<host-local-path>")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        required=True,
        help="Host-owned Go2 native-rate capture directory. Raw contents stay local.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for sanitized ControlBench artifacts.",
    )
    args = parser.parse_args(argv)
    summary = run_go2_controlbench_workflow(args.capture_dir, args.out)
    print(f"run_id={summary['run_id']}")
    print(f"task_a_best_baseline={summary['task_a_best_baseline']}")
    print(f"task_b_best_baseline={summary['task_b_best_baseline']}")
    print(f"trial_count={summary['trial_count']}")
    print(f"command_cell_count={summary['command_cell_count']}")
    print(f"decision_trace={summary['decision_trace_path']}")
    print(f"report={summary['report_path']}")
    return 0


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "Go2ControlBenchResult",
    "render_go2_controlbench_report",
    "run_go2_controlbench",
    "run_go2_controlbench_workflow",
]


if __name__ == "__main__":
    raise SystemExit(main())
