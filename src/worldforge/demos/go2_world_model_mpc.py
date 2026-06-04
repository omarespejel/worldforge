"""Go2 ControlBench world-model MPC demo with a DimOS shadow bridge.

This module consumes the public Go2 Air ControlBench v1 dataset artifacts and
builds a checkout-safe one-step MPC demo. It does not import DimOS, connect to a
robot, or send commands. DimOS is represented as the host-owned runtime bridge
that would execute the selected bounded Sport Move after the offline and shadow
gates pass.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from worldforge.artifact_io import write_json_artifact
from worldforge.decision_trace import (
    DECISION_TRACE_ARTIFACT_KIND,
    DECISION_TRACE_SCHEMA_VERSION,
    decision_trace_digest,
    validate_decision_trace,
)
from worldforge.models import JSONDict, WorldForgeError, WorldStateError

DEFAULT_DATASET_ID = "espejelomar/go2-air-controlbench-v1"
DEFAULT_HF_BASE_URL = f"https://huggingface.co/datasets/{DEFAULT_DATASET_ID}/resolve/main"
DEFAULT_TRIAL_TABLE_URL = f"{DEFAULT_HF_BASE_URL}/tables/all_trials_normalized.csv"
DEFAULT_OUTPUT_DIR = Path(".worldforge/go2-world-model-mpc-dimos")
RUN_ID = "go2-controlbench-world-model-mpc-dimos"
CODE_REF = "feat/go2-world-model-mpc-dimos-bridge"
CLI_DESCRIPTION = (
    "Run an offline Go2 world-model MPC demo over the public ControlBench dataset and emit "
    "DimOS shadow-bridge plus DecisionTrace artifacts."
)

_YAW_WEIGHT_M_PER_RAD = 0.25
_RIDGE = 1e-9
_MOTION_THRESHOLD = 0.02
_DEFAULT_TARGETS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("forward_target_020", (0.20, 0.0, 0.0)),
    ("forward_target_030", (0.30, 0.0, 0.0)),
    ("yaw_target_5deg", (0.0, 0.0, math.radians(5.0))),
)
_SAFE_ARTIFACT_LABELS = {
    "all_trials_normalized.csv": "<dataset>/tables/all_trials_normalized.csv",
    "world-model-mpc-summary.json": "<output-dir>/world-model-mpc-summary.json",
    "world-model-mpc-report.md": "<output-dir>/world-model-mpc-report.md",
    "dimos-shadow-bridge-plan.json": "<output-dir>/dimos-shadow-bridge-plan.json",
}


@dataclass(frozen=True, slots=True)
class Go2WorldModelMPCResult:
    """Artifacts emitted by one Go2 world-model MPC run."""

    summary: JSONDict
    report_markdown: str
    dimos_bridge_plan: JSONDict
    decision_traces: list[JSONDict]
    summary_path: Path
    report_path: Path
    dimos_bridge_path: Path
    decision_trace_paths: list[Path]


@dataclass(frozen=True, slots=True)
class _Trial:
    trial_id: str
    release: str
    name: str
    command: tuple[float, float, float]
    sport_move: tuple[float, float, float, float]
    measured: tuple[float, float, float]
    hardware_executed: bool
    outcome_kind: str


@dataclass(frozen=True, slots=True)
class _CommandCell:
    candidate_id: str
    label: str
    command: tuple[float, float, float]
    sport_move: tuple[float, float, float, float]
    measured: tuple[float, float, float]
    trial_count: int


@dataclass(frozen=True, slots=True)
class _DeadbandWorldModel:
    thresholds: tuple[float, float, float]
    weights: list[list[float]]
    residual_rmse: tuple[float, float, float]

    def predict_outcome(self, command: tuple[float, float, float]) -> tuple[float, float, float]:
        return _predict(self.weights, _deadband_features(command, self.thresholds))


def run_go2_world_model_mpc(
    *,
    dataset_csv: Path | None = None,
    trial_table_url: str = DEFAULT_TRIAL_TABLE_URL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    targets: Sequence[tuple[str, tuple[float, float, float]]] = _DEFAULT_TARGETS,
) -> Go2WorldModelMPCResult:
    """Run the checkout-safe Go2 world-model MPC demo."""

    trials = load_controlbench_trials(dataset_csv=dataset_csv, trial_table_url=trial_table_url)
    command_cells = _command_cells_from_trials(trials)
    world_model = _fit_deadband_world_model(trials)
    decisions = [
        _rank_candidates(
            target_id=target_id,
            target=target,
            command_cells=command_cells,
            world_model=world_model,
        )
        for target_id, target in targets
    ]
    summary = _build_summary(
        dataset_csv=dataset_csv,
        trial_table_url=trial_table_url,
        trials=trials,
        command_cells=command_cells,
        world_model=world_model,
        decisions=decisions,
    )
    dimos_bridge_plan = _build_dimos_bridge_plan(summary=summary, decisions=decisions)
    decision_traces = [
        _build_decision_trace(
            summary=summary,
            decision=decision,
            command_cells=command_cells,
            world_model=world_model,
        )
        for decision in decisions
    ]
    report = render_go2_world_model_mpc_report(summary)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorldStateError(
            f"Go2 world-model MPC output_dir.mkdir failed: {_safe_artifact_path(output_dir)}"
        ) from exc

    try:
        summary_path = write_json_artifact(output_dir / "world-model-mpc-summary.json", summary)
        dimos_bridge_path = write_json_artifact(
            output_dir / "dimos-shadow-bridge-plan.json",
            dimos_bridge_plan,
        )
        trace_paths = [
            write_json_artifact(
                output_dir / f"decision-trace-go2-world-model-mpc-{decision['target_id']}.json",
                trace,
            )
            for decision, trace in zip(decisions, decision_traces, strict=True)
        ]
        report_path = output_dir / "world-model-mpc-report.md"
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        raise WorldStateError(
            f"Go2 world-model MPC artifact write failed under {_safe_artifact_path(output_dir)}."
        ) from exc

    return Go2WorldModelMPCResult(
        summary=summary,
        report_markdown=report,
        dimos_bridge_plan=dimos_bridge_plan,
        decision_traces=decision_traces,
        summary_path=summary_path,
        report_path=report_path,
        dimos_bridge_path=dimos_bridge_path,
        decision_trace_paths=trace_paths,
    )


def run_go2_world_model_mpc_workflow(
    *,
    dataset_csv: Path | None = None,
    trial_table_url: str = DEFAULT_TRIAL_TABLE_URL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> JSONDict:
    """CLI-friendly wrapper with portable output handles."""

    result = run_go2_world_model_mpc(
        dataset_csv=dataset_csv,
        trial_table_url=trial_table_url,
        output_dir=output_dir,
    )
    return {
        "run_id": result.summary["run_id"],
        "dataset_id": result.summary["dataset"]["dataset_id"],
        "trial_count": result.summary["dataset"]["trial_count"],
        "command_cell_count": result.summary["dataset"]["command_cell_count"],
        "world_model": result.summary["world_model"]["model_family"],
        "dimos_mode": result.dimos_bridge_plan["runtime"]["mode"],
        "summary_path": result.summary_path.name,
        "report_path": result.report_path.name,
        "dimos_bridge_path": result.dimos_bridge_path.name,
        "decision_trace_paths": [path.name for path in result.decision_trace_paths],
    }


def load_controlbench_trials(
    *,
    dataset_csv: Path | None = None,
    trial_table_url: str = DEFAULT_TRIAL_TABLE_URL,
) -> list[_Trial]:
    """Load Go2 ControlBench trials from a local CSV or the public HF raw CSV."""

    rows = _read_csv_rows(dataset_csv=dataset_csv, trial_table_url=trial_table_url)
    trials: list[_Trial] = []
    for index, row in enumerate(rows):
        hardware_executed = _bool(row, "hardware_executed")
        if not hardware_executed:
            continue
        command = (
            _float(row, "cmd_integral_x_m"),
            _float(row, "cmd_integral_y_m"),
            _float(row, "cmd_integral_yaw_rad"),
        )
        sport_move = (
            _float(row, "cmd_x"),
            _float(row, "cmd_y"),
            _float(row, "cmd_z"),
            _float(row, "duration_s"),
        )
        measured = _body_frame_measured(row, command=command)
        trial_index = _int(row, "trial_index", default=index)
        release = str(row.get("release") or "unknown_release").strip()
        name = str(row.get("name") or _command_cell_id(command)).strip()
        trials.append(
            _Trial(
                trial_id=f"{release}-trial-{trial_index:04d}",
                release=release,
                name=name,
                command=command,
                sport_move=sport_move,
                measured=measured,
                hardware_executed=hardware_executed,
                outcome_kind=str(row.get("outcome_kind") or "real_measured_native_odom"),
            )
        )
    if len(trials) < 2:
        raise WorldForgeError("Go2 world-model MPC requires at least two hardware trials.")
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise WorldStateError("Go2 world-model MPC trial ids must be unique.")
    return trials


def render_go2_world_model_mpc_report(summary: JSONDict) -> str:
    """Render a concise report for the offline MPC and DimOS bridge demo."""

    world_model = summary["world_model"]
    dataset = summary["dataset"]
    lines = [
        "# Go2 World-Model MPC + DimOS Shadow Bridge",
        "",
        "This demo uses the public Go2 Air ControlBench dataset to fit a transparent "
        "deadband-aware command-outcome world model, then ranks bounded Unitree Sport Move "
        "commands for target motions. DimOS is represented as the host-owned runtime bridge in "
        "shadow mode; no hardware command is sent.",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{dataset['dataset_id']}`",
        f"- Trial rows used: `{dataset['trial_count']}`",
        f"- Command cells: `{dataset['command_cell_count']}`",
        f"- Outcome source: `{dataset['outcome_source']}`",
        "",
        "## World Model",
        "",
        f"- Family: `{world_model['model_family']}`",
        f"- Thresholds: `{world_model['thresholds']}`",
        f"- Residual RMSE dx/dy/dyaw: `{world_model['residual_rmse']}`",
        "- This is the v0 world model: transparent system-ID, not a learned neural model.",
        "",
        "## MPC Decisions",
        "",
        "| Target | Selected | Score | Measured Regret | DimOS Mode |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    lines.extend(
        "| "
        f"`{decision['target_id']}` | "
        f"`{decision['selected_candidate_id']}` | "
        f"{decision['selected_score']:.6f} | "
        f"{decision['inverse_control_regret']:.6f} | "
        "`shadow_replay_no_execution` |"
        for decision in summary["decisions"]
    )
    lines.extend(
        [
            "",
            "## DimOS Boundary",
            "",
            "- DimOS is the intended runtime and memory adapter.",
            "- This artifact emits a DimOS shadow bridge plan rather than importing DimOS or "
            "controlling a live robot.",
            "- A live run would require a host-owned bounded Sport Move skill, StopMove after "
            "each command, operator approval, and live telemetry logging.",
            "",
            "## Claim Boundary",
            "",
            "- Proves: WorldForge can use measured Go2 outcomes to rank commands and emit "
            "DecisionTrace evidence.",
            "- Does not prove: autonomous Go2 control, learned model superiority, Cosmos 3 "
            "superiority, or safety certification.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv_rows(*, dataset_csv: Path | None, trial_table_url: str) -> list[dict[str, str]]:
    if dataset_csv is not None:
        try:
            with dataset_csv.open(newline="", encoding="utf-8") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        except FileNotFoundError as exc:
            raise WorldForgeError(
                f"Go2 world-model MPC CSV not found: {_safe_artifact_path(dataset_csv)}"
            ) from exc
        except (csv.Error, OSError, UnicodeDecodeError) as exc:
            raise WorldStateError(
                f"Go2 world-model MPC CSV could not be read: {_safe_artifact_path(dataset_csv)}"
            ) from exc
    try:
        with urlopen(trial_table_url, timeout=20) as response:
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as exc:
        raise WorldForgeError(
            "Go2 world-model MPC could not fetch the public dataset CSV."
        ) from exc
    return [dict(row) for row in csv.DictReader(io.StringIO(payload))]


def _command_cells_from_trials(trials: list[_Trial]) -> list[_CommandCell]:
    grouped: dict[tuple[float, ...], list[_Trial]] = {}
    for trial in trials:
        grouped.setdefault(_command_group_key(trial), []).append(trial)
    command_cells: list[_CommandCell] = []
    candidate_ids = set()
    for _, members in sorted(grouped.items()):
        command = _mean_tuple([member.command for member in members])
        sport_move = _mean_sport_move([member.sport_move for member in members])
        measured = _mean_tuple([member.measured for member in members])
        label = _command_group_label(members, command)
        candidate_id = _unique_candidate_id(f"cmdcell-{_slug_text(label)}", candidate_ids)
        candidate_ids.add(candidate_id)
        command_cells.append(
            _CommandCell(
                candidate_id=candidate_id,
                label=label,
                command=command,
                sport_move=sport_move,
                measured=measured,
                trial_count=len(members),
            )
        )
    if len(command_cells) < 2:
        raise WorldForgeError("Go2 world-model MPC requires at least two command cells.")
    return command_cells


def _fit_deadband_world_model(trials: list[_Trial]) -> _DeadbandWorldModel:
    thresholds = _select_deadband_thresholds(trials)
    model = _fit_model(
        [_deadband_features(trial.command, thresholds) for trial in trials],
        [trial.measured for trial in trials],
    )
    predictions = [
        _predict(model, _deadband_features(trial.command, thresholds)) for trial in trials
    ]
    return _DeadbandWorldModel(
        thresholds=thresholds,
        weights=model,
        residual_rmse=_per_axis_rmse(predictions, [trial.measured for trial in trials]),
    )


def _select_deadband_thresholds(trials: list[_Trial]) -> tuple[float, float, float]:
    x_thresholds = (0.0, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15)
    y_thresholds = (0.0, 0.02, 0.05)
    yaw_thresholds = (0.0, 0.05, 0.08, 0.095, 0.1, 0.12, 0.15, 0.2)
    best_thresholds = (0.0, 0.0, 0.0)
    best_error: float | None = None
    for x_threshold in x_thresholds:
        for y_threshold in y_thresholds:
            for yaw_threshold in yaw_thresholds:
                thresholds = (x_threshold, y_threshold, yaw_threshold)
                model = _fit_model(
                    [_deadband_features(trial.command, thresholds) for trial in trials],
                    [trial.measured for trial in trials],
                )
                predictions = [
                    _predict(model, _deadband_features(trial.command, thresholds))
                    for trial in trials
                ]
                error = sum(_per_axis_rmse(predictions, [trial.measured for trial in trials])) / 3
                if best_error is None or (error, thresholds) < (
                    best_error,
                    best_thresholds,
                ):
                    best_error = error
                    best_thresholds = thresholds
    return best_thresholds


def _rank_candidates(
    *,
    target_id: str,
    target: tuple[float, float, float],
    command_cells: list[_CommandCell],
    world_model: _DeadbandWorldModel,
) -> JSONDict:
    scores = []
    command_integral_scores = []
    true_errors = {}
    for cell in command_cells:
        predicted = world_model.predict_outcome(cell.command)
        score = _outcome_distance(predicted, target)
        command_integral_score = _outcome_distance(cell.command, target)
        true_error = _outcome_distance(cell.measured, target)
        scores.append(
            {
                "candidate_id": cell.candidate_id,
                "label": cell.label,
                "score": score,
                "predicted": predicted,
                "measured": cell.measured,
                "command": cell.command,
                "sport_move": cell.sport_move,
                "trial_count": cell.trial_count,
            }
        )
        command_integral_scores.append(
            {"candidate_id": cell.candidate_id, "score": command_integral_score}
        )
        true_errors[cell.candidate_id] = true_error
    ranked = sorted(scores, key=lambda row: (row["score"], row["candidate_id"]))
    selected = ranked[0]
    baseline = min(command_integral_scores, key=lambda row: (row["score"], row["candidate_id"]))
    true_best_id = min(
        true_errors, key=lambda candidate_id: (true_errors[candidate_id], candidate_id)
    )
    return _round_json_floats(
        {
            "target_id": target_id,
            "target": {
                "dx_m": target[0],
                "dy_m": target[1],
                "dyaw_rad": target[2],
            },
            "ranked_candidates": [
                {
                    **row,
                    "rank": index + 1,
                    "value_signal": 1.0 / (1.0 + float(row["score"])),
                }
                for index, row in enumerate(ranked)
            ],
            "selected_candidate_id": selected["candidate_id"],
            "selected_label": selected["label"],
            "selected_score": selected["score"],
            "score_margin": (
                float(ranked[1]["score"]) - float(ranked[0]["score"]) if len(ranked) > 1 else 0.0
            ),
            "baseline_candidate_id": baseline["candidate_id"],
            "baseline_score": baseline["score"],
            "true_best_candidate_id": true_best_id,
            "selected_true_error": true_errors[str(selected["candidate_id"])],
            "true_best_error": true_errors[true_best_id],
            "inverse_control_regret": max(
                0.0,
                true_errors[str(selected["candidate_id"])] - true_errors[true_best_id],
            ),
        }
    )


def _build_summary(
    *,
    dataset_csv: Path | None,
    trial_table_url: str,
    trials: list[_Trial],
    command_cells: list[_CommandCell],
    world_model: _DeadbandWorldModel,
    decisions: list[JSONDict],
) -> JSONDict:
    releases = sorted({trial.release for trial in trials})
    return _round_json_floats(
        {
            "schema_version": 1,
            "artifact_kind": "worldforge.go2_world_model_mpc_summary",
            "run_id": RUN_ID,
            "dataset": {
                "dataset_id": DEFAULT_DATASET_ID,
                "source": (
                    _SAFE_ARTIFACT_LABELS["all_trials_normalized.csv"]
                    if dataset_csv is not None
                    else trial_table_url
                ),
                "trial_count": len(trials),
                "command_cell_count": len(command_cells),
                "release_count": len(releases),
                "releases": releases,
                "outcome_source": "native_go2_odom_signed_body_projection_public_preview",
                "external_reference": "bounded_aruco_tables_available_but_not_required_for_v0_mpc",
            },
            "world_model": {
                "model_family": "deadband_affine_command_outcome_v0",
                "score_kind": "hand_cost",
                "learned_model_used": False,
                "thresholds": {
                    "dx_m": world_model.thresholds[0],
                    "dy_m": world_model.thresholds[1],
                    "dyaw_rad": world_model.thresholds[2],
                },
                "residual_rmse": {
                    "dx_m": world_model.residual_rmse[0],
                    "dy_m": world_model.residual_rmse[1],
                    "dyaw_rad": world_model.residual_rmse[2],
                },
            },
            "decisions": [
                {
                    "target_id": decision["target_id"],
                    "selected_candidate_id": decision["selected_candidate_id"],
                    "selected_label": decision["selected_label"],
                    "selected_score": decision["selected_score"],
                    "score_margin": decision["score_margin"],
                    "baseline_candidate_id": decision["baseline_candidate_id"],
                    "true_best_candidate_id": decision["true_best_candidate_id"],
                    "inverse_control_regret": decision["inverse_control_regret"],
                }
                for decision in decisions
            ],
            "dimos": {
                "role": "host_owned_runtime_and_memory_bridge",
                "mode": "shadow_replay_no_execution",
                "live_execution_allowed": False,
                "required_live_skill": "bounded_sport_move_then_stop",
            },
            "claim_boundary": {
                "autonomous_robot_control": False,
                "live_dimos_execution": False,
                "learned_model_claim": False,
                "cosmos3_claim": False,
                "raw_room_video_included": False,
            },
        }
    )


def _build_dimos_bridge_plan(*, summary: JSONDict, decisions: list[JSONDict]) -> JSONDict:
    bridge_decisions = []
    for decision in decisions:
        selected = next(
            row
            for row in decision["ranked_candidates"]
            if row["candidate_id"] == decision["selected_candidate_id"]
        )
        sport_move = selected["sport_move"]
        bridge_decisions.append(
            {
                "target_id": decision["target_id"],
                "selected_candidate_id": decision["selected_candidate_id"],
                "selected_label": selected["label"],
                "shadow_command": {
                    "type": "unitree_go2_sport_move",
                    "params": {
                        "x": sport_move[0],
                        "y": sport_move[1],
                        "z": sport_move[2],
                        "duration_s": sport_move[3],
                    },
                    "post_condition": "StopMove required after bounded command",
                },
                "dimos_skill_contract": {
                    "module": "Go2DirectSkills",
                    "method": "bounded_sport_move_then_stop",
                    "status": "required_for_live_execution",
                    "note": (
                        "Existing DimOS joystick wrappers are not treated as equivalent to the "
                        "Sport Move command used by ControlBench."
                    ),
                },
            }
        )
    return _round_json_floats(
        {
            "schema_version": 1,
            "artifact_kind": "worldforge.dimos_go2_world_model_mpc_shadow_bridge",
            "run_id": summary["run_id"],
            "runtime": {
                "name": "DimOS",
                "mode": "shadow_replay_no_execution",
                "imports_dimos": False,
                "hardware_commands_sent": False,
            },
            "source_dataset": summary["dataset"],
            "memory_stream_contract": {
                "pre_state": ["ROBOTODOM", "LOW_STATE", "ULIDAR_STATE"],
                "post_state": ["ROBOTODOM", "LOW_STATE", "ULIDAR_STATE"],
                "optional_reference": ["overhead_aruco_pose"],
            },
            "decisions": bridge_decisions,
            "safety_gate": {
                "offline_controlbench_gate_passed": True,
                "shadow_mode_required_before_live": True,
                "operator_approval_required": True,
                "stopmove_after_each_command": True,
            },
        }
    )


def _build_decision_trace(
    *,
    summary: JSONDict,
    decision: JSONDict,
    command_cells: list[_CommandCell],
    world_model: _DeadbandWorldModel,
) -> JSONDict:
    ranked = list(decision["ranked_candidates"])
    selected_id = str(decision["selected_candidate_id"])
    selected = ranked[0]
    target = decision["target"]
    cells_by_id = {cell.candidate_id: cell for cell in command_cells}
    trace: JSONDict = {
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "artifact_kind": DECISION_TRACE_ARTIFACT_KIND,
        "trace_id": f"go2-world-model-mpc-{decision['target_id']}",
        "run_id": RUN_ID,
        "step_index": 0,
        "prev_trace_id": None,
        "embodiment": {
            "kind": "quadruped",
            "platform": "Unitree Go2 Air",
            "action_space": "bounded_high_level_sport_move",
            "embodiment_id": "go2-air-public-controlbench",
        },
        "host_runtime": {
            "name": "DimOS shadow bridge",
            "mode": "offline_shadow_replay_no_execution",
            "version": CODE_REF,
        },
        "task": {
            "task_id": "go2-world-model-mpc-command-selection",
            "description": (
                "Rank bounded Go2 Sport Move candidates with a ControlBench world model."
            ),
        },
        "observation": {
            "dataset_id": DEFAULT_DATASET_ID,
            "trial_count": summary["dataset"]["trial_count"],
            "command_cell_count": summary["dataset"]["command_cell_count"],
            "dimos_mode": "shadow_replay_no_execution",
        },
        "goal": {
            "type": "target_body_motion",
            "description": "Select the bounded command predicted to best match target body motion.",
            "target": target,
            "sub_goals": [
                {
                    "id": "predict_candidate_outcomes",
                    "description": "Use the deadband-affine world model to predict each command.",
                    "required": True,
                    "weight": 0.5,
                },
                {
                    "id": "minimize_measured_regret",
                    "description": "Compare selected command to real measured command cells.",
                    "required": True,
                    "weight": 0.5,
                },
            ],
            "success_criteria": {
                "metric": "inverse_control_regret",
                "partial_credit": True,
            },
        },
        "candidate_actions": [
            _candidate_action(cells_by_id[str(row["candidate_id"])]) for row in ranked
        ],
        "scores": [
            _score_record(row, rank=index + 1, world_model=world_model)
            for index, row in enumerate(ranked)
        ],
        "selected_action": {
            "candidate_id": selected_id,
            "score": selected["score"],
            "score_margin": decision["score_margin"],
            "why_selected": (
                "Lowest predicted distance to the target under the measured ControlBench "
                "deadband-affine world model. DimOS execution remains shadow-only."
            ),
        },
        "counterfactuals": [
            {
                "candidate_id": str(row["candidate_id"]),
                "score": row["score"],
                "delta_vs_selected": float(row["score"] - selected["score"]),
                "why_rejected": "Higher predicted target error.",
            }
            for row in ranked
            if row["candidate_id"] != selected_id
        ],
        "candidate_outcomes": [
            _candidate_outcome(cells_by_id[str(row["candidate_id"])], decision=decision)
            for row in ranked
        ],
        "baseline": {
            "candidate_id": decision["baseline_candidate_id"],
            "score": next(
                row["score"]
                for row in ranked
                if row["candidate_id"] == decision["baseline_candidate_id"]
            ),
            "regret_vs_selected": next(
                row["score"]
                for row in ranked
                if row["candidate_id"] == decision["baseline_candidate_id"]
            )
            - selected["score"],
            "policy": "command_integral_baseline",
        },
        "outcome": {
            "kind": "real_measured",
            "status": "offline_benchmark_regret_computed",
            "metrics": {
                "outcome_source": "go2_controlbench_native_odom_public_preview",
                "selected_true_error": decision["selected_true_error"],
                "true_best_error": decision["true_best_error"],
                "inverse_control_regret": decision["inverse_control_regret"],
                "true_best_candidate_id": decision["true_best_candidate_id"],
                "live_dimos_execution": False,
            },
        },
        "planner_diagnostics": {
            "planner": "one_step_mpc_candidate_ranker",
            "world_model": "deadband_affine_command_outcome_v0",
            "candidate_count": len(ranked),
            "dimos_shadow_bridge": True,
        },
        "reproducibility": {
            "provider_version": "Go2 ControlBench public v1 preview",
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
            "safety_controller": "offline_shadow_only_dimos_bridge",
            "limitations": [
                (
                    "Hardware execution refers to historical ControlBench trials, not this "
                    "shadow plan."
                ),
                "DimOS bridge artifact does not import DimOS or send live robot commands.",
                "World model is transparent deadband-affine system identification, not Cosmos 3.",
                "Native Go2 odometry is the measured source for this v0 trace.",
            ],
        },
        "interop": {
            "dimos": {
                "role": "host_owned_runtime_and_memory_bridge",
                "mode": "shadow_replay_no_execution",
                "required_live_skill": "bounded_sport_move_then_stop",
            },
            "wmcp": {
                "role": "future rollout/evaluate interface for predicted Go2 outcomes",
                "methods": ["rollout", "evaluate"],
            },
        },
    }
    return validate_decision_trace(
        _round_json_floats(trace),
        name=f"Go2 world-model MPC DecisionTrace {decision['target_id']}",
    )


def _candidate_action(cell: _CommandCell) -> JSONDict:
    x, y, z, duration = cell.sport_move
    return {
        "candidate_id": cell.candidate_id,
        "label": cell.label,
        "action": {
            "type": "dimos_go2_bounded_sport_move_shadow",
            "params": {
                "x_mps": x,
                "y_mps": y,
                "z_radps": z,
                "duration_s": duration,
            },
            "units": {
                "x_mps": "m/s",
                "y_mps": "m/s",
                "z_radps": "rad/s",
                "duration_s": "s",
            },
        },
        "measurement_context": {
            "trial_count": cell.trial_count,
            "runtime_bridge": "DimOS shadow replay",
        },
    }


def _score_record(
    row: JSONDict,
    *,
    rank: int,
    world_model: _DeadbandWorldModel,
) -> JSONDict:
    predicted = row["predicted"]
    return {
        "candidate_id": str(row["candidate_id"]),
        "rank": rank,
        "score": float(row["score"]),
        "lower_is_better": True,
        "components": {
            "predicted_dx_m": predicted[0],
            "predicted_dy_m": predicted[1],
            "predicted_dyaw_rad": predicted[2],
            "predicted_distance_to_target": row["score"],
            "yaw_weight_m_per_rad": _YAW_WEIGHT_M_PER_RAD,
            "model_rmse_dx_m": world_model.residual_rmse[0],
            "model_rmse_dy_m": world_model.residual_rmse[1],
            "model_rmse_dyaw_rad": world_model.residual_rmse[2],
        },
        "normalized": {
            "value_signal": row["value_signal"],
            "calibration_target": "real_measured_controlbench_command_cells",
        },
    }


def _candidate_outcome(cell: _CommandCell, *, decision: JSONDict) -> JSONDict:
    return {
        "candidate_id": cell.candidate_id,
        "kind": "real_measured",
        "status": "executed_in_controlbench_capture",
        "outcome_source": "native_go2_odom_signed_body_projection_public_preview",
        "action_executed": True,
        "commanded": {
            "cmd_integral_x_m": cell.command[0],
            "cmd_integral_y_m": cell.command[1],
            "cmd_integral_yaw_rad": cell.command[2],
            "sport_move_x_mps": cell.sport_move[0],
            "sport_move_y_mps": cell.sport_move[1],
            "sport_move_z_radps": cell.sport_move[2],
            "duration_s": cell.sport_move[3],
        },
        "measured": {
            "dx_m_mean": cell.measured[0],
            "dy_m_mean": cell.measured[1],
            "dyaw_rad_mean": cell.measured[2],
        },
        "metrics": {
            "trial_count": cell.trial_count,
            "true_distance_to_target": _outcome_distance(
                cell.measured,
                (
                    decision["target"]["dx_m"],
                    decision["target"]["dy_m"],
                    decision["target"]["dyaw_rad"],
                ),
            ),
        },
    }


def _fit_model(
    features: list[list[float]],
    targets: list[tuple[float, float, float]],
) -> list[list[float]]:
    if not features or not targets or len(features) != len(targets):
        raise WorldStateError("Go2 world-model MPC fit requires aligned features and targets.")
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


def _deadband_features(
    command: tuple[float, float, float],
    thresholds: tuple[float, float, float],
) -> list[float]:
    return [
        math.copysign(max(abs(value) - threshold, 0.0), value)
        for value, threshold in zip(command, thresholds, strict=True)
    ]


def _per_axis_rmse(
    predicted: Iterable[tuple[float, float, float]],
    actual: Iterable[tuple[float, float, float]],
) -> tuple[float, float, float]:
    predicted_rows = list(predicted)
    actual_rows = list(actual)
    if not predicted_rows or len(predicted_rows) != len(actual_rows):
        raise WorldStateError("Go2 world-model MPC RMSE requires aligned rows.")
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


def _outcome_distance(
    value: tuple[float, float, float],
    target: tuple[float, float, float],
) -> float:
    return math.sqrt(
        (value[0] - target[0]) ** 2
        + (value[1] - target[1]) ** 2
        + ((_YAW_WEIGHT_M_PER_RAD * (value[2] - target[2])) ** 2)
    )


def _mean_tuple(
    values: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    return tuple(sum(value[axis] for value in values) / len(values) for axis in range(3))  # type: ignore[return-value]


def _mean_sport_move(
    values: Sequence[tuple[float, float, float, float]],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    return tuple(sum(value[axis] for value in values) / len(values) for axis in range(4))  # type: ignore[return-value]


def _body_frame_measured(
    row: Mapping[str, object],
    *,
    command: tuple[float, float, float],
) -> tuple[float, float, float]:
    planar = _optional_float(row, "odom_planar_m")
    if planar is None:
        planar = math.hypot(_float(row, "odom_dx_m"), _float(row, "odom_dy_m"))
    dx = 0.0
    dy = 0.0
    if abs(command[0]) >= _MOTION_THRESHOLD and abs(command[0]) >= abs(command[1]):
        dx = math.copysign(planar, command[0])
    elif abs(command[1]) >= _MOTION_THRESHOLD:
        dy = math.copysign(planar, command[1])
    return (dx, dy, _float(row, "odom_dyaw_rad"))


def _command_group_key(trial: _Trial) -> tuple[float, ...]:
    return tuple(round(value, 6) for value in (*trial.sport_move, *trial.command))


def _command_group_label(members: Sequence[_Trial], command: tuple[float, float, float]) -> str:
    labels = {member.name for member in members}
    if len(labels) == 1:
        return next(iter(labels))
    return _command_cell_id(command)


def _unique_candidate_id(base_id: str, existing: set[str]) -> str:
    if base_id not in existing:
        return base_id
    suffix = 2
    while f"{base_id}-{suffix}" in existing:
        suffix += 1
    return f"{base_id}-{suffix}"


def _command_cell_id(command: tuple[float, float, float]) -> str:
    axis = "forward"
    value = command[0]
    if abs(command[1]) > abs(value):
        axis = "lateral"
        value = command[1]
    if abs(command[2]) > abs(value):
        axis = "yaw"
        value = command[2]
    sign = "pos" if value >= 0.0 else "neg"
    return f"{axis}_{sign}_{abs(value):.3f}"


def _slug_text(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")


def _float(row: Mapping[str, object], key: str) -> float:
    raw = row.get(key)
    if isinstance(raw, str):
        raw = raw.strip()
    if raw in (None, ""):
        raise WorldStateError(f"Go2 world-model MPC row is missing numeric field {key!r}.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise WorldStateError(f"Go2 world-model MPC field {key!r} must be numeric.") from exc
    if not math.isfinite(value):
        raise WorldStateError(f"Go2 world-model MPC field {key!r} must be finite.")
    return value


def _optional_float(row: Mapping[str, object], key: str) -> float | None:
    raw = row.get(key)
    if isinstance(raw, str):
        raw = raw.strip()
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise WorldStateError(f"Go2 world-model MPC field {key!r} must be numeric.") from exc
    if not math.isfinite(value):
        raise WorldStateError(f"Go2 world-model MPC field {key!r} must be finite.")
    return value


def _int(row: Mapping[str, object], key: str, *, default: int = 0) -> int:
    raw = row.get(key)
    if raw in (None, ""):
        return default
    try:
        value = int(float(str(raw)))
    except (TypeError, ValueError) as exc:
        raise WorldStateError(f"Go2 world-model MPC field {key!r} must be an integer.") from exc
    return value


def _bool(row: Mapping[str, object], key: str) -> bool:
    raw = row.get(key)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise WorldStateError(f"Go2 world-model MPC field {key!r} must be boolean.")


def _round_json_floats(value: object) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("JSON float values must be finite.")
        return round(value, 6)
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return [_round_json_floats(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _round_json_floats(item) for key, item in value.items()}
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}.")


def _safe_artifact_path(path: Path) -> str:
    return _SAFE_ARTIFACT_LABELS.get(path.name, "<host-local-path>")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=None,
        help=(
            "Optional local all_trials_normalized.csv. If omitted, the public Hugging Face "
            "dataset table is fetched."
        ),
    )
    parser.add_argument(
        "--trial-table-url",
        default=DEFAULT_TRIAL_TABLE_URL,
        help="Raw CSV URL for the Go2 ControlBench trial table.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for sanitized world-model MPC artifacts.",
    )
    args = parser.parse_args(argv)
    result = run_go2_world_model_mpc_workflow(
        dataset_csv=args.dataset_csv,
        trial_table_url=args.trial_table_url,
        output_dir=args.out,
    )
    print(f"run_id={result['run_id']}")
    print(f"dataset_id={result['dataset_id']}")
    print(f"trial_count={result['trial_count']}")
    print(f"command_cell_count={result['command_cell_count']}")
    print(f"world_model={result['world_model']}")
    print(f"dimos_mode={result['dimos_mode']}")
    print(f"dimos_bridge={result['dimos_bridge_path']}")
    print(f"report={result['report_path']}")
    for trace_path in result["decision_trace_paths"]:
        print(f"decision_trace={trace_path}")
    return 0


__all__ = [
    "DEFAULT_DATASET_ID",
    "DEFAULT_OUTPUT_DIR",
    "Go2WorldModelMPCResult",
    "load_controlbench_trials",
    "render_go2_world_model_mpc_report",
    "run_go2_world_model_mpc",
    "run_go2_world_model_mpc_workflow",
]


if __name__ == "__main__":
    raise SystemExit(main())
