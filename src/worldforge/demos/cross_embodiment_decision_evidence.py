"""Cross-embodiment DecisionTrace v1 evidence bundle.

This demo is fork-integration proof work for issue #38. It runs the checkout-safe
Go2 replay arena, the PimSim export adapter, and the SO-101 replay trace, then
normalizes all three outputs into the same DecisionTrace v1 contract.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from worldforge.artifact_io import write_json_artifact
from worldforge.decision_trace import (
    DECISION_TRACE_ARTIFACT_KIND,
    DECISION_TRACE_SCHEMA_VERSION,
    decision_trace_digest,
    validate_decision_trace,
)
from worldforge.demos.dimos_go2_replay_arena import (
    DEFAULT_FIXTURE_PATH,
    DEFAULT_PIMSIM_EXPORT_PATH,
    run_dimos_go2_pimsim_export,
    run_dimos_go2_replay_arena,
)
from worldforge.demos.so101_replay_trace import (
    SO101_DATASET_REFERENCE,
    SO101_JOINT_NAMES,
)
from worldforge.demos.so101_replay_trace import (
    run_demo as run_so101_demo,
)
from worldforge.models import JSONDict, WorldForgeError

DEFAULT_OUTPUT_DIR = Path(".worldforge/cross-embodiment-decision-evidence")
RUN_ID = "cross-embodiment-decision-evidence"
CODE_REF = "integration/worldforge-go2-so101-evidence"


def run_cross_embodiment_decision_evidence(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    go2_fixture_path: Path = DEFAULT_FIXTURE_PATH,
    pimsim_export_path: Path = DEFAULT_PIMSIM_EXPORT_PATH,
) -> JSONDict:
    """Run all checkout-safe evidence lanes and write normalized artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    go2_result = run_dimos_go2_replay_arena(go2_fixture_path, output_dir / "raw-go2-replay")
    pimsim_result = run_dimos_go2_pimsim_export(
        pimsim_export_path,
        output_dir / "raw-pimsim-export",
    )
    so101_summary = run_so101_demo(state_dir=output_dir / "so101-state", emit=False)

    traces = {
        "go2": normalize_go2_decision_trace(
            go2_result.trace,
            trace_id="decision-trace-go2",
            step_index=0,
            host_runtime_name="DimOS Go2 replay fixture",
        ),
        "pimsim": normalize_go2_decision_trace(
            pimsim_result.trace,
            trace_id="decision-trace-pimsim",
            step_index=1,
            host_runtime_name="DimOS PimSim export adapter",
            prev_trace_id="decision-trace-go2",
        ),
        "so101": normalize_so101_decision_trace(
            so101_summary["trace"],
            trace_id="decision-trace-so101",
            step_index=2,
            prev_trace_id="decision-trace-pimsim",
        ),
    }

    artifact_paths = {
        "go2": write_json_artifact(output_dir / "decision-trace-go2.json", traces["go2"]),
        "pimsim": write_json_artifact(
            output_dir / "decision-trace-pimsim.json",
            traces["pimsim"],
        ),
        "so101": write_json_artifact(output_dir / "decision-trace-so101.json", traces["so101"]),
    }
    report = render_cross_embodiment_report(traces)
    report_path = output_dir / "cross-embodiment-report.md"
    report_path.write_text(report, encoding="utf-8")

    summary: JSONDict = {
        "schema_version": 1,
        "artifact_kind": "worldforge.cross_embodiment_decision_evidence",
        "run_id": RUN_ID,
        "trace_count": len(traces),
        "traces": {
            name: {
                "trace_id": trace["trace_id"],
                "path": str(artifact_paths[name]),
                "selected_action_id": trace["selected_action"]["candidate_id"],
                "selected_value_signal": _selected_score_record(trace)["normalized"][
                    "value_signal"
                ],
                "score_margin": trace["selected_action"]["score_margin"],
                "baseline_regret": trace["baseline"]["regret_vs_selected"],
                "outcome_kind": trace["outcome"]["kind"],
                "score_kind": trace["claim_boundary"]["score_kind"],
            }
            for name, trace in traces.items()
        },
        "report_path": str(report_path),
        "kill_criterion": (
            "If WorldForge cannot choose, explain, compare, or expose counterfactual robot "
            "actions better than a hardcoded command or plain script, stop pushing this "
            "integration. Learned-scorer value requires an independent held-out referee result "
            "that beats proprio/hand-cost baselines on fair decoy/progress metrics; exact-match "
            "top-1 is diagnostic for near-duplicate decoys, not the pass/fail gate."
        ),
    }
    write_json_artifact(output_dir / "summary.json", summary)
    return summary


def normalize_go2_decision_trace(
    trace: JSONDict,
    *,
    trace_id: str,
    step_index: int,
    host_runtime_name: str,
    prev_trace_id: str | None = None,
) -> JSONDict:
    """Convert a Go2 replay/PimSim decision trace into DecisionTrace v1."""

    try:
        return _normalize_go2_decision_trace(
            trace,
            trace_id=trace_id,
            step_index=step_index,
            host_runtime_name=host_runtime_name,
            prev_trace_id=prev_trace_id,
        )
    except WorldForgeError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise WorldForgeError(f"Failed to normalize Go2 decision trace: {exc}") from exc


def _normalize_go2_decision_trace(
    trace: JSONDict,
    *,
    trace_id: str,
    step_index: int,
    host_runtime_name: str,
    prev_trace_id: str | None = None,
) -> JSONDict:
    """Convert a Go2 replay/PimSim decision trace into DecisionTrace v1."""

    scored = _scored_go2_candidates(trace)
    selected = scored[0]
    baseline = _candidate_by_id(scored, trace.get("baseline_action_id"))
    scores = [float(candidate["total_cost"]) for candidate in scored]
    score_margin = float(trace["score_margin"])
    baseline_regret = float(trace["baseline_regret"])
    normalized = _normalized_score_records(
        candidates=scored,
        id_key="action_id",
        score_key="total_cost",
        components_key="components",
        baseline_score=float(baseline["total_cost"]) if baseline is not None else None,
        score_margin=score_margin,
    )
    decision_trace: JSONDict = {
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "artifact_kind": DECISION_TRACE_ARTIFACT_KIND,
        "trace_id": trace_id,
        "run_id": RUN_ID,
        "step_index": step_index,
        "prev_trace_id": prev_trace_id,
        "embodiment": {
            "kind": "quadruped_navigation",
            "platform": "unitree_go2_air",
            "embodiment_id": "go2-replay",
            "action_space": "planar_body_velocity",
        },
        "host_runtime": {
            "name": host_runtime_name,
            "mode": "checkout_safe_replay",
            "version": None,
        },
        "task": {
            "task_id": str(trace["scenario_id"]),
            "description": str(trace["goal"]["description"]),
        },
        "observation": {
            "ref": {
                "source": trace.get("source", {}),
                "frame_id": trace["observation"]["frame_id"],
            },
            "summary": dict(trace["observation"]),
        },
        "goal": _go2_goal(trace["goal"]),
        "candidate_actions": [
            {
                "candidate_id": str(candidate["action_id"]),
                "label": str(candidate["action_id"]),
                "action": _go2_action(candidate["action"]),
                "predicted_outcome": {"endpoint": candidate["endpoint"]},
            }
            for candidate in scored
        ],
        "scores": normalized,
        "selected_action": {
            "candidate_id": str(selected["action_id"]),
            "score": float(selected["total_cost"]),
            "score_margin": score_margin,
            "why_selected": _go2_selection_reason(selected),
        },
        "counterfactuals": [
            {
                "candidate_id": str(candidate["action_id"]),
                "score": float(candidate["total_cost"]),
                "delta_vs_selected": round(
                    float(candidate["total_cost"]) - float(selected["total_cost"]),
                    6,
                ),
                "why_rejected": _go2_rejection_reason(candidate),
            }
            for candidate in scored
            if candidate["action_id"] != selected["action_id"]
        ],
        "candidate_outcomes": [
            {
                "candidate_id": str(candidate["action_id"]),
                "kind": "analytic",
                "status": "predicted_endpoint",
                "outcome_source": "analytic_replay_estimate",
                "action_executed": False,
                "commanded": _go2_action(candidate["action"])["params"],
                "measured": {},
                "metrics": {
                    "total_cost": float(candidate["total_cost"]),
                    "progress_m": float(candidate["components"]["progress_m"]),
                    "obstacle_risk": float(candidate["components"]["obstacle_risk"]),
                    "map_cost": float(candidate["components"]["map_cost"]),
                },
            }
            for candidate in scored
        ],
        "baseline": {
            "candidate_id": trace.get("baseline_action_id"),
            "score": float(baseline["total_cost"]) if baseline is not None else None,
            "regret_vs_selected": baseline_regret,
            "policy": "fixture_baseline_command",
        },
        "outcome": {
            "kind": "analytic",
            "status": "predicted_lower_cost_action" if scores else "not_evaluated",
            "metrics": {
                "candidate_count": len(scored),
                "score_margin": score_margin,
                "baseline_regret": baseline_regret,
                "partial_subgoal_credit": _partial_credit_from_regret(
                    baseline_regret,
                    score_margin,
                ),
            },
        },
        "planner_diagnostics": {
            "planner": trace["plan_metadata"]["planning_mode"],
            "score_provider": trace["plan_metadata"]["score_provider"],
            "candidate_count": trace["candidate_count"],
            "source_schema": trace["artifact_kind"],
            "workflow_trace": trace["plan_metadata"]["workflow_trace"],
        },
        "reproducibility": {
            "provider_version": "Go2ReplayScoreProvider deterministic hand-cost",
            "checkpoint_hash": None,
            "model_card_ref": None,
            "input_digest": f"sha256:{decision_trace_digest(trace)}",
            "seed": 0,
            "code_ref": CODE_REF,
        },
        "claim_boundary": {
            "score_kind": "hand_cost",
            "outcome_kind": "analytic",
            "hardware_executed": False,
            "learned_model_used": False,
            "safety_controller": None,
            "limitations": [
                "No live Go2 command was sent.",
                "Outcome is an analytic endpoint estimate from replay metadata.",
                "The scorer is transparent hand cost, not a learned latent world model.",
            ],
        },
        "interop": _interop_block(
            trace_id=trace_id,
            provider="Go2ReplayScoreProvider",
            trajectory_ref={
                "standard": "DimOS/PimSim-derived replay fixture",
                "frame_id": trace["observation"]["frame_id"],
            },
        ),
    }
    return validate_decision_trace(
        _round_json_floats(decision_trace),
        name=f"{trace_id} DecisionTrace",
    )


def normalize_so101_decision_trace(
    trace: JSONDict,
    *,
    trace_id: str,
    step_index: int,
    prev_trace_id: str | None = None,
) -> JSONDict:
    """Convert the SO-101 replay trace into DecisionTrace v1."""

    try:
        return _normalize_so101_decision_trace(
            trace,
            trace_id=trace_id,
            step_index=step_index,
            prev_trace_id=prev_trace_id,
        )
    except WorldForgeError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise WorldForgeError(f"Failed to normalize SO-101 decision trace: {exc}") from exc


def _normalize_so101_decision_trace(
    trace: JSONDict,
    *,
    trace_id: str,
    step_index: int,
    prev_trace_id: str | None = None,
) -> JSONDict:
    """Convert the SO-101 replay trace into DecisionTrace v1."""

    scores = list(trace["candidate_scores"])
    selected = dict(trace["selected_action"])
    selected_id = str(selected["candidate_id"])
    selected_score = float(selected["score"])
    baseline = _candidate_by_id(scores, "stop-relocalize", id_key="candidate_id")
    score_margin = float(selected["score_margin_to_runner_up"])
    normalized = _normalized_score_records(
        candidates=scores,
        id_key="candidate_id",
        score_key="score",
        components_key="components",
        baseline_score=float(baseline["score"]) if baseline is not None else None,
        score_margin=score_margin,
    )
    decision_trace: JSONDict = {
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "artifact_kind": DECISION_TRACE_ARTIFACT_KIND,
        "trace_id": trace_id,
        "run_id": RUN_ID,
        "step_index": step_index,
        "prev_trace_id": prev_trace_id,
        "embodiment": {
            "kind": "manipulation",
            "platform": "so101",
            "embodiment_id": "so101-replay",
            "action_space": "six_axis_joint_delta",
        },
        "host_runtime": {
            "name": "LeRobot-shaped SO-101 replay",
            "mode": "checkout_safe_replay",
            "version": None,
        },
        "task": {
            "task_id": "so101-pick-place",
            "description": "Select a replay action chunk for tabletop pick-and-place.",
        },
        "observation": {
            "ref": {
                "standard": "LeRobotDataset v3-shaped fixture",
                "repo_id": SO101_DATASET_REFERENCE["repo_id"],
                "episode_index": trace["observation"]["episode_index"],
                "frame_index": trace["observation"]["frame_index"],
            },
            "summary": dict(trace["observation"]),
        },
        "goal": _so101_goal(trace["goal"]),
        "candidate_actions": [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "label": str(candidate["label"]),
                "action": {
                    "type": "so101_joint_delta",
                    "params": {
                        "joint_names": list(SO101_JOINT_NAMES),
                        "joint_delta": list(candidate["joint_delta"]),
                        "duration_s": float(candidate["duration_s"]),
                    },
                    "units": {
                        "joint_delta": "normalized_replay_delta",
                        "duration_s": "s",
                    },
                },
                "predicted_outcome": {
                    "object_pose": candidate["predicted_object_pose"],
                    "expected_outcome": candidate["expected_outcome"],
                },
            }
            for candidate in trace["candidate_actions"]
        ],
        "scores": normalized,
        "selected_action": {
            "candidate_id": selected_id,
            "score": selected_score,
            "score_margin": score_margin,
            "why_selected": selected["why_selected"],
        },
        "counterfactuals": [
            {
                "candidate_id": str(counterfactual["candidate_id"]),
                "score": float(counterfactual["score"]),
                "delta_vs_selected": round(float(counterfactual["score"]) - selected_score, 6),
                "why_rejected": str(counterfactual["why_rejected"]),
            }
            for counterfactual in trace["counterfactuals"]
        ],
        "candidate_outcomes": [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "kind": "analytic",
                "status": "mock_replay_estimate",
                "outcome_source": str(
                    trace["outcome"].get("outcome_source", "mock_replay_execution")
                ),
                "action_executed": False,
                "commanded": {
                    "joint_names": list(SO101_JOINT_NAMES),
                    "joint_delta": list(candidate["joint_delta"]),
                    "duration_s": float(candidate["duration_s"]),
                },
                "measured": {},
                "metrics": {
                    "placement_error_m": float(
                        _candidate_score_by_id(scores, candidate["candidate_id"])["components"][
                            "placement_error_m"
                        ]
                    ),
                    "contact_risk": float(
                        _candidate_score_by_id(scores, candidate["candidate_id"])["components"][
                            "contact_risk"
                        ]
                    ),
                },
            }
            for candidate in trace["candidate_actions"]
        ],
        "baseline": {
            "candidate_id": "stop-relocalize",
            "score": float(baseline["score"]) if baseline is not None else None,
            "regret_vs_selected": (
                round(float(baseline["score"]) - selected_score, 6) if baseline is not None else 0.0
            ),
            "policy": "safe_hold_baseline",
        },
        "outcome": {
            "kind": "analytic",
            "status": str(trace["outcome"]["success_label"]),
            "metrics": {
                "outcome_source": str(
                    trace["outcome"].get("outcome_source", "unknown_replay_outcome")
                ),
                "placement_error_m": float(trace["outcome"]["placement_error_m"]),
                "success": bool(trace["outcome"]["success"]),
                "partial_subgoal_credit": 1.0 if trace["outcome"]["success"] else 0.5,
            },
        },
        "planner_diagnostics": {
            "planner": "so101-replay-cem",
            "score_provider": "so101-replay-score",
            "candidate_count": len(trace["candidate_actions"]),
            "source_schema": trace["schema_version"],
            "dataset_reference": trace["dataset_reference"],
        },
        "reproducibility": {
            "provider_version": "SO101ReplayScoreProvider deterministic hand-cost",
            "checkpoint_hash": None,
            "model_card_ref": "https://huggingface.co/datasets/lerobot/svla_so101_pickplace",
            "input_digest": f"sha256:{decision_trace_digest(trace)}",
            "seed": 0,
            "code_ref": CODE_REF,
        },
        "claim_boundary": {
            "score_kind": "hand_cost",
            "outcome_kind": "analytic",
            "hardware_executed": False,
            "learned_model_used": False,
            "safety_controller": None,
            "limitations": [
                "No SO-101 hardware, camera, calibration, or LeRobot runtime was used.",
                "Outcome is a replay/mock placement check, not a physical execution result.",
                "Joint deltas are fixture units and must not be treated as calibrated commands.",
            ],
        },
        "interop": _interop_block(
            trace_id=trace_id,
            provider="SO101ReplayScoreProvider",
            trajectory_ref={
                "standard": "LeRobotDataset v3",
                "repo_id": SO101_DATASET_REFERENCE["repo_id"],
                "episode_index": trace["observation"]["episode_index"],
                "frame_index": trace["observation"]["frame_index"],
            },
        ),
    }
    return validate_decision_trace(
        _round_json_floats(decision_trace),
        name=f"{trace_id} DecisionTrace",
    )


def render_cross_embodiment_report(traces: Mapping[str, JSONDict]) -> str:
    """Render the compact team-facing report for the evidence bundle."""

    lines = [
        "# Cross-Embodiment Decision Evidence",
        "",
        "This bundle validates Go2 replay, PimSim export, and SO-101 manipulation traces against "
        "one `DecisionTrace v1` contract. It is replay evidence, not a live-robot or learned-model "
        "claim.",
        "",
        "The comparable cross-embodiment column is `Value Signal` on a normalized `[0, 1]` "
        "scale. `Local Margin` and `Local Baseline Regret` are embodiment-local hand-cost "
        "diagnostics and should not be compared across embodiments as raw units.",
        "",
        "`Outcome Kind` is the DecisionTrace measurement class. `Outcome Provenance` is the "
        "concrete source; `mock_replay_execution` under `analytic` means a fixture/mock analytic "
        "check, not sim-measured or real-measured success.",
        "",
        "| Trace | Embodiment | Selected | Value Signal | Desirability | Local Margin | Local "
        "Baseline Regret | Score Kind | Outcome Kind | Outcome Provenance |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    lines.extend(
        "| "
        f"`{trace['trace_id']}` | "
        f"{trace['embodiment']['platform']} / {trace['embodiment']['kind']} | "
        f"`{trace['selected_action']['candidate_id']}` | "
        f"{float(_selected_score_record(trace)['normalized']['value_signal']):.6f} | "
        f"{float(_selected_score_record(trace)['normalized']['desirability']):.6f} | "
        f"{float(trace['selected_action']['score_margin']):.6f} | "
        f"{float(trace['baseline']['regret_vs_selected']):.6f} | "
        f"{trace['claim_boundary']['score_kind']} | "
        f"{trace['outcome']['kind']} |"
        f" {trace['outcome']['metrics'].get('outcome_source', 'analytic_replay_estimate')} |"
        for trace in traces.values()
    )
    lines.extend(
        [
            "",
            "## DecisionTrace v1 Contract",
            "",
            "- One DecisionTrace v1 semantic validator accepts navigation and manipulation "
            "decisions.",
            "- Actions are self-describing with `type`, `params`, and `units`.",
            "- Goals include ordered `sub_goals` and partial-credit success criteria.",
            "- Reproducibility records provider version, checkpoint/model-card refs, input digest, "
            "seed, and code ref.",
            "- Claim boundaries explicitly separate hand-cost scoring from learned-latent scoring "
            "and analytic outcomes from sim/real measured outcomes.",
            "",
            "## Kill Criterion",
            "",
            (
                "If WorldForge cannot choose, explain, compare, or expose counterfactual robot "
                "actions better than a hardcoded DimOS command or plain script, stop pushing this "
                "integration and pivot."
            ),
            "",
            (
                "For learned-scorer claims, the stricter gate is independent held-out referee "
                "performance above the proprio/hand-cost baselines on fair decoy/progress "
                "metrics. Exact-match top-1 is diagnostic for near-duplicate decoys; "
                "rank-correlation against proprio progress is calibration, not an independent "
                "success signal."
            ),
            "",
        ]
    )
    for name, trace in traces.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Task: {trace['task']['description']}",
                f"- Selected: `{trace['selected_action']['candidate_id']}`",
                "- Value signal: "
                f"{float(_selected_score_record(trace)['normalized']['value_signal']):.6f}",
                f"- Why: {trace['selected_action']['why_selected']}",
                f"- Counterfactuals: {len(trace['counterfactuals'])}",
                "- Outcome provenance: "
                f"{trace['outcome']['metrics'].get('outcome_source', 'analytic_replay_estimate')}",
                f"- Limitations: {'; '.join(trace['claim_boundary']['limitations'])}",
                "",
            ]
        )
    return "\n".join(lines)


def _selected_score_record(trace: JSONDict) -> JSONDict:
    selected_id = str(trace["selected_action"]["candidate_id"])
    for score in trace["scores"]:
        if str(score["candidate_id"]) == selected_id:
            return score
    raise WorldForgeError(f"Trace {trace['trace_id']} selected action has no score row.")


def _go2_goal(goal: JSONDict) -> JSONDict:
    return {
        "type": "navigation",
        "description": str(goal["description"]),
        "target": {"x": float(goal["x"]), "y": float(goal["y"])},
        "sub_goals": [
            {
                "id": "advance_to_goal",
                "description": "Reduce planar distance to the goal.",
                "required": True,
                "weight": 0.45,
            },
            {
                "id": "avoid_obstacles",
                "description": "Avoid obstacle, cost, and uncertainty zones.",
                "required": True,
                "weight": 0.35,
            },
            {
                "id": "maintain_localization",
                "description": "Prefer relocalization when confidence is low.",
                "required": False,
                "weight": 0.2,
            },
        ],
        "success_criteria": {
            "metric": "partial_subgoal_credit",
            "partial_credit": True,
            "protocol": "WorldForge replay analytic scoring",
        },
    }


def _so101_goal(goal: JSONDict) -> JSONDict:
    return {
        "type": "manipulation",
        "description": str(goal.get("description", "Place object at target pose.")),
        "target": goal,
        "sub_goals": [
            {
                "id": "approach",
                "description": "Move end effector toward the object.",
                "required": True,
                "weight": 0.15,
            },
            {
                "id": "pre_grasp",
                "description": "Reach a stable pre-grasp pose with clearance.",
                "required": True,
                "weight": 0.2,
            },
            {
                "id": "contact",
                "description": "Make low-risk contact and preserve grasp confidence.",
                "required": True,
                "weight": 0.2,
            },
            {
                "id": "lift",
                "description": "Lift or move the object without unsafe contact risk.",
                "required": True,
                "weight": 0.2,
            },
            {
                "id": "place",
                "description": "Place the object inside target tolerance.",
                "required": True,
                "weight": 0.25,
            },
        ],
        "success_criteria": {
            "metric": "partial_subgoal_credit",
            "partial_credit": True,
            "protocol": "RoboWM-Bench-style step/task outcome structure",
        },
    }


def _go2_action(action: JSONDict) -> JSONDict:
    params = dict(action["parameters"])
    params.pop("action_id", None)
    units: JSONDict = {}
    for key in params:
        if key.endswith("_m"):
            units[key] = "m"
        elif key.endswith("_mps"):
            units[key] = "m/s"
        elif key.endswith("_rad"):
            units[key] = "rad"
        elif key.endswith("_s"):
            units[key] = "s"
        else:
            units[key] = "unitless"
    return {
        "type": str(action["type"]),
        "params": params,
        "units": units,
    }


def _normalized_score_records(
    *,
    candidates: list[JSONDict],
    id_key: str,
    score_key: str,
    components_key: str,
    baseline_score: float | None,
    score_margin: float,
) -> list[JSONDict]:
    if not candidates:
        raise WorldForgeError("Expected at least one candidate score record.")
    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (float(candidate[score_key]), str(candidate[id_key])),
    )
    scores = [float(candidate[score_key]) for candidate in ranked_candidates]
    best_score = min(scores)
    worst_score = max(scores)
    span = max(worst_score - best_score, 0.0)
    records: list[JSONDict] = []
    for index, candidate in enumerate(ranked_candidates):
        score = float(candidate[score_key])
        desirability = 0.0 if span == 0.0 else (worst_score - score) / span
        regret = 0.0 if baseline_score is None else max(0.0, baseline_score - score)
        regret_scale = max(span, abs(baseline_score or 0.0), 1.0)
        separability = 0.0 if span == 0.0 else max(0.0, score_margin) / span
        value_signal = _clamp01(
            (0.5 * desirability) + (0.3 * regret / regret_scale) + (0.2 * separability)
        )
        records.append(
            {
                "candidate_id": str(candidate[id_key]),
                "rank": index + 1,
                "score": score,
                "lower_is_better": True,
                "components": dict(candidate[components_key]),
                "normalized": {
                    "value_signal": round(value_signal, 6),
                    "desirability": round(_clamp01(desirability), 6),
                    "regret_vs_baseline": round(regret, 6),
                    "separability": round(_clamp01(separability), 6),
                    "calibration_target": "analytic_cost_gap",
                },
            }
        )
    return records


def _round_json_floats(value: object) -> object:
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
        rounded: JSONDict = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings.")
            rounded[key] = _round_json_floats(item)
        return rounded
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}.")


def _go2_selection_reason(candidate: JSONDict) -> str:
    components = candidate["components"]
    return (
        "Lowest transparent replay cost after combining goal distance, obstacle risk, map cost, "
        f"uncertainty, and relocalization terms; progress={components['progress_m']:.3f}m."
    )


def _go2_rejection_reason(candidate: JSONDict) -> str:
    components = candidate["components"]
    penalties = {
        "obstacle risk": float(components["obstacle_risk"]),
        "map cost": float(components["map_cost"]),
        "uncertainty": float(components["uncertainty_cost"]),
        "relocalization": float(components["relocalization_cost"]),
    }
    worst_name, worst_value = max(penalties.items(), key=lambda item: item[1])
    if worst_value > 0.0:
        return f"Higher replay cost, led by {worst_name} penalty."
    if float(components["progress_m"]) <= 0.0:
        return "Higher replay cost with no useful goal progress."
    return "Higher total replay cost than the selected action."


def _partial_credit_from_regret(baseline_regret: float, score_margin: float) -> float:
    if baseline_regret > 0.0 and score_margin > 0.0:
        return 1.0
    if score_margin > 0.0:
        return 0.5
    return 0.0


def _scored_go2_candidates(trace: JSONDict) -> list[JSONDict]:
    candidates = list(trace["scored_candidates"])
    if not candidates:
        raise WorldForgeError("Go2 trace must contain scored_candidates.")
    return candidates


def _candidate_by_id(
    candidates: list[JSONDict],
    candidate_id: object,
    *,
    id_key: str = "action_id",
) -> JSONDict | None:
    if candidate_id is None:
        return None
    for candidate in candidates:
        if str(candidate[id_key]) == str(candidate_id):
            return candidate
    return None


def _candidate_score_by_id(candidates: list[JSONDict], candidate_id: object) -> JSONDict:
    match = _candidate_by_id(candidates, candidate_id, id_key="candidate_id")
    if match is None:
        raise WorldForgeError(f"SO-101 candidate score not found for {candidate_id!r}.")
    return match


def _interop_block(
    *,
    trace_id: str,
    provider: str,
    trajectory_ref: JSONDict,
) -> JSONDict:
    return {
        "trajectory_ref": trajectory_ref,
        "prov": {
            "entities": ["observation", "candidate_actions", "scores", "outcome"],
            "activities": ["score_candidates", "select_action"],
            "agents": [provider, "WorldForge"],
            "selected_action_was_derived_from": ["observation", "candidate_actions", "scores"],
        },
        "otel_span": {
            "name": "worldforge.decision_trace.score_and_select",
            "attributes": {
                "worldforge.trace_id": trace_id,
                "worldforge.schema_version": DECISION_TRACE_SCHEMA_VERSION,
                "gen_ai.operation.name": "invoke_agent",
            },
        },
    }


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for DecisionTrace v1 artifacts and report.",
    )
    args = parser.parse_args(argv)
    summary = run_cross_embodiment_decision_evidence(output_dir=args.out)
    print(f"trace_count={summary['trace_count']}")
    for name, row in summary["traces"].items():
        print(
            f"{name}: selected={row['selected_action_id']} "
            f"margin={float(row['score_margin']):.6f} "
            f"baseline_regret={float(row['baseline_regret']):.6f} "
            f"trace={row['path']}"
        )
    print(f"report={summary['report_path']}")
    return 0


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "normalize_go2_decision_trace",
    "normalize_so101_decision_trace",
    "render_cross_embodiment_report",
    "run_cross_embodiment_decision_evidence",
]
