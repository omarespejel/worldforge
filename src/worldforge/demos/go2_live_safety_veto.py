"""Go2 live safety-veto shadow demo.

This module builds the no-motion side of a live Go2 safety-veto demo. It uses
the public ControlBench dataset to fit the same transparent command-outcome
model as the DimOS shadow MPC demo, combines it with live-shaped obstacle
evidence, and emits a DecisionTrace explaining why a proposed forward motion is
authorized or rejected.

The default workflow is intentionally checkout-safe: it does not import DimOS,
connect to a robot, or send hardware commands. A live host can later supply the
same bridge-shaped fields from DimOS odom/LiDAR/costmap streams.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from worldforge.artifact_io import write_json_artifact
from worldforge.decision_trace import (
    DECISION_TRACE_ARTIFACT_KIND,
    DECISION_TRACE_SCHEMA_VERSION,
    decision_trace_digest,
    validate_decision_trace,
)
from worldforge.demos.go2_world_model_mpc import (
    DEFAULT_DATASET_ID,
    DEFAULT_TRIAL_TABLE_URL,
    _DeadbandWorldModel,
    _fit_deadband_world_model,
    _outcome_distance,
    _round_json_floats,
    load_controlbench_trials,
)
from worldforge.models import JSONDict, WorldStateError

DEFAULT_OUTPUT_DIR = Path(".worldforge/go2-live-safety-veto")
RUN_ID = "go2-live-safety-veto-shadow"
CODE_REF = "feat/go2-live-safety-veto"
CLI_DESCRIPTION = (
    "Run a checkout-safe Go2 live safety-veto shadow demo and emit DimOS bridge "
    "plus DecisionTrace artifacts."
)

_PROPOSED_TARGET = (0.50, 0.0, 0.0)
_STOP_TARGET = (0.0, 0.0, 0.0)
_TINY_LEFT_YAW = (0.0, 0.0, math.radians(8.0))
_YAW_WEIGHT_M_PER_RAD = 0.25


@dataclass(frozen=True, slots=True)
class Go2LiveSafetyVetoResult:
    """Artifacts emitted by one live safety-veto shadow run."""

    summary: JSONDict
    bridge_plan: JSONDict
    decision_trace: JSONDict
    report_markdown: str
    summary_path: Path
    bridge_path: Path
    decision_trace_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class Go2LiveSafetyVetoDemoPairResult:
    """Artifacts emitted by the two-case demo pair."""

    summary: JSONDict
    report_markdown: str
    obstacle_result: Go2LiveSafetyVetoResult
    open_space_result: Go2LiveSafetyVetoResult
    summary_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class ObstacleEvidence:
    """Live-shaped obstacle evidence used by the shadow safety gate."""

    source_kind: str
    forward_clearance_m: float
    lidar_freshness_ms: float
    costmap_freshness_ms: float
    odom_freshness_ms: float
    unknown_cells_in_forward_corridor: bool
    stopmove_verified: bool
    hardware_commands_sent: bool = False


@dataclass(frozen=True, slots=True)
class _DemoCase:
    case_id: str
    label: str
    result: Go2LiveSafetyVetoResult


@dataclass(frozen=True, slots=True)
class _Candidate:
    candidate_id: str
    label: str
    action_type: str
    command: tuple[float, float, float]
    predicted: tuple[float, float, float]
    target_error: float
    risk_cost: float
    total_score: float
    authorized_for_execution: bool
    rejection_reasons: tuple[str, ...]


def run_go2_live_safety_veto(
    *,
    dataset_csv: Path | None = None,
    trial_table_url: str = DEFAULT_TRIAL_TABLE_URL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    obstacle_evidence: ObstacleEvidence | None = None,
) -> Go2LiveSafetyVetoResult:
    """Run the checkout-safe Go2 live safety-veto shadow demo."""

    evidence = obstacle_evidence or ObstacleEvidence(
        source_kind="fixture_live_shaped_lidar_costmap",
        forward_clearance_m=0.28,
        lidar_freshness_ms=120.0,
        costmap_freshness_ms=180.0,
        odom_freshness_ms=80.0,
        unknown_cells_in_forward_corridor=False,
        stopmove_verified=False,
    )
    trials = load_controlbench_trials(dataset_csv=dataset_csv, trial_table_url=trial_table_url)
    world_model = _fit_deadband_world_model(trials)
    candidates = _score_candidates(world_model=world_model, evidence=evidence)
    selected = candidates[0]
    summary = _build_summary(
        dataset_csv=dataset_csv,
        trial_table_url=trial_table_url,
        trial_count=len(trials),
        world_model=world_model,
        evidence=evidence,
        candidates=candidates,
        selected=selected,
    )
    bridge_plan = _build_bridge_plan(summary=summary, evidence=evidence, selected=selected)
    decision_trace = _build_decision_trace(
        summary=summary,
        evidence=evidence,
        candidates=candidates,
        selected=selected,
        world_model=world_model,
    )
    report = render_go2_live_safety_veto_report(summary)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorldStateError(
            "Go2 live safety-veto output directory could not be created."
        ) from exc

    try:
        summary_path = write_json_artifact(output_dir / "live-safety-veto-summary.json", summary)
        bridge_path = write_json_artifact(
            output_dir / "dimos-live-safety-bridge-plan.json",
            bridge_plan,
        )
        trace_path = write_json_artifact(
            output_dir / "decision-trace-go2-live-safety-veto.json",
            decision_trace,
        )
        report_path = output_dir / "live-safety-veto-report.md"
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        raise WorldStateError("Go2 live safety-veto artifact write failed.") from exc

    return Go2LiveSafetyVetoResult(
        summary=summary,
        bridge_plan=bridge_plan,
        decision_trace=decision_trace,
        report_markdown=report,
        summary_path=summary_path,
        bridge_path=bridge_path,
        decision_trace_path=trace_path,
        report_path=report_path,
    )


def run_go2_live_safety_veto_demo_pair(
    *,
    dataset_csv: Path | None = None,
    trial_table_url: str = DEFAULT_TRIAL_TABLE_URL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    obstacle_clearance_m: float = 0.28,
    open_space_clearance_m: float = 2.0,
) -> Go2LiveSafetyVetoDemoPairResult:
    """Run the two-case demo pair: obstacle veto plus open-space control."""

    obstacle_result = run_go2_live_safety_veto(
        dataset_csv=dataset_csv,
        trial_table_url=trial_table_url,
        output_dir=output_dir / "obstacle",
        obstacle_evidence=ObstacleEvidence(
            source_kind="demo_pair_obstacle_lidar_costmap",
            forward_clearance_m=obstacle_clearance_m,
            lidar_freshness_ms=120.0,
            costmap_freshness_ms=180.0,
            odom_freshness_ms=80.0,
            unknown_cells_in_forward_corridor=False,
            stopmove_verified=True,
        ),
    )
    open_space_result = run_go2_live_safety_veto(
        dataset_csv=dataset_csv,
        trial_table_url=trial_table_url,
        output_dir=output_dir / "open-space",
        obstacle_evidence=ObstacleEvidence(
            source_kind="demo_pair_open_space_lidar_costmap",
            forward_clearance_m=open_space_clearance_m,
            lidar_freshness_ms=120.0,
            costmap_freshness_ms=180.0,
            odom_freshness_ms=80.0,
            unknown_cells_in_forward_corridor=False,
            stopmove_verified=True,
        ),
    )
    cases = [
        _DemoCase(case_id="obstacle", label="obstacle veto", result=obstacle_result),
        _DemoCase(case_id="open_space", label="open-space control", result=open_space_result),
    ]
    summary = _build_demo_pair_summary(
        output_dir=output_dir,
        obstacle_clearance_m=obstacle_clearance_m,
        open_space_clearance_m=open_space_clearance_m,
        cases=cases,
    )
    report = render_go2_live_safety_veto_demo_pair_report(summary)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = write_json_artifact(output_dir / "demo-pair-summary.json", summary)
        report_path = output_dir / "demo-pair-report.md"
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        raise WorldStateError("Go2 live safety-veto demo artifact write failed.") from exc

    return Go2LiveSafetyVetoDemoPairResult(
        summary=summary,
        report_markdown=report,
        obstacle_result=obstacle_result,
        open_space_result=open_space_result,
        summary_path=summary_path,
        report_path=report_path,
    )


def run_go2_live_safety_veto_workflow(
    *,
    dataset_csv: Path | None = None,
    trial_table_url: str = DEFAULT_TRIAL_TABLE_URL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    obstacle_evidence: ObstacleEvidence | None = None,
) -> JSONDict:
    """CLI-friendly wrapper with portable output handles."""

    result = run_go2_live_safety_veto(
        dataset_csv=dataset_csv,
        trial_table_url=trial_table_url,
        output_dir=output_dir,
        obstacle_evidence=obstacle_evidence,
    )
    selected = result.summary["decision"]["selected_candidate_id"]
    proposed = result.summary["decision"]["proposed_candidate_id"]
    return {
        "run_id": result.summary["run_id"],
        "mode": result.summary["runtime"]["mode"],
        "proposed_candidate_id": proposed,
        "selected_candidate_id": selected,
        "decision": result.summary["decision"]["authorization"],
        "hardware_commands_sent": result.summary["runtime"]["hardware_commands_sent"],
        "summary_path": result.summary_path.name,
        "bridge_path": result.bridge_path.name,
        "decision_trace_path": result.decision_trace_path.name,
        "report_path": result.report_path.name,
    }


def run_go2_live_safety_veto_demo_pair_workflow(
    *,
    dataset_csv: Path | None = None,
    trial_table_url: str = DEFAULT_TRIAL_TABLE_URL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    obstacle_clearance_m: float = 0.28,
    open_space_clearance_m: float = 2.0,
) -> JSONDict:
    """CLI-friendly demo pair wrapper with portable output handles."""

    result = run_go2_live_safety_veto_demo_pair(
        dataset_csv=dataset_csv,
        trial_table_url=trial_table_url,
        output_dir=output_dir,
        obstacle_clearance_m=obstacle_clearance_m,
        open_space_clearance_m=open_space_clearance_m,
    )
    return {
        "run_id": result.summary["run_id"],
        "mode": result.summary["runtime"]["mode"],
        "hardware_commands_sent": result.summary["runtime"]["hardware_commands_sent"],
        "obstacle_decision": result.summary["cases"][0]["decision"],
        "obstacle_selected": result.summary["cases"][0]["selected_candidate_id"],
        "obstacle_trace_digest": result.summary["cases"][0]["decision_trace_digest"],
        "obstacle_trace_path": result.summary["cases"][0]["decision_trace_path"],
        "open_space_decision": result.summary["cases"][1]["decision"],
        "open_space_selected": result.summary["cases"][1]["selected_candidate_id"],
        "open_space_trace_digest": result.summary["cases"][1]["decision_trace_digest"],
        "open_space_trace_path": result.summary["cases"][1]["decision_trace_path"],
        "summary_path": result.summary_path.name,
        "report_path": result.report_path.name,
    }


def render_go2_live_safety_veto_report(summary: JSONDict) -> str:
    """Render a concise report for the live safety-veto shadow run."""

    decision = summary["decision"]
    evidence = summary["obstacle_evidence"]
    lines = [
        "# Go2 Live Safety Veto Shadow Demo",
        "",
        "This checkout-safe run uses the public Go2 Air ControlBench dataset to fit a "
        "transparent short-horizon command-outcome model, then applies a live-shaped "
        "obstacle gate to a proposed `forward_50cm` action.",
        "",
        "## Decision",
        "",
        f"- Proposed: `{decision['proposed_candidate_id']}`",
        f"- Selected: `{decision['selected_candidate_id']}`",
        f"- Authorization: `{decision['authorization']}`",
        f"- Hardware commands sent: `{summary['runtime']['hardware_commands_sent']}`",
        f"- Primary reason: `{decision['primary_reason']}`",
        "",
        "## Obstacle Evidence",
        "",
        f"- Source kind: `{evidence['source_kind']}`",
        f"- Forward clearance: `{evidence['forward_clearance_m']}` m",
        f"- Odom freshness: `{evidence['odom_freshness_ms']}` ms",
        f"- LiDAR freshness: `{evidence['lidar_freshness_ms']}` ms",
        f"- Costmap freshness: `{evidence['costmap_freshness_ms']}` ms",
        f"- StopMove verified: `{evidence['stopmove_verified']}`",
        "",
        "## Boundary",
        "",
        "- This is a predictive safety-filter shadow artifact, not certified safety.",
        "- It does not import DimOS, call Unitree APIs, or move a live robot.",
        "- A live execution path must verify StopMove before any bounded command.",
        "",
    ]
    return "\n".join(lines)


def render_go2_live_safety_veto_demo_pair_report(summary: JSONDict) -> str:
    """Render a concise report for the two-case shadow demo pair."""

    lines = [
        "# Go2 Live Safety Veto Demo Pair",
        "",
        "Two checkout-safe shadow cases were run from the same ControlBench world model:",
        "",
    ]
    for case in summary["cases"]:
        lines.extend(
            [
                f"## {case['label'].title()}",
                "",
                f"- Decision: `{case['decision']}`",
                f"- Proposed: `{case['proposed_candidate_id']}`",
                f"- Selected: `{case['selected_candidate_id']}`",
                f"- Reason: `{case['primary_reason']}`",
                f"- DecisionTrace digest: `{case['decision_trace_digest']}`",
                f"- DecisionTrace path: `{case['decision_trace_path']}`",
                f"- Hardware commands sent: `{case['hardware_commands_sent']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "- No DimOS import.",
            "- No Unitree connection.",
            "- No robot command.",
            "- No Cosmos 3 control path.",
            "- No `relative_move`.",
            "- This is not certified safety.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_demo_pair_summary(
    *,
    output_dir: Path,
    obstacle_clearance_m: float,
    open_space_clearance_m: float,
    cases: list[_DemoCase],
) -> JSONDict:
    return _round_json_floats(
        {
            "schema_version": 1,
            "artifact_kind": "worldforge.go2_live_safety_veto_demo_pair_summary",
            "run_id": f"{RUN_ID}-demo-pair",
            "runtime": {
                "name": "DimOS bridge contract",
                "mode": "shadow_no_execution",
                "imports_dimos": False,
                "hardware_commands_sent": False,
            },
            "dataset": {
                "dataset_id": DEFAULT_DATASET_ID,
                "outcome_source": "native_go2_odom_signed_body_projection_public_preview",
            },
            "demo_parameters": {
                "obstacle_clearance_m": obstacle_clearance_m,
                "open_space_clearance_m": open_space_clearance_m,
                "stopmove_verified": True,
            },
            "cases": [_demo_pair_case_summary(case=case, output_dir=output_dir) for case in cases],
            "claim_boundary": {
                "predictive_safety_filter_demo": True,
                "certified_safety": False,
                "live_dimos_execution": False,
                "cosmos3_in_control_path": False,
                "relative_move_used": False,
                "autonomous_navigation": False,
            },
        }
    )


def _demo_pair_case_summary(*, case: _DemoCase, output_dir: Path) -> JSONDict:
    summary = case.result.summary
    decision = summary["decision"]
    relative_case_dir = case.result.summary_path.parent.relative_to(output_dir)
    trace_digest = f"sha256:{decision_trace_digest(case.result.decision_trace)}"
    return {
        "case_id": case.case_id,
        "label": case.label,
        "decision": decision["authorization"],
        "proposed_candidate_id": decision["proposed_candidate_id"],
        "selected_candidate_id": decision["selected_candidate_id"],
        "primary_reason": decision["primary_reason"],
        "unsafe_forward_rejected": decision["unsafe_forward_rejected"],
        "hardware_commands_sent": summary["runtime"]["hardware_commands_sent"],
        "summary_path": f"{relative_case_dir}/live-safety-veto-summary.json",
        "decision_trace_path": f"{relative_case_dir}/decision-trace-go2-live-safety-veto.json",
        "decision_trace_digest": trace_digest,
        "decision_trace_digest_short": trace_digest.removeprefix("sha256:")[:12],
    }


def _score_candidates(
    *,
    world_model: _DeadbandWorldModel,
    evidence: ObstacleEvidence,
) -> list[_Candidate]:
    raw_candidates = [
        ("forward_50cm", "planner requested forward 50cm", "planner_request", _PROPOSED_TARGET),
        ("stop_hold", "hold position / StopMove", "stop_hold", _STOP_TARGET),
        ("turn_left_small", "tiny left yaw alternative", "bounded_yaw_shadow", _TINY_LEFT_YAW),
    ]
    candidates = [
        _score_candidate(
            candidate_id=candidate_id,
            label=label,
            action_type=action_type,
            command=command,
            world_model=world_model,
            evidence=evidence,
        )
        for candidate_id, label, action_type, command in raw_candidates
    ]
    return sorted(candidates, key=lambda item: (item.total_score, item.candidate_id))


def _score_candidate(
    *,
    candidate_id: str,
    label: str,
    action_type: str,
    command: tuple[float, float, float],
    world_model: _DeadbandWorldModel,
    evidence: ObstacleEvidence,
) -> _Candidate:
    predicted = world_model.predict_outcome(command)
    target_error = _outcome_distance(predicted, _PROPOSED_TARGET)
    risk_cost, authorized, rejection_reasons = _risk_for_candidate(
        action_type=action_type,
        predicted=predicted,
        evidence=evidence,
    )
    return _Candidate(
        candidate_id=candidate_id,
        label=label,
        action_type=action_type,
        command=command,
        predicted=predicted,
        target_error=target_error,
        risk_cost=risk_cost,
        total_score=target_error + risk_cost,
        authorized_for_execution=authorized,
        rejection_reasons=tuple(rejection_reasons),
    )


def _risk_for_candidate(
    *,
    action_type: str,
    predicted: tuple[float, float, float],
    evidence: ObstacleEvidence,
) -> tuple[float, bool, list[str]]:
    reasons: list[str] = []
    risk_cost = 0.0
    is_motion_candidate = action_type != "stop_hold"
    if is_motion_candidate and evidence.odom_freshness_ms > 250.0:
        reasons.append("stale_odom")
        risk_cost += 10.0
    if (
        is_motion_candidate
        and evidence.lidar_freshness_ms > 500.0
        and evidence.costmap_freshness_ms > 500.0
    ):
        reasons.append("stale_lidar_and_costmap")
        risk_cost += 10.0
    if evidence.unknown_cells_in_forward_corridor and action_type == "planner_request":
        reasons.append("unknown_cells_in_forward_corridor")
        risk_cost += 5.0
    if action_type == "planner_request":
        required_clearance = abs(predicted[0]) + 0.35
        if evidence.forward_clearance_m < required_clearance:
            reasons.append("predicted_swept_footprint_intersects_obstacle_zone")
            risk_cost += 20.0
    if not evidence.stopmove_verified:
        reasons.append("stopmove_path_not_verified_for_live_execution")
        if action_type != "stop_hold":
            risk_cost += 3.0
    authorized = not reasons
    return risk_cost, authorized, reasons


def _build_summary(
    *,
    dataset_csv: Path | None,
    trial_table_url: str,
    trial_count: int,
    world_model: _DeadbandWorldModel,
    evidence: ObstacleEvidence,
    candidates: list[_Candidate],
    selected: _Candidate,
) -> JSONDict:
    proposed = next(
        candidate for candidate in candidates if candidate.candidate_id == "forward_50cm"
    )
    unsafe_forward_rejected = bool(
        proposed.rejection_reasons and selected.candidate_id != proposed.candidate_id
    )
    primary_reason = (
        proposed.rejection_reasons[0]
        if proposed.rejection_reasons
        else "forward_request_authorized_in_shadow"
    )
    return _round_json_floats(
        {
            "schema_version": 1,
            "artifact_kind": "worldforge.go2_live_safety_veto_summary",
            "run_id": RUN_ID,
            "dataset": {
                "dataset_id": DEFAULT_DATASET_ID,
                "source": (
                    "<dataset>/tables/all_trials_normalized.csv"
                    if dataset_csv is not None
                    else trial_table_url
                ),
                "trial_count": trial_count,
                "outcome_source": "native_go2_odom_signed_body_projection_public_preview",
            },
            "runtime": {
                "name": "DimOS bridge contract",
                "mode": "shadow_no_execution",
                "imports_dimos": False,
                "hardware_commands_sent": evidence.hardware_commands_sent,
            },
            "world_model": {
                "model_family": "deadband_affine_command_outcome_v0",
                "learned_model_used": False,
                "score_kind": "hand_cost",
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
            "obstacle_evidence": _evidence_json(evidence),
            "decision": {
                "proposed_candidate_id": proposed.candidate_id,
                "selected_candidate_id": selected.candidate_id,
                "authorization": _authorization_status(
                    selected=selected,
                    proposed=proposed,
                    unsafe_forward_rejected=unsafe_forward_rejected,
                ),
                "primary_reason": primary_reason,
                "unsafe_forward_rejected": unsafe_forward_rejected,
                "score_margin": (
                    candidates[1].total_score - candidates[0].total_score
                    if len(candidates) > 1
                    else 0.0
                ),
            },
            "candidates": [_candidate_summary(candidate) for candidate in candidates],
            "claim_boundary": {
                "predictive_safety_filter_demo": True,
                "certified_safety": False,
                "live_dimos_execution": False,
                "cosmos3_in_control_path": False,
                "relative_move_used": False,
                "autonomous_navigation": False,
            },
        }
    )


def _build_bridge_plan(
    *,
    summary: JSONDict,
    evidence: ObstacleEvidence,
    selected: _Candidate,
) -> JSONDict:
    return _round_json_floats(
        {
            "schema_version": 1,
            "artifact_kind": "worldforge.dimos_go2_live_safety_bridge_plan",
            "run_id": summary["run_id"],
            "runtime": summary["runtime"],
            "required_bridge_methods": {
                "read_state": {
                    "returns": ["pose", "velocity", "freshness_ms"],
                    "fail_closed_if_missing": True,
                },
                "read_lidar_or_costmap": {
                    "returns": ["forward_clearance_m", "occupancy_or_costmap", "freshness_ms"],
                    "fail_closed_if_missing": True,
                },
                "stop_move": {
                    "preferred": "UnitreeGo2TwistAdapter.write_stop -> SportClient.StopMove",
                    "status": (
                        "verified" if evidence.stopmove_verified else "not_verified_shadow_only"
                    ),
                },
                "bounded_sport_move_then_stop": {
                    "max_duration_s": 0.6,
                    "max_vx_mps": 0.15,
                    "max_abs_wz_radps": 0.25,
                    "post_condition": "StopMove acknowledged and velocity near zero",
                },
            },
            "selected_shadow_action": {
                "candidate_id": selected.candidate_id,
                "label": selected.label,
                "execution_authorized": False,
                "shadow_authorized_if_live_gate_rechecks_pass": selected.authorized_for_execution,
                "reason": "shadow_no_execution_default",
            },
            "hard_gates": {
                "stale_odom_reject_ms": 250,
                "stale_lidar_and_costmap_reject_ms": 500,
                "unknown_forward_cells_reject": True,
                "stopmove_required_before_motion": True,
                "operator_approval_required": True,
            },
        }
    )


def _build_decision_trace(
    *,
    summary: JSONDict,
    evidence: ObstacleEvidence,
    candidates: list[_Candidate],
    selected: _Candidate,
    world_model: _DeadbandWorldModel,
) -> JSONDict:
    selected_score = selected.total_score
    proposed = next(
        candidate for candidate in candidates if candidate.candidate_id == "forward_50cm"
    )
    trace: JSONDict = {
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "artifact_kind": DECISION_TRACE_ARTIFACT_KIND,
        "trace_id": "go2-live-safety-veto-shadow-0001",
        "run_id": RUN_ID,
        "step_index": 0,
        "prev_trace_id": None,
        "embodiment": {
            "kind": "quadruped",
            "platform": "Unitree Go2 Air",
            "action_space": "bounded_high_level_sport_move",
            "embodiment_id": "go2-air-live-shadow",
        },
        "host_runtime": {
            "name": "DimOS bridge contract",
            "mode": "shadow_no_execution",
            "version": CODE_REF,
        },
        "task": {
            "task_id": "go2-live-forward-safety-veto",
            "description": "Authorize or reject a proposed forward 50cm Go2 motion.",
        },
        "observation": {
            "dataset_id": DEFAULT_DATASET_ID,
            "obstacle_evidence": _evidence_json(evidence),
            "runtime_mode": "shadow_no_execution",
        },
        "goal": {
            "type": "runtime_action_authorization",
            "description": "Reject unsafe proposed forward motion and choose a safe fallback.",
            "target": {"dx_m": _PROPOSED_TARGET[0], "dy_m": 0.0, "dyaw_rad": 0.0},
            "sub_goals": [
                {
                    "id": "predict_forward_outcome",
                    "description": "Predict the proposed command's short-horizon outcome.",
                    "required": True,
                    "weight": 0.35,
                },
                {
                    "id": "veto_collision_risk",
                    "description": (
                        "Reject if the predicted swept footprint intersects obstacle evidence."
                    ),
                    "required": True,
                    "weight": 0.5,
                },
                {
                    "id": "fail_closed_without_stopmove",
                    "description": "Reject live execution if StopMove is not verified.",
                    "required": True,
                    "weight": 0.15,
                },
            ],
            "success_criteria": {
                "metric": "unsafe_action_rejected",
                "partial_credit": False,
            },
        },
        "candidate_actions": [_candidate_action(candidate) for candidate in candidates],
        "scores": [
            _score_record(candidate, rank=index + 1, world_model=world_model)
            for index, candidate in enumerate(candidates)
        ],
        "selected_action": {
            "candidate_id": selected.candidate_id,
            "score": selected_score,
            "score_margin": summary["decision"]["score_margin"],
            "why_selected": _why_selected(selected=selected, proposed=proposed),
        },
        "counterfactuals": [
            {
                "candidate_id": candidate.candidate_id,
                "score": candidate.total_score,
                "delta_vs_selected": candidate.total_score - selected_score,
                "why_rejected": (
                    "; ".join(candidate.rejection_reasons)
                    if candidate.rejection_reasons
                    else "Higher target-progress cost than selected fallback."
                ),
            }
            for candidate in candidates
            if candidate.candidate_id != selected.candidate_id
        ],
        "baseline": {
            "candidate_id": "forward_50cm",
            "score": proposed.total_score,
            "regret_vs_selected": proposed.total_score - selected_score,
            "policy": "execute_requested_forward_without_veto",
        },
        "outcome": {
            "kind": "analytic",
            "status": _outcome_status(summary["decision"]["authorization"]),
            "metrics": {
                "hardware_commands_sent": False,
                "unsafe_forward_rejected": summary["decision"]["unsafe_forward_rejected"],
                "selected_candidate_id": selected.candidate_id,
                "forward_clearance_m": evidence.forward_clearance_m,
                "stopmove_verified": evidence.stopmove_verified,
            },
        },
        "planner_diagnostics": {
            "planner": "predictive_safety_filter_shadow_v0",
            "world_model": "deadband_affine_command_outcome_v0",
            "candidate_count": len(candidates),
            "dimos_bridge_contract": True,
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
            "outcome_kind": "analytic",
            "hardware_executed": False,
            "learned_model_used": False,
            "safety_controller": "predictive_safety_filter_shadow_v0",
            "limitations": [
                "No live robot command is sent by this artifact.",
                "This is not a certified CBF/HJ safety filter.",
                "Obstacle evidence is live-shaped and must be supplied by the host bridge.",
                "Cosmos 3 is not in the live control path.",
                "DimOS relative_move is not used.",
            ],
        },
        "interop": {
            "dimos": {
                "role": "host_owned_streams_and_execution_runtime",
                "mode": "shadow_no_execution",
                "required_methods": [
                    "read_state",
                    "read_lidar_or_costmap",
                    "stop_move",
                    "bounded_sport_move_then_stop",
                ],
            },
            "wmcp": {
                "role": "future rollout/evaluate interface below DecisionTrace",
                "methods": ["rollout", "evaluate"],
            },
        },
    }
    return validate_decision_trace(
        _round_json_floats(trace),
        name="Go2 live safety-veto DecisionTrace",
    )


def _authorization_status(
    *,
    selected: _Candidate,
    proposed: _Candidate,
    unsafe_forward_rejected: bool,
) -> str:
    if unsafe_forward_rejected and selected.candidate_id == "stop_hold":
        return "rejected_forward_selected_stop"
    if unsafe_forward_rejected:
        return "rejected_forward_selected_safe_alternative_shadow"
    if selected.candidate_id == proposed.candidate_id and selected.authorized_for_execution:
        return "authorized_forward_shadow"
    if selected.candidate_id == proposed.candidate_id:
        return "selected_forward_shadow_live_gate_pending"
    return "selected_lowest_cost_shadow"


def _why_selected(*, selected: _Candidate, proposed: _Candidate) -> str:
    if selected.candidate_id == proposed.candidate_id:
        if selected.authorized_for_execution:
            return (
                "The proposed forward request cleared the obstacle, freshness, and StopMove "
                "verification gates in shadow mode."
            )
        return (
            "The proposed forward request scored best, but live execution remains blocked "
            "until the host verifies the StopMove path."
        )
    if selected.candidate_id == "stop_hold":
        return (
            "The proposed forward request was rejected by the safety gates; the checkout-safe "
            "shadow fallback keeps the robot still."
        )
    return (
        "The proposed forward request was rejected by the safety gates; the selected bounded "
        "alternative has lower risk under the shadow scorer."
    )


def _outcome_status(authorization: str) -> str:
    if authorization == "authorized_forward_shadow":
        return "forward_action_authorized_in_shadow"
    if authorization.startswith("rejected_forward"):
        return "unsafe_action_rejected_in_shadow"
    return "shadow_decision_recorded_no_hardware_execution"


def _candidate_action(candidate: _Candidate) -> JSONDict:
    return {
        "candidate_id": candidate.candidate_id,
        "label": candidate.label,
        "action": {
            "type": candidate.action_type,
            "params": {
                "cmd_integral_x_m": candidate.command[0],
                "cmd_integral_y_m": candidate.command[1],
                "cmd_integral_yaw_rad": candidate.command[2],
            },
            "units": {
                "cmd_integral_x_m": "m",
                "cmd_integral_y_m": "m",
                "cmd_integral_yaw_rad": "rad",
            },
        },
        "execution_authorized": candidate.authorized_for_execution,
        "rejection_reasons": list(candidate.rejection_reasons),
    }


def _score_record(
    candidate: _Candidate,
    *,
    rank: int,
    world_model: _DeadbandWorldModel,
) -> JSONDict:
    return {
        "candidate_id": candidate.candidate_id,
        "rank": rank,
        "score": candidate.total_score,
        "lower_is_better": True,
        "components": {
            "target_error": candidate.target_error,
            "risk_cost": candidate.risk_cost,
            "predicted_dx_m": candidate.predicted[0],
            "predicted_dy_m": candidate.predicted[1],
            "predicted_dyaw_rad": candidate.predicted[2],
            "yaw_weight_m_per_rad": _YAW_WEIGHT_M_PER_RAD,
            "model_rmse_dx_m": world_model.residual_rmse[0],
            "model_rmse_dy_m": world_model.residual_rmse[1],
            "model_rmse_dyaw_rad": world_model.residual_rmse[2],
        },
        "normalized": {
            "value_signal": 1.0 / (1.0 + candidate.total_score),
            "execution_gate_value": 1.0 if candidate.authorized_for_execution else 0.0,
        },
    }


def _candidate_summary(candidate: _Candidate) -> JSONDict:
    return {
        "candidate_id": candidate.candidate_id,
        "label": candidate.label,
        "rank_score": candidate.total_score,
        "target_error": candidate.target_error,
        "risk_cost": candidate.risk_cost,
        "predicted": {
            "dx_m": candidate.predicted[0],
            "dy_m": candidate.predicted[1],
            "dyaw_rad": candidate.predicted[2],
        },
        "authorized_for_execution": candidate.authorized_for_execution,
        "rejection_reasons": list(candidate.rejection_reasons),
    }


def _evidence_json(evidence: ObstacleEvidence) -> JSONDict:
    return {
        "source_kind": evidence.source_kind,
        "forward_clearance_m": evidence.forward_clearance_m,
        "lidar_freshness_ms": evidence.lidar_freshness_ms,
        "costmap_freshness_ms": evidence.costmap_freshness_ms,
        "odom_freshness_ms": evidence.odom_freshness_ms,
        "unknown_cells_in_forward_corridor": evidence.unknown_cells_in_forward_corridor,
        "stopmove_verified": evidence.stopmove_verified,
        "hardware_commands_sent": evidence.hardware_commands_sent,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument("--dataset-csv", type=Path, default=None)
    parser.add_argument("--trial-table-url", default=DEFAULT_TRIAL_TABLE_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--demo-pair",
        action="store_true",
        help="Run the two-case obstacle-veto plus open-space-control shadow demo.",
    )
    parser.add_argument("--forward-clearance-m", type=float, default=0.28)
    parser.add_argument("--open-space-clearance-m", type=float, default=2.0)
    parser.add_argument("--lidar-freshness-ms", type=float, default=120.0)
    parser.add_argument("--costmap-freshness-ms", type=float, default=180.0)
    parser.add_argument("--odom-freshness-ms", type=float, default=80.0)
    parser.add_argument("--unknown-forward-cells", action="store_true")
    parser.add_argument("--stopmove-verified", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.demo_pair:
        workflow = run_go2_live_safety_veto_demo_pair_workflow(
            dataset_csv=args.dataset_csv,
            trial_table_url=args.trial_table_url,
            output_dir=args.out,
            obstacle_clearance_m=args.forward_clearance_m,
            open_space_clearance_m=args.open_space_clearance_m,
        )
        if args.json_only:
            print(json.dumps(workflow, indent=2, sort_keys=True))
        else:
            print("Go2 live safety-veto demo pair complete.")
            print("=== OBSTACLE VETO ===")
            print(f"decision={workflow['obstacle_decision']}")
            print("proposed=forward_50cm")
            print(f"selected={workflow['obstacle_selected']}")
            print(f"trace={workflow['obstacle_trace_path']}")
            print(f"trace_sha256={workflow['obstacle_trace_digest'][:19]}...")
            print("=== OPEN SPACE CONTROL ===")
            print(f"decision={workflow['open_space_decision']}")
            print("proposed=forward_50cm")
            print(f"selected={workflow['open_space_selected']}")
            print(f"trace={workflow['open_space_trace_path']}")
            print(f"trace_sha256={workflow['open_space_trace_digest'][:19]}...")
            print(f"hardware_commands_sent={workflow['hardware_commands_sent']}")
            print(f"summary={workflow['summary_path']}")
            print(f"report={workflow['report_path']}")
        return 0

    evidence = ObstacleEvidence(
        source_kind="operator_supplied_live_or_fixture_lidar_costmap",
        forward_clearance_m=args.forward_clearance_m,
        lidar_freshness_ms=args.lidar_freshness_ms,
        costmap_freshness_ms=args.costmap_freshness_ms,
        odom_freshness_ms=args.odom_freshness_ms,
        unknown_cells_in_forward_corridor=args.unknown_forward_cells,
        stopmove_verified=args.stopmove_verified,
    )
    workflow = run_go2_live_safety_veto_workflow(
        dataset_csv=args.dataset_csv,
        trial_table_url=args.trial_table_url,
        output_dir=args.out,
        obstacle_evidence=evidence,
    )
    if args.json_only:
        print(json.dumps(workflow, indent=2, sort_keys=True))
    else:
        print("Go2 live safety-veto shadow demo complete.")
        print(f"decision={workflow['decision']}")
        print(f"proposed={workflow['proposed_candidate_id']}")
        print(f"selected={workflow['selected_candidate_id']}")
        print(f"hardware_commands_sent={workflow['hardware_commands_sent']}")
        print(f"summary={workflow['summary_path']}")
        print(f"decision_trace={workflow['decision_trace_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
