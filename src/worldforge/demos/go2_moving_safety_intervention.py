"""Go2 moving safety-intervention demo.

This module builds the next step after the shadow safety-veto demo: a
checkout-safe moving-intervention plan that predicts the displacement from a
tiny Go2 forward command, inflates it with a measured or fallback stop budget,
and emits chained DecisionTrace artifacts for continue, veto, hold, and resume
decisions.

The default workflow remains host-safe. It does not import DimOS or Unitree SDKs,
does not connect to hardware, and does not send movement commands. A host-owned
runtime can later inject measured execution receipts from DimOS, but the safety
decision and evidence contract are deterministic and testable here.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from worldforge.artifact_io import write_json_artifact
from worldforge.demos.go2_world_model_mpc import (
    DECISION_TRACE_SCHEMA_VERSION,
    DEFAULT_DATASET_ID,
    DEFAULT_TRIAL_TABLE_URL,
    _dataset_metadata,
    _DeadbandWorldModel,
    _finite_number,
    _fit_deadband_world_model,
    _json_list,
    _json_object,
    _positive_int,
    _require_fields,
    _round_json_floats,
    decision_trace_digest,
    load_controlbench_trials,
)
from worldforge.models import JSONDict, WorldForgeError, WorldStateError

DEFAULT_OUTPUT_DIR = Path(".worldforge/go2-moving-safety-intervention")
RUN_ID = "go2-moving-safety-intervention"
CODE_REF = "feat/go2-moving-safety-intervention"
DECISION_TRACE_ARTIFACT_KIND = "worldforge.go2_moving_safety_intervention_decision_trace"
CLI_DESCRIPTION = (
    "Run a checkout-safe Go2 moving safety-intervention plan and emit chained "
    "DecisionTrace artifacts for DimOS-hosted execution."
)

Mode = Literal[
    "dry-run",
    "stop-proof",
    "open-space-bounded-motion",
    "moving-intervention",
]

_MAX_ODOM_AGE_MS = 250.0
_MAX_LIDAR_OR_COSTMAP_AGE_MS = 500.0
_MAX_LIVE_OBSTACLE_LIDAR_AGE_MS = 200.0
_MAX_HEARTBEAT_AGE_MS = 200.0
_HOST_EXECUTION_RECEIPT_KIND = "dimos_bounded_motion_receipt_v1"
_RECEIPT_HMAC_FIELD = "hmac_sha256"


@dataclass(frozen=True, slots=True)
class Go2MovingSafetyResult:
    """Artifacts emitted by a moving safety-intervention run."""

    summary: JSONDict
    bridge_plan: JSONDict
    traces: list[JSONDict]
    report_markdown: str
    summary_path: Path
    bridge_path: Path
    trace_paths: list[Path]
    report_path: Path


@dataclass(frozen=True, slots=True)
class MovingSafetyConfig:
    """Conservative envelope for the moving demo."""

    vx_mps: float = 0.10
    vy_mps: float = 0.0
    wz_radps: float = 0.0
    chunk_duration_s: float = 0.15
    robot_half_extent_m: float = 0.35
    obstacle_margin_m: float = 0.20
    decision_budget_ms: float = 40.0
    command_rtt_ms: float = 180.0
    fallback_decel_time_s: float = 0.50
    fallback_decel_distance_m: float = 0.08
    resume_clear_frames_required: int = 5
    veto_clearance_hysteresis_m: float = 0.12
    firmware_obstacle_avoidance_backup_expected: bool = True


@dataclass(frozen=True, slots=True)
class StopProofMeasurement:
    """Measured stop-proof result for a demo speed."""

    available: bool = False
    stop_time_s: float | None = None
    stop_distance_m: float | None = None
    command_rtt_ms: float | None = None
    velocity_before_stop_mps: float | None = None
    velocity_after_stop_mps: float | None = None


@dataclass(frozen=True, slots=True)
class MovingObservation:
    """Live-shaped observation for one decision tick."""

    step_id: str
    label: str
    forward_clearance_m: float | None
    odom_freshness_ms: float | None
    lidar_freshness_ms: float | None
    costmap_freshness_ms: float | None
    heartbeat_freshness_ms: float | None
    measured_vx_mps: float
    unknown_cells_in_forward_corridor: bool = False
    deadman_held: bool = True
    stopmove_verified: bool = True
    clear_frames_seen: int = 0
    operator_resume_authorized: bool = False
    firmware_obstacle_avoidance_backup_enabled: bool = True
    hardware_execution_receipt: JSONDict | None = None
    measured_stop_time_s: float | None = None
    measured_stop_distance_m: float | None = None
    stopmove_ack: bool | None = None


@dataclass(frozen=True, slots=True)
class _Candidate:
    candidate_id: str
    action_type: str
    command: tuple[float, float, float, float]
    predicted: tuple[float, float, float]
    required_clearance_m: float
    score: float
    authorized: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _StepDecision:
    observation: MovingObservation
    candidates: list[_Candidate]
    selected: _Candidate
    safety_budget: JSONDict
    receipt_verification: JSONDict
    hardware_commands_sent: bool


def run_go2_moving_safety_intervention(
    *,
    mode: Mode = "moving-intervention",
    dataset_csv: Path | None = None,
    trial_table_url: str = DEFAULT_TRIAL_TABLE_URL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: MovingSafetyConfig | None = None,
    stop_proof: StopProofMeasurement | None = None,
    observations: list[MovingObservation] | None = None,
    receipt_hmac_key: str | bytes | None = None,
) -> Go2MovingSafetyResult:
    """Run the checkout-safe moving safety-intervention workflow."""

    cfg = config or MovingSafetyConfig()
    _validate_config(cfg)
    proof = stop_proof or StopProofMeasurement()
    trials = load_controlbench_trials(dataset_csv=dataset_csv, trial_table_url=trial_table_url)
    world_model = _fit_deadband_world_model(trials)
    sequence = observations or _default_observations(mode)
    decisions = [
        _decide_step(
            observation=observation,
            world_model=world_model,
            config=cfg,
            stop_proof=proof,
            receipt_hmac_key=receipt_hmac_key,
        )
        for observation in sequence
    ]
    traces = _build_trace_chain(
        mode=mode,
        dataset_csv=dataset_csv,
        trial_table_url=trial_table_url,
        decisions=decisions,
        world_model=world_model,
        config=cfg,
        stop_proof=proof,
    )
    summary = _build_summary(
        mode=mode,
        dataset_csv=dataset_csv,
        trial_table_url=trial_table_url,
        trial_count=len(trials),
        world_model=world_model,
        config=cfg,
        stop_proof=proof,
        decisions=decisions,
        traces=traces,
    )
    bridge_plan = _build_bridge_plan(mode=mode, summary=summary, config=cfg, stop_proof=proof)
    report = render_go2_moving_safety_report(summary)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorldStateError(
            "Go2 moving safety-intervention output directory could not be created."
        ) from exc

    try:
        summary_path = write_json_artifact(
            output_dir / "moving-safety-intervention-summary.json",
            summary,
        )
        bridge_path = write_json_artifact(
            output_dir / "dimos-moving-safety-bridge-plan.json",
            bridge_plan,
        )
        trace_paths = [
            write_json_artifact(
                output_dir
                / (f"decision-trace-go2-moving-safety-{index:02d}-{trace['trace_id']}.json"),
                trace,
            )
            for index, trace in enumerate(traces)
        ]
        report_path = output_dir / "moving-safety-intervention-report.md"
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        raise WorldStateError("Go2 moving safety-intervention artifact write failed.") from exc

    return Go2MovingSafetyResult(
        summary=summary,
        bridge_plan=bridge_plan,
        traces=traces,
        report_markdown=report,
        summary_path=summary_path,
        bridge_path=bridge_path,
        trace_paths=trace_paths,
        report_path=report_path,
    )


def run_go2_moving_safety_intervention_workflow(
    *,
    mode: Mode = "moving-intervention",
    dataset_csv: Path | None = None,
    trial_table_url: str = DEFAULT_TRIAL_TABLE_URL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> JSONDict:
    """Run the workflow and return a portable summary for hosts and docs."""

    result = run_go2_moving_safety_intervention(
        mode=mode,
        dataset_csv=dataset_csv,
        trial_table_url=trial_table_url,
        output_dir=output_dir,
    )
    return {
        "summary_path": str(result.summary_path),
        "bridge_path": str(result.bridge_path),
        "report_path": str(result.report_path),
        "trace_paths": [str(path) for path in result.trace_paths],
        "mode": result.summary["mode"],
        "decision_sequence": [
            step["decision"]["selected_candidate_id"] for step in result.summary["steps"]
        ],
        "hardware_commands_sent": result.summary["runtime"]["hardware_commands_sent"],
    }


def build_dimos_execution_receipt(
    *,
    step_id: str,
    command: tuple[float, float, float, float],
    issued_at_unix_s: float,
    stopmove_ack: bool,
    measured_stop_time_s: float,
    measured_stop_distance_m: float,
    outcome_source: str,
    receipt_hmac_key: str | bytes,
) -> JSONDict:
    """Build a host-owned HMAC receipt for measured DimOS execution."""

    payload = _receipt_payload(
        step_id=step_id,
        command=command,
        issued_at_unix_s=issued_at_unix_s,
        stopmove_ack=stopmove_ack,
        measured_stop_time_s=measured_stop_time_s,
        measured_stop_distance_m=measured_stop_distance_m,
        outcome_source=outcome_source,
    )
    payload[_RECEIPT_HMAC_FIELD] = _receipt_hmac(payload, receipt_hmac_key)
    return payload


def render_go2_moving_safety_report(summary: JSONDict) -> str:
    """Render a compact Markdown report."""

    lines = [
        "# Go2 Moving Safety Intervention",
        "",
        "## Claim Boundary",
        "",
        "- Predictive safety-filter inspired demo: `true`",
        "- Certified safety controller: `false`",
        f"- Hardware commands sent: `{summary['runtime']['hardware_commands_sent']}`",
        f"- Outcome kind: `{summary['claim_boundary']['outcome_kind']}`",
        "- Cosmos 3 in control path: `false`",
        "- Autonomous navigation: `false`",
        "",
        "## Safety Budget",
        "",
        f"- Required clearance budget: `{summary['safety_budget']['required_clearance_m']}` m",
        f"- Model prediction scope: `{summary['world_model']['prediction_scope']}`",
        f"- Model residual dx RMSE: `{summary['world_model']['residual_rmse']['dx_m']}` m",
        f"- Stop budget source: `{summary['safety_budget']['stop_budget_source']}`",
        f"- Stop-proof available: `{summary['stop_proof']['available']}`",
        "",
        "## Sequence",
        "",
    ]
    for step in summary["steps"]:
        lines.extend(
            [
                f"### {step['step_id']}",
                "",
                f"- Clearance: `{step['observation']['forward_clearance_m']}` m",
                f"- Selected: `{step['decision']['selected_candidate_id']}`",
                f"- Authorization: `{step['decision']['authorization']}`",
                f"- Primary reason: `{step['decision']['primary_reason']}`",
                f"- Trace digest: `{step['decision_trace_digest']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _validate_config(config: MovingSafetyConfig) -> None:
    if config.vx_mps <= 0.0 or config.vx_mps > 0.10:
        raise WorldStateError("Go2 moving safety demo vx_mps must be in (0, 0.10].")
    if config.chunk_duration_s <= 0.0 or config.chunk_duration_s > 0.15:
        raise WorldStateError("Go2 moving safety demo chunk_duration_s must be in (0, 0.15].")
    if config.robot_half_extent_m <= 0.0 or config.obstacle_margin_m <= 0.0:
        raise WorldStateError("Go2 moving safety demo footprint and margin must be positive.")


def _default_observations(mode: Mode) -> list[MovingObservation]:
    if mode == "dry-run":
        return [
            MovingObservation(
                step_id="dry_run_obstacle_veto",
                label="Obstacle present; no execution.",
                forward_clearance_m=0.45,
                odom_freshness_ms=80.0,
                lidar_freshness_ms=120.0,
                costmap_freshness_ms=160.0,
                heartbeat_freshness_ms=80.0,
                measured_vx_mps=0.0,
            )
        ]
    if mode == "stop-proof":
        return [
            MovingObservation(
                step_id="stop_proof_plan",
                label="Stop-proof ladder step; host must measure stop before live demo.",
                forward_clearance_m=2.0,
                odom_freshness_ms=80.0,
                lidar_freshness_ms=120.0,
                costmap_freshness_ms=160.0,
                heartbeat_freshness_ms=80.0,
                measured_vx_mps=0.0,
                operator_resume_authorized=True,
                clear_frames_seen=5,
            )
        ]
    if mode == "open-space-bounded-motion":
        return [
            MovingObservation(
                step_id="open_space_continue",
                label="Open corridor; tiny bounded forward chunk authorized in shadow.",
                forward_clearance_m=2.0,
                odom_freshness_ms=80.0,
                lidar_freshness_ms=120.0,
                costmap_freshness_ms=160.0,
                heartbeat_freshness_ms=80.0,
                measured_vx_mps=0.0,
                operator_resume_authorized=True,
                clear_frames_seen=5,
            )
        ]
    return [
        MovingObservation(
            step_id="moving_clear",
            label="Path clear; keep moving in tiny chunks.",
            forward_clearance_m=2.0,
            odom_freshness_ms=80.0,
            lidar_freshness_ms=120.0,
            costmap_freshness_ms=160.0,
            heartbeat_freshness_ms=80.0,
            measured_vx_mps=0.08,
            operator_resume_authorized=True,
            clear_frames_seen=5,
        ),
        MovingObservation(
            step_id="obstacle_veto",
            label="Obstacle enters the forward corridor; veto continued motion.",
            forward_clearance_m=0.45,
            odom_freshness_ms=70.0,
            lidar_freshness_ms=100.0,
            costmap_freshness_ms=150.0,
            heartbeat_freshness_ms=70.0,
            measured_vx_mps=0.08,
            stopmove_ack=True,
        ),
        MovingObservation(
            step_id="hold_blocked",
            label="Obstacle still present; hold stopped.",
            forward_clearance_m=0.42,
            odom_freshness_ms=70.0,
            lidar_freshness_ms=100.0,
            costmap_freshness_ms=150.0,
            heartbeat_freshness_ms=70.0,
            measured_vx_mps=0.0,
            stopmove_ack=True,
        ),
        MovingObservation(
            step_id="resume_clear",
            label="Obstacle removed; resume only after clear hysteresis and operator approval.",
            forward_clearance_m=2.0,
            odom_freshness_ms=70.0,
            lidar_freshness_ms=100.0,
            costmap_freshness_ms=150.0,
            heartbeat_freshness_ms=70.0,
            measured_vx_mps=0.0,
            operator_resume_authorized=True,
            clear_frames_seen=5,
        ),
    ]


def _decide_step(
    *,
    observation: MovingObservation,
    world_model: _DeadbandWorldModel,
    config: MovingSafetyConfig,
    stop_proof: StopProofMeasurement,
    receipt_hmac_key: str | bytes | None,
) -> _StepDecision:
    forward_command = (config.vx_mps, config.vy_mps, config.wz_radps, config.chunk_duration_s)
    receipt_verification = _verify_host_execution_receipt(
        observation.hardware_execution_receipt,
        receipt_hmac_key=receipt_hmac_key,
        expected_step_id=observation.step_id,
        expected_command=forward_command,
    )
    hardware_commands_sent = bool(receipt_verification["verified"])
    forward_integral = (
        config.vx_mps * config.chunk_duration_s,
        config.vy_mps * config.chunk_duration_s,
        config.wz_radps * config.chunk_duration_s,
    )
    predicted_forward = world_model.predict_outcome(forward_integral)
    budget = _safety_budget(
        predicted=predicted_forward,
        observation=observation,
        config=config,
        stop_proof=stop_proof,
        model_rmse_dx_m=world_model.residual_rmse[0],
    )
    forward_reasons = _forward_rejection_reasons(
        observation=observation,
        budget=budget,
        config=config,
        stop_proof=stop_proof,
        hardware_commands_sent=hardware_commands_sent,
    )
    forward_authorized = not forward_reasons
    forward_score = 0.0 if forward_authorized else 100.0 + len(forward_reasons)
    stop_score = 0.0 if not forward_authorized else 10.0
    candidates = [
        _Candidate(
            candidate_id="continue_forward_chunk",
            action_type="bounded_sport_move_chunk",
            command=forward_command,
            predicted=predicted_forward,
            required_clearance_m=budget["required_clearance_m"],
            score=forward_score,
            authorized=forward_authorized,
            rejection_reasons=tuple(forward_reasons),
        ),
        _Candidate(
            candidate_id="stop_move",
            action_type="stop_move",
            command=(0.0, 0.0, 0.0, 0.0),
            predicted=(0.0, 0.0, 0.0),
            required_clearance_m=0.0,
            score=stop_score,
            authorized=observation.stopmove_verified,
            rejection_reasons=(
                () if observation.stopmove_verified else ("stopmove_path_not_verified",)
            ),
        ),
    ]
    ranked = sorted(candidates, key=lambda candidate: (candidate.score, candidate.candidate_id))
    return _StepDecision(
        observation=observation,
        candidates=ranked,
        selected=ranked[0],
        safety_budget=budget,
        receipt_verification=receipt_verification,
        hardware_commands_sent=hardware_commands_sent,
    )


def _forward_rejection_reasons(
    *,
    observation: MovingObservation,
    budget: JSONDict,
    config: MovingSafetyConfig,
    stop_proof: StopProofMeasurement,
    hardware_commands_sent: bool,
) -> list[str]:
    reasons: list[str] = []
    if _age_or_inf(observation.odom_freshness_ms) > _MAX_ODOM_AGE_MS:
        reasons.append("stale_odom")
    lidar_age = _age_or_inf(observation.lidar_freshness_ms)
    costmap_age = _age_or_inf(observation.costmap_freshness_ms)
    if lidar_age > _MAX_LIDAR_OR_COSTMAP_AGE_MS and costmap_age > _MAX_LIDAR_OR_COSTMAP_AGE_MS:
        reasons.append("stale_lidar_and_costmap")
    if _age_or_inf(observation.heartbeat_freshness_ms) > _MAX_HEARTBEAT_AGE_MS:
        reasons.append("stale_authorization_heartbeat")
    if not observation.deadman_held:
        reasons.append("operator_deadman_not_held")
    if not observation.stopmove_verified:
        reasons.append("stopmove_path_not_verified")
    if config.firmware_obstacle_avoidance_backup_expected and (
        not observation.firmware_obstacle_avoidance_backup_enabled
    ):
        reasons.append("firmware_obstacle_avoidance_backup_not_enabled")
    if observation.unknown_cells_in_forward_corridor:
        reasons.append("unknown_cells_in_forward_corridor")
    clearance = _positive_measurement_or_none(observation.forward_clearance_m)
    resume_threshold = budget["required_clearance_m"] + config.veto_clearance_hysteresis_m
    if clearance is None:
        reasons.append("missing_forward_clearance")
    elif clearance < budget["required_clearance_m"]:
        reasons.append("clearance_budget_intersects_obstacle_zone")
    elif clearance < resume_threshold:
        reasons.append("clearance_hysteresis_band_hold")
    else:
        if observation.clear_frames_seen < config.resume_clear_frames_required:
            reasons.append("resume_clear_hysteresis_not_satisfied")
        if not observation.operator_resume_authorized:
            reasons.append("operator_resume_not_authorized")
    if hardware_commands_sent and not stop_proof.available:
        reasons.append("live_motion_without_stop_proof_measurement")
    return reasons


def _safety_budget(
    *,
    predicted: tuple[float, float, float],
    observation: MovingObservation,
    config: MovingSafetyConfig,
    stop_proof: StopProofMeasurement,
    model_rmse_dx_m: float,
) -> JSONDict:
    perception_age_ms = min(
        _age_or_inf(observation.lidar_freshness_ms),
        _age_or_inf(observation.costmap_freshness_ms),
    )
    if not math.isfinite(perception_age_ms):
        perception_age_ms = _MAX_LIDAR_OR_COSTMAP_AGE_MS
    command_rtt_ms = (
        stop_proof.command_rtt_ms
        if _finite_positive(stop_proof.command_rtt_ms)
        else config.command_rtt_ms
    )
    stop_time_s = (
        stop_proof.stop_time_s
        if _finite_positive(stop_proof.stop_time_s)
        else config.fallback_decel_time_s
    )
    stop_distance_m = (
        stop_proof.stop_distance_m
        if _finite_positive(stop_proof.stop_distance_m)
        else config.fallback_decel_distance_m
    )
    latency_distance_m = config.vx_mps * (
        (perception_age_ms + config.decision_budget_ms + command_rtt_ms) / 1000.0
    )
    predicted_forward_m = max(float(predicted[0]), 0.0)
    required = (
        predicted_forward_m
        + latency_distance_m
        + stop_distance_m
        + config.robot_half_extent_m
        + config.obstacle_margin_m
        + abs(model_rmse_dx_m)
    )
    return _round_json_floats(
        {
            "predicted_forward_dx_m": predicted_forward_m,
            "perception_age_ms": perception_age_ms,
            "decision_budget_ms": config.decision_budget_ms,
            "command_rtt_ms": command_rtt_ms,
            "latency_distance_m": latency_distance_m,
            "stop_time_s": stop_time_s,
            "stop_distance_m": stop_distance_m,
            "stop_budget_source": (
                "measured_stop_proof"
                if stop_proof.available
                else "conservative_fallback_unmeasured"
            ),
            "robot_half_extent_m": config.robot_half_extent_m,
            "obstacle_margin_m": config.obstacle_margin_m,
            "model_rmse_dx_m": abs(model_rmse_dx_m),
            "required_clearance_m": required,
            "stop_proof_measurement_available": stop_proof.available,
            "model_prediction_scope": "command_displacement_only",
            "external_gt": "pending",
        }
    )


def _build_summary(
    *,
    mode: Mode,
    dataset_csv: Path | None,
    trial_table_url: str,
    trial_count: int,
    world_model: _DeadbandWorldModel,
    config: MovingSafetyConfig,
    stop_proof: StopProofMeasurement,
    decisions: list[_StepDecision],
    traces: list[JSONDict],
) -> JSONDict:
    any_hardware = any(decision.hardware_commands_sent for decision in decisions)
    any_real_measured = any(
        _decision_outcome_kind(decision) == "real_measured" for decision in decisions
    )
    dataset_metadata = _dataset_metadata(dataset_csv=dataset_csv, trial_table_url=trial_table_url)
    return _round_json_floats(
        {
            "schema_version": 1,
            "artifact_kind": "worldforge.go2_moving_safety_intervention_summary",
            "run_id": RUN_ID,
            "mode": mode,
            "dataset": {
                **dataset_metadata,
                "trial_count": trial_count,
                "outcome_source": "native_go2_odom_signed_body_projection_public_preview",
            },
            "runtime": {
                "name": "DimOS bridge contract",
                "mode": "shadow_no_execution" if not any_hardware else "host_receipt_supplied",
                "imports_dimos": False,
                "hardware_commands_sent": any_hardware,
            },
            "world_model": {
                "model_family": "deadband_affine_command_outcome_v0",
                "learned_model_used": False,
                "score_kind": "hand_cost",
                "prediction_scope": "command_displacement_only",
                "calibration_source": "native_go2_odom_public_preview",
                "external_gt": "pending",
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
            "control_envelope": {
                "vx_mps": config.vx_mps,
                "chunk_duration_s": config.chunk_duration_s,
                "fail_safe_default": "stopped_between_chunks",
                "long_continuous_move_allowed": False,
                "rage_mode_allowed": False,
            },
            "stop_proof": _stop_proof_json(stop_proof),
            "stop_proof_protocol": _stop_proof_protocol_json(config),
            "safety_budget": decisions[0].safety_budget if decisions else {},
            "steps": [
                _step_summary(decision=decision, trace=trace)
                for decision, trace in zip(decisions, traces, strict=True)
            ],
            "claim_boundary": {
                "predictive_safety_filter_demo": True,
                "certified_safety": False,
                "outcome_kind": "real_measured" if any_real_measured else "analytic",
                "live_dimos_execution": any_hardware,
                "cosmos3_in_control_path": False,
                "relative_move_used": False,
                "autonomous_navigation": False,
            },
        }
    )


def _build_bridge_plan(
    *,
    mode: Mode,
    summary: JSONDict,
    config: MovingSafetyConfig,
    stop_proof: StopProofMeasurement,
) -> JSONDict:
    return _round_json_floats(
        {
            "schema_version": 1,
            "artifact_kind": "worldforge.dimos_go2_moving_safety_bridge_plan",
            "run_id": summary["run_id"],
            "mode": mode,
            "runtime_boundary": {
                "worldforge": "scores candidates, authorizes or vetoes, emits DecisionTrace",
                "dimos": "host-owned streams, heartbeat watchdog, Move and StopMove execution",
                "imports_dimos_in_worldforge_demo": False,
            },
            "required_bridge_methods": {
                "read_odometry": "UnitreeGo2TwistAdapter.read_odometry",
                "read_velocities": "UnitreeGo2TwistAdapter.read_velocities",
                "write_velocities": "UnitreeGo2TwistAdapter.write_velocities",
                "write_stop": "UnitreeGo2TwistAdapter.write_stop -> SportClient.StopMove",
                "read_lidar_or_costmap": "DimOS lidar stream or CostMapper.global_costmap",
                "positive_heartbeat_watchdog": "host thread sends StopMove without WorldForge",
            },
            "execution_receipt_contract": {
                "receipt_kind": _HOST_EXECUTION_RECEIPT_KIND,
                "requires_hmac_sha256": True,
                "host_verification_key_not_serialized": True,
                "must_match_step_id_and_command": True,
                "must_include_stopmove_ack_and_measured_stop_outcome": True,
                "copyable_string_receipts_are_rejected": True,
            },
            "live_perception_contract": {
                "forward_clearance_wired_in_this_checkout_demo": False,
                "host_must_supply_forward_clearance": True,
                "real_obstacle_demo_max_lidar_age_ms": _MAX_LIVE_OBSTACLE_LIDAR_AGE_MS,
                "web_rtc_lidar_rate_may_be_too_low_for_real_obstacle": True,
            },
            "bounded_motion_contract": {
                "max_vx_mps": config.vx_mps,
                "max_chunk_duration_s": config.chunk_duration_s,
                "stop_between_chunks": True,
                "stopmove_on_exception": True,
                "stopmove_on_stale_sensor": True,
                "operator_deadman_required": True,
                "rage_mode_must_be_disabled": True,
                "firmware_obstacle_avoidance_backup_expected": (
                    config.firmware_obstacle_avoidance_backup_expected
                ),
            },
            "hard_gates": {
                "stop_proof_required_before_live_intervention": True,
                "stop_proof_available": stop_proof.available,
                "stopping_distance_measurement_required": True,
                "live_forward_clearance_wiring_required": True,
                "rage_mode_disabled_required": True,
                "max_odom_age_ms": _MAX_ODOM_AGE_MS,
                "max_lidar_or_costmap_age_ms": _MAX_LIDAR_OR_COSTMAP_AGE_MS,
                "max_live_obstacle_lidar_age_ms": _MAX_LIVE_OBSTACLE_LIDAR_AGE_MS,
                "max_heartbeat_age_ms": _MAX_HEARTBEAT_AGE_MS,
                "missing_zero_or_nonfinite_freshness_is_stale": True,
                "unknown_forward_cells_reject": True,
                "resume_requires_hysteresis_and_operator_authorization": True,
            },
        }
    )


def _build_trace_chain(
    *,
    mode: Mode,
    dataset_csv: Path | None,
    trial_table_url: str,
    decisions: list[_StepDecision],
    world_model: _DeadbandWorldModel,
    config: MovingSafetyConfig,
    stop_proof: StopProofMeasurement,
) -> list[JSONDict]:
    traces: list[JSONDict] = []
    prev_trace_id: str | None = None
    for index, decision in enumerate(decisions):
        trace_id = f"go2-moving-safety-{index:02d}-{decision.observation.step_id}"
        trace = _build_decision_trace(
            mode=mode,
            trace_id=trace_id,
            prev_trace_id=prev_trace_id,
            step_index=index,
            dataset_csv=dataset_csv,
            trial_table_url=trial_table_url,
            decision=decision,
            world_model=world_model,
            config=config,
            stop_proof=stop_proof,
        )
        traces.append(trace)
        prev_trace_id = trace_id
    return traces


def _build_decision_trace(
    *,
    mode: Mode,
    trace_id: str,
    prev_trace_id: str | None,
    step_index: int,
    dataset_csv: Path | None,
    trial_table_url: str,
    decision: _StepDecision,
    world_model: _DeadbandWorldModel,
    config: MovingSafetyConfig,
    stop_proof: StopProofMeasurement,
) -> JSONDict:
    observation = decision.observation
    selected = decision.selected
    scores = [
        _score_record(candidate, rank=index + 1)
        for index, candidate in enumerate(decision.candidates)
    ]
    outcome_kind = _decision_outcome_kind(decision)
    trace: JSONDict = {
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "artifact_kind": DECISION_TRACE_ARTIFACT_KIND,
        "trace_id": trace_id,
        "run_id": RUN_ID,
        "step_index": step_index,
        "prev_trace_id": prev_trace_id,
        "embodiment": {
            "kind": "quadruped",
            "platform": "Unitree Go2 Air",
            "action_space": "bounded_high_level_sport_move_chunks",
        },
        "host_runtime": {
            "name": "DimOS",
            "mode": "shadow_no_execution"
            if not decision.hardware_commands_sent
            else "host_receipt_supplied",
            "adapter": "UnitreeGo2TwistAdapter",
        },
        "task": {
            "task_id": f"go2_moving_safety_{mode}",
            "description": "Authorize or veto a tiny forward Go2 command before collision risk.",
        },
        "observation": {
            "step_id": observation.step_id,
            "label": observation.label,
            "forward_clearance_m": _positive_measurement_or_none(observation.forward_clearance_m),
            "measured_vx_mps": observation.measured_vx_mps,
            "freshness_ms": {
                "odom": _freshness_value_or_none(observation.odom_freshness_ms),
                "lidar": _freshness_value_or_none(observation.lidar_freshness_ms),
                "costmap": _freshness_value_or_none(observation.costmap_freshness_ms),
                "heartbeat": _freshness_value_or_none(observation.heartbeat_freshness_ms),
            },
            "deadman_held": observation.deadman_held,
            "stopmove_verified": observation.stopmove_verified,
            "operator_resume_authorized": observation.operator_resume_authorized,
            "clear_frames_seen": observation.clear_frames_seen,
            "firmware_obstacle_avoidance_backup_enabled": (
                observation.firmware_obstacle_avoidance_backup_enabled
            ),
            "external_gt": "pending",
        },
        "goal": {
            "type": "safety_filter",
            "description": (
                "Continue moving only while the inflated clearance budget remains clear."
            ),
            "sub_goals": [
                {
                    "id": "avoid_collision",
                    "description": "Keep obstacle outside the inflated clearance budget.",
                },
                {
                    "id": "minimal_intervention",
                    "description": "Continue tiny motion only when safe.",
                },
            ],
            "success_criteria": {
                "metric": "inflated_clearance_budget_and_stop_ack",
                "partial_credit": True,
            },
        },
        "candidate_actions": [_candidate_action(candidate) for candidate in decision.candidates],
        "scores": scores,
        "selected_action": {
            "candidate_id": selected.candidate_id,
            "score": selected.score,
            "score_margin": _score_margin(scores),
            "why_selected": _why_selected(selected),
        },
        "counterfactuals": [
            {
                "candidate_id": candidate.candidate_id,
                "score": candidate.score,
                "delta_vs_selected": candidate.score - selected.score,
                "why_rejected": _why_rejected(candidate),
            }
            for candidate in decision.candidates
            if candidate.candidate_id != selected.candidate_id
        ],
        "candidate_outcomes": [
            _candidate_outcome(candidate, decision=decision) for candidate in decision.candidates
        ],
        "baseline": {
            "candidate_id": "continue_forward_chunk",
            "score": next(
                candidate.score
                for candidate in decision.candidates
                if candidate.candidate_id == "continue_forward_chunk"
            ),
            "regret_vs_selected": next(
                candidate.score
                for candidate in decision.candidates
                if candidate.candidate_id == "continue_forward_chunk"
            )
            - selected.score,
            "description": "Naive baseline keeps moving forward unless explicitly vetoed.",
        },
        "outcome": {
            "kind": outcome_kind,
            "status": _outcome_status(selected=selected, decision=decision),
            "metrics": {
                "hardware_commands_sent": decision.hardware_commands_sent,
                "stopmove_ack": _outcome_stopmove_ack(decision),
                "measured_stop_time_s": _outcome_measured_stop_time_s(decision),
                "measured_stop_distance_m": _outcome_measured_stop_distance_m(decision),
                "selected_action": selected.candidate_id,
                "outcome_source": _outcome_source(decision),
                "builtin_obstacle_avoidance_enabled": (
                    observation.firmware_obstacle_avoidance_backup_enabled
                ),
                "stop_attribution": _stop_attribution(selected=selected, decision=decision),
            },
        },
        "planner_diagnostics": {
            "safety_budget": decision.safety_budget,
            "execution_receipt_verification": decision.receipt_verification,
            "world_model": {
                "model_family": "deadband_affine_command_outcome_v0",
                "prediction_scope": "command_displacement_only",
                "calibration_source": "native_go2_odom_public_preview",
                "external_gt": "pending",
                "thresholds": world_model.thresholds,
                "residual_rmse": world_model.residual_rmse,
            },
            "stop_proof": _stop_proof_json(stop_proof),
        },
        "reproducibility": {
            "provider_version": None,
            "checkpoint_hash": None,
            "model_card_ref": DEFAULT_DATASET_ID,
            "input_digest": _input_digest(
                mode=mode,
                observation=observation,
                config=config,
                dataset_csv=dataset_csv,
                trial_table_url=trial_table_url,
            ),
            "seed": 0,
            "code_ref": CODE_REF,
        },
        "claim_boundary": {
            "score_kind": "hand_cost",
            "outcome_kind": outcome_kind,
            "hardware_executed": decision.hardware_commands_sent,
            "learned_model_used": False,
            "safety_controller": "predictive_safety_filter_inspired_moving_v0",
            "builtin_obstacle_avoidance_enabled": (
                observation.firmware_obstacle_avoidance_backup_enabled
            ),
            "outcome_source": _outcome_source(decision),
            "limitations": [
                "Not a certified predictive safety filter, CBF, or HJ reachability controller.",
                "Uses transparent ControlBench deadband-affine command-outcome model.",
                "DimOS and the operator remain responsible for live execution and emergency stop.",
                "Cosmos 3 is not in the live control path.",
            ],
        },
        "interop": {
            "dimos": {
                "role": "host_owned_streams_heartbeat_and_execution_runtime",
                "adapter": "UnitreeGo2TwistAdapter",
                "required_stop": "write_stop -> SportClient.StopMove",
                "live_forward_clearance": "host_supplied_not_wired_in_checkout_demo",
                "rage_mode": "must_remain_disabled",
            },
            "wmcp": {
                "role": "future rollout/evaluate protocol below DecisionTrace",
                "used_in_this_demo": False,
            },
        },
    }
    return validate_decision_trace(_round_json_floats(trace))


def validate_decision_trace(
    payload: object,
    *,
    name: str = "Go2 moving safety-intervention DecisionTrace",
) -> JSONDict:
    """Validate the Go2 moving safety-intervention DecisionTrace artifact shape."""

    trace = _json_object(payload, name)
    _require_fields(
        trace,
        (
            "schema_version",
            "artifact_kind",
            "trace_id",
            "run_id",
            "host_runtime",
            "observation",
            "goal",
            "candidate_actions",
            "scores",
            "selected_action",
            "counterfactuals",
            "candidate_outcomes",
            "outcome",
            "planner_diagnostics",
            "claim_boundary",
            "interop",
        ),
        name,
    )
    if trace["schema_version"] != DECISION_TRACE_SCHEMA_VERSION:
        raise WorldForgeError(f"{name}.schema_version must be {DECISION_TRACE_SCHEMA_VERSION}.")
    if trace["artifact_kind"] != DECISION_TRACE_ARTIFACT_KIND:
        raise WorldForgeError(f"{name}.artifact_kind must be {DECISION_TRACE_ARTIFACT_KIND}.")

    candidates = _json_list(trace["candidate_actions"], f"{name}.candidate_actions")
    scores = _json_list(trace["scores"], f"{name}.scores")
    outcomes = _json_list(trace["candidate_outcomes"], f"{name}.candidate_outcomes")
    if not candidates or len(candidates) != len(scores) or len(candidates) != len(outcomes):
        raise WorldForgeError(
            f"{name} candidate_actions, scores, and candidate_outcomes must align."
        )

    candidate_ids = {
        str(_json_object(candidate, f"{name}.candidate_actions[{index}]")["candidate_id"])
        for index, candidate in enumerate(candidates)
    }
    selected = _json_object(trace["selected_action"], f"{name}.selected_action")
    selected_id = str(selected.get("candidate_id", ""))
    if selected_id not in candidate_ids:
        raise WorldForgeError(f"{name}.selected_action.candidate_id must match a candidate.")

    for index, score in enumerate(scores):
        score_record = _json_object(score, f"{name}.scores[{index}]")
        if str(score_record.get("candidate_id", "")) not in candidate_ids:
            raise WorldForgeError(f"{name}.scores[{index}].candidate_id must match a candidate.")
        _finite_number(score_record.get("score"), f"{name}.scores[{index}].score")
        _positive_int(score_record.get("rank"), f"{name}.scores[{index}].rank")

    for index, outcome in enumerate(outcomes):
        outcome_record = _json_object(outcome, f"{name}.candidate_outcomes[{index}]")
        if str(outcome_record.get("candidate_id", "")) not in candidate_ids:
            raise WorldForgeError(
                f"{name}.candidate_outcomes[{index}].candidate_id must match a candidate."
            )

    outcome = _json_object(trace["outcome"], f"{name}.outcome")
    metrics = _json_object(outcome["metrics"], f"{name}.outcome.metrics")
    claim_boundary = _json_object(trace["claim_boundary"], f"{name}.claim_boundary")
    hardware_executed = claim_boundary.get("hardware_executed")
    if not isinstance(hardware_executed, bool):
        raise WorldForgeError(f"{name}.claim_boundary.hardware_executed must be boolean.")
    if metrics.get("hardware_commands_sent") is not hardware_executed:
        raise WorldForgeError(
            f"{name}.outcome.metrics.hardware_commands_sent must match claim boundary."
        )
    if hardware_executed:
        if outcome.get("kind") != "real_measured":
            raise WorldForgeError(f"{name}.outcome.kind must be real_measured when executed.")
        if claim_boundary.get("outcome_kind") != "real_measured":
            raise WorldForgeError(
                f"{name}.claim_boundary.outcome_kind must be real_measured when executed."
            )
        if metrics.get("stopmove_ack") is not True:
            raise WorldForgeError(f"{name}.outcome.metrics.stopmove_ack must be true.")
        _finite_number(
            metrics.get("measured_stop_time_s"),
            f"{name}.outcome.metrics.measured_stop_time_s",
        )
        _finite_number(
            metrics.get("measured_stop_distance_m"),
            f"{name}.outcome.metrics.measured_stop_distance_m",
        )
        diagnostics = _json_object(trace["planner_diagnostics"], f"{name}.planner_diagnostics")
        receipt = _json_object(
            diagnostics["execution_receipt_verification"],
            f"{name}.planner_diagnostics.execution_receipt_verification",
        )
        if receipt.get("verified") is not True or receipt.get("measurement_present") is not True:
            raise WorldForgeError(f"{name}.receipt verification must prove measured execution.")
    elif outcome.get("kind") != "analytic":
        raise WorldForgeError(f"{name}.outcome.kind must be analytic when not executed.")
    return trace


def _candidate_action(candidate: _Candidate) -> JSONDict:
    vx, vy, wz, duration = candidate.command
    return {
        "candidate_id": candidate.candidate_id,
        "action": {
            "type": candidate.action_type,
            "params": {
                "vx": vx,
                "vy": vy,
                "wz": wz,
                "duration_s": duration,
            },
            "units": {
                "vx": "m/s",
                "vy": "m/s",
                "wz": "rad/s",
                "duration_s": "s",
            },
        },
    }


def _candidate_outcome(candidate: _Candidate, *, decision: _StepDecision) -> JSONDict:
    real_executed = decision.hardware_commands_sent and (
        candidate.candidate_id == decision.selected.candidate_id
    )
    return {
        "candidate_id": candidate.candidate_id,
        "kind": _decision_outcome_kind(decision) if real_executed else "analytic",
        "status": (
            _outcome_status(selected=candidate, decision=decision)
            if real_executed
            else ("shadow_authorized" if candidate.authorized else "shadow_rejected_by_safety_gate")
        ),
        "outcome_source": _outcome_source(decision) if real_executed else "analytic_shadow",
        "action_executed": real_executed,
        "predicted": {
            "dx_m": candidate.predicted[0],
            "dy_m": candidate.predicted[1],
            "dyaw_rad": candidate.predicted[2],
        },
        "metrics": {
            "required_clearance_m": candidate.required_clearance_m,
            "score": candidate.score,
            "authorized": candidate.authorized,
            "rejection_reasons": list(candidate.rejection_reasons),
            "measured_stop_time_s": (
                _outcome_measured_stop_time_s(decision) if real_executed else None
            ),
            "measured_stop_distance_m": (
                _outcome_measured_stop_distance_m(decision) if real_executed else None
            ),
            "stopmove_ack": _outcome_stopmove_ack(decision) if real_executed else None,
        },
    }


def _score_record(candidate: _Candidate, *, rank: int) -> JSONDict:
    value_signal = 1.0 / (1.0 + candidate.score)
    return {
        "candidate_id": candidate.candidate_id,
        "rank": rank,
        "score": candidate.score,
        "lower_is_better": True,
        "components": {
            "authorized": candidate.authorized,
            "risk_cost": candidate.score,
            "predicted_dx_m": candidate.predicted[0],
            "predicted_dy_m": candidate.predicted[1],
            "predicted_dyaw_rad": candidate.predicted[2],
            "required_clearance_m": candidate.required_clearance_m,
            "rejection_reasons": list(candidate.rejection_reasons),
        },
        "normalized": {
            "value_signal": value_signal,
            "calibration_target": "controlbench_command_displacement_plus_stop_budget",
        },
    }


def _step_summary(*, decision: _StepDecision, trace: JSONDict) -> JSONDict:
    forward = next(
        candidate
        for candidate in decision.candidates
        if candidate.candidate_id == "continue_forward_chunk"
    )
    selected = decision.selected
    return {
        "step_id": decision.observation.step_id,
        "observation": {
            "forward_clearance_m": _positive_measurement_or_none(
                decision.observation.forward_clearance_m
            ),
            "measured_vx_mps": decision.observation.measured_vx_mps,
            "odom_freshness_ms": _freshness_value_or_none(decision.observation.odom_freshness_ms),
            "lidar_freshness_ms": _freshness_value_or_none(decision.observation.lidar_freshness_ms),
            "costmap_freshness_ms": _freshness_value_or_none(
                decision.observation.costmap_freshness_ms
            ),
        },
        "decision": {
            "selected_candidate_id": selected.candidate_id,
            "authorization": (
                "authorized_continue_forward"
                if selected.candidate_id == "continue_forward_chunk"
                else "vetoed_forward_selected_stop"
            ),
            "primary_reason": (
                forward.rejection_reasons[0]
                if forward.rejection_reasons
                else "clearance_budget_clear"
            ),
            "forward_rejection_reasons": list(forward.rejection_reasons),
        },
        "safety_budget": decision.safety_budget,
        "decision_trace_digest": f"sha256:{decision_trace_digest(trace)}",
        "decision_trace_digest_short": decision_trace_digest(trace)[:12],
    }


def _input_digest(
    *,
    mode: Mode,
    observation: MovingObservation,
    config: MovingSafetyConfig,
    dataset_csv: Path | None,
    trial_table_url: str,
) -> str:
    payload = {
        "mode": mode,
        "observation": {
            "step_id": observation.step_id,
            "clearance": _positive_measurement_or_none(observation.forward_clearance_m),
            "freshness": [
                _freshness_value_or_none(observation.odom_freshness_ms),
                _freshness_value_or_none(observation.lidar_freshness_ms),
                _freshness_value_or_none(observation.costmap_freshness_ms),
            ],
        },
        "config": {
            "vx_mps": config.vx_mps,
            "chunk_duration_s": config.chunk_duration_s,
        },
        "dataset_source": "<dataset>/tables/all_trials_normalized.csv"
        if dataset_csv is not None
        else trial_table_url,
    }
    return decision_trace_digest(payload)


def _receipt_payload(
    *,
    step_id: str,
    command: tuple[float, float, float, float],
    issued_at_unix_s: float,
    stopmove_ack: bool,
    measured_stop_time_s: float,
    measured_stop_distance_m: float,
    outcome_source: str,
) -> JSONDict:
    if not step_id:
        raise WorldStateError("DimOS execution receipt step_id must be non-empty.")
    if not _finite_positive(issued_at_unix_s):
        raise WorldStateError("DimOS execution receipt issued_at_unix_s must be positive.")
    if not _finite_positive(measured_stop_time_s):
        raise WorldStateError("DimOS execution receipt measured_stop_time_s must be positive.")
    if not _finite_positive(measured_stop_distance_m):
        raise WorldStateError("DimOS execution receipt measured_stop_distance_m must be positive.")
    if outcome_source not in {"native_odom", "external_aruco", "native_odom_plus_external_aruco"}:
        raise WorldStateError("DimOS execution receipt outcome_source is unsupported.")
    vx, vy, wz, duration = command
    return {
        "receipt_kind": _HOST_EXECUTION_RECEIPT_KIND,
        "step_id": step_id,
        "command": {
            "vx": float(vx),
            "vy": float(vy),
            "wz": float(wz),
            "duration_s": float(duration),
        },
        "issued_at_unix_s": float(issued_at_unix_s),
        "stopmove_ack": bool(stopmove_ack),
        "measured_stop_time_s": float(measured_stop_time_s),
        "measured_stop_distance_m": float(measured_stop_distance_m),
        "outcome_source": outcome_source,
    }


def _verify_host_execution_receipt(
    receipt: object,
    *,
    receipt_hmac_key: str | bytes | None,
    expected_step_id: str,
    expected_command: tuple[float, float, float, float],
) -> JSONDict:
    status: JSONDict = {
        "provided": receipt is not None,
        "verified": False,
        "scheme": "hmac-sha256",
        "receipt_kind": _HOST_EXECUTION_RECEIPT_KIND,
        "receipt_digest": None,
        "failure_reason": None,
        "measurement_present": False,
        "stopmove_ack": False,
        "outcome_source": None,
        "measured_stop_time_s": None,
        "measured_stop_distance_m": None,
    }
    if receipt is None:
        return status
    if not isinstance(receipt, Mapping):
        status["failure_reason"] = "receipt_not_json_object"
        return status
    public_receipt = dict(receipt)
    supplied_hmac = public_receipt.pop(_RECEIPT_HMAC_FIELD, None)
    try:
        status["receipt_digest"] = hashlib.sha256(
            _canonical_receipt_json(public_receipt).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        status["failure_reason"] = "receipt_not_json_serializable"
        return status
    if public_receipt.get("receipt_kind") != _HOST_EXECUTION_RECEIPT_KIND:
        status["failure_reason"] = "receipt_kind_mismatch"
        return status
    if public_receipt.get("step_id") != expected_step_id:
        status["failure_reason"] = "step_id_mismatch"
        return status
    if not _receipt_command_matches(public_receipt.get("command"), expected_command):
        status["failure_reason"] = "command_mismatch"
        return status
    if not _finite_positive(_number_from_receipt(public_receipt.get("issued_at_unix_s"))):
        status["failure_reason"] = "invalid_issued_at"
        return status
    measured_stop_time_s = _number_from_receipt(public_receipt.get("measured_stop_time_s"))
    measured_stop_distance_m = _number_from_receipt(public_receipt.get("measured_stop_distance_m"))
    if not _finite_positive(measured_stop_time_s) or not _finite_positive(measured_stop_distance_m):
        status["failure_reason"] = "missing_measured_stop_outcome"
        return status
    if public_receipt.get("stopmove_ack") is not True:
        status["failure_reason"] = "missing_stopmove_ack"
        return status
    outcome_source = public_receipt.get("outcome_source")
    if outcome_source not in {"native_odom", "external_aruco", "native_odom_plus_external_aruco"}:
        status["failure_reason"] = "unsupported_outcome_source"
        return status
    if receipt_hmac_key is None:
        status["failure_reason"] = "missing_receipt_hmac_key"
        return status
    if not isinstance(supplied_hmac, str):
        status["failure_reason"] = "missing_receipt_hmac"
        return status
    expected_hmac = _receipt_hmac(public_receipt, receipt_hmac_key)
    if not hmac.compare_digest(supplied_hmac, expected_hmac):
        status["failure_reason"] = "receipt_hmac_mismatch"
        return status
    status.update(
        {
            "verified": True,
            "failure_reason": None,
            "measurement_present": True,
            "stopmove_ack": True,
            "outcome_source": outcome_source,
            "measured_stop_time_s": float(measured_stop_time_s),
            "measured_stop_distance_m": float(measured_stop_distance_m),
        }
    )
    return _round_json_floats(status)


def _receipt_hmac(payload: Mapping[str, object], receipt_hmac_key: str | bytes) -> str:
    if isinstance(receipt_hmac_key, str):
        key = receipt_hmac_key.encode("utf-8")
    else:
        key = receipt_hmac_key
    if not key:
        raise WorldStateError("DimOS execution receipt HMAC key must be non-empty.")
    return hmac.new(
        key,
        _canonical_receipt_json(dict(payload)).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _canonical_receipt_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _receipt_command_matches(
    command: object, expected_command: tuple[float, float, float, float]
) -> bool:
    if not isinstance(command, Mapping):
        return False
    expected = {
        "vx": expected_command[0],
        "vy": expected_command[1],
        "wz": expected_command[2],
        "duration_s": expected_command[3],
    }
    for key, expected_value in expected.items():
        actual = _number_from_receipt(command.get(key))
        if actual is None or not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-9):
            return False
    return True


def _number_from_receipt(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _score_margin(scores: list[JSONDict]) -> float:
    if len(scores) < 2:
        return 0.0
    return float(scores[1]["score"]) - float(scores[0]["score"])


def _why_selected(candidate: _Candidate) -> str:
    if candidate.candidate_id == "stop_move":
        return "Forward chunk was unsafe or not sufficiently proven; selected StopMove."
    return "Forward chunk stayed inside the inflated clearance budget."


def _why_rejected(candidate: _Candidate) -> str:
    if candidate.rejection_reasons:
        return ", ".join(candidate.rejection_reasons)
    return "Higher intervention cost than the selected action."


def _decision_outcome_kind(decision: _StepDecision) -> str:
    if (
        decision.hardware_commands_sent
        and decision.receipt_verification.get("measurement_present") is True
    ):
        return "real_measured"
    return "analytic"


def _outcome_stopmove_ack(decision: _StepDecision) -> bool | None:
    if decision.receipt_verification.get("verified") is True:
        return bool(decision.receipt_verification.get("stopmove_ack"))
    return decision.observation.stopmove_ack


def _outcome_measured_stop_time_s(decision: _StepDecision) -> float | None:
    receipt_time = _positive_measurement_or_none(
        decision.receipt_verification.get("measured_stop_time_s")
    )
    if receipt_time is not None:
        return receipt_time
    return _positive_measurement_or_none(decision.observation.measured_stop_time_s)


def _outcome_measured_stop_distance_m(decision: _StepDecision) -> float | None:
    receipt_distance = _positive_measurement_or_none(
        decision.receipt_verification.get("measured_stop_distance_m")
    )
    if receipt_distance is not None:
        return receipt_distance
    return _positive_measurement_or_none(decision.observation.measured_stop_distance_m)


def _outcome_source(decision: _StepDecision) -> str:
    if decision.receipt_verification.get("verified") is True:
        return str(decision.receipt_verification.get("outcome_source") or "host_verified_receipt")
    if _outcome_measured_stop_distance_m(decision) is not None:
        return "host_supplied_unverified_measurement"
    return "analytic_shadow"


def _stop_attribution(*, selected: _Candidate, decision: _StepDecision) -> str:
    if selected.candidate_id != "stop_move":
        return "continued_forward"
    if decision.hardware_commands_sent:
        return "dimos_stopmove_ack_with_host_receipt"
    return "worldforge_shadow_selected_stop"


def _outcome_status(*, selected: _Candidate, decision: _StepDecision) -> str:
    if selected.candidate_id == "stop_move":
        return "stopped_or_stop_selected_before_obstacle"
    if decision.hardware_commands_sent:
        return "forward_chunk_executed_with_host_receipt"
    return "forward_chunk_authorized_in_shadow"


def _stop_proof_json(stop_proof: StopProofMeasurement) -> JSONDict:
    return {
        "available": stop_proof.available,
        "stop_time_s": _positive_measurement_or_none(stop_proof.stop_time_s),
        "stop_distance_m": _positive_measurement_or_none(stop_proof.stop_distance_m),
        "command_rtt_ms": _positive_measurement_or_none(stop_proof.command_rtt_ms),
        "velocity_before_stop_mps": _positive_measurement_or_none(
            stop_proof.velocity_before_stop_mps
        ),
        "velocity_after_stop_mps": _nonnegative_measurement_or_none(
            stop_proof.velocity_after_stop_mps
        ),
    }


def _stop_proof_protocol_json(config: MovingSafetyConfig) -> JSONDict:
    return {
        "required_before_live_moving_intervention": True,
        "purpose": "Measure the actual Go2 StopMove coast distance and time at demo speed.",
        "model_prediction_scope": "command displacement only; stop distance is measured separately",
        "minimum_repetitions_per_speed": 15,
        "speeds_mps": [0.08, config.vx_mps],
        "chunk_duration_s": config.chunk_duration_s,
        "measurements": [
            "overhead_aruco_stop_distance_m",
            "native_odom_stop_distance_m",
            "command_to_stop_ack_latency_ms",
            "time_to_velocity_near_zero_s",
        ],
        "acceptance_gate": (
            "Use moving intervention only if measured stop distance plus residual margin is "
            "comfortably below the configured clearance threshold."
        ),
    }


def _freshness_is_present(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0.0


def _freshness_value_or_none(value: float | None) -> float | None:
    if not _freshness_is_present(value):
        return None
    return float(value)


def _age_or_inf(value: float | None) -> float:
    if not _freshness_is_present(value):
        return math.inf
    return float(value)


def _finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0.0


def _positive_measurement_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def _nonnegative_measurement_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value < 0.0:
        return None
    return float(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "stop-proof", "open-space-bounded-motion", "moving-intervention"),
        default="moving-intervention",
    )
    parser.add_argument("--dataset-csv", type=Path)
    parser.add_argument("--trial-table-url", default=DEFAULT_TRIAL_TABLE_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vx-mps", type=float, default=0.10)
    parser.add_argument("--chunk-duration-s", type=float, default=0.15)
    parser.add_argument("--stop-proof-distance-m", type=float)
    parser.add_argument("--stop-proof-time-s", type=float)
    parser.add_argument("--command-rtt-ms", type=float)
    args = parser.parse_args(argv)

    stop_proof = StopProofMeasurement(
        available=args.stop_proof_distance_m is not None and args.stop_proof_time_s is not None,
        stop_distance_m=args.stop_proof_distance_m,
        stop_time_s=args.stop_proof_time_s,
        command_rtt_ms=args.command_rtt_ms,
    )
    result = run_go2_moving_safety_intervention(
        mode=args.mode,
        dataset_csv=args.dataset_csv,
        trial_table_url=args.trial_table_url,
        output_dir=args.out,
        config=MovingSafetyConfig(vx_mps=args.vx_mps, chunk_duration_s=args.chunk_duration_s),
        stop_proof=stop_proof,
    )
    print(f"mode={result.summary['mode']}")
    print(f"world_model={result.summary['world_model']['model_family']}")
    print("runtime=DimOS bridge contract")
    print(f"hardware_commands_sent={result.summary['runtime']['hardware_commands_sent']}")
    for step in result.summary["steps"]:
        print(
            f"{step['step_id']} -> decision={step['decision']['authorization']} "
            f"selected={step['decision']['selected_candidate_id']} "
            f"clearance={step['observation']['forward_clearance_m']} "
            f"required={step['safety_budget']['required_clearance_m']} "
            f"trace_sha256={step['decision_trace_digest'][:19]}..."
        )
    print(f"summary={result.summary_path}")
    print(f"bridge={result.bridge_path}")
    print(f"report={result.report_path}")


__all__ = [
    "Go2MovingSafetyResult",
    "MovingObservation",
    "MovingSafetyConfig",
    "StopProofMeasurement",
    "build_dimos_execution_receipt",
    "render_go2_moving_safety_report",
    "run_go2_moving_safety_intervention",
    "run_go2_moving_safety_intervention_workflow",
]
