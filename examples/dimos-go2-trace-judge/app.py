"""Offline DimOS Go2 trace-judge example for host-owned robot integrations.

The example intentionally does not import DimOS or connect to a robot. It turns
mocked DimOS-style observation summaries and candidate actions into WorldForge
score evidence, then writes the artifacts proposed in issue #329.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from worldforge import ActionScoreResult, WorldForge, WorldForgeError
from worldforge.models import JSONDict, ProviderCapabilities, ProviderHealth, dump_json
from worldforge.providers import BaseProvider, ProviderProfileSpec

JSON = dict[str, Any]
DEFAULT_OUTPUT_DIR = Path(".worldforge/dimos-go2-trace-judge")
DEFAULT_RUN_ID = "dimos-go2-mock-001"
SCORE_WEIGHTS: JSON = {
    "goal_alignment": 0.28,
    "information_gain": 0.27,
    "progress": 0.25,
    "clearance": 0.15,
    "not_stuck": 0.05,
    "execution_cost": -0.05,
}
REQUIRED_SCORE_WEIGHT_KEYS = (
    "goal_alignment",
    "information_gain",
    "progress",
    "clearance",
    "not_stuck",
    "execution_cost",
)
REQUIRED_FEATURES = (
    "goal_alignment",
    "information_gain",
    "progress",
    "obstacle_risk",
    "stuck_risk",
    "execution_cost",
)


class TransparentGo2ScoreProvider(BaseProvider):
    """Deterministic utility scorer for mocked Unitree Go2 navigation candidates."""

    def __init__(self) -> None:
        _validate_score_weights()
        super().__init__(
            name="transparent-go2-score",
            capabilities=ProviderCapabilities(predict=False, score=True),
            profile=ProviderProfileSpec(
                is_local=True,
                description="Transparent mock Go2 decision scorer for host-owned DimOS examples.",
                implementation_status="example",
                deterministic=True,
                requires_credentials=False,
            ),
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(name=self.name, healthy=True, latency_ms=0.1, details="configured")

    def score_actions(self, *, info: JSONDict, action_candidates: object) -> ActionScoreResult:
        candidates = _require_candidates(action_candidates)
        task = info.get("task")
        if not isinstance(task, dict):
            raise WorldForgeError("transparent Go2 scorer requires info.task.")

        scores = [round(_candidate_score(candidate["features"]), 4) for candidate in candidates]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        return ActionScoreResult(
            provider=self.name,
            scores=scores,
            best_index=best_index,
            lower_is_better=False,
            metadata={
                "score_type": "transparent_navigation_utility",
                "candidate_count": len(candidates),
                "weights": SCORE_WEIGHTS,
                "host_runtime": info.get("host_runtime", "mock-dimos"),
            },
        )


def sample_observation_summary() -> JSON:
    """Return a sanitized DimOS-style observation summary for offline runs."""

    return {
        "pose": {"x": 0.0, "y": 0.0, "yaw_degrees": 0.0},
        "costmap_summary": {
            "min_clearance_m": 0.8,
            "unknown_area_ratio": 0.42,
            "frontier_count": 4,
            "blocked_ahead": False,
        },
        "visual_summary": {
            "target_confidence": 0.64,
            "target_bearing_degrees": 30,
            "gesture_direction_degrees": 35,
        },
        "navigation_state": {
            "localized": True,
            "stuck_probability": 0.06,
            "active_goal": None,
        },
    }


def sample_task(goal: str) -> JSON:
    return {
        "human_goal": goal,
        "goal_representation": {
            "type": "host_interpreted_goal",
            "target_label": "marker",
            "gesture_direction_degrees": 35,
        },
    }


def sample_candidates() -> list[JSON]:
    return [
        {
            "id": "scan_left",
            "action": "relative_move",
            "params": {"degrees": 35},
            "features": {
                "goal_alignment": 0.88,
                "obstacle_risk": 0.05,
                "information_gain": 0.83,
                "progress": 0.21,
                "stuck_risk": 0.02,
                "execution_cost": 0.16,
            },
            "reason_hint": "high information gain and aligned with the gesture bearing",
        },
        {
            "id": "go_forward_small",
            "action": "relative_move",
            "params": {"forward": 0.35},
            "features": {
                "goal_alignment": 0.58,
                "obstacle_risk": 0.31,
                "information_gain": 0.24,
                "progress": 0.72,
                "stuck_risk": 0.18,
                "execution_cost": 0.08,
            },
            "reason_hint": "good physical progress, but higher obstacle and stuck risk",
        },
        {
            "id": "detour_right",
            "action": "relative_move",
            "params": {"forward": 0.25, "left": -0.2},
            "features": {
                "goal_alignment": 0.81,
                "obstacle_risk": 0.04,
                "information_gain": 0.67,
                "progress": 0.76,
                "stuck_risk": 0.03,
                "execution_cost": 0.14,
            },
            "reason_hint": "best balance of safe progress, target alignment, and new coverage",
        },
        {
            "id": "stop_relocalize",
            "action": "wait",
            "params": {"seconds": 1.0},
            "features": {
                "goal_alignment": 0.18,
                "obstacle_risk": 0.01,
                "information_gain": 0.12,
                "progress": 0.0,
                "stuck_risk": 0.01,
                "execution_cost": 0.2,
            },
            "reason_hint": "safe fallback, but not needed while the host reports localization",
        },
    ]


def run_trace_judge(
    *,
    output_dir: Path,
    run_id: str = DEFAULT_RUN_ID,
    goal: str | None = None,
    input_json: Path | None = None,
) -> JSON:
    """Score mocked Go2 candidates and write issue-safe evidence artifacts."""

    run_id = _require_non_empty_str(run_id, name="run_id")
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_bundle = _load_trace_input(input_json=input_json, goal=goal)
    goal = input_bundle["task"]["human_goal"]
    observation = input_bundle["observation_summary"]
    task = input_bundle["task"]
    candidates = input_bundle["candidates"]
    embodiment = input_bundle["embodiment"]
    host_runtime = input_bundle["host_runtime"]
    provider = TransparentGo2ScoreProvider()
    forge = WorldForge(auto_register_remote=False)
    forge.register_provider(provider)

    score_info: JSON = {
        "schema_version": 1,
        "run_id": run_id,
        "embodiment": embodiment,
        "host_runtime": host_runtime,
        "input_source": input_bundle["input_source"],
        "task": task,
        "observation_summary": observation,
    }
    score_info_artifact: JSON = {
        "schema_version": 1,
        "run_id": run_id,
        "provider": provider.name,
        "capability": "score",
        "score_info": score_info,
        "action_candidates": candidates,
    }
    score_result = forge.score_actions(provider.name, info=score_info, action_candidates=candidates)
    selected = candidates[score_result.best_index]
    candidate_scores = _candidate_scores_payload(
        run_id=run_id,
        embodiment=embodiment,
        host_runtime=host_runtime,
        task=task,
        observation=observation,
        candidates=candidates,
        score_result=score_result,
    )
    selected_action = {
        "schema_version": 1,
        "run_id": run_id,
        "selected_candidate_id": selected["id"],
        "candidate_index": score_result.best_index,
        "action": selected["action"],
        "params": selected["params"],
        "score": score_result.best_score,
        "execute_with": f"host_runtime:{host_runtime}",
        "live_execute_with": "host_runtime:dimos",
        "worldforge_executes_robot": False,
    }
    outcome = _mock_outcome(run_id, selected["id"], host_runtime=host_runtime)
    manifest = _run_manifest(
        run_id=run_id,
        goal=goal,
        input_source=input_bundle["input_source"],
        host_runtime=host_runtime,
        selected_action=selected_action,
        score_result=score_result,
    )
    report = _markdown_report(candidate_scores, selected_action, outcome)

    _write_json(output_dir / "score_info.json", score_info_artifact)
    _write_json(output_dir / "observation_summary.json", observation)
    _write_json(output_dir / "candidate_scores.json", candidate_scores)
    _write_json(output_dir / "selected_action.json", selected_action)
    _write_json(output_dir / "outcome_after_execution.json", outcome)
    _write_json(output_dir / "run_manifest.json", manifest)
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    return {
        "status": "passed",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "selected_candidate_id": selected["id"],
        "selected_score": score_result.best_score,
        "artifact_paths": {
            "score_info": "score_info.json",
            "observation_summary": "observation_summary.json",
            "candidate_scores": "candidate_scores.json",
            "selected_action": "selected_action.json",
            "outcome_after_execution": "outcome_after_execution.json",
            "run_manifest": "run_manifest.json",
            "report": "report.md",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the offline DimOS Go2 trace judge.")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--run-id", default=DEFAULT_RUN_ID)
    run.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional venue_input.json with observation_summary, task, and candidates.",
    )
    run.add_argument(
        "--goal",
        default=None,
        help="Optional human-readable goal override recorded in the evidence artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_trace_judge(
            output_dir=args.output_dir,
            run_id=args.run_id,
            goal=args.goal,
            input_json=args.input_json,
        )
    except WorldForgeError as exc:
        print(json.dumps({"error": {"type": "validation_error", "message": str(exc)}}))
        return 2
    print(dump_json(result))
    return 0


def _candidate_scores_payload(
    *,
    run_id: str,
    embodiment: str,
    host_runtime: str,
    task: JSON,
    observation: JSON,
    candidates: list[JSON],
    score_result: ActionScoreResult,
) -> JSON:
    rows = []
    for index, candidate in enumerate(candidates):
        selected = index == score_result.best_index
        rows.append(
            {
                "candidate_id": candidate["id"],
                "candidate_index": index,
                "score": score_result.scores[index],
                "selected": selected,
                "features": candidate["features"],
                "reason": _reason(candidate, selected=selected),
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "embodiment": embodiment,
        "host_runtime": host_runtime,
        "task": task,
        "observation_summary": observation,
        "candidates": [
            {
                "id": item["id"],
                "action": item["action"],
                "params": item["params"],
            }
            for item in candidates
        ],
        "scores": rows,
        "selected_candidate_id": candidates[score_result.best_index]["id"],
        "worldforge_score_result": score_result.to_dict(),
    }


def _mock_outcome(run_id: str, candidate_id: str, *, host_runtime: str) -> JSON:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "selected_candidate_id": candidate_id,
        "executed_by": f"mock-{host_runtime.removeprefix('mock-')}",
        "worldforge_executes_robot": False,
        "outcome_after_execution": {
            "duration_s": 5.0,
            "progress_delta_m": 0.28,
            "coverage_delta": 0.12,
            "target_confidence_delta": 0.18,
            "blocked": False,
            "manual_intervention": False,
        },
    }


def _run_manifest(
    *,
    run_id: str,
    goal: str,
    input_source: str,
    host_runtime: str,
    selected_action: JSON,
    score_result: ActionScoreResult,
) -> JSON:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "kind": "dimos_go2_trace_judge",
        "status": "completed",
        "provider": score_result.provider,
        "capability": "score",
        "operation": "candidate_trace_judge",
        "goal": goal,
        "host_runtime": host_runtime,
        "input_source": input_source,
        "selected_candidate_id": selected_action["selected_candidate_id"],
        "selected_score": selected_action["score"],
        "artifact_paths": {
            "score_info": "score_info.json",
            "observation_summary": "observation_summary.json",
            "candidate_scores": "candidate_scores.json",
            "selected_action": "selected_action.json",
            "outcome_after_execution": "outcome_after_execution.json",
            "run_manifest": "run_manifest.json",
            "report": "report.md",
        },
        "safety_boundary": {
            "host_owns_robot_networking": True,
            "host_owns_operator_supervision": True,
            "host_owns_emergency_stop": True,
            "worldforge_executes_robot": False,
            "worldforge_certifies_robot_safety": False,
        },
        "issue": "https://github.com/AbdelStark/worldforge/issues/329",
        "experimental": True,
        "open_to_maintainer_changes": True,
    }


def _load_trace_input(*, input_json: Path | None, goal: str | None) -> JSON:
    if input_json is None:
        resolved_goal = (
            _require_non_empty_str(goal, name="goal")
            if goal is not None
            else "inspect the area and move toward the indicated target"
        )
        return {
            "input_source": "built_in_sample",
            "embodiment": "unitree_go2",
            "host_runtime": "mock-dimos",
            "task": sample_task(resolved_goal),
            "observation_summary": sample_observation_summary(),
            "candidates": sample_candidates(),
        }

    payload = _read_json_object(input_json)
    observation = _require_json_object(
        payload.get("observation_summary"),
        name="observation_summary",
    )
    task = _require_task(payload.get("task"), goal=goal)
    candidates = _require_candidates(payload.get("candidates"))
    return {
        "input_source": "venue_input_json",
        "input_json": str(input_json),
        "embodiment": _optional_non_empty_str(
            payload.get("embodiment"),
            name="embodiment",
            default="unitree_go2",
        ),
        "host_runtime": _optional_non_empty_str(
            payload.get("host_runtime"),
            name="host_runtime",
            default="dimos",
        ),
        "task": task,
        "observation_summary": observation,
        "candidates": candidates,
    }


def _read_json_object(path: Path) -> JSON:
    path = path.expanduser()
    if not path.is_file():
        raise WorldForgeError(
            "host-owned DimOS trace judge input_json is not a readable file; "
            "check --input-json points to a JSON object file and retry."
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorldForgeError(
            "host-owned DimOS trace judge failed to read input_json; "
            "check file permissions and retry."
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorldForgeError(
            "host-owned DimOS trace judge input_json must be valid JSON; "
            f"fix the JSON syntax near {exc.msg} and retry."
        ) from exc
    if not isinstance(payload, dict):
        raise WorldForgeError("input_json must contain a JSON object.")
    return payload


def _require_json_object(value: object, *, name: str) -> JSON:
    if not isinstance(value, dict):
        raise WorldForgeError(f"{name} must be a JSON object.")
    return dict(value)


def _require_task(value: object, *, goal: str | None) -> JSON:
    task = _require_json_object(value, name="task")
    if goal is not None:
        task["human_goal"] = _require_non_empty_str(goal, name="goal")
    else:
        task["human_goal"] = _require_non_empty_str(
            task.get("human_goal"),
            name="task.human_goal",
        )
    goal_representation = task.get("goal_representation")
    if goal_representation is not None and not isinstance(goal_representation, dict):
        raise WorldForgeError("task.goal_representation must be a JSON object when present.")
    if goal_representation is None:
        task["goal_representation"] = {"type": "host_interpreted_goal"}
    return task


def _optional_non_empty_str(value: object, *, name: str, default: str) -> str:
    if value is None:
        return default
    return _require_non_empty_str(value, name=name)


def _candidate_score(features: JSON) -> float:
    return (
        0.1
        + float(features["goal_alignment"]) * float(SCORE_WEIGHTS["goal_alignment"])
        + float(features["information_gain"]) * float(SCORE_WEIGHTS["information_gain"])
        + float(features["progress"]) * float(SCORE_WEIGHTS["progress"])
        + (1.0 - float(features["obstacle_risk"])) * float(SCORE_WEIGHTS["clearance"])
        + (1.0 - float(features["stuck_risk"])) * float(SCORE_WEIGHTS["not_stuck"])
        + float(features["execution_cost"]) * float(SCORE_WEIGHTS["execution_cost"])
    )


def _require_candidates(action_candidates: object) -> list[JSON]:
    if not isinstance(action_candidates, list) or not action_candidates:
        raise WorldForgeError("transparent Go2 scorer requires non-empty candidate list.")
    candidates: list[JSON] = []
    for index, raw in enumerate(action_candidates):
        if not isinstance(raw, dict):
            raise WorldForgeError(f"candidate {index} must be an object.")
        candidate = dict(raw)
        for field_name in ("id", "action", "params", "features"):
            if field_name not in candidate:
                raise WorldForgeError(f"candidate {index} missing {field_name}.")
        if not isinstance(candidate["id"], str) or not candidate["id"].strip():
            raise WorldForgeError(f"candidate {index} id must be a non-empty string.")
        if not isinstance(candidate["action"], str) or not candidate["action"].strip():
            raise WorldForgeError(f"candidate {index} action must be a non-empty string.")
        if not isinstance(candidate["params"], dict):
            raise WorldForgeError(f"candidate {index} params must be an object.")
        if not isinstance(candidate["features"], dict):
            raise WorldForgeError(f"candidate {index} features must be an object.")
        reason_hint = candidate.get("reason_hint")
        if not isinstance(reason_hint, str) or not reason_hint.strip():
            raise WorldForgeError(
                f"candidate {candidate['id']} reason_hint must be a non-empty string."
            )
        candidate["reason_hint"] = reason_hint.strip()
        _require_features(candidate["features"], candidate_id=candidate["id"])
        candidates.append(candidate)
    return candidates


def _validate_score_weights() -> None:
    required = set(REQUIRED_SCORE_WEIGHT_KEYS)
    actual = set(SCORE_WEIGHTS)
    missing = sorted(required - actual)
    if missing:
        raise WorldForgeError(f"missing score weights: {', '.join(missing)}.")
    unexpected = sorted(actual - required)
    if unexpected:
        raise WorldForgeError(f"unexpected score weights: {', '.join(unexpected)}.")
    for name in REQUIRED_SCORE_WEIGHT_KEYS:
        value = SCORE_WEIGHTS.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise WorldForgeError(f"score weight {name} must be numeric.")
        if not math.isfinite(float(value)):
            raise WorldForgeError(f"score weight {name} must be finite.")


def _require_features(features: JSON, *, candidate_id: str) -> None:
    for name in REQUIRED_FEATURES:
        value = features.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise WorldForgeError(f"candidate {candidate_id} feature {name} must be numeric.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise WorldForgeError(f"candidate {candidate_id} feature {name} must be finite.")
        if numeric < 0.0 or numeric > 1.0:
            raise WorldForgeError(f"candidate {candidate_id} feature {name} must be in [0, 1].")


def _reason(candidate: JSON, *, selected: bool) -> str:
    prefix = "selected" if selected else "rejected"
    return f"{prefix}: {candidate['reason_hint']}"


def _markdown_report(candidate_scores: JSON, selected_action: JSON, outcome: JSON) -> str:
    host_runtime = candidate_scores.get("host_runtime", "mock-dimos")
    lines = [
        "# DimOS Go2 Trace Judge",
        "",
        f"- run_id: {candidate_scores['run_id']}",
        f"- selected_candidate_id: {selected_action['selected_candidate_id']}",
        f"- selected_score: {selected_action['score']}",
        "- worldforge_executes_robot: false",
        f"- host_runtime: {host_runtime}",
        "",
        "| Candidate | Score | Decision | Reason |",
        "| --- | ---: | --- | --- |",
    ]
    for row in candidate_scores["scores"]:
        decision = "selected" if row["selected"] else "rejected"
        lines.append(
            f"| `{row['candidate_id']}` | {row['score']:.4f} | {decision} | {row['reason']} |"
        )
    after = outcome["outcome_after_execution"]
    lines.extend(
        [
            "",
            "## Mock Outcome",
            "",
            f"- progress_delta_m: {after['progress_delta_m']}",
            f"- coverage_delta: {after['coverage_delta']}",
            f"- target_confidence_delta: {after['target_confidence_delta']}",
            f"- blocked: {str(after['blocked']).lower()}",
            f"- manual_intervention: {str(after['manual_intervention']).lower()}",
            "",
            "This artifact is an offline, experimental host-integration example. DimOS or another "
            "host runtime owns robot networking, controller execution, operator supervision, "
            "and emergency stop behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: JSON) -> None:
    path.write_text(dump_json(payload) + "\n", encoding="utf-8")


def _require_non_empty_str(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorldForgeError(f"{name} must be a non-empty string.")
    return value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
