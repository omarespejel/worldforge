"""DecisionTrace v1 validation helpers.

DecisionTrace records one robot planning decision: what the runtime observed, which
candidate actions were scored, which action was selected, what alternatives were
rejected, and what outcome was available. It complements trajectory datasets and
runtime spans; it does not replace either one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
from itertools import pairwise
from typing import Any

from worldforge.models import (
    JSONDict,
    WorldForgeError,
    dump_json,
    require_bool,
    require_finite_number,
    require_json_dict,
    require_non_empty_text,
    require_non_negative_int,
    require_positive_int,
)

DECISION_TRACE_SCHEMA_VERSION = "worldforge.decision_trace.v1"
DECISION_TRACE_ARTIFACT_KIND = "worldforge.decision_trace"
DECISION_TRACE_SCHEMA_RESOURCE = "decision_trace.v1.schema.json"

SCORE_KINDS: tuple[str, ...] = ("hand_cost", "learned_latent", "simulator", "human_label")
OUTCOME_KINDS: tuple[str, ...] = ("analytic", "sim_measured", "real_measured")

_REQUIRED_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "schema_version",
    "artifact_kind",
    "trace_id",
    "run_id",
    "step_index",
    "embodiment",
    "host_runtime",
    "task",
    "observation",
    "goal",
    "candidate_actions",
    "scores",
    "selected_action",
    "counterfactuals",
    "baseline",
    "outcome",
    "planner_diagnostics",
    "reproducibility",
    "claim_boundary",
)


@dataclass(frozen=True, slots=True)
class _ScoreTable:
    candidate_ids: set[str]
    score_by_candidate: dict[str, float]
    lower_is_better: bool
    ranked_candidate_ids: list[str]
    ranked_scores: list[float]
    best_candidate_ids: set[str]
    best_score: float
    runner_up_score: float


def load_decision_trace_schema() -> JSONDict:
    """Return the packaged DecisionTrace v1 JSON Schema."""

    try:
        schema_text = (
            resources.files("worldforge.schemas")
            .joinpath(DECISION_TRACE_SCHEMA_RESOURCE)
            .read_text()
        )
        decoded = json.loads(schema_text)
    except (FileNotFoundError, ModuleNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise WorldForgeError(
            "Could not load DecisionTrace v1 JSON Schema resource "
            f"{DECISION_TRACE_SCHEMA_RESOURCE!r} from package 'worldforge.schemas'. "
            "Verify the installed wheel includes packaged schema data."
        ) from exc
    return require_json_dict(decoded, name="DecisionTrace v1 JSON Schema", allow_empty=False)


def decision_trace_digest(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for a JSON-native trace payload."""

    return sha256(dump_json(dict(payload)).encode("utf-8")).hexdigest()


def validate_decision_trace(payload: object, *, name: str = "DecisionTrace v1") -> JSONDict:
    """Return a JSON-native DecisionTrace v1 payload or raise ``WorldForgeError``."""

    trace = require_json_dict(payload, name=name, allow_empty=False)
    _require_fields(trace, _REQUIRED_TOP_LEVEL_FIELDS, name=name)
    _require_const(trace["schema_version"], DECISION_TRACE_SCHEMA_VERSION, f"{name}.schema_version")
    _require_const(trace["artifact_kind"], DECISION_TRACE_ARTIFACT_KIND, f"{name}.artifact_kind")
    require_non_empty_text(trace["trace_id"], name=f"{name}.trace_id")
    require_non_empty_text(trace["run_id"], name=f"{name}.run_id")
    require_non_negative_int(trace["step_index"], name=f"{name}.step_index")
    if trace.get("prev_trace_id") is not None:
        require_non_empty_text(trace["prev_trace_id"], name=f"{name}.prev_trace_id")

    _validate_embodiment(trace["embodiment"], name=f"{name}.embodiment")
    _validate_host_runtime(trace["host_runtime"], name=f"{name}.host_runtime")
    _validate_task(trace["task"], name=f"{name}.task")
    require_json_dict(trace["observation"], name=f"{name}.observation")
    _validate_goal(trace["goal"], name=f"{name}.goal")
    candidate_ids = _validate_candidate_actions(
        trace["candidate_actions"], name=f"{name}.candidate_actions"
    )
    score_table = _validate_scores(trace["scores"], name=f"{name}.scores")
    _require_same_ids(candidate_ids, score_table.candidate_ids, name=f"{name}.scores")
    selected_id = _validate_selected_action(
        trace["selected_action"],
        candidate_ids=candidate_ids,
        score_table=score_table,
        name=f"{name}.selected_action",
    )
    _validate_counterfactuals(
        trace["counterfactuals"],
        candidate_ids=candidate_ids,
        selected_id=selected_id,
        name=f"{name}.counterfactuals",
    )
    _validate_baseline(trace["baseline"], candidate_ids=candidate_ids, name=f"{name}.baseline")
    outcome_kind = _validate_outcome(trace["outcome"], name=f"{name}.outcome")
    require_json_dict(trace["planner_diagnostics"], name=f"{name}.planner_diagnostics")
    _validate_reproducibility(trace["reproducibility"], name=f"{name}.reproducibility")
    _validate_claim_boundary(
        trace["claim_boundary"],
        outcome_kind=outcome_kind,
        name=f"{name}.claim_boundary",
    )
    if "interop" in trace:
        require_json_dict(trace["interop"], name=f"{name}.interop")
    return trace


def _validate_embodiment(value: object, *, name: str) -> None:
    embodiment = require_json_dict(value, name=name, allow_empty=False)
    _require_fields(embodiment, ("kind", "platform", "action_space"), name=name)
    require_non_empty_text(embodiment["kind"], name=f"{name}.kind")
    require_non_empty_text(embodiment["platform"], name=f"{name}.platform")
    require_non_empty_text(embodiment["action_space"], name=f"{name}.action_space")
    if embodiment.get("embodiment_id") is not None:
        require_non_empty_text(embodiment["embodiment_id"], name=f"{name}.embodiment_id")


def _validate_host_runtime(value: object, *, name: str) -> None:
    runtime = require_json_dict(value, name=name, allow_empty=False)
    _require_fields(runtime, ("name", "mode"), name=name)
    require_non_empty_text(runtime["name"], name=f"{name}.name")
    require_non_empty_text(runtime["mode"], name=f"{name}.mode")
    if runtime.get("version") is not None:
        require_non_empty_text(runtime["version"], name=f"{name}.version")


def _validate_task(value: object, *, name: str) -> None:
    task = require_json_dict(value, name=name, allow_empty=False)
    _require_fields(task, ("task_id", "description"), name=name)
    require_non_empty_text(task["task_id"], name=f"{name}.task_id")
    require_non_empty_text(task["description"], name=f"{name}.description")


def _validate_goal(value: object, *, name: str) -> None:
    goal = require_json_dict(value, name=name, allow_empty=False)
    _require_fields(goal, ("type", "description", "sub_goals", "success_criteria"), name=name)
    require_non_empty_text(goal["type"], name=f"{name}.type")
    require_non_empty_text(goal["description"], name=f"{name}.description")
    sub_goals = _require_non_empty_sequence(goal["sub_goals"], name=f"{name}.sub_goals")
    for index, sub_goal_value in enumerate(sub_goals):
        sub_goal = require_json_dict(
            sub_goal_value, name=f"{name}.sub_goals[{index}]", allow_empty=False
        )
        _require_fields(sub_goal, ("id", "description"), name=f"{name}.sub_goals[{index}]")
        require_non_empty_text(sub_goal["id"], name=f"{name}.sub_goals[{index}].id")
        require_non_empty_text(
            sub_goal["description"], name=f"{name}.sub_goals[{index}].description"
        )
        if "required" in sub_goal:
            require_bool(sub_goal["required"], name=f"{name}.sub_goals[{index}].required")
        if "weight" in sub_goal:
            require_finite_number(sub_goal["weight"], name=f"{name}.sub_goals[{index}].weight")
    criteria = require_json_dict(
        goal["success_criteria"], name=f"{name}.success_criteria", allow_empty=False
    )
    _require_fields(criteria, ("metric", "partial_credit"), name=f"{name}.success_criteria")
    require_non_empty_text(criteria["metric"], name=f"{name}.success_criteria.metric")
    require_bool(criteria["partial_credit"], name=f"{name}.success_criteria.partial_credit")


def _validate_candidate_actions(value: object, *, name: str) -> set[str]:
    candidates = _require_non_empty_sequence(value, name=name)
    candidate_ids: set[str] = set()
    for index, candidate_value in enumerate(candidates):
        candidate = require_json_dict(candidate_value, name=f"{name}[{index}]", allow_empty=False)
        _require_fields(candidate, ("candidate_id", "action"), name=f"{name}[{index}]")
        candidate_id = require_non_empty_text(
            candidate["candidate_id"], name=f"{name}[{index}].candidate_id"
        )
        if candidate_id in candidate_ids:
            raise WorldForgeError(f"{name} contains duplicate candidate_id '{candidate_id}'.")
        candidate_ids.add(candidate_id)
        action = require_json_dict(candidate["action"], name=f"{name}[{index}].action")
        _require_fields(action, ("type", "params", "units"), name=f"{name}[{index}].action")
        require_non_empty_text(action["type"], name=f"{name}[{index}].action.type")
        require_json_dict(action["params"], name=f"{name}[{index}].action.params")
        require_json_dict(action["units"], name=f"{name}[{index}].action.units")
    return candidate_ids


def _validate_scores(value: object, *, name: str) -> _ScoreTable:
    scores = _require_non_empty_sequence(value, name=name)
    score_ids: set[str] = set()
    score_by_candidate: dict[str, float] = {}
    seen_ranks: set[int] = set()
    lower_is_better: bool | None = None
    ranked_rows: list[tuple[str, float, int]] = []
    for index, score_value in enumerate(scores):
        score = require_json_dict(score_value, name=f"{name}[{index}]", allow_empty=False)
        _require_fields(
            score,
            ("candidate_id", "rank", "score", "lower_is_better", "components", "normalized"),
            name=f"{name}[{index}]",
        )
        candidate_id = require_non_empty_text(
            score["candidate_id"], name=f"{name}[{index}].candidate_id"
        )
        if candidate_id in score_ids:
            raise WorldForgeError(f"{name} contains duplicate candidate_id '{candidate_id}'.")
        score_ids.add(candidate_id)
        rank = require_positive_int(score["rank"], name=f"{name}[{index}].rank")
        if rank in seen_ranks:
            raise WorldForgeError(f"{name} contains duplicate rank {rank}.")
        seen_ranks.add(rank)
        numeric_score = require_finite_number(score["score"], name=f"{name}[{index}].score")
        score_by_candidate[candidate_id] = numeric_score
        current_lower_is_better = require_bool(
            score["lower_is_better"], name=f"{name}[{index}].lower_is_better"
        )
        if lower_is_better is None:
            lower_is_better = current_lower_is_better
        elif lower_is_better != current_lower_is_better:
            raise WorldForgeError(f"{name} lower_is_better must be consistent for all scores.")
        ranked_rows.append((candidate_id, numeric_score, rank))
        require_json_dict(score["components"], name=f"{name}[{index}].components")
        normalized = require_json_dict(
            score["normalized"], name=f"{name}[{index}].normalized", allow_empty=False
        )
        if "value_signal" not in normalized:
            raise WorldForgeError(f"{name}[{index}].normalized is missing value_signal.")
        value_signal = require_finite_number(
            normalized["value_signal"], name=f"{name}[{index}].normalized.value_signal"
        )
        if value_signal < 0.0 or value_signal > 1.0:
            raise WorldForgeError(f"{name}[{index}].normalized.value_signal must be in [0, 1].")
    expected_ranks = set(range(1, len(scores) + 1))
    if seen_ranks != expected_ranks:
        raise WorldForgeError(f"{name} ranks must be contiguous from 1 to {len(scores)}.")
    if lower_is_better is None:  # pragma: no cover - guarded by non-empty sequence
        raise WorldForgeError(f"{name} must not be empty.")
    _validate_score_ranks_monotonic(ranked_rows, lower_is_better=lower_is_better, name=name)
    rank_sorted_rows = sorted(ranked_rows, key=lambda row: row[2])
    ranked_candidate_ids = [candidate_id for candidate_id, _score, _rank in rank_sorted_rows]
    ranked_scores = [score_value for _candidate_id, score_value, _rank in rank_sorted_rows]
    if lower_is_better:
        best_score = min(score_by_candidate.values())
    else:
        best_score = max(score_by_candidate.values())
    best_candidate_ids = {
        candidate_id
        for candidate_id, score_value in score_by_candidate.items()
        if _scores_equal(score_value, best_score)
    }
    non_best_scores = [
        score_value
        for score_value in score_by_candidate.values()
        if not _scores_equal(score_value, best_score)
    ]
    if not non_best_scores or len(best_candidate_ids) > 1:
        runner_up_score = best_score
    elif lower_is_better:
        runner_up_score = min(non_best_scores)
    else:
        runner_up_score = max(non_best_scores)
    return _ScoreTable(
        candidate_ids=score_ids,
        score_by_candidate=score_by_candidate,
        lower_is_better=lower_is_better,
        ranked_candidate_ids=ranked_candidate_ids,
        ranked_scores=ranked_scores,
        best_candidate_ids=best_candidate_ids,
        best_score=best_score,
        runner_up_score=runner_up_score,
    )


def _validate_score_ranks_monotonic(
    ranked_rows: list[tuple[str, float, int]],
    *,
    lower_is_better: bool,
    name: str,
) -> None:
    rank_sorted_rows = sorted(ranked_rows, key=lambda row: row[2])
    for previous, current in pairwise(rank_sorted_rows):
        previous_score = previous[1]
        current_id, current_score, _current_rank = current
        if _scores_equal(previous_score, current_score):
            continue
        if _score_is_better(current_score, previous_score, lower_is_better=lower_is_better):
            raise WorldForgeError(
                f"{name} rank for candidate_id '{current_id}' must match score ordering."
            )


def _validate_selected_action(
    value: object,
    *,
    candidate_ids: set[str],
    score_table: _ScoreTable,
    name: str,
) -> str:
    selected = require_json_dict(value, name=name, allow_empty=False)
    _require_fields(selected, ("candidate_id", "score", "score_margin", "why_selected"), name=name)
    candidate_id = require_non_empty_text(selected["candidate_id"], name=f"{name}.candidate_id")
    if candidate_id not in candidate_ids:
        raise WorldForgeError(f"{name}.candidate_id '{candidate_id}' is not a candidate action.")
    if candidate_id not in score_table.best_candidate_ids:
        raise WorldForgeError(f"{name}.candidate_id must reference a best-scoring candidate.")
    if candidate_id != score_table.ranked_candidate_ids[0]:
        raise WorldForgeError(f"{name}.candidate_id must reference the rank-1 score candidate.")
    selected_score = require_finite_number(selected["score"], name=f"{name}.score")
    _require_close(
        selected_score,
        score_table.score_by_candidate[candidate_id],
        name=f"{name}.score",
    )
    score_margin = require_finite_number(selected["score_margin"], name=f"{name}.score_margin")
    expected_margin = _score_margin(
        selected=score_table.best_score,
        runner_up=score_table.runner_up_score,
        lower_is_better=score_table.lower_is_better,
    )
    _require_close(score_margin, expected_margin, name=f"{name}.score_margin", tolerance=1e-3)
    require_non_empty_text(selected["why_selected"], name=f"{name}.why_selected")
    return candidate_id


def _validate_counterfactuals(
    value: object,
    *,
    candidate_ids: set[str],
    selected_id: str,
    name: str,
) -> None:
    counterfactuals = _require_sequence(value, name=name)
    counterfactual_ids: set[str] = set()
    for index, counterfactual_value in enumerate(counterfactuals):
        counterfactual = require_json_dict(
            counterfactual_value, name=f"{name}[{index}]", allow_empty=False
        )
        _require_fields(
            counterfactual,
            ("candidate_id", "score", "delta_vs_selected", "why_rejected"),
            name=f"{name}[{index}]",
        )
        candidate_id = require_non_empty_text(
            counterfactual["candidate_id"], name=f"{name}[{index}].candidate_id"
        )
        if candidate_id == selected_id:
            raise WorldForgeError(f"{name}[{index}] must not repeat the selected action.")
        if candidate_id not in candidate_ids:
            raise WorldForgeError(f"{name}[{index}].candidate_id '{candidate_id}' is unknown.")
        if candidate_id in counterfactual_ids:
            raise WorldForgeError(f"{name} contains duplicate candidate_id '{candidate_id}'.")
        counterfactual_ids.add(candidate_id)
        require_finite_number(counterfactual["score"], name=f"{name}[{index}].score")
        require_finite_number(
            counterfactual["delta_vs_selected"], name=f"{name}[{index}].delta_vs_selected"
        )
        require_non_empty_text(counterfactual["why_rejected"], name=f"{name}[{index}].why_rejected")


def _validate_baseline(value: object, *, candidate_ids: set[str], name: str) -> None:
    baseline = require_json_dict(value, name=name, allow_empty=False)
    _require_fields(baseline, ("candidate_id", "regret_vs_selected"), name=name)
    if baseline["candidate_id"] is not None:
        candidate_id = require_non_empty_text(baseline["candidate_id"], name=f"{name}.candidate_id")
        if candidate_id not in candidate_ids:
            raise WorldForgeError(
                f"{name}.candidate_id '{candidate_id}' is not a candidate action."
            )
    require_finite_number(baseline["regret_vs_selected"], name=f"{name}.regret_vs_selected")


def _validate_outcome(value: object, *, name: str) -> str:
    outcome = require_json_dict(value, name=name, allow_empty=False)
    _require_fields(outcome, ("kind", "status", "metrics"), name=name)
    outcome_kind = _require_enum(outcome["kind"], OUTCOME_KINDS, name=f"{name}.kind")
    require_non_empty_text(outcome["status"], name=f"{name}.status")
    require_json_dict(outcome["metrics"], name=f"{name}.metrics")
    return outcome_kind


def _validate_reproducibility(value: object, *, name: str) -> None:
    reproducibility = require_json_dict(value, name=name, allow_empty=False)
    _require_fields(
        reproducibility,
        (
            "provider_version",
            "checkpoint_hash",
            "model_card_ref",
            "input_digest",
            "seed",
            "code_ref",
        ),
        name=name,
    )
    for optional_text in ("provider_version", "checkpoint_hash", "model_card_ref"):
        if reproducibility.get(optional_text) is not None:
            require_non_empty_text(reproducibility[optional_text], name=f"{name}.{optional_text}")
    require_non_empty_text(reproducibility["input_digest"], name=f"{name}.input_digest")
    if reproducibility["seed"] is not None:
        require_non_negative_int(reproducibility["seed"], name=f"{name}.seed")
    require_non_empty_text(reproducibility["code_ref"], name=f"{name}.code_ref")


def _validate_claim_boundary(value: object, *, outcome_kind: str, name: str) -> None:
    boundary = require_json_dict(value, name=name, allow_empty=False)
    _require_fields(
        boundary,
        (
            "score_kind",
            "outcome_kind",
            "hardware_executed",
            "learned_model_used",
            "safety_controller",
            "limitations",
        ),
        name=name,
    )
    _require_enum(boundary["score_kind"], SCORE_KINDS, name=f"{name}.score_kind")
    boundary_outcome_kind = _require_enum(
        boundary["outcome_kind"], OUTCOME_KINDS, name=f"{name}.outcome_kind"
    )
    if boundary_outcome_kind != outcome_kind:
        raise WorldForgeError(f"{name}.outcome_kind must match outcome.kind.")
    require_bool(boundary["hardware_executed"], name=f"{name}.hardware_executed")
    require_bool(boundary["learned_model_used"], name=f"{name}.learned_model_used")
    if boundary["safety_controller"] is not None:
        require_non_empty_text(boundary["safety_controller"], name=f"{name}.safety_controller")
    limitations = _require_sequence(boundary["limitations"], name=f"{name}.limitations")
    for index, limitation in enumerate(limitations):
        require_non_empty_text(limitation, name=f"{name}.limitations[{index}]")


def _require_fields(payload: Mapping[str, Any], fields: Iterable[str], *, name: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise WorldForgeError(f"{name} is missing required field(s): {', '.join(missing)}.")


def _require_const(value: object, expected: str, name: str) -> None:
    if value != expected:
        raise WorldForgeError(f"{name} must be {expected!r}.")


def _require_enum(value: object, options: tuple[str, ...], *, name: str) -> str:
    text = require_non_empty_text(value, name=name)
    if text not in options:
        raise WorldForgeError(f"{name} must be one of: {', '.join(options)}.")
    return text


def _require_non_empty_sequence(value: object, *, name: str) -> list[Any]:
    sequence = _require_sequence(value, name=name)
    if not sequence:
        raise WorldForgeError(f"{name} must not be empty.")
    return sequence


def _require_sequence(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise WorldForgeError(f"{name} must be a JSON array.")
    return list(value)


def _require_same_ids(first: set[str], second: set[str], *, name: str) -> None:
    if first == second:
        return
    missing = sorted(first - second)
    extra = sorted(second - first)
    pieces: list[str] = []
    if missing:
        pieces.append(f"missing score(s) for {', '.join(missing)}")
    if extra:
        pieces.append(f"unknown score id(s) {', '.join(extra)}")
    raise WorldForgeError(f"{name} candidate ids mismatch: {'; '.join(pieces)}.")


def _score_margin(*, selected: float, runner_up: float, lower_is_better: bool) -> float:
    return runner_up - selected if lower_is_better else selected - runner_up


def _score_is_better(candidate: float, incumbent: float, *, lower_is_better: bool) -> bool:
    return candidate < incumbent if lower_is_better else candidate > incumbent


def _require_close(
    actual: float,
    expected: float,
    *,
    name: str,
    tolerance: float = 1e-9,
) -> None:
    if abs(actual - expected) > tolerance:
        raise WorldForgeError(f"{name} must match the scores table.")


def _scores_equal(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance


__all__ = [
    "DECISION_TRACE_ARTIFACT_KIND",
    "DECISION_TRACE_SCHEMA_RESOURCE",
    "DECISION_TRACE_SCHEMA_VERSION",
    "OUTCOME_KINDS",
    "SCORE_KINDS",
    "decision_trace_digest",
    "load_decision_trace_schema",
    "validate_decision_trace",
]
