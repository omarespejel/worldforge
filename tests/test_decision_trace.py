from __future__ import annotations

import pytest

import worldforge.decision_trace as decision_trace_module
from worldforge.decision_trace import (
    DECISION_TRACE_ARTIFACT_KIND,
    DECISION_TRACE_SCHEMA_VERSION,
    decision_trace_digest,
    load_decision_trace_schema,
    validate_decision_trace,
)
from worldforge.models import JSONDict, WorldForgeError


def _valid_trace() -> JSONDict:
    return {
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "artifact_kind": DECISION_TRACE_ARTIFACT_KIND,
        "trace_id": "trace-go2-office-0001",
        "run_id": "cross-embodiment-demo",
        "step_index": 0,
        "prev_trace_id": None,
        "embodiment": {
            "kind": "quadruped_navigation",
            "platform": "unitree_go2_air",
            "embodiment_id": "go2-air-replay",
            "action_space": "planar_velocity",
        },
        "host_runtime": {"name": "DimOS replay", "mode": "checkout_safe_replay", "version": None},
        "task": {"task_id": "office-scan", "description": "Choose a local navigation action."},
        "observation": {"ref": {"episode_id": "episode-1", "frame_id": "frame-1"}},
        "goal": {
            "type": "navigation",
            "description": "Move toward the scan goal while avoiding obstacles.",
            "sub_goals": [
                {
                    "id": "advance",
                    "description": "Reduce distance to the goal.",
                    "required": True,
                    "weight": 0.7,
                }
            ],
            "success_criteria": {"metric": "partial_subgoal_credit", "partial_credit": True},
        },
        "candidate_actions": [
            {
                "candidate_id": "forward",
                "action": {
                    "type": "body_velocity",
                    "params": {"dx_m": 0.4, "dyaw_rad": 0.0},
                    "units": {"dx_m": "m", "dyaw_rad": "rad"},
                },
            },
            {
                "candidate_id": "stop",
                "action": {
                    "type": "stop_relocalize",
                    "params": {"duration_s": 1.0},
                    "units": {"duration_s": "s"},
                },
            },
        ],
        "scores": [
            {
                "candidate_id": "forward",
                "rank": 1,
                "score": 0.2,
                "lower_is_better": True,
                "components": {"distance_cost": 0.1, "risk_cost": 0.1},
                "normalized": {
                    "value_signal": 0.8,
                    "regret_vs_baseline": 0.5,
                    "separability": 0.6,
                },
            },
            {
                "candidate_id": "stop",
                "rank": 2,
                "score": 0.7,
                "lower_is_better": True,
                "components": {"distance_cost": 0.6, "risk_cost": 0.1},
                "normalized": {
                    "value_signal": 0.3,
                    "regret_vs_baseline": 0.0,
                    "separability": 0.0,
                },
            },
        ],
        "selected_action": {
            "candidate_id": "forward",
            "score": 0.2,
            "score_margin": 0.5,
            "why_selected": "Lowest transparent replay cost with positive baseline regret.",
        },
        "counterfactuals": [
            {
                "candidate_id": "stop",
                "score": 0.7,
                "delta_vs_selected": 0.5,
                "why_rejected": "Higher goal-progress cost.",
            }
        ],
        "baseline": {
            "candidate_id": "stop",
            "score": 0.7,
            "regret_vs_selected": 0.5,
            "policy": "hardcoded_stop",
        },
        "outcome": {
            "kind": "analytic",
            "status": "predicted_success",
            "metrics": {"partial_subgoal_credit": 0.8},
        },
        "planner_diagnostics": {"planner": "score-rank", "candidate_count": 2},
        "reproducibility": {
            "provider_version": "worldforge-integration",
            "checkpoint_hash": None,
            "model_card_ref": None,
            "input_digest": "sha256:fixture",
            "seed": 0,
            "code_ref": "integration/worldforge-go2-so101-evidence",
        },
        "claim_boundary": {
            "score_kind": "hand_cost",
            "outcome_kind": "analytic",
            "hardware_executed": False,
            "learned_model_used": False,
            "safety_controller": None,
            "limitations": ["No live robot execution."],
        },
    }


def test_decision_trace_schema_resource_loads() -> None:
    schema = load_decision_trace_schema()

    assert schema["title"] == "WorldForge DecisionTrace v1"
    assert schema["properties"]["schema_version"]["const"] == DECISION_TRACE_SCHEMA_VERSION


def test_validate_decision_trace_accepts_valid_trace() -> None:
    trace = validate_decision_trace(_valid_trace())

    assert trace["schema_version"] == DECISION_TRACE_SCHEMA_VERSION
    assert decision_trace_digest(trace) == decision_trace_digest(trace)


def test_validate_decision_trace_rejects_missing_subgoals() -> None:
    trace = _valid_trace()
    trace["goal"]["sub_goals"] = []

    with pytest.raises(WorldForgeError, match="sub_goals"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_score_without_candidate() -> None:
    trace = _valid_trace()
    trace["scores"][1]["candidate_id"] = "unknown"

    with pytest.raises(WorldForgeError, match="candidate ids mismatch"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_score_rank_drift() -> None:
    trace = _valid_trace()
    trace["scores"][0]["rank"] = 2
    trace["scores"][1]["rank"] = 1

    with pytest.raises(WorldForgeError, match=r"rank .* must match score ordering"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_selected_score_mismatch() -> None:
    trace = _valid_trace()
    trace["selected_action"]["score"] = 0.9

    with pytest.raises(WorldForgeError, match=r"selected_action\.score"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_selected_margin_mismatch() -> None:
    trace = _valid_trace()
    trace["selected_action"]["score_margin"] = 0.1

    with pytest.raises(WorldForgeError, match=r"selected_action\.score_margin"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_counterfactual_score_mismatch() -> None:
    trace = _valid_trace()
    trace["counterfactuals"][0]["score"] = 0.6

    with pytest.raises(WorldForgeError, match=r"counterfactuals\[0\]\.score"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_counterfactual_delta_mismatch() -> None:
    trace = _valid_trace()
    trace["counterfactuals"][0]["delta_vs_selected"] = 0.4

    with pytest.raises(WorldForgeError, match=r"counterfactuals\[0\]\.delta_vs_selected"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_baseline_regret_mismatch() -> None:
    trace = _valid_trace()
    trace["baseline"]["regret_vs_selected"] = 0.4

    with pytest.raises(WorldForgeError, match=r"baseline\.regret_vs_selected"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_baseline_score_without_baseline_candidate() -> None:
    trace = _valid_trace()
    trace["baseline"]["candidate_id"] = None
    trace["baseline"]["score"] = 123.0
    trace["baseline"]["regret_vs_selected"] = 0.0

    with pytest.raises(WorldForgeError, match=r"baseline\.score"):
        validate_decision_trace(trace)


def test_validate_decision_trace_allows_absent_baseline_candidate_without_score() -> None:
    trace = _valid_trace()
    trace["baseline"]["candidate_id"] = None
    trace["baseline"]["score"] = None
    trace["baseline"]["regret_vs_selected"] = 0.0

    assert validate_decision_trace(trace)["baseline"]["candidate_id"] is None


def test_validate_decision_trace_allows_rank_declared_tied_best_without_lexical_bias() -> None:
    trace = _valid_trace()
    trace["candidate_actions"][0]["candidate_id"] = "z-best"
    trace["scores"][0]["candidate_id"] = "z-best"
    trace["scores"][0]["score"] = 0.2
    trace["scores"][0]["rank"] = 1
    trace["candidate_actions"][1]["candidate_id"] = "a-best"
    trace["scores"][1]["candidate_id"] = "a-best"
    trace["scores"][1]["score"] = 0.2
    trace["scores"][1]["rank"] = 2
    trace["selected_action"]["candidate_id"] = "z-best"
    trace["selected_action"]["score"] = 0.2
    trace["selected_action"]["score_margin"] = 0.0
    trace["counterfactuals"] = [
        {
            "candidate_id": "a-best",
            "score": 0.2,
            "delta_vs_selected": 0.0,
            "why_rejected": "Tie broken by provider order.",
        }
    ]
    trace["baseline"]["candidate_id"] = "a-best"
    trace["baseline"]["score"] = 0.2
    trace["baseline"]["regret_vs_selected"] = 0.0

    assert validate_decision_trace(trace)["selected_action"]["candidate_id"] == "z-best"


def test_validate_decision_trace_rejects_tied_best_when_selected_is_not_rank_one() -> None:
    trace = _valid_trace()
    trace["scores"][0]["score"] = 0.2
    trace["scores"][0]["rank"] = 1
    trace["scores"][1]["score"] = 0.2
    trace["scores"][1]["rank"] = 2
    trace["selected_action"]["candidate_id"] = "stop"
    trace["selected_action"]["score"] = 0.2
    trace["selected_action"]["score_margin"] = 0.0
    trace["counterfactuals"] = [
        {
            "candidate_id": "forward",
            "score": 0.2,
            "delta_vs_selected": 0.0,
            "why_rejected": "Trace declared stop as selected even though it is not rank 1.",
        }
    ]
    trace["baseline"]["score"] = 0.2
    trace["baseline"]["regret_vs_selected"] = 0.0

    with pytest.raises(WorldForgeError, match="rank-1"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_overclaimed_real_outcome() -> None:
    trace = _valid_trace()
    trace["claim_boundary"]["outcome_kind"] = "real_measured"

    with pytest.raises(WorldForgeError, match="outcome_kind must match"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_real_measured_without_hardware() -> None:
    trace = _valid_trace()
    trace["outcome"]["kind"] = "real_measured"
    trace["claim_boundary"]["outcome_kind"] = "real_measured"

    with pytest.raises(WorldForgeError, match="hardware_executed"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_learned_score_without_model_flag() -> None:
    trace = _valid_trace()
    trace["claim_boundary"]["score_kind"] = "learned_latent"

    with pytest.raises(WorldForgeError, match="learned_model_used"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_secret_like_text() -> None:
    trace = _valid_trace()
    trace["task"]["description"] = "Choose a local action with password=hunter2."

    with pytest.raises(WorldForgeError, match="credentials"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_secret_like_keys() -> None:
    trace = _valid_trace()
    trace["observation"]["password"] = "hunter2"

    with pytest.raises(WorldForgeError, match="secret-like key"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_private_ip_text() -> None:
    trace = _valid_trace()
    trace["observation"]["robot_ip"] = "10.0.0.5"

    with pytest.raises(WorldForgeError, match="private or local IP"):
        validate_decision_trace(trace)


@pytest.mark.parametrize("ip_text", ["::1", "fe80::1", "[::1]"])
def test_validate_decision_trace_rejects_private_ipv6_text(ip_text: str) -> None:
    trace = _valid_trace()
    trace["observation"]["robot_ip"] = ip_text

    with pytest.raises(WorldForgeError, match="private or local IP"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_host_local_path_text() -> None:
    trace = _valid_trace()
    trace["observation"]["summary"] = "Loaded replay from /Users/secret/robot.key"

    with pytest.raises(WorldForgeError, match="host-local paths"):
        validate_decision_trace(trace)


def test_validate_decision_trace_allows_benign_url_paths_and_queries() -> None:
    trace = _valid_trace()
    trace["reproducibility"]["model_card_ref"] = "https://example.com/tmp/model?view=full#card"

    assert validate_decision_trace(trace)["reproducibility"]["model_card_ref"].startswith(
        "https://example.com/tmp/model"
    )


def test_validate_decision_trace_rejects_sensitive_url_query() -> None:
    trace = _valid_trace()
    trace["reproducibility"]["model_card_ref"] = "https://example.com/model?token=hunter2"

    with pytest.raises(WorldForgeError, match="sensitive URL query"):
        validate_decision_trace(trace)


def test_validate_decision_trace_rejects_sensitive_url_path() -> None:
    trace = _valid_trace()
    trace["reproducibility"]["model_card_ref"] = "https://example.com/token/hunter2"

    with pytest.raises(WorldForgeError, match="sensitive URL path"):
        validate_decision_trace(trace)


def test_load_decision_trace_schema_wraps_resource_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingResource:
        def joinpath(self, _name: str) -> MissingResource:
            return self

        def read_text(self) -> str:
            raise FileNotFoundError("missing schema")

    monkeypatch.setattr(
        decision_trace_module.resources,
        "files",
        lambda _package: MissingResource(),
    )

    with pytest.raises(WorldForgeError, match=r"decision_trace\.v1\.schema\.json"):
        load_decision_trace_schema()
