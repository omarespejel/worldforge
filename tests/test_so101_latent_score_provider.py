from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from worldforge.demos.so101_latent_score_provider import SO101LatentScoreProvider
from worldforge.models import WorldForgeError


def test_so101_latent_score_provider_imports_without_loading_numpy() -> None:
    assert SO101LatentScoreProvider.name == "so101-latent-score-provider"


def test_so101_referee_smoke_script_help_imports_without_referee(
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_so101_referee_smoke.py"
    spec = importlib.util.spec_from_file_location("run_so101_referee_smoke_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])

    assert exc_info.value.code == 0
    assert "external held-out referee" in capsys.readouterr().out


def test_so101_referee_smoke_metric_summary_uses_clear_bad_decoys() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_so101_referee_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "run_so101_referee_smoke_metric_test", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class RefereeWithoutMetricSummary:
        pass

    summary = module._metric_summary(
        RefereeWithoutMetricSummary(),
        {
            "decoy_beat_rate": {
                "no_motion": 0.1,
                "scale_half": 0.3,
                "overshoot": 0.6,
                "reverse": 0.5,
                "random_other": 0.8,
                "jitter": 0.7,
            }
        },
    )

    assert summary == {
        "fair_beat_rate_clearbad": 0.65,
        "near_duplicate_beat_rate": 0.2,
    }


def test_so101_referee_smoke_metric_summary_prefers_referee_owned_summary() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_so101_referee_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "run_so101_referee_smoke_referee_metric_test", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class RefereeWithMetricSummary:
        @staticmethod
        def metric_summary(row: dict[str, object]) -> dict[str, float]:
            assert row["decoy_beat_rate"] == {"overshoot": 0.1}
            return {
                "fair_beat_rate_clearbad": 0.12345,
                "near_duplicate_beat_rate": 0.54321,
            }

    summary = module._metric_summary(
        RefereeWithMetricSummary(),
        {"decoy_beat_rate": {"overshoot": 0.1}},
    )

    assert summary == {
        "fair_beat_rate_clearbad": 0.1235,
        "near_duplicate_beat_rate": 0.5432,
    }


def test_so101_latent_score_provider_scores_candidate_deltas(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    model_name = "vision_mlp_h1"
    weights_path = tmp_path / "weights.npz"
    metadata_path = tmp_path / "metadata.json"
    np.savez_compressed(
        weights_path,
        selected_model=np.asarray(model_name),
        latent_dim=np.asarray(2, dtype=np.int64),
        **{
            f"{model_name}__x_mean": np.zeros(3, dtype=np.float32),
            f"{model_name}__x_scale": np.ones(3, dtype=np.float32),
            f"{model_name}__w1": np.asarray([[0.0], [0.0], [1.0]], dtype=np.float32),
            f"{model_name}__b1": np.zeros(1, dtype=np.float32),
            f"{model_name}__w2": np.asarray([[-1.0, 1.0]], dtype=np.float32),
            f"{model_name}__b2": np.zeros(2, dtype=np.float32),
        },
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "worldforge.so101_latent_score_provider",
                "selected_model": model_name,
                "models": {
                    model_name: {
                        "variant": "vision_mlp",
                        "horizon": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    provider = SO101LatentScoreProvider(
        weights_path=weights_path,
        metadata_path=metadata_path,
    )

    result = provider.score_actions(
        info={
            "current_latent": [1.0, 0.0],
            "goal_latent": [0.0, 1.0],
        },
        action_candidates=[
            {"candidate_id": "stay", "action_delta": [0.0]},
            {"candidate_id": "toward_goal", "action_delta": [3.0]},
        ],
    )

    assert result.best_index == 1
    assert result.scores[1] < result.scores[0]
    assert result.metadata["score_kind"] == "learned_latent"
    assert result.metadata["target"] == "future_latent_residual"


def test_so101_latent_score_provider_scores_ranked_residual_model(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    model_name = "ranked_vision_proprio_mlp_h5"
    weights_path = tmp_path / "weights.npz"
    metadata_path = tmp_path / "metadata.json"
    np.savez_compressed(
        weights_path,
        selected_model=np.asarray(model_name),
        latent_dim=np.asarray(2, dtype=np.int64),
        **{
            f"{model_name}__x_mean": np.zeros(4, dtype=np.float32),
            f"{model_name}__x_scale": np.ones(4, dtype=np.float32),
            f"{model_name}__w1": np.asarray([[0.0], [0.0], [1.0], [0.0]], dtype=np.float32),
            f"{model_name}__b1": np.zeros(1, dtype=np.float32),
            f"{model_name}__w2": np.asarray([[-1.0, 1.0]], dtype=np.float32),
            f"{model_name}__b2": np.zeros(2, dtype=np.float32),
        },
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "worldforge.so101_latent_score_provider",
                "selected_model": model_name,
                "models": {
                    model_name: {
                        "variant": "ranked_vision_proprio_mlp",
                        "horizon": 5,
                        "target": "ranked_future_latent_residual",
                        "target_scale": "pairwise_clearbad_decoy_margin",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    provider = SO101LatentScoreProvider(
        weights_path=weights_path,
        metadata_path=metadata_path,
    )

    with pytest.raises(WorldForgeError, match="current_state"):
        provider.score_actions(
            info={
                "current_latent": [1.0, 0.0],
                "goal_latent": [0.0, 1.0],
            },
            action_candidates=[{"candidate_id": "toward_goal", "action_delta": [3.0]}],
        )

    result = provider.score_actions(
        info={
            "current_latent": [1.0, 0.0],
            "goal_latent": [0.0, 1.0],
            "current_state": [0.0],
        },
        action_candidates=[
            {"candidate_id": "stay", "action_delta": [0.0]},
            {"candidate_id": "toward_goal", "action_delta": [3.0]},
        ],
    )

    assert result.best_index == 1
    assert result.scores[1] < result.scores[0]
    assert result.metadata["target"] == "ranked_future_latent_residual"
    assert result.metadata["target_scale"] == "pairwise_clearbad_decoy_margin"
    assert provider.predict_latent(
        current_latent=[1.0, 0.0],
        action_delta=[3.0],
        current_state=[0.0],
    )


def test_so101_latent_score_provider_scores_goal_conditioned_cost(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    model_name = "goal_score_mlp_h1"
    weights_path = tmp_path / "weights.npz"
    metadata_path = tmp_path / "metadata.json"
    np.savez_compressed(
        weights_path,
        selected_model=np.asarray(model_name),
        latent_dim=np.asarray(2, dtype=np.int64),
        **{
            f"{model_name}__x_mean": np.zeros(5, dtype=np.float32),
            f"{model_name}__x_scale": np.ones(5, dtype=np.float32),
            f"{model_name}__w1": np.asarray([[0.0], [0.0], [0.0], [0.0], [1.0]], dtype=np.float32),
            f"{model_name}__b1": np.zeros(1, dtype=np.float32),
            f"{model_name}__w2": np.asarray([[-1.0]], dtype=np.float32),
            f"{model_name}__b2": np.asarray([1.0], dtype=np.float32),
        },
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "worldforge.so101_latent_score_provider",
                "selected_model": model_name,
                "models": {
                    model_name: {
                        "variant": "goal_score_mlp",
                        "horizon": 1,
                        "target": "goal_conditioned_cost",
                        "target_scale": "joint_distance_to_goal",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    provider = SO101LatentScoreProvider(
        weights_path=weights_path,
        metadata_path=metadata_path,
    )

    result = provider.score_actions(
        info={
            "current_latent": [1.0, 0.0],
            "goal_latent": [0.0, 1.0],
        },
        action_candidates=[
            {"candidate_id": "stay", "action_delta": [0.0]},
            {"candidate_id": "toward_goal", "action_delta": [3.0]},
        ],
    )

    assert result.best_index == 1
    assert result.scores[1] < result.scores[0]
    assert result.metadata["target"] == "goal_conditioned_cost"
    assert result.metadata["target_scale"] == "joint_distance_to_goal"
    with pytest.raises(WorldForgeError, match="predict_latent"):
        provider.predict_latent(current_latent=[1.0, 0.0], action_delta=[3.0])


def test_so101_latent_score_provider_scores_history_goal_cost(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    model_name = "goal_history_score_mlp_h1"
    weights_path = tmp_path / "weights.npz"
    metadata_path = tmp_path / "metadata.json"
    np.savez_compressed(
        weights_path,
        selected_model=np.asarray(model_name),
        latent_dim=np.asarray(2, dtype=np.int64),
        **{
            f"{model_name}__x_mean": np.zeros(9, dtype=np.float32),
            f"{model_name}__x_scale": np.ones(9, dtype=np.float32),
            f"{model_name}__w1": np.asarray(
                [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0]],
                dtype=np.float32,
            ),
            f"{model_name}__b1": np.zeros(1, dtype=np.float32),
            f"{model_name}__w2": np.asarray([[-1.0]], dtype=np.float32),
            f"{model_name}__b2": np.asarray([1.0], dtype=np.float32),
        },
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "worldforge.so101_latent_score_provider",
                "selected_model": model_name,
                "models": {
                    model_name: {
                        "variant": "goal_history_score_mlp",
                        "horizon": 1,
                        "target": "goal_conditioned_cost",
                        "target_scale": "joint_distance_to_goal",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    provider = SO101LatentScoreProvider(
        weights_path=weights_path,
        metadata_path=metadata_path,
    )

    with pytest.raises(WorldForgeError, match="history_latents"):
        provider.score_actions(
            info={
                "current_latent": [1.0, 0.0],
                "goal_latent": [0.0, 1.0],
            },
            action_candidates=[{"candidate_id": "toward_goal", "action_delta": [3.0]}],
        )

    result = provider.score_actions(
        info={
            "current_latent": [1.0, 0.0],
            "goal_latent": [0.0, 1.0],
            "history_latents": [[1.0, 0.0], [0.8, 0.2], [0.6, 0.4]],
        },
        action_candidates=[
            {"candidate_id": "stay", "action_delta": [0.0]},
            {"candidate_id": "toward_goal", "action_delta": [3.0]},
        ],
    )

    assert result.best_index == 1
    assert result.metadata["variant"] == "goal_history_score_mlp"
