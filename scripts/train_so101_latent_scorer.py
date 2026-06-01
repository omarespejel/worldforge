#!/usr/bin/env python3
"""Train SO-101 latent score-provider artifacts from an existing DINOv2 cache.

This is a scorer handoff script, not a referee. It trains candidate learned scorers on train
episodes, selects with target-aware validation metrics, and writes provider-ready weights for
Claude's independent held-out referee to grade. Ranked residual variants optimize the same
candidate-ranking objective used by the SO-101 referee's fair clear-bad decoy gate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASET_DIR = Path(
    "/Users/espejelomar/StarkNet/zk-ai/hackathons/worldforge-so101-trace-judge/data/"
    "svla_so101_pickplace"
)
DEFAULT_CACHE_PATH = Path(
    "/Users/espejelomar/StarkNet/zk-ai/hackathons/worldforge-eval-referee/latent_cache.npz"
)
DEFAULT_SIDECAR_PATH = DEFAULT_CACHE_PATH.with_name("latent_cache_meta.json")
DEFAULT_OUTPUT_DIR = Path(".worldforge/so101-latent-scorer")
GRIPPER_INDEX = 5
HISTORY_FRAME_COUNT = 3
DECOY_NAMES = ("no_motion", "reverse", "scale_half", "overshoot", "random_other", "jitter")
CLEAR_BAD_DECOYS = ("overshoot", "reverse", "random_other", "jitter")
LATENT_DYNAMICS_VARIANTS = {"vision_mlp", "vision_proprio_mlp"}
RANKED_RESIDUAL_VARIANTS = {"ranked_vision_mlp", "ranked_vision_proprio_mlp"}
GOAL_CONDITIONED_SCORE_VARIANTS = {
    "goal_score_mlp",
    "goal_proprio_score_mlp",
    "goal_history_score_mlp",
    "goal_history_proprio_score_mlp",
}
GOAL_HISTORY_SCORE_VARIANTS = {"goal_history_score_mlp", "goal_history_proprio_score_mlp"}
SUPPORTED_VARIANTS = (
    LATENT_DYNAMICS_VARIANTS | RANKED_RESIDUAL_VARIANTS | GOAL_CONDITIONED_SCORE_VARIANTS
)


@dataclass(frozen=True)
class DatasetArrays:
    states: Any
    actions: Any
    episode_indices: Any
    frame_indices: Any


@dataclass(frozen=True)
class SplitEpisodes:
    train: list[int]
    validation: list[int]
    test: list[int]


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    weight_decay: float
    early_stop_patience: int
    ranking_margin: float
    auxiliary_weight: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", default="vision_mlp,vision_proprio_mlp")
    parser.add_argument("--horizons", default="1,5,15")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--auxiliary-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    variants = _split_csv(args.variants)
    horizons = [int(value) for value in _split_csv(args.horizons)]
    config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        ranking_margin=args.ranking_margin,
        auxiliary_weight=args.auxiliary_weight,
    )
    result = train_so101_latent_scorers(
        dataset_dir=args.dataset_dir,
        cache_path=args.cache,
        sidecar_path=args.sidecar,
        output_dir=args.out_dir,
        variants=variants,
        horizons=horizons,
        config=config,
    )
    print(f"WROTE {result['weights_path']}")
    print(f"WROTE {result['metadata_path']}")
    selected_metrics = result["models"][result["selected_model"]]["validation"]
    if "mean_cosine" in selected_metrics:
        summary_metric = f"val_cosine={selected_metrics['mean_cosine']:.6f}"
    elif "fair_beat_rate_clearbad" in selected_metrics:
        summary_metric = (
            f"val_fair_clearbad={selected_metrics['fair_beat_rate_clearbad']:.6f} "
            f"val_mrr={selected_metrics['mrr']:.6f}"
        )
    else:
        summary_metric = (
            f"val_mse={selected_metrics['mse']:.6f} "
            f"val_corr={selected_metrics['target_correlation']:.6f}"
        )
    print(f"selected={result['selected_model']} {summary_metric}")
    return 0


def train_so101_latent_scorers(
    *,
    dataset_dir: Path,
    cache_path: Path,
    sidecar_path: Path,
    output_dir: Path,
    variants: list[str],
    horizons: list[int],
    config: TrainingConfig,
) -> dict[str, Any]:
    np = _import_numpy()
    pq = _import_pyarrow_parquet()
    output_dir.mkdir(parents=True, exist_ok=True)

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    cache = np.load(cache_path, allow_pickle=False)
    latents = cache["latents"].astype(np.float32)
    arrays = _load_dataset_arrays(dataset_dir, pq, np)
    if len(arrays.states) != latents.shape[0]:
        raise ValueError("SO-101 state rows and latent cache rows are not aligned.")

    splits = _split_episodes_from_sidecar(sidecar)
    rng = np.random.default_rng(config.seed)
    weights: dict[str, Any] = {}
    models: dict[str, Any] = {}
    selected_name: str | None = None
    selected_key: tuple[float, float] | None = None

    for variant in variants:
        _validate_variant(variant)
        for horizon in horizons:
            name = f"{variant}_h{horizon}"
            train = _build_pairs(
                arrays=arrays,
                latents=latents,
                episodes=splits.train,
                horizon=horizon,
                variant=variant,
                seed=config.seed,
                np=np,
            )
            validation = _build_pairs(
                arrays=arrays,
                latents=latents,
                episodes=splits.validation,
                horizon=horizon,
                variant=variant,
                seed=config.seed,
                np=np,
            )
            test = _build_pairs(
                arrays=arrays,
                latents=latents,
                episodes=splits.test,
                horizon=horizon,
                variant=variant,
                seed=config.seed,
                np=np,
            )
            if train["target_kind"] == "ranked_future_latent_residual":
                model = _fit_ranked_residual_mlp(
                    train,
                    validation=validation,
                    config=config,
                    rng=rng,
                    np=np,
                )
            else:
                model = _fit_mlp(train, config=config, rng=rng, np=np)
            metrics = {
                "train": _evaluate_model(model, train, np=np),
                "validation": _evaluate_model(model, validation, np=np),
                "test": _evaluate_model(model, test, np=np),
            }
            _store_model_arrays(weights, name=name, model=model)
            models[name] = {
                "variant": variant,
                "horizon": horizon,
                "input_dim": int(model["x_mean"].shape[0]),
                "hidden_dim": config.hidden_dim,
                "output_dim": int(model["b2"].shape[0]),
                "target": _variant_target(variant),
                "target_scale": _variant_target_scale(variant),
                "activation": "tanh",
                **metrics,
            }
            score_key = _selection_key(models[name])
            if selected_key is None or score_key > selected_key:
                selected_key = score_key
                selected_name = name

    if selected_name is None:  # pragma: no cover - guarded by variants/horizons
        raise ValueError("No scorer models were trained.")

    weights["selected_model"] = np.asarray(selected_name)
    weights["latent_dim"] = np.asarray(latents.shape[1], dtype=np.int64)
    weights_path = output_dir / "so101_latent_scorers.npz"
    np.savez_compressed(weights_path, **weights)
    metadata = {
        "schema_version": 1,
        "artifact_kind": "worldforge.so101_latent_score_provider",
        "selected_model": selected_name,
        "models": models,
        "cache": {
            "source": "existing_side_camera_dinov2_cache",
            "model": sidecar["cache"]["model"],
            "camera": sidecar["cache"]["camera"],
            "token": sidecar["cache"]["token"],
            "normalization": sidecar["cache"]["normalization"],
            "sha256": sidecar["cache"]["sha256"],
            "frames_sha256": sidecar["frames_sha256"],
            "content_sha256": sidecar["content_sha256"],
        },
        "dataset": sidecar["dataset"],
        "split": sidecar["split"],
        "training": {
            "seed": config.seed,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "early_stop_patience": config.early_stop_patience,
            "ranking_margin": config.ranking_margin,
            "auxiliary_weight": config.auxiliary_weight,
            "selection": (
                "Target-aware validation selection: ranked residual models use highest fair "
                "clear-bad decoy beat rate, then MRR; latent-dynamics models use highest "
                "mean_cosine then lowest mse; goal-conditioned cost models use lowest mse then "
                "highest target_correlation."
            ),
            "referee_boundary": (
                "This artifact is a scorer handoff only. The independent held-out referee "
                "must decide whether it beats baselines and chance."
            ),
        },
        "kill_criterion": (
            "Do not claim learned-scorer value unless the independent referee shows improvement "
            "over proprio/hand-cost baselines on fair decoy/progress metrics, with no "
            "shuffled-label leak. Exact-match top-1 is diagnostic for near-duplicate decoys, "
            "not the pass/fail gate."
        ),
    }
    metadata_path = output_dir / "so101_latent_scorer_meta.json"
    metadata_path.write_text(_dump_json(metadata), encoding="utf-8")
    return {
        "weights_path": str(weights_path),
        "metadata_path": str(metadata_path),
        "selected_model": selected_name,
        "models": models,
    }


def _build_pairs(
    *,
    arrays: DatasetArrays,
    latents: Any,
    episodes: list[int],
    horizon: int,
    variant: str,
    seed: int,
    np: Any,
) -> dict[str, Any]:
    if variant in GOAL_CONDITIONED_SCORE_VARIANTS:
        return _build_goal_conditioned_score_pairs(
            arrays=arrays,
            latents=latents,
            episodes=episodes,
            variant=variant,
            seed=seed,
            np=np,
        )
    if variant in RANKED_RESIDUAL_VARIANTS:
        return _build_ranked_residual_pairs(
            arrays=arrays,
            latents=latents,
            episodes=episodes,
            horizon=horizon,
            variant=variant,
            seed=seed,
            np=np,
        )
    return _build_latent_dynamics_pairs(
        arrays=arrays,
        latents=latents,
        episodes=episodes,
        horizon=horizon,
        variant=variant,
        np=np,
    )


def _build_latent_dynamics_pairs(
    *,
    arrays: DatasetArrays,
    latents: Any,
    episodes: list[int],
    horizon: int,
    variant: str,
    np: Any,
) -> dict[str, Any]:
    indices: list[int] = []
    future_indices: list[int] = []
    episode_set = set(episodes)
    for episode in episodes:
        rows = np.flatnonzero(arrays.episode_indices == episode)
        if rows.size <= horizon:
            continue
        indices.extend(int(row) for row in rows[:-horizon])
        future_indices.extend(int(row) for row in rows[horizon:])
    if not indices:
        raise ValueError(f"No pairs available for episodes={episodes!r} horizon={horizon}")
    current = np.asarray(indices, dtype=np.int64)
    future = np.asarray(future_indices, dtype=np.int64)
    if not set(arrays.episode_indices[current].tolist()).issubset(episode_set):
        raise ValueError("Pair construction crossed episode boundaries.")
    action_delta = arrays.actions[current] - arrays.states[current]
    parts = [latents[current], action_delta]
    if variant == "vision_proprio_mlp":
        parts.append(arrays.states[current])
    x = np.concatenate(parts, axis=1).astype(np.float32)
    current_latents = latents[current].astype(np.float32)
    y = (latents[future] - current_latents).astype(np.float32)
    return {
        "target_kind": "future_latent_residual",
        "x": x,
        "y": y,
        "current_latents": current_latents,
        "future_latents": latents[future].astype(np.float32),
    }


def _build_ranked_residual_pairs(
    *,
    arrays: DatasetArrays,
    latents: Any,
    episodes: list[int],
    horizon: int,
    variant: str,
    seed: int,
    np: Any,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x_demo: list[Any] = []
    x_decoy: list[Any] = []
    pair_current_latents: list[Any] = []
    pair_goal_latents: list[Any] = []
    pair_future_latents: list[Any] = []
    pair_decoy_names: list[str] = []
    eval_x: list[Any] = []
    eval_current_latents: list[Any] = []
    eval_goal_latents: list[Any] = []
    candidate_names = ("demonstrated", *DECOY_NAMES)
    for episode in episodes:
        rows = np.flatnonzero(arrays.episode_indices == episode)
        if rows.size <= horizon or rows.size < 4:
            continue
        episode_states = arrays.states[rows]
        episode_actions = arrays.actions[rows]
        subgoals = _subgoal_frames(episode_states, np=np)
        for local_t in range(len(rows) - horizon):
            future_subgoals = [index for index in subgoals if index > local_t]
            if not future_subgoals:
                continue
            local_goal = future_subgoals[0]
            global_t = int(rows[local_t])
            current_latent = latents[global_t].astype(np.float32)
            goal_latent = latents[int(rows[local_goal])].astype(np.float32)
            future_latent = latents[int(rows[local_t + horizon])].astype(np.float32)
            current_state = episode_states[local_t]
            demonstrated = episode_actions[local_t]
            delta = demonstrated - current_state
            random_other = episode_actions[int(rng.integers(0, len(episode_actions)))]
            candidates = {
                "demonstrated": demonstrated,
                "no_motion": current_state.copy(),
                "reverse": current_state - delta,
                "scale_half": current_state + 0.5 * delta,
                "overshoot": current_state + 1.5 * delta,
                "random_other": random_other,
                "jitter": demonstrated + rng.normal(0.0, 8.0, size=demonstrated.shape),
            }
            demo_features = _ranked_residual_features(
                current_latent=current_latent,
                action_delta=demonstrated - current_state,
                current_state=current_state,
                variant=variant,
                np=np,
            )
            eval_x.append(
                np.asarray(
                    [
                        _ranked_residual_features(
                            current_latent=current_latent,
                            action_delta=candidates[name] - current_state,
                            current_state=current_state,
                            variant=variant,
                            np=np,
                        )
                        for name in candidate_names
                    ],
                    dtype=np.float32,
                )
            )
            eval_current_latents.append(current_latent)
            eval_goal_latents.append(goal_latent)
            for decoy_name in CLEAR_BAD_DECOYS:
                x_demo.append(demo_features)
                x_decoy.append(
                    _ranked_residual_features(
                        current_latent=current_latent,
                        action_delta=candidates[decoy_name] - current_state,
                        current_state=current_state,
                        variant=variant,
                        np=np,
                    )
                )
                pair_current_latents.append(current_latent)
                pair_goal_latents.append(goal_latent)
                pair_future_latents.append(future_latent)
                pair_decoy_names.append(decoy_name)
    if not x_demo:
        raise ValueError(
            f"No ranked residual pairs available for episodes={episodes!r} horizon={horizon}"
        )
    return {
        "target_kind": "ranked_future_latent_residual",
        "x_demo": np.asarray(x_demo, dtype=np.float32),
        "x_decoy": np.asarray(x_decoy, dtype=np.float32),
        "current_latents": np.asarray(pair_current_latents, dtype=np.float32),
        "goal_latents": np.asarray(pair_goal_latents, dtype=np.float32),
        "future_latents": np.asarray(pair_future_latents, dtype=np.float32),
        "decoy_names": pair_decoy_names,
        "eval_x": np.asarray(eval_x, dtype=np.float32),
        "eval_current_latents": np.asarray(eval_current_latents, dtype=np.float32),
        "eval_goal_latents": np.asarray(eval_goal_latents, dtype=np.float32),
        "candidate_names": candidate_names,
    }


def _ranked_residual_features(
    *,
    current_latent: Any,
    action_delta: Any,
    current_state: Any,
    variant: str,
    np: Any,
) -> Any:
    parts = [current_latent, action_delta]
    if variant == "ranked_vision_proprio_mlp":
        parts.append(current_state)
    return np.concatenate(parts).astype(np.float32)


def _build_goal_conditioned_score_pairs(
    *,
    arrays: DatasetArrays,
    latents: Any,
    episodes: list[int],
    variant: str,
    seed: int,
    np: Any,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    xs: list[Any] = []
    ys: list[list[float]] = []
    for episode in episodes:
        rows = np.flatnonzero(arrays.episode_indices == episode)
        if rows.size < 4:
            continue
        episode_states = arrays.states[rows]
        episode_actions = arrays.actions[rows]
        subgoals = _subgoal_frames(episode_states, np=np)
        for local_t in range(len(rows) - 1):
            future_subgoals = [index for index in subgoals if index > local_t]
            if not future_subgoals:
                continue
            local_goal = future_subgoals[0]
            global_t = int(rows[local_t])
            global_goal = int(rows[local_goal])
            current_state = episode_states[local_t]
            goal_state = episode_states[local_goal]
            demonstrated = episode_actions[local_t]
            delta = demonstrated - current_state
            random_other = episode_actions[int(rng.integers(0, len(episode_actions)))]
            candidates = {
                "demonstrated": demonstrated,
                "no_motion": current_state.copy(),
                "reverse": current_state - delta,
                "scale_half": current_state + 0.5 * delta,
                "overshoot": current_state + 1.5 * delta,
                "random_other": random_other,
                "jitter": demonstrated + rng.normal(0.0, 8.0, size=demonstrated.shape),
            }
            for candidate in candidates.values():
                action_delta = candidate - current_state
                if variant in GOAL_HISTORY_SCORE_VARIANTS:
                    observation = _history_latents(latents, rows, local_t, np=np)
                else:
                    observation = latents[global_t]
                parts = [observation, latents[global_goal], action_delta]
                if variant in {"goal_proprio_score_mlp", "goal_history_proprio_score_mlp"}:
                    parts.append(current_state)
                xs.append(np.concatenate(parts))
                ys.append([float(np.linalg.norm(candidate - goal_state))])
    if not xs:
        raise ValueError(f"No goal-conditioned score pairs available for episodes={episodes!r}")
    return {
        "target_kind": "goal_conditioned_cost",
        "x": np.asarray(xs, dtype=np.float32),
        "y": np.asarray(ys, dtype=np.float32),
    }


def _subgoal_frames(states: Any, *, np: Any) -> list[int]:
    gripper = states[:, GRIPPER_INDEX]
    deltas = np.diff(gripper)
    if len(deltas) < 2:
        return [len(gripper) - 1]
    transition_indices = [int(index) + 1 for index in np.argsort(-np.abs(deltas))[:2]]
    return sorted({*transition_indices, len(gripper) - 1})


def _history_latents(latents: Any, episode_rows: Any, local_index: int, *, np: Any) -> Any:
    indices = [
        int(episode_rows[max(0, local_index - offset)])
        for offset in range(HISTORY_FRAME_COUNT - 1, -1, -1)
    ]
    return latents[np.asarray(indices, dtype=np.int64)].reshape(-1)


def _fit_ranked_residual_mlp(
    data: dict[str, Any],
    *,
    validation: dict[str, Any],
    config: TrainingConfig,
    rng: Any,
    np: Any,
) -> dict[str, Any]:
    x_all = np.concatenate([data["x_demo"], data["x_decoy"]], axis=0).astype(np.float32)
    x_mean = x_all.mean(axis=0, keepdims=True).astype(np.float32)
    x_scale = x_all.std(axis=0, keepdims=True).astype(np.float32)
    x_scale = np.where(x_scale < 1e-6, 1.0, x_scale).astype(np.float32)
    input_dim = x_all.shape[1]
    output_dim = data["current_latents"].shape[1]
    hidden_dim = config.hidden_dim
    w1 = rng.normal(0.0, 1.0 / max(1.0, input_dim**0.5), size=(input_dim, hidden_dim)).astype(
        np.float32
    )
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    w2 = rng.normal(0.0, 1.0 / max(1.0, hidden_dim**0.5), size=(hidden_dim, output_dim)).astype(
        np.float32
    )
    b2 = np.zeros(output_dim, dtype=np.float32)
    params = [w1, b1, w2, b2]
    adam_m = [np.zeros_like(param) for param in params]
    adam_v = [np.zeros_like(param) for param in params]
    count = data["x_demo"].shape[0]
    batch_size = min(config.batch_size, count)
    step = 0
    losses: list[float] = []
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    best_params = [param.copy() for param in params]
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        order = rng.permutation(count)
        epoch_loss = 0.0
        for start in range(0, count, batch_size):
            batch = order[start : start + batch_size]
            batch_result = _ranked_residual_batch_loss_and_grads(
                params=params,
                x_mean=x_mean,
                x_scale=x_scale,
                x_demo=data["x_demo"][batch],
                x_decoy=data["x_decoy"][batch],
                current_latents=data["current_latents"][batch],
                goal_latents=data["goal_latents"][batch],
                future_latents=data["future_latents"][batch],
                config=config,
                np=np,
            )
            epoch_loss += batch_result["loss"] * len(batch)
            step += 1
            _adam_update(
                params,
                batch_result["grads"],
                adam_m,
                adam_v,
                step=step,
                learning_rate=config.learning_rate,
                np=np,
            )
        losses.append(epoch_loss / count)
        candidate = {
            "x_mean": x_mean.reshape(-1),
            "x_scale": x_scale.reshape(-1),
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "loss_first": float(losses[0]),
            "loss_last": float(losses[-1]),
            "best_epoch": epoch + 1,
            "epochs_trained": epoch + 1,
            "ranking_margin": config.ranking_margin,
        }
        metrics = _evaluate_ranked_residual_model(candidate, validation, np=np)
        key = (
            float(metrics["fair_beat_rate_clearbad"]),
            float(metrics["mrr"]),
            -float(metrics["ranking_loss"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch + 1
            best_params = [param.copy() for param in params]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            config.early_stop_patience > 0
            and epochs_without_improvement >= config.early_stop_patience
        ):
            break
    w1[:], b1[:], w2[:], b2[:] = best_params
    return {
        "x_mean": x_mean.reshape(-1),
        "x_scale": x_scale.reshape(-1),
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "best_epoch": best_epoch,
        "epochs_trained": len(losses),
        "ranking_margin": config.ranking_margin,
    }


def _ranked_residual_batch_loss_and_grads(
    *,
    params: list[Any],
    x_mean: Any,
    x_scale: Any,
    x_demo: Any,
    x_decoy: Any,
    current_latents: Any,
    goal_latents: Any,
    future_latents: Any,
    config: TrainingConfig,
    np: Any,
) -> dict[str, Any]:
    w1, b1, w2, b2 = params
    demo = _forward_ranked_residual(
        x_demo, current_latents, x_mean=x_mean, x_scale=x_scale, w1=w1, b1=b1, w2=w2, b2=b2, np=np
    )
    decoy = _forward_ranked_residual(
        x_decoy, current_latents, x_mean=x_mean, x_scale=x_scale, w1=w1, b1=b1, w2=w2, b2=b2, np=np
    )
    demo_cost = _latent_cost(demo["pred"], goal_latents, np=np)
    decoy_cost = _latent_cost(decoy["pred"], goal_latents, np=np)
    violations = config.ranking_margin + demo_cost - decoy_cost
    active = violations > 0.0
    count = max(1, len(violations))
    ranking_loss = float(np.maximum(violations, 0.0).mean())
    grad_demo_pred = np.zeros_like(demo["pred"], dtype=np.float32)
    grad_decoy_pred = np.zeros_like(decoy["pred"], dtype=np.float32)
    if active.any():
        grad_demo_pred += _latent_cost_grad(demo["pred"], goal_latents, np=np) * (
            active[:, None] / count
        )
        grad_decoy_pred -= _latent_cost_grad(decoy["pred"], goal_latents, np=np) * (
            active[:, None] / count
        )
    aux_loss = 0.0
    if config.auxiliary_weight > 0.0:
        aux_err = demo["pred"] - future_latents
        aux_loss = float(np.mean(aux_err * aux_err))
        grad_demo_pred += (
            config.auxiliary_weight * 2.0 * aux_err / max(1, aux_err.shape[0] * aux_err.shape[1])
        )
    demo_grads = _backward_ranked_residual(demo, grad_demo_pred, w2=w2, np=np)
    decoy_grads = _backward_ranked_residual(decoy, grad_decoy_pred, w2=w2, np=np)
    grads = [
        demo_grads[0] + decoy_grads[0] + config.weight_decay * w1,
        demo_grads[1] + decoy_grads[1],
        demo_grads[2] + decoy_grads[2] + config.weight_decay * w2,
        demo_grads[3] + decoy_grads[3],
    ]
    return {
        "loss": ranking_loss + config.auxiliary_weight * aux_loss,
        "grads": grads,
    }


def _forward_ranked_residual(
    x: Any,
    current_latents: Any,
    *,
    x_mean: Any,
    x_scale: Any,
    w1: Any,
    b1: Any,
    w2: Any,
    b2: Any,
    np: Any,
) -> dict[str, Any]:
    x_norm = (x - x_mean) / x_scale
    hidden = np.tanh(x_norm @ w1 + b1)
    residual = hidden @ w2 + b2
    raw = current_latents + residual
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    norm = np.where(norm < 1e-9, 1.0, norm)
    pred = raw / norm
    return {
        "x_norm": x_norm,
        "hidden": hidden,
        "pred": pred,
        "norm": norm,
    }


def _backward_ranked_residual(
    result: dict[str, Any], grad_pred: Any, *, w2: Any, np: Any
) -> list[Any]:
    pred = result["pred"]
    norm = result["norm"]
    grad_raw = (grad_pred - pred * np.sum(grad_pred * pred, axis=1, keepdims=True)) / norm
    hidden = result["hidden"]
    grad_w2 = hidden.T @ grad_raw
    grad_b2 = grad_raw.sum(axis=0)
    grad_hidden = grad_raw @ w2.T
    grad_pre = grad_hidden * (1.0 - hidden * hidden)
    grad_w1 = result["x_norm"].T @ grad_pre
    grad_b1 = grad_pre.sum(axis=0)
    return [grad_w1, grad_b1, grad_w2, grad_b2]


def _latent_cost(predicted: Any, goal_latents: Any, *, np: Any) -> Any:
    err = predicted - goal_latents
    return np.sqrt(np.sum(err * err, axis=1) + 1e-9)


def _latent_cost_grad(predicted: Any, goal_latents: Any, *, np: Any) -> Any:
    err = predicted - goal_latents
    cost = _latent_cost(predicted, goal_latents, np=np)
    return err / np.maximum(cost[:, None], 1e-9)


def _fit_mlp(data: dict[str, Any], *, config: TrainingConfig, rng: Any, np: Any) -> dict[str, Any]:
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    x_mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    x_scale = x.std(axis=0, keepdims=True).astype(np.float32)
    x_scale = np.where(x_scale < 1e-6, 1.0, x_scale).astype(np.float32)
    x_norm = (x - x_mean) / x_scale
    input_dim = x_norm.shape[1]
    output_dim = y.shape[1]
    hidden_dim = config.hidden_dim
    w1 = rng.normal(0.0, 1.0 / max(1.0, input_dim**0.5), size=(input_dim, hidden_dim)).astype(
        np.float32
    )
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    w2 = rng.normal(0.0, 1.0 / max(1.0, hidden_dim**0.5), size=(hidden_dim, output_dim)).astype(
        np.float32
    )
    b2 = np.zeros(output_dim, dtype=np.float32)
    params = [w1, b1, w2, b2]
    adam_m = [np.zeros_like(param) for param in params]
    adam_v = [np.zeros_like(param) for param in params]
    step = 0
    losses: list[float] = []
    count = x_norm.shape[0]
    batch_size = min(config.batch_size, count)
    for _epoch in range(config.epochs):
        order = rng.permutation(count)
        epoch_loss = 0.0
        for start in range(0, count, batch_size):
            batch = order[start : start + batch_size]
            xb = x_norm[batch]
            yb = y[batch]
            hidden = np.tanh(xb @ w1 + b1)
            pred = hidden @ w2 + b2
            err = pred - yb
            epoch_loss += float(np.mean(err * err)) * len(batch)
            grad_pred = (2.0 / max(1, len(batch))) * err
            grad_w2 = hidden.T @ grad_pred + config.weight_decay * w2
            grad_b2 = grad_pred.sum(axis=0)
            grad_hidden = grad_pred @ w2.T
            grad_pre = grad_hidden * (1.0 - hidden * hidden)
            grad_w1 = xb.T @ grad_pre + config.weight_decay * w1
            grad_b1 = grad_pre.sum(axis=0)
            step += 1
            _adam_update(
                params,
                [grad_w1, grad_b1, grad_w2, grad_b2],
                adam_m,
                adam_v,
                step=step,
                learning_rate=config.learning_rate,
                np=np,
            )
        losses.append(epoch_loss / count)
    return {
        "x_mean": x_mean.reshape(-1),
        "x_scale": x_scale.reshape(-1),
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
    }


def _adam_update(
    params: list[Any],
    grads: list[Any],
    adam_m: list[Any],
    adam_v: list[Any],
    *,
    step: int,
    learning_rate: float,
    np: Any,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for index, (param, grad) in enumerate(zip(params, grads)):  # noqa: B905 - keep py3.9 host
        adam_m[index][:] = beta1 * adam_m[index] + (1.0 - beta1) * grad
        adam_v[index][:] = beta2 * adam_v[index] + (1.0 - beta2) * (grad * grad)
        m_hat = adam_m[index] / (1.0 - beta1**step)
        v_hat = adam_v[index] / (1.0 - beta2**step)
        param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def _evaluate_model(model: dict[str, Any], data: dict[str, Any], *, np: Any) -> dict[str, Any]:
    if data["target_kind"] == "goal_conditioned_cost":
        return _evaluate_score_model(model, data, np=np)
    if data["target_kind"] == "ranked_future_latent_residual":
        return _evaluate_ranked_residual_model(model, data, np=np)
    pred = _predict_residual_model(model, data["x"], data["current_latents"], np=np)
    target = data["future_latents"]
    err = pred - target
    cosine = np.sum(pred * target, axis=1)
    return {
        "pairs": len(pred),
        "mse": round(float(np.mean(err * err)), 8),
        "mean_cosine": round(float(np.mean(cosine)), 8),
        "loss_first": round(float(model["loss_first"]), 8),
        "loss_last": round(float(model["loss_last"]), 8),
    }


def _evaluate_ranked_residual_model(
    model: dict[str, Any], data: dict[str, Any], *, np: Any
) -> dict[str, Any]:
    x = data["eval_x"]
    item_count, candidate_count, input_dim = x.shape
    current = np.repeat(data["eval_current_latents"], candidate_count, axis=0)
    goal = np.repeat(data["eval_goal_latents"], candidate_count, axis=0)
    pred = _predict_residual_model(
        model, x.reshape(item_count * candidate_count, input_dim), current, np=np
    )
    costs = _latent_cost(pred, goal, np=np).reshape(item_count, candidate_count)
    top1 = 0.0
    mrr = 0.0
    decoy_beat_rates: dict[str, float] = {}
    candidate_names = list(data["candidate_names"])
    for index, name in enumerate(candidate_names[1:], start=1):
        decoy_beat_rates[name] = float(np.mean(costs[:, 0] < costs[:, index]))
    order = np.argsort(costs, axis=1)
    top1 = float(np.mean(order[:, 0] == 0))
    ranks = np.argmax(order == 0, axis=1) + 1
    mrr = float(np.mean(1.0 / ranks))
    clear_bad = [decoy_beat_rates[name] for name in CLEAR_BAD_DECOYS if name in decoy_beat_rates]
    ranking_loss = _ranked_residual_ranking_loss(model, data, np=np)
    pred_future = _predict_residual_model(
        model,
        data["x_demo"],
        data["current_latents"],
        np=np,
    )
    aux_err = pred_future - data["future_latents"]
    return {
        "pairs": int(data["x_demo"].shape[0]),
        "items": int(item_count),
        "ranking_loss": round(float(ranking_loss), 8),
        "future_latent_mse": round(float(np.mean(aux_err * aux_err)), 8),
        "top1_vs_demonstrated": round(top1, 4),
        "mrr": round(mrr, 4),
        "fair_beat_rate_clearbad": round(float(sum(clear_bad) / max(1, len(clear_bad))), 4),
        "decoy_beat_rate": {
            name: round(value, 4) for name, value in sorted(decoy_beat_rates.items())
        },
        "loss_first": round(float(model["loss_first"]), 8),
        "loss_last": round(float(model["loss_last"]), 8),
        "best_epoch": int(model.get("best_epoch", 0)),
        "epochs_trained": int(model.get("epochs_trained", 0)),
    }


def _ranked_residual_ranking_loss(model: dict[str, Any], data: dict[str, Any], *, np: Any) -> float:
    pred_demo = _predict_residual_model(model, data["x_demo"], data["current_latents"], np=np)
    pred_decoy = _predict_residual_model(model, data["x_decoy"], data["current_latents"], np=np)
    demo_cost = _latent_cost(pred_demo, data["goal_latents"], np=np)
    decoy_cost = _latent_cost(pred_decoy, data["goal_latents"], np=np)
    margin = float(model.get("ranking_margin", 0.05))
    return float(np.maximum(margin + demo_cost - decoy_cost, 0.0).mean())


def _evaluate_score_model(
    model: dict[str, Any], data: dict[str, Any], *, np: Any
) -> dict[str, Any]:
    pred = _predict_score_model(model, data["x"], np=np)
    target = data["y"]
    err = pred - target
    if pred.std() > 1e-9 and target.std() > 1e-9:
        corr = float(np.corrcoef(pred.reshape(-1), target.reshape(-1))[0, 1])
    else:
        corr = 0.0
    return {
        "pairs": len(pred),
        "mse": round(float(np.mean(err * err)), 8),
        "mean_abs_error": round(float(np.mean(np.abs(err))), 8),
        "target_correlation": round(corr, 8),
        "loss_first": round(float(model["loss_first"]), 8),
        "loss_last": round(float(model["loss_last"]), 8),
    }


def _predict_residual_model(model: dict[str, Any], x: Any, current_latents: Any, *, np: Any) -> Any:
    x_norm = (x - model["x_mean"]) / model["x_scale"]
    hidden = np.tanh(x_norm @ model["w1"] + model["b1"])
    residual = hidden @ model["w2"] + model["b2"]
    pred = current_latents + residual
    norm = np.linalg.norm(pred, axis=1, keepdims=True)
    norm = np.where(norm < 1e-9, 1.0, norm)
    return (pred / norm).astype(np.float32)


def _predict_score_model(model: dict[str, Any], x: Any, *, np: Any) -> Any:
    x_norm = (x - model["x_mean"]) / model["x_scale"]
    hidden = np.tanh(x_norm @ model["w1"] + model["b1"])
    return (hidden @ model["w2"] + model["b2"]).astype(np.float32)


def _selection_key(model_row: dict[str, Any]) -> tuple[float, float]:
    validation = model_row["validation"]
    if model_row["target"] == "ranked_future_latent_residual":
        return (
            float(validation["fair_beat_rate_clearbad"]),
            float(validation["mrr"]),
        )
    if model_row["target"] == "goal_conditioned_cost":
        return (
            -float(validation["mse"]),
            float(validation["target_correlation"]),
        )
    return (
        float(validation["mean_cosine"]),
        -float(validation["mse"]),
    )


def _variant_target(variant: str) -> str:
    if variant in RANKED_RESIDUAL_VARIANTS:
        return "ranked_future_latent_residual"
    if variant in GOAL_CONDITIONED_SCORE_VARIANTS:
        return "goal_conditioned_cost"
    return "future_latent_residual"


def _variant_target_scale(variant: str) -> str:
    if variant in RANKED_RESIDUAL_VARIANTS:
        return "pairwise_clearbad_decoy_margin"
    if variant in GOAL_CONDITIONED_SCORE_VARIANTS:
        return "joint_distance_to_goal"
    return "latent_residual"


def _store_model_arrays(weights: dict[str, Any], *, name: str, model: dict[str, Any]) -> None:
    for key in ("x_mean", "x_scale", "w1", "b1", "w2", "b2"):
        weights[f"{name}__{key}"] = model[key]


def _load_dataset_arrays(dataset_dir: Path, pq: Any, np: Any) -> DatasetArrays:
    paths = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files found under {dataset_dir / 'data'}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["index"]))
    states = np.asarray([row["observation.state"] for row in rows], dtype=np.float32)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
    episode_indices = np.asarray([row["episode_index"] for row in rows], dtype=np.int64)
    frame_indices = np.asarray([row["frame_index"] for row in rows], dtype=np.int64)
    return DatasetArrays(
        states=states,
        actions=actions,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
    )


def _split_episodes_from_sidecar(sidecar: dict[str, Any]) -> SplitEpisodes:
    split = sidecar["split"]
    return SplitEpisodes(
        train=_expand_episode_range(split["train_episodes"]),
        validation=_expand_episode_range(split["validation_episodes"]),
        test=_expand_episode_range(split["test_episodes"]),
    )


def _expand_episode_range(value: dict[str, Any]) -> list[int]:
    if value["start"] is None:
        return []
    return list(range(int(value["start"]), int(value["end_inclusive"]) + 1))


def _validate_variant(variant: str) -> None:
    if variant not in SUPPORTED_VARIANTS:
        expected = ", ".join(sorted(SUPPORTED_VARIANTS))
        raise ValueError(f"Unknown variant {variant!r}; expected one of: {expected}.")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dump_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on host runtime
        raise RuntimeError("numpy is required to train the SO-101 latent scorer.") from exc
    return np


def _import_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on host runtime
        raise RuntimeError(
            "pyarrow is required to read the LeRobot dataset parquet files."
        ) from exc
    return pq


if __name__ == "__main__":
    raise SystemExit(main())
