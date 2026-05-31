#!/usr/bin/env python3
"""Train SO-101 latent score-provider artifacts from an existing DINOv2 cache.

This is a scorer handoff script, not a referee. It trains candidate learned scorers on train
episodes, selects by validation latent prediction quality, and writes provider-ready weights for
Claude's independent held-out referee to grade.
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
    print(
        "selected="
        f"{result['selected_model']} "
        f"val_cosine={result['models'][result['selected_model']]['validation']['mean_cosine']:.6f}"
    )
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
                np=np,
            )
            validation = _build_pairs(
                arrays=arrays,
                latents=latents,
                episodes=splits.validation,
                horizon=horizon,
                variant=variant,
                np=np,
            )
            test = _build_pairs(
                arrays=arrays,
                latents=latents,
                episodes=splits.test,
                horizon=horizon,
                variant=variant,
                np=np,
            )
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
                "output_dim": int(latents.shape[1]),
                "target": "future_latent_residual",
                "activation": "tanh",
                **metrics,
            }
            score_key = (
                float(metrics["validation"]["mean_cosine"]),
                -float(metrics["validation"]["mse"]),
            )
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
            "selection": "highest validation mean_cosine, then lowest validation mse",
            "referee_boundary": (
                "This artifact is a scorer handoff only. The independent held-out referee "
                "must decide whether it beats baselines and chance."
            ),
        },
        "kill_criterion": (
            "Do not claim learned-scorer value unless the independent referee shows above-chance "
            "performance and improvement over proprio/hand-cost baselines. Below-chance marginal "
            "improvement is negative evidence."
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
        "x": x,
        "y": y,
        "current_latents": current_latents,
        "future_latents": latents[future].astype(np.float32),
    }


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


def _predict_residual_model(model: dict[str, Any], x: Any, current_latents: Any, *, np: Any) -> Any:
    x_norm = (x - model["x_mean"]) / model["x_scale"]
    hidden = np.tanh(x_norm @ model["w1"] + model["b1"])
    residual = hidden @ model["w2"] + model["b2"]
    pred = current_latents + residual
    norm = np.linalg.norm(pred, axis=1, keepdims=True)
    norm = np.where(norm < 1e-9, 1.0, norm)
    return (pred / norm).astype(np.float32)


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
    if variant not in {"vision_mlp", "vision_proprio_mlp"}:
        raise ValueError(f"Unknown variant {variant!r}; expected vision_mlp or vision_proprio_mlp.")


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
