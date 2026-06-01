#!/usr/bin/env python3
"""Build a metadata sidecar for a precomputed SO-101 latent cache.

This script intentionally reuses an existing frozen-encoder cache. It does not decode video,
download models, or recompute DINOv2 embeddings. The sidecar gives an independent referee enough
metadata to align latent rows with LeRobot frame rows, episode splits, and content hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_DIR = Path("data/svla_so101_pickplace")
DEFAULT_CACHE_PATH = Path(".worldforge/so101-latent-cache/latent_cache.npz")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = args.out or args.cache.with_name("latent_cache_meta.json")
    sidecar = build_so101_latent_cache_sidecar(
        dataset_dir=args.dataset_dir,
        cache_path=args.cache,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dump_json(sidecar), encoding="utf-8")
    print(f"WROTE {out}")
    print(
        "frames="
        f"{sidecar['dataset']['total_frames']} "
        f"latent_dim={sidecar['cache']['latent_dim']} "
        f"split=train:{sidecar['split']['train_episode_count']} "
        f"val:{sidecar['split']['validation_episode_count']} "
        f"test:{sidecar['split']['test_episode_count']}"
    )
    return 0


def build_so101_latent_cache_sidecar(
    *,
    dataset_dir: Path,
    cache_path: Path,
    test_frac: float = 0.3,
    seed: int = 0,
) -> dict[str, Any]:
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError("--test-frac must be in (0, 1)")

    np = _import_numpy()
    pq = _import_pyarrow_parquet()

    info = _read_json(dataset_dir / "meta" / "info.json")
    data_rows = _read_data_rows(dataset_dir, pq)
    episode_rows = _read_episode_rows(dataset_dir, pq)
    cache = np.load(cache_path, allow_pickle=False)
    latents = cache["latents"]
    if latents.ndim != 2:
        raise ValueError(f"latent cache must be rank-2, got shape {latents.shape!r}")
    if len(data_rows) != latents.shape[0]:
        raise ValueError(
            f"latent row count must match LeRobot rows: {latents.shape[0]} != {len(data_rows)}"
        )

    total_episodes = len(episode_rows)
    train_cut = int(total_episodes * (1.0 - test_frac))
    validation_cut = int(train_cut * 0.8)
    split_by_episode = _episode_split_map(
        episode_rows,
        validation_cut=validation_cut,
        train_cut=train_cut,
    )
    frame_rows = [
        _frame_sidecar_row(row, split=split_by_episode[int(row["episode_index"])])
        for row in data_rows
    ]
    split_counts = _split_counts(frame_rows)
    cache_shape = [int(latents.shape[0]), int(latents.shape[1])]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "worldforge.so101_latent_cache_sidecar",
        "cache": {
            "model": _cache_scalar(cache, "model", "facebook/dinov2-small"),
            "camera": _cache_scalar(cache, "cam", "side"),
            "token": _cache_scalar(cache, "token", "cls"),
            "normalization": _cache_scalar(cache, "norm", "l2"),
            "dtype": str(latents.dtype),
            "shape": cache_shape,
            "latent_dim": int(latents.shape[1]),
            "sha256": _sha256_file(cache_path),
        },
        "dataset": {
            "repo_id": "lerobot/svla_so101_pickplace",
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "fps": info.get("fps"),
            "total_frames": int(info.get("total_frames", len(data_rows))),
            "total_episodes": int(info.get("total_episodes", total_episodes)),
            "action_names": list(info["features"]["action"]["names"]),
            "state_names": list(info["features"]["observation.state"]["names"]),
        },
        "split": {
            "strategy": "episode_prefix_train_val_test",
            "seed": seed,
            "test_frac": test_frac,
            "train_episode_count": validation_cut,
            "validation_episode_count": train_cut - validation_cut,
            "test_episode_count": total_episodes - train_cut,
            "train_episodes": _episode_range(0, validation_cut),
            "validation_episodes": _episode_range(validation_cut, train_cut),
            "test_episodes": _episode_range(train_cut, total_episodes),
            "frame_counts": split_counts,
        },
        "frames": frame_rows,
    }
    payload["frames_sha256"] = _sha256_json(frame_rows)
    payload["content_sha256"] = _sha256_json(
        {
            "cache": payload["cache"],
            "dataset": payload["dataset"],
            "split": payload["split"],
            "frames_sha256": payload["frames_sha256"],
        }
    )
    return payload


def _frame_sidecar_row(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    return {
        "index": int(row["index"]),
        "episode_index": int(row["episode_index"]),
        "frame_index": int(row["frame_index"]),
        "timestamp_s": round(float(row["timestamp"]), 6),
        "task_index": int(row["task_index"]),
        "split": split,
    }


def _episode_split_map(
    episodes: list[dict[str, Any]],
    *,
    validation_cut: int,
    train_cut: int,
) -> dict[int, str]:
    split_by_episode: dict[int, str] = {}
    for row in episodes:
        episode_index = int(row["episode_index"])
        if episode_index < validation_cut:
            split = "train"
        elif episode_index < train_cut:
            split = "validation"
        else:
            split = "test"
        split_by_episode[episode_index] = split
    return split_by_episode


def _split_counts(frame_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in frame_rows:
        counts[str(row["split"])] += 1
    return counts


def _episode_range(start: int, end: int) -> dict[str, int | None]:
    if start >= end:
        return {"start": None, "end_inclusive": None}
    return {"start": start, "end_inclusive": end - 1}


def _read_data_rows(dataset_dir: Path, pq: Any) -> list[dict[str, Any]]:
    paths = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files found under {dataset_dir / 'data'}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["index"]))
    for expected, row in enumerate(rows):
        actual = int(row["index"])
        if actual != expected:
            raise ValueError(f"Frame index alignment failed at row {expected}: {actual}")
    return rows


def _read_episode_rows(dataset_dir: Path, pq: Any) -> list[dict[str, Any]]:
    paths = sorted((dataset_dir / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError("No episode parquet files found under meta/episodes")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_scalar(cache: Any, key: str, default: str) -> str:
    if key not in cache.files:
        return default
    return str(cache[key].item())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _dump_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on host runtime
        raise RuntimeError("numpy is required to inspect the latent cache.") from exc
    return np


def _import_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on host runtime
        raise RuntimeError(
            "pyarrow is required to inspect the LeRobot dataset parquet files."
        ) from exc
    return pq


if __name__ == "__main__":
    raise SystemExit(main())
