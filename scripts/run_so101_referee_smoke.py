#!/usr/bin/env python3
"""Run SO-101 scorer artifacts through an external held-out referee.

This is a host-owned handoff helper. It does not define the referee, grade itself, or add numpy /
pyarrow / robot dependencies to the base package. The external referee path owns the evaluation
protocol; this script only adapts WorldForge scorer artifacts to that callable interface.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from worldforge.demos.so101_latent_score_provider import SO101LatentScoreProvider

DEFAULT_DATASET_DIR = Path(
    "/Users/espejelomar/StarkNet/zk-ai/hackathons/worldforge-so101-trace-judge/data/"
    "svla_so101_pickplace"
)
DEFAULT_CACHE_PATH = Path(
    "/Users/espejelomar/StarkNet/zk-ai/hackathons/worldforge-eval-referee/latent_cache.npz"
)
DEFAULT_REFEREE_PATH = Path(
    "/Users/espejelomar/StarkNet/zk-ai/hackathons/worldforge-eval-referee/referee_headline.py"
)
DEFAULT_SCORER_DIR = Path(".worldforge/so101-goal-score-scorer")
CLEAR_BAD_DECOYS = ("overshoot", "reverse", "random_other", "jitter")
NEAR_DUPLICATE_DECOYS = ("no_motion", "scale_half")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--referee", type=Path, default=DEFAULT_REFEREE_PATH)
    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_SCORER_DIR / "so101_latent_scorers.npz"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_SCORER_DIR / "so101_latent_scorer_meta.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SCORER_DIR / "claude-referee-goal-score-smoke-result.json",
    )
    parser.add_argument(
        "--models", default="", help="Comma-separated model names. Defaults to all."
    )
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=7)
    args = parser.parse_args(argv)

    result = run_referee_smoke(
        dataset_dir=args.dataset_dir,
        cache_path=args.cache,
        referee_path=args.referee,
        weights_path=args.weights,
        metadata_path=args.metadata,
        output_path=args.out,
        model_names=_split_csv(args.models),
        test_frac=args.test_frac,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
    )
    print(f"items={result['n_items']} test_episodes={result['test_episode_count']}")
    for name, row in result["results"].items():
        summary = result["metric_summary"][name]
        print(
            f"{name}: fair_clearbad={summary['fair_beat_rate_clearbad']:.4f} "
            f"mrr={row['mrr']:.4f} top1={row['top1_vs_demonstrated']:.4f} "
            f"rank_corr_calibration={row['rank_corr_vs_proprio_truth']:.4f}"
        )
    print(f"WROTE {result['output_path']}")
    return 0


def run_referee_smoke(
    *,
    dataset_dir: Path,
    cache_path: Path,
    referee_path: Path,
    weights_path: Path,
    metadata_path: Path,
    output_path: Path,
    model_names: list[str],
    test_frac: float,
    seed: int,
    shuffle_seed: int,
) -> dict[str, Any]:
    np = _import_numpy()
    referee = _load_referee(referee_path)
    cache = np.load(cache_path, allow_pickle=False)["latents"].astype(np.float32)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected_models = model_names or sorted(metadata["models"])

    data, episodes = referee.load(dataset_dir)
    split = int(len(episodes) * (1 - test_frac))
    test_episodes = episodes[split:]
    items = referee.build_items(data, test_episodes, seed=seed)
    results = {
        "proprio_baseline": referee.evaluate(
            items,
            referee.proprio_scorer,
            np.random.RandomState(shuffle_seed),
        )
    }
    for model_name in selected_models:
        results[model_name] = referee.evaluate(
            items,
            _make_provider_scorer(
                referee=referee,
                np=np,
                cache=cache,
                weights_path=weights_path,
                metadata_path=metadata_path,
                model_name=model_name,
            ),
            np.random.RandomState(shuffle_seed),
        )
    metric_summary = {name: _metric_summary(referee, row) for name, row in results.items()}

    payload = {
        "boundary": (
            "Smoke handoff using external referee functions; the external referee owner's "
            "independent run remains authoritative."
        ),
        "dataset_dir": str(dataset_dir),
        "cache_path": str(cache_path),
        "referee_path": str(referee_path),
        "scorer_metadata_path": str(metadata_path),
        "weights_sha256": _sha256(weights_path),
        "metadata_sha256": _sha256(metadata_path),
        "n_items": len(items),
        "test_episode_count": len(test_episodes),
        "metric_boundary": (
            "Exact-match top-1 is diagnostic for this near-duplicate decoy set. The primary "
            "handoff metric is fair_beat_rate_clearbad over overshoot, reverse, random_other, "
            "and jitter. rank_corr_vs_proprio_truth is calibration against proprio progress, "
            "not an independent success signal."
        ),
        "metric_summary": metric_summary,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**payload, "output_path": str(output_path)}


def _make_provider_scorer(
    *,
    referee: Any,
    np: Any,
    cache: Any,
    weights_path: Path,
    metadata_path: Path,
    model_name: str,
) -> Any:
    provider = SO101LatentScoreProvider(
        weights_path=weights_path,
        metadata_path=metadata_path,
        model_name=model_name,
    )
    names = ["demonstrated", *referee.DECOYS]

    def scorer(item: dict[str, Any]) -> dict[str, float]:
        candidates = [
            {"candidate_id": name, "action": item["cands"][name].tolist()} for name in names
        ]
        result = provider.score_actions(
            info={
                "current_latent": cache[item["cur_g"]].tolist(),
                "goal_latent": cache[item["goal_g"]].tolist(),
                "history_latents": _history_latents(cache, item).tolist(),
                "current_state": item["cur"].tolist(),
            },
            action_candidates=candidates,
        )
        return dict(zip(names, result.scores, strict=True))

    return scorer


def _metric_summary(referee: Any, row: dict[str, Any]) -> dict[str, float]:
    metric_summary = getattr(referee, "metric_summary", None)
    if callable(metric_summary):
        try:
            raw_summary = metric_summary(row)
            return {
                "fair_beat_rate_clearbad": round(
                    _finite_metric(raw_summary, "fair_beat_rate_clearbad"), 4
                ),
                "near_duplicate_beat_rate": round(
                    _finite_metric(raw_summary, "near_duplicate_beat_rate"), 4
                ),
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
    beat_rates = row.get("decoy_beat_rate", {})
    clear_bad = _mean_named_rates(beat_rates, CLEAR_BAD_DECOYS)
    near_duplicate = _mean_named_rates(beat_rates, NEAR_DUPLICATE_DECOYS)
    return {
        "fair_beat_rate_clearbad": round(clear_bad, 4),
        "near_duplicate_beat_rate": round(near_duplicate, 4),
    }


def _finite_metric(summary: Any, key: str) -> float:
    value = float(summary[key])
    if not math.isfinite(value):
        raise ValueError(f"SO-101 referee metric_summary returned non-finite {key}.")
    return value


def _mean_named_rates(beat_rates: Any, names: tuple[str, ...]) -> float:
    values = [float(beat_rates[name]) for name in names if name in beat_rates]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _history_latents(cache: Any, item: dict[str, Any]) -> Any:
    np = _import_numpy()
    local_index = int(item["t"])
    current_index = int(item["cur_g"])
    indices = [current_index - min(offset, local_index) for offset in (2, 1, 0)]
    return cache[np.asarray(indices, dtype=np.int64)]


def _load_referee(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"SO-101 referee path does not exist: {path}")
    spec = importlib.util.spec_from_file_location("worldforge_external_so101_referee", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SO-101 referee module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("load", "build_items", "evaluate", "proprio_scorer", "DECOYS"):
        if not hasattr(module, name):
            raise RuntimeError(f"SO-101 referee module is missing required member: {name}")
    return module


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on host runtime
        raise RuntimeError("numpy is required to run the SO-101 referee smoke.") from exc
    return np


if __name__ == "__main__":
    raise SystemExit(main())
