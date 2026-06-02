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
import numbers
from pathlib import Path
from typing import Any

from worldforge.demos.so101_latent_score_provider import SO101LatentScoreProvider

DEFAULT_DATASET_DIR = Path("data/svla_so101_pickplace")
DEFAULT_CACHE_PATH = Path(".worldforge/so101-latent-cache/latent_cache.npz")
DEFAULT_REFEREE_PATH = Path(".worldforge/so101-referee/referee_headline.py")
DEFAULT_SCORER_DIR = Path(".worldforge/so101-latent-scorer")
CLEAR_BAD_DECOYS = ("overshoot", "reverse", "random_other", "jitter")
NEAR_DUPLICATE_DECOYS = ("no_motion", "scale_half")
ON_MANIFOLD_DECOYS = (
    "previous_demo_action",
    "next_demo_action",
    "same_episode_demo_action",
    "other_episode_demo_action",
)
GATE_METRIC = "fair_beat_rate_clearbad"
DEFAULT_RESIDUAL_RIDGE_FAIR_CLEARBAD = 0.6762
PROGRESS_TOLERANCE = 1e-9


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
    parser.add_argument(
        "--residual-ridge-fair-clearbad",
        type=_finite_probability_float,
        default=DEFAULT_RESIDUAL_RIDGE_FAIR_CLEARBAD,
        help=(
            "Pre-registered residual-ridge sanity baseline for fair_beat_rate_clearbad. "
            "Default comes from the prior frozen-referee residual-ridge run."
        ),
    )
    parser.add_argument(
        "--skip-stress-tests",
        action="store_true",
        help=(
            "Skip adapter-side held-out-generator and on-manifold hard-negative diagnostics. "
            "The canonical frozen-referee smoke still runs."
        ),
    )
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
        residual_ridge_fair_clearbad=args.residual_ridge_fair_clearbad,
        include_stress_tests=not args.skip_stress_tests,
    )
    print(f"items={result['n_items']} test_episodes={result['test_episode_count']}")
    for name, row in result["results"].items():
        summary = result["metric_summary"][name]
        print(
            f"{name}: fair_clearbad={summary['fair_beat_rate_clearbad']:.4f} "
            f"mrr={_result_metric(row, 'mrr', scorer_name=name):.4f} "
            f"top1={_result_metric(row, 'top1_vs_demonstrated', scorer_name=name):.4f} "
            "rank_corr_calibration="
            f"{_result_metric(row, 'rank_corr_vs_proprio_truth', scorer_name=name):.4f}"
        )
    gate = result["gate"]
    print(
        "gate="
        f"{gate['verdict']} "
        f"selected={gate['selected_model']} "
        f"{gate['metric']}={gate['selected_model_score']:.4f} "
        f"proprio={gate['proprio_baseline']:.4f} "
        f"ridge={gate['residual_ridge_baseline']:.4f}"
    )
    if result.get("stress_tests"):
        hard_gate = result["stress_tests"]["hard_negative_gate"]
        print(
            "hard_negative_gate="
            f"{hard_gate['verdict']} "
            f"selected={hard_gate.get('selected_model', '-')} "
            f"{hard_gate['metric']}={hard_gate.get('selected_model_score', 0.0):.4f} "
            f"proprio_oracle={hard_gate.get('proprio_baseline', 0.0):.4f}"
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
    residual_ridge_fair_clearbad: float,
    include_stress_tests: bool = True,
) -> dict[str, Any]:
    np = _import_numpy()
    referee = _load_referee(referee_path)
    cache = _load_latent_cache(np, cache_path)
    metadata = _load_scorer_metadata(metadata_path)
    selected_models = model_names or sorted(metadata["models"])

    data, episodes = referee.load(dataset_dir)
    split = int(len(episodes) * (1 - test_frac))
    test_episodes = episodes[split:]
    items = referee.build_items(data, test_episodes, seed=seed)
    scorers = {"proprio_baseline": referee.proprio_scorer}
    results = {
        "proprio_baseline": referee.evaluate(
            items,
            scorers["proprio_baseline"],
            np.random.RandomState(shuffle_seed),
        )
    }
    for model_name in selected_models:
        scorers[model_name] = _make_provider_scorer(
            referee=referee,
            np=np,
            cache=cache,
            weights_path=weights_path,
            metadata_path=metadata_path,
            model_name=model_name,
        )
        results[model_name] = referee.evaluate(
            items,
            scorers[model_name],
            np.random.RandomState(shuffle_seed),
        )
    metric_summary = {name: _metric_summary(referee, row) for name, row in results.items()}
    stress_tests = (
        _build_stress_tests(referee=referee, np=np, items=items, scorers=scorers, seed=seed)
        if include_stress_tests
        else {}
    )
    gate = _build_gate_summary(
        metadata=metadata,
        metric_summary=metric_summary,
        results=results,
        residual_ridge_fair_clearbad=residual_ridge_fair_clearbad,
        stress_tests=stress_tests,
    )

    payload = {
        "boundary": (
            "Smoke handoff using external referee functions; the external referee owner's "
            "independent run remains authoritative."
        ),
        "inputs": {
            "dataset_dir": _safe_path_for_report(dataset_dir),
            "cache_path": _safe_path_for_report(cache_path),
            "referee_path": _safe_path_for_report(referee_path),
            "scorer_metadata_path": _safe_path_for_report(metadata_path),
        },
        "external_referee": {
            "sha256": _sha256(referee_path),
            "frozen_for_gate": True,
            "required_members": ["load", "build_items", "evaluate", "proprio_scorer", "DECOYS"],
            "boundary": (
                "Host-owned referee module. This helper adapts scorer artifacts to the "
                "referee; it does not define the evaluation protocol."
            ),
        },
        "scorer_artifacts": {
            "weights_sha256": _sha256(weights_path),
            "metadata_sha256": _sha256(metadata_path),
        },
        "n_items": len(items),
        "test_episode_count": len(test_episodes),
        "split": {
            "test_frac": test_frac,
            "test_episodes": [
                _episode_identifier(episode, fallback_index=index)
                for index, episode in enumerate(test_episodes, start=split)
            ],
        },
        "metric_boundary": (
            "Exact-match top-1 is diagnostic for this near-duplicate decoy set. "
            "fair_beat_rate_clearbad over overshoot, reverse, random_other, and jitter is a "
            "train-like synthetic clear-bad smoke metric, not a robust learned-scorer proof by "
            "itself. rank_corr_vs_proprio_truth is calibration against proprio progress, not an "
            "independent success signal. The stress_tests block records held-out-generator and "
            "on-manifold hard-negative diagnostics that must be audited before any external "
            "learned-value claim."
        ),
        "gate": gate,
        "stress_tests": stress_tests,
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

    def scorer(item: dict[str, Any]) -> dict[str, float]:
        names = list(item["cands"])
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


def _build_stress_tests(
    *,
    referee: Any,
    np: Any,
    items: list[dict[str, Any]],
    scorers: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    heldout_items = _heldout_shifted_synthetic_items(items, np=np, seed=seed)
    on_manifold_items = _on_manifold_real_action_items(items)
    profiles = {
        "heldout_shifted_synthetic": {
            "boundary": (
                "Same held-out episodes, but decoy parameters differ from the train-like "
                "generator: altered reverse/scale/overshoot factors, smaller jitter, and "
                "a real demonstrated action from another item for random_other."
            ),
            "decoys": [*CLEAR_BAD_DECOYS, *NEAR_DUPLICATE_DECOYS],
            "results": {
                name: _evaluate_progress_candidate_set(
                    referee=referee,
                    np=np,
                    items=heldout_items,
                    scorer=scorer,
                    decoy_names=(*CLEAR_BAD_DECOYS, *NEAR_DUPLICATE_DECOYS),
                )
                for name, scorer in scorers.items()
            },
        },
        "on_manifold_real_actions": {
            "boundary": (
                "Hard-negative diagnostic using real demonstrated actions from neighboring, "
                "same-episode, and different-episode frames. These negatives are on-manifold, "
                "so the metric is progress-pairwise rank fidelity rather than assuming the "
                "original demonstrated action is always the best candidate."
            ),
            "decoys": list(ON_MANIFOLD_DECOYS),
            "results": {
                name: _evaluate_progress_candidate_set(
                    referee=referee,
                    np=np,
                    items=on_manifold_items,
                    scorer=scorer,
                    decoy_names=ON_MANIFOLD_DECOYS,
                )
                for name, scorer in scorers.items()
            },
        },
    }
    return {
        "boundary": (
            "Adapter-side diagnostics only. They reuse the frozen referee's held-out items and "
            "progress calculation, but they are not a replacement for Claude's independent "
            "referee verdict or execution-grounded policy ranking."
        ),
        "profiles": profiles,
        "hard_negative_gate": _build_hard_negative_gate(profiles),
    }


def _heldout_shifted_synthetic_items(
    items: list[dict[str, Any]], *, np: Any, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 10_003)
    demonstrated_pool = [item["cands"]["demonstrated"] for item in items]
    shifted = []
    for index, item in enumerate(items):
        current = item["cur"]
        demonstrated = item["cands"]["demonstrated"]
        delta = demonstrated - current
        other = demonstrated_pool[(index * 9973 + 17) % len(demonstrated_pool)]
        cands = {
            "demonstrated": demonstrated,
            "overshoot": current + 2.0 * delta,
            "reverse": current - 0.5 * delta,
            "random_other": other,
            "jitter": demonstrated + rng.normal(0.0, 4.0, size=demonstrated.shape),
            "no_motion": current.copy(),
            "scale_half": current + 0.25 * delta,
        }
        shifted.append(_item_with_candidates(item, cands))
    return shifted


def _on_manifold_real_action_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[Any, list[int]] = {}
    for index, item in enumerate(items):
        by_episode.setdefault(item.get("ep"), []).append(index)
    episode_keys = list(by_episode)
    transformed = []
    for index, item in enumerate(items):
        episode_indices = by_episode[item.get("ep")]
        local_position = episode_indices.index(index)
        previous_index = episode_indices[max(0, local_position - 3)]
        next_index = episode_indices[min(len(episode_indices) - 1, local_position + 3)]
        same_episode_offset = local_position + max(1, len(episode_indices) // 3)
        same_episode_index = episode_indices[same_episode_offset % len(episode_indices)]
        other_episode_position = episode_keys.index(item.get("ep")) + 1
        other_episode_key = episode_keys[other_episode_position % len(episode_keys)]
        other_episode_indices = by_episode[other_episode_key]
        other_episode_index = other_episode_indices[index % len(other_episode_indices)]
        cands = {
            "demonstrated": item["cands"]["demonstrated"],
            "previous_demo_action": items[previous_index]["cands"]["demonstrated"],
            "next_demo_action": items[next_index]["cands"]["demonstrated"],
            "same_episode_demo_action": items[same_episode_index]["cands"]["demonstrated"],
            "other_episode_demo_action": items[other_episode_index]["cands"]["demonstrated"],
        }
        transformed.append(_item_with_candidates(item, cands))
    return transformed


def _item_with_candidates(item: dict[str, Any], cands: dict[str, Any]) -> dict[str, Any]:
    return {**item, "cands": cands}


def _evaluate_progress_candidate_set(
    *,
    referee: Any,
    np: Any,
    items: list[dict[str, Any]],
    scorer: Any,
    decoy_names: tuple[str, ...],
) -> dict[str, Any]:
    names = ("demonstrated", *decoy_names)
    top1_demo = 0
    top1_progress = 0
    pairwise_correct = 0
    pairwise_total = 0
    demonstrated_better_total = 0
    demonstrated_better_correct = 0
    decoy_beat = dict.fromkeys(decoy_names, 0)
    for item in items:
        costs = _score_named_candidates(scorer(item), names=names)
        progress = _true_progress(referee, np, item, names=names)
        selected = min(names, key=costs.__getitem__)
        progress_best = max(names, key=progress.__getitem__)
        top1_demo += selected == "demonstrated"
        top1_progress += selected == progress_best
        for decoy_name in decoy_names:
            decoy_beat[decoy_name] += costs["demonstrated"] < costs[decoy_name]
        for better in names:
            for worse in names:
                if better == worse:
                    continue
                if progress[better] <= progress[worse] + PROGRESS_TOLERANCE:
                    continue
                pairwise_total += 1
                pairwise_correct += costs[better] < costs[worse]
                if better == "demonstrated":
                    demonstrated_better_total += 1
                    demonstrated_better_correct += costs[better] < costs[worse]
    item_count = max(1, len(items))
    pairwise_rate = pairwise_correct / max(1, pairwise_total)
    demonstrated_rate = demonstrated_better_correct / max(1, demonstrated_better_total)
    return {
        "n_items": len(items),
        "decoys": list(decoy_names),
        "top1_vs_demonstrated": round(top1_demo / item_count, 4),
        "top1_vs_progress_best": round(top1_progress / item_count, 4),
        "progress_pairwise_accuracy": round(pairwise_rate, 4),
        "progress_pairwise_accuracy_raw": pairwise_rate,
        "demonstrated_better_pair_count": demonstrated_better_total,
        "demonstrated_better_pair_beat_rate": round(demonstrated_rate, 4),
        "demonstrated_better_pair_beat_rate_raw": demonstrated_rate,
        "decoy_beat_rate": {name: round(decoy_beat[name] / item_count, 4) for name in decoy_names},
    }


def _score_named_candidates(costs: Any, *, names: tuple[str, ...]) -> dict[str, float]:
    if not isinstance(costs, dict):
        raise RuntimeError("SO-101 scorer must return a dictionary of candidate costs.")
    result = {}
    for name in names:
        try:
            value = float(costs[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"SO-101 scorer result is missing numeric cost for {name}.") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"SO-101 scorer result has non-finite cost for {name}.")
        result[name] = value
    return result


def _true_progress(
    referee: Any,
    np: Any,
    item: dict[str, Any],
    *,
    names: tuple[str, ...],
) -> dict[str, float]:
    true_progress = getattr(referee, "true_progress", None)
    if callable(true_progress):
        raw = true_progress(item)
    else:
        base = float(np.linalg.norm(item["cur"] - item["goal"]))
        raw = {
            name: base - float(np.linalg.norm(item["cands"][name] - item["goal"])) for name in names
        }
    return _score_named_candidates(raw, names=names)


def _build_hard_negative_gate(profiles: dict[str, Any]) -> dict[str, Any]:
    on_manifold = profiles["on_manifold_real_actions"]["results"]
    selected_candidates = [name for name in on_manifold if name != "proprio_baseline"]
    if not selected_candidates:
        return {
            "metric": "progress_pairwise_accuracy",
            "verdict": "not_run_no_learned_scorer",
        }
    selected_model = max(
        selected_candidates,
        key=lambda name: float(on_manifold[name]["progress_pairwise_accuracy_raw"]),
    )
    selected_score = _finite_rate_value(
        on_manifold[selected_model]["progress_pairwise_accuracy_raw"],
        field_name="selected hard-negative progress-pairwise accuracy",
    )
    proprio_score = _finite_rate_value(
        on_manifold["proprio_baseline"]["progress_pairwise_accuracy_raw"],
        field_name="proprio hard-negative progress-pairwise accuracy",
    )
    clears_proprio = selected_score > proprio_score
    return {
        "metric": "progress_pairwise_accuracy",
        "profile": "on_manifold_real_actions",
        "selected_model": selected_model,
        "selected_model_score": round(selected_score, 4),
        "proprio_baseline": round(proprio_score, 4),
        "clears_proprio_baseline": clears_proprio,
        "verdict": (
            "clears_hard_on_manifold_gate"
            if clears_proprio
            else "does_not_clear_hard_on_manifold_gate"
        ),
        "claim_boundary": (
            "This is a hard-negative diagnostic using proprio progress as an oracle proxy, not "
            "an execution-grounded success metric. Failure means the scorer did not preserve "
            "on-manifold candidate progress ordering under this proxy. A learned-scorer value "
            "claim should require this diagnostic, a near-duplicate gate, or a stronger "
            "execution-grounded rank-fidelity gate, not only train-like synthetic clear-bad "
            "decoys."
        ),
    }


def _metric_summary(referee: Any, row: dict[str, Any]) -> dict[str, float]:
    metric_summary = getattr(referee, "metric_summary", None)
    if callable(metric_summary):
        try:
            raw_summary = metric_summary(row)
            fair_clearbad = _finite_metric(raw_summary, "fair_beat_rate_clearbad")
            near_duplicate = _finite_metric(raw_summary, "near_duplicate_beat_rate")
            return {
                "fair_beat_rate_clearbad": round(fair_clearbad, 4),
                "fair_beat_rate_clearbad_raw": fair_clearbad,
                "near_duplicate_beat_rate": round(near_duplicate, 4),
                "near_duplicate_beat_rate_raw": near_duplicate,
            }
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("SO-101 referee metric_summary returned invalid metrics.") from exc
    beat_rates = row.get("decoy_beat_rate", {})
    clear_bad = _mean_named_rates(beat_rates, CLEAR_BAD_DECOYS)
    near_duplicate = _mean_named_rates(beat_rates, NEAR_DUPLICATE_DECOYS)
    return {
        "fair_beat_rate_clearbad": round(clear_bad, 4),
        "fair_beat_rate_clearbad_raw": clear_bad,
        "near_duplicate_beat_rate": round(near_duplicate, 4),
        "near_duplicate_beat_rate_raw": near_duplicate,
    }


def _build_gate_summary(
    *,
    metadata: dict[str, Any],
    metric_summary: dict[str, dict[str, float]],
    results: dict[str, dict[str, Any]],
    residual_ridge_fair_clearbad: float,
    stress_tests: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_model = str(metadata.get("selected_model", ""))
    if selected_model not in metric_summary:
        raise RuntimeError(
            "SO-101 scorer gate requires the selected_model to be present in referee results."
        )
    if "proprio_baseline" not in metric_summary:
        raise RuntimeError("SO-101 scorer gate requires the proprio_baseline referee result.")
    selected_score = _finite_summary_metric(
        metric_summary[selected_model],
        f"{GATE_METRIC}_raw",
        scorer_name=selected_model,
        fallback_key=GATE_METRIC,
    )
    proprio_score = _finite_summary_metric(
        metric_summary["proprio_baseline"],
        f"{GATE_METRIC}_raw",
        scorer_name="proprio_baseline",
        fallback_key=GATE_METRIC,
    )
    residual_ridge_score = _finite_rate_value(
        residual_ridge_fair_clearbad,
        field_name="residual-ridge baseline",
    )
    selected_result = results[selected_model]
    chance_top1 = _result_metric(selected_result, "chance", scorer_name=selected_model)
    shuffled_label_top1 = _result_metric(
        selected_result,
        "shuffled_label_top1",
        scorer_name=selected_model,
    )
    clears_proprio = selected_score > proprio_score
    clears_residual_ridge = selected_score >= residual_ridge_score
    smoke_verdict = (
        "clears_train_like_clearbad_smoke_gate"
        if clears_proprio and clears_residual_ridge
        else "does_not_clear_train_like_clearbad_smoke_gate"
    )
    hard_gate = (stress_tests or {}).get("hard_negative_gate", {})
    hard_verdict = hard_gate.get("verdict", "not_run")
    return {
        "registered_before_result": True,
        "metric": GATE_METRIC,
        "metric_boundary": "train_like_synthetic_clearbad_smoke",
        "clear_bad_decoys": list(CLEAR_BAD_DECOYS),
        "near_duplicate_decoys_excluded_from_gate": list(NEAR_DUPLICATE_DECOYS),
        "selected_model": selected_model,
        "selected_model_score": round(selected_score, 4),
        "proprio_baseline": round(proprio_score, 4),
        "residual_ridge_baseline": round(residual_ridge_score, 4),
        "clears_proprio_baseline": clears_proprio,
        "clears_residual_ridge_baseline": clears_residual_ridge,
        "chance_top1": round(chance_top1, 4),
        "shuffled_label_top1": round(shuffled_label_top1, 4),
        "control_boundary": (
            "Shuffled-label top-1 is reported as a leakage control and should stay near chance; "
            "Claude's independent audit owns the final leakage verdict."
        ),
        "verdict": smoke_verdict,
        "hard_negative_verdict": hard_verdict,
        "claim_boundary": (
            "Passing this train-like synthetic clear-bad smoke gate is not enough for a robust "
            "learned-scorer value claim. That claim also needs held-out-generator or "
            "on-manifold hard-negative rank fidelity, plus the independent leakage audit. This "
            "is not a physical execution, sim-measured task-success, or safety-controller claim."
        ),
    }


def _finite_summary_metric(
    row: dict[str, Any],
    key: str,
    *,
    scorer_name: str,
    fallback_key: str | None = None,
) -> float:
    try:
        value = float(row[key])
    except KeyError:
        if fallback_key is None:
            raise RuntimeError(
                f"SO-101 referee summary for {scorer_name!r} is missing numeric metric {key!r}."
            ) from None
        return _finite_summary_metric(
            row,
            fallback_key,
            scorer_name=scorer_name,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"SO-101 referee summary for {scorer_name!r} is missing numeric metric {key!r}."
        ) from exc
    if not math.isfinite(value):
        raise RuntimeError(
            f"SO-101 referee summary for {scorer_name!r} has non-finite metric {key!r}."
        )
    return value


def _finite_metric(summary: Any, key: str) -> float:
    if not isinstance(summary, dict):
        raise TypeError("SO-101 referee metric_summary must return a dictionary.")
    value = float(summary[key])
    if not math.isfinite(value):
        raise ValueError(f"SO-101 referee metric_summary returned non-finite {key}.")
    return value


def _episode_identifier(episode: Any, *, fallback_index: int) -> int:
    if isinstance(episode, bool):
        return fallback_index
    if isinstance(episode, numbers.Integral):
        return int(episode)
    if isinstance(episode, dict) and "episode_index" in episode:
        return _coerce_episode_identifier(episode["episode_index"], fallback_index=fallback_index)
    if hasattr(episode, "episode_index"):
        return _coerce_episode_identifier(
            episode.episode_index,
            fallback_index=fallback_index,
        )
    return fallback_index


def _coerce_episode_identifier(value: Any, *, fallback_index: int) -> int:
    if isinstance(value, bool):
        return fallback_index
    if isinstance(value, numbers.Integral):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback_index


def _mean_named_rates(beat_rates: Any, names: tuple[str, ...]) -> float:
    if not isinstance(beat_rates, dict):
        raise ValueError("SO-101 referee result decoy_beat_rate must be a dictionary.")
    missing = [name for name in names if name not in beat_rates]
    if missing:
        raise ValueError(
            "SO-101 referee result decoy_beat_rate is missing required decoys: "
            + ", ".join(missing)
        )
    values = []
    for name in names:
        value = float(beat_rates[name])
        if not math.isfinite(value):
            raise ValueError(f"SO-101 referee decoy_beat_rate.{name} must be finite.")
        values.append(value)
    return sum(values) / len(values)


def _history_latents(cache: Any, item: dict[str, Any]) -> Any:
    np = _import_numpy()
    local_index = int(item["t"])
    current_index = int(item["cur_g"])
    indices = [current_index - min(offset, local_index) for offset in (2, 1, 0)]
    return cache[np.asarray(indices, dtype=np.int64)]


def _load_referee(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"SO-101 referee path does not exist: {_safe_path_for_error(path)}. "
            "Pass --referee with the host-owned referee module path."
        )
    spec = importlib.util.spec_from_file_location("worldforge_external_so101_referee", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load SO-101 referee module from {_safe_path_for_error(path)}."
        )
    module = importlib.util.module_from_spec(spec)
    # Host-owned referee modules execute arbitrary Python. This helper only loads an explicit
    # path supplied by the operator and must not be used with untrusted downloaded files.
    spec.loader.exec_module(module)
    for name in ("load", "build_items", "evaluate", "proprio_scorer", "DECOYS"):
        if not hasattr(module, name):
            raise RuntimeError(f"SO-101 referee module is missing required member: {name}")
    return module


def _load_latent_cache(np: Any, path: Path) -> Any:
    try:
        return np.load(path, allow_pickle=False)["latents"].astype(np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"Could not load SO-101 latent cache from {_safe_path_for_error(path)}. "
            "Pass --cache with the host-owned latent_cache.npz path."
        ) from exc


def _load_scorer_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not load SO-101 scorer metadata from {_safe_path_for_error(path)}. "
            "Pass --metadata with the scorer metadata artifact."
        ) from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("models"), dict):
        raise RuntimeError("SO-101 scorer metadata must contain a models dictionary.")
    return metadata


def _safe_path_for_error(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return path.name


def _safe_path_for_report(path: Path) -> str:
    return _safe_path_for_error(path)


def _result_metric(row: dict[str, Any], key: str, *, scorer_name: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"SO-101 referee result for {scorer_name!r} is missing numeric metric {key!r}."
        ) from exc
    if not math.isfinite(value):
        raise RuntimeError(
            f"SO-101 referee result for {scorer_name!r} has non-finite metric {key!r}."
        )
    return value


def _finite_rate_value(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"SO-101 scorer gate requires a finite {field_name}.") from exc
    if not math.isfinite(parsed):
        raise RuntimeError(f"SO-101 scorer gate requires a finite {field_name}.")
    if parsed < 0.0 or parsed > 1.0:
        raise RuntimeError(f"SO-101 scorer gate requires {field_name} to be within [0, 1].")
    return parsed


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_probability_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a finite float") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite float")
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a non-negative float")
    if parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be at most 1.0")
    return parsed


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
