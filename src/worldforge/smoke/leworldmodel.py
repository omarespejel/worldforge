"""Run a LeWorldModel provider smoke test with a real checkpoint.

Invoke this command through uv, for example:

    uv run --python 3.10 --with "<git stable-worldmodel>" --with "datasets>=2.21"
      lewm-real --checkpoint ~/.stable-wm/pusht/lewm_object.ckpt

This smoke requires the upstream LeWorldModel runtime dependencies and an
extracted ``<policy>_object.ckpt`` under ``--stablewm-home`` or ``--cache-dir``.
Use the exact dependency command from the README. It is not part of
WorldForge's base dependency set.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from worldforge.providers import LeWorldModelProvider

DEFAULT_STABLEWM_HOME = "~/.stable-wm"


def _checkpoint_path(cache_dir: Path, policy: str) -> Path:
    return cache_dir / f"{policy}_object.ckpt"


def _infer_cache_dir_from_checkpoint(checkpoint: Path, policy: str) -> Path:
    suffix = Path(f"{policy}_object.ckpt")
    suffix_parts = suffix.parts
    checkpoint_parts = checkpoint.parts
    if (
        len(checkpoint_parts) <= len(suffix_parts)
        or tuple(checkpoint_parts[-len(suffix_parts) :]) != suffix_parts
    ):
        raise SystemExit(
            f"Checkpoint path {checkpoint} does not match policy '{policy}'. "
            f"Expected a path ending with {suffix}. Pass --cache-dir/--stablewm-home "
            "with a matching --policy, or adjust --policy to match the checkpoint layout."
        )
    return Path(*checkpoint_parts[: -len(suffix_parts)])


def _require_object_checkpoint(*, policy: str, cache_dir: Path) -> Path:
    object_path = _checkpoint_path(cache_dir, policy)
    if object_path.exists():
        return object_path

    raise SystemExit(
        f"LeWorldModel object checkpoint not found: {object_path}. "
        "Download the checkpoint archive from the upstream LeWorldModel README and extract it "
        "under STABLEWM_HOME so the policy resolves to <policy>_object.ckpt, or pass "
        "--cache-dir to the directory that contains the policy subdirectory."
    )


def _resolve_checkpoint(
    *,
    policy: str,
    stablewm_home: Path,
    cache_dir: Path | None,
    checkpoint: Path | None,
) -> tuple[Path, Path]:
    if checkpoint is None:
        resolved_cache_dir = (cache_dir or stablewm_home).expanduser()
        return _require_object_checkpoint(
            policy=policy, cache_dir=resolved_cache_dir
        ), resolved_cache_dir

    object_path = checkpoint.expanduser()
    if not object_path.exists():
        raise SystemExit(f"LeWorldModel object checkpoint not found: {object_path}")

    inferred_cache_dir = _infer_cache_dir_from_checkpoint(object_path, policy)
    resolved_cache_dir = cache_dir.expanduser() if cache_dir is not None else inferred_cache_dir
    expected_path = _checkpoint_path(resolved_cache_dir, policy)
    if expected_path != object_path:
        raise SystemExit(
            f"Checkpoint path {object_path} does not match cache root {resolved_cache_dir} for "
            f"policy '{policy}'. Expected {expected_path}."
        )
    return object_path, resolved_cache_dir


def _build_inputs(
    *,
    batch: int,
    samples: int,
    history: int,
    horizon: int,
    action_dim: int,
    image_size: int,
):
    if horizon <= history:
        raise SystemExit("horizon must be greater than history for LeWorldModel rollout.")
    import torch

    info = {
        "pixels": torch.rand(batch, 1, history, 3, image_size, image_size),
        "goal": torch.rand(batch, 1, history, 3, image_size, image_size),
        "action": torch.rand(batch, 1, history, action_dim),
    }
    action_candidates = torch.rand(batch, samples, horizon, action_dim)
    return info, action_candidates


def _shape_text(value: object) -> str:
    shape = getattr(value, "shape", None)
    if shape is None and isinstance(value, dict):
        shape = value.get("shape")
    if shape is None:
        return type(value).__name__
    return " x ".join(str(part) for part in tuple(shape))


def _input_shape_summary(info: dict[str, object], action_candidates: object) -> dict[str, str]:
    return {
        "pixels": _shape_text(info["pixels"]),
        "goal": _shape_text(info["goal"]),
        "action_history": _shape_text(info["action"]),
        "action_candidates": _shape_text(action_candidates),
    }


def _score_chart(result: dict[str, Any]) -> list[str]:
    scores = [float(score) for score in result.get("scores", [])]
    if not scores:
        return ["  no scores returned"]
    lower_is_better = bool(result.get("lower_is_better", True))
    best_index = int(result.get("best_index", 0))
    score_min = min(scores)
    score_max = max(scores)
    span = score_max - score_min
    width = 24
    lines = []
    for index, score in enumerate(scores):
        if span == 0:
            fill = width
        elif lower_is_better:
            fill = round(((score_max - score) / span) * width)
        else:
            fill = round(((score - score_min) / span) * width)
        bar = "#" * fill
        marker = "  BEST" if index == best_index else ""
        lines.append(f"  #{index:<2} {score:>12.6f} |{bar:<24}|{marker}")
    return lines


def _log_step(
    index: int, total: int, title: str, rows: list[tuple[str, object]] | None = None
) -> None:
    print(f"\n[{index}/{total}] {title}", flush=True)
    for label, value in rows or []:
        print(f"  {label:<18} {value}", flush=True)


def _print_header() -> None:
    print("WorldForge LeWorldModel real checkpoint inference", flush=True)
    print("=" * 48, flush=True)
    print(
        "Mode: real upstream checkpoint inference, not the injected checkout-safe demo.", flush=True
    )


def _runtime_command(*, checkpoint: Path, device: str) -> str:
    return (
        f"scripts/lewm-real --checkpoint {checkpoint} --device {device}\n"
        "\n"
        "or, without the wrapper:\n"
        'uv run --python 3.10 --with "stable-worldmodel[train,env] @ '
        'git+https://github.com/galilai-group/stable-worldmodel.git" '
        f'--with "datasets>=2.21" lewm-real --checkpoint {checkpoint} --device {device}'
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", default=os.environ.get("LEWORLDMODEL_POLICY", "pusht/lewm"))
    parser.add_argument(
        "--stablewm-home",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", DEFAULT_STABLEWM_HOME)).expanduser(),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=("Checkpoint root passed to LeWorldModelProvider. Defaults to STABLEWM_HOME."),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            Path(os.environ["LEWORLDMODEL_CHECKPOINT"]).expanduser()
            if os.environ.get("LEWORLDMODEL_CHECKPOINT")
            else None
        ),
        help=(
            "Exact <policy>_object.ckpt path. When provided, the cache root is inferred from "
            "the policy-shaped suffix unless --cache-dir is also supplied."
        ),
    )
    parser.add_argument("--device", default=os.environ.get("LEWORLDMODEL_DEVICE", "cpu"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the machine-readable JSON summary.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.json_only:
        _print_header()

    object_path, cache_dir = _resolve_checkpoint(
        policy=args.policy,
        stablewm_home=args.stablewm_home,
        cache_dir=args.cache_dir,
        checkpoint=args.checkpoint,
    )
    if not args.json_only:
        _log_step(
            1,
            6,
            "Resolve checkpoint and runtime settings",
            [
                ("policy", args.policy),
                ("checkpoint", object_path),
                ("cache root", cache_dir),
                ("device", args.device),
            ],
        )

    if not args.json_only:
        _log_step(2, 6, "Create LeWorldModelProvider", [("provider", "leworldmodel")])
    provider = LeWorldModelProvider(
        policy=args.policy,
        cache_dir=str(cache_dir),
        device=args.device,
    )
    health = provider.health().to_dict()
    if not args.json_only:
        _log_step(
            3,
            6,
            "Preflight optional runtime dependencies",
            [
                ("healthy", health.get("healthy")),
                ("details", health.get("details")),
                ("latency ms", health.get("latency_ms")),
            ],
        )
    if not health.get("healthy"):
        if args.json_only:
            print(
                json.dumps(
                    {
                        "checkpoint": str(object_path),
                        "error": "runtime preflight failed",
                        "health": health,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(
            "\nLeWorldModel runtime preflight failed: "
            f"{health.get('details')}\n\n"
            "Run the complete uv-backed task instead:\n"
            f"{_runtime_command(checkpoint=object_path, device=args.device)}",
            flush=True,
        )
        return 1

    if not args.json_only:
        _log_step(
            4,
            6,
            "Build synthetic LeWorldModel tensors",
            [
                ("batch", args.batch),
                ("samples", args.samples),
                ("history", args.history),
                ("horizon", args.horizon),
                ("action dim", args.action_dim),
                ("image size", args.image_size),
            ],
        )
    info, action_candidates = _build_inputs(
        batch=args.batch,
        samples=args.samples,
        history=args.history,
        horizon=args.horizon,
        action_dim=args.action_dim,
        image_size=args.image_size,
    )
    input_shapes = _input_shape_summary(info, action_candidates)
    if not args.json_only:
        for label, shape in input_shapes.items():
            print(f"  {label:<18} {shape}", flush=True)

    if not args.json_only:
        _log_step(
            5,
            6,
            "Run score_actions through the real checkpoint",
            [
                ("operation", "leworldmodel.score_actions"),
                ("candidate count", args.samples),
            ],
        )
    started = perf_counter()
    result = provider.score_actions(info=info, action_candidates=action_candidates)
    score_latency_ms = (perf_counter() - started) * 1000
    result_payload = result.to_dict()
    payload = {
        "checkpoint": str(object_path),
        "health": health,
        "result": result_payload,
    }
    if args.json_only:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    metadata = result_payload.get("metadata", {})
    _log_step(
        6,
        6,
        "Rank action candidates",
        [
            ("lower is better", result_payload.get("lower_is_better")),
            ("best index", result_payload.get("best_index")),
            ("best score", result_payload.get("best_score")),
            ("score latency ms", f"{score_latency_ms:.2f}"),
            ("score type", metadata.get("score_type", "cost")),
        ],
    )
    print("\nCandidate scores", flush=True)
    print("----------------", flush=True)
    for line in _score_chart(result_payload):
        print(line, flush=True)
    print("\nCompleted real LeWorldModel checkpoint inference.", flush=True)
    print("Use --json-only for the machine-readable summary.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
