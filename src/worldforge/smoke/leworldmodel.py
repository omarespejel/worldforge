"""Run a LeWorldModel provider smoke test with a real checkpoint.

Invoke this command through uv, for example:

    uv run --python 3.10 --with "<git stable-worldmodel>" --with "datasets>=2.21"
      worldforge-smoke-leworldmodel

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

from worldforge.providers import LeWorldModelProvider

DEFAULT_STABLEWM_HOME = "~/.stable-wm"


def _checkpoint_path(cache_dir: Path, policy: str) -> Path:
    return cache_dir / f"{policy}_object.ckpt"


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
    parser.add_argument("--device", default=os.environ.get("LEWORLDMODEL_DEVICE", "cpu"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    return parser


def main() -> int:
    args = _parser().parse_args()
    cache_dir = args.cache_dir or args.stablewm_home
    object_path = _require_object_checkpoint(policy=args.policy, cache_dir=cache_dir)
    info, action_candidates = _build_inputs(
        batch=args.batch,
        samples=args.samples,
        history=args.history,
        horizon=args.horizon,
        action_dim=args.action_dim,
        image_size=args.image_size,
    )
    provider = LeWorldModelProvider(
        policy=args.policy,
        cache_dir=str(cache_dir),
        device=args.device,
    )
    result = provider.score_actions(info=info, action_candidates=action_candidates)
    print(
        json.dumps(
            {
                "checkpoint": str(object_path),
                "health": provider.health().to_dict(),
                "result": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
