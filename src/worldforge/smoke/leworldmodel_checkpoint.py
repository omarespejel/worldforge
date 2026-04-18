"""Build a LeWorldModel object checkpoint from Hugging Face assets.

Run with the same upstream runtime used by the real smoke:

    uv run --python 3.10 --with "<git stable-worldmodel>" --with "datasets>=2.21"
      --with huggingface_hub worldforge-build-leworldmodel-checkpoint

The command downloads ``config.json`` and ``weights.pt`` from the model repo,
instantiates the LeWM module, loads the weights, and saves the object checkpoint
where ``stable_worldmodel.policy.AutoCostModel`` expects it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from worldforge.smoke.leworldmodel import DEFAULT_STABLEWM_HOME, _checkpoint_path

DEFAULT_REPO_ID = "quentinll/lewm-pusht"


def _repo_cache_dir(repo_id: str) -> Path:
    safe_repo = repo_id.replace("/", "__")
    return Path("~/.cache/worldforge/leworldmodel").expanduser() / safe_repo


def _load_optional_build_dependencies():
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - exercised by host smoke usage
        raise SystemExit(
            "Building a LeWorldModel object checkpoint requires torch, huggingface_hub, "
            "hydra-core, omegaconf, and the upstream stable-worldmodel LeWM modules. "
            "Run the command with the dependency flags documented in the README."
        ) from exc
    return torch, hf_hub_download, instantiate, OmegaConf


def _incompatible_keys(load_result: object) -> tuple[list[str], list[str]]:
    if load_result is None:
        return [], []
    if hasattr(load_result, "missing_keys") and hasattr(load_result, "unexpected_keys"):
        return list(load_result.missing_keys), list(load_result.unexpected_keys)
    missing, unexpected = load_result
    return list(missing), list(unexpected)


def build_checkpoint(
    *,
    repo_id: str,
    policy: str,
    stablewm_home: Path,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Build and persist an object checkpoint from a Hugging Face LeWM repo."""

    output_path = _checkpoint_path(stablewm_home.expanduser(), policy)
    if output_path.exists() and not force:
        return {
            "created": False,
            "output": str(output_path),
            "policy": policy,
            "repo_id": repo_id,
            "reason": "checkpoint already exists",
        }

    torch, hf_hub_download, instantiate, OmegaConf = _load_optional_build_dependencies()
    asset_dir = (cache_dir or _repo_cache_dir(repo_id)).expanduser()
    config_path = Path(
        hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=str(asset_dir))
    )
    weights_path = Path(
        hf_hub_download(repo_id=repo_id, filename="weights.pt", local_dir=str(asset_dir))
    )

    config = OmegaConf.load(config_path)
    model = instantiate(config)
    weights = torch.load(weights_path, map_location="cpu")
    incompatible = model.load_state_dict(weights, strict=False)
    missing, unexpected = _incompatible_keys(incompatible)
    if missing or unexpected:
        raise SystemExit(
            "LeWorldModel weights did not match the instantiated model: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_path)
    return {
        "created": True,
        "config": str(config_path),
        "output": str(output_path),
        "policy": policy,
        "repo_id": repo_id,
        "weights": str(weights_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--policy", default=os.environ.get("LEWORLDMODEL_POLICY", "pusht/lewm"))
    parser.add_argument(
        "--stablewm-home",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", DEFAULT_STABLEWM_HOME)).expanduser(),
    )
    parser.add_argument(
        "--asset-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for downloaded Hugging Face config/weights. Defaults to ~/.cache/worldforge."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild and overwrite the object checkpoint when it already exists.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = build_checkpoint(
        repo_id=args.repo_id,
        policy=args.policy,
        stablewm_home=args.stablewm_home,
        cache_dir=args.asset_cache_dir.expanduser() if args.asset_cache_dir else None,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
