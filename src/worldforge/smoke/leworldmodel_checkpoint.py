"""Build a LeWorldModel object checkpoint from Hugging Face assets.

Run with the same upstream runtime used by the real smoke:

    uv run --python 3.13 --with "<git stable-worldmodel>" --with "datasets>=2.21"
      --with huggingface_hub worldforge-build-leworldmodel-checkpoint

The command downloads ``config.json`` and ``weights.pt`` from the model repo,
instantiates the LeWM module, loads the weights, and saves the object checkpoint
where ``stable_worldmodel.policy.AutoCostModel`` expects it.
"""

from __future__ import annotations

import argparse
import importlib
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
        importlib.import_module("stable_pretraining")  # probe upstream transitive imports
        import torch
        from huggingface_hub import hf_hub_download
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - exercised by host smoke usage
        missing = exc.name or ""
        if missing == "matplotlib":
            raise SystemExit(
                "Building a LeWorldModel object checkpoint requires matplotlib because upstream "
                "stable_pretraining imports it at module load time. Add `--with matplotlib` to "
                "the uv run invocation (see docs/src/operations.md)."
            ) from exc
        raise SystemExit(
            "Building a LeWorldModel object checkpoint requires torch, huggingface_hub, "
            "hydra-core, omegaconf, matplotlib, and the upstream stable-worldmodel LeWM modules. "
            "Run the command with the dependency flags documented in docs/src/operations.md."
        ) from exc
    return torch, hf_hub_download, instantiate, OmegaConf


def _incompatible_keys(load_result: object) -> tuple[list[str], list[str]]:
    if load_result is None:
        return [], []
    if hasattr(load_result, "missing_keys") and hasattr(load_result, "unexpected_keys"):
        return list(load_result.missing_keys), list(load_result.unexpected_keys)
    missing, unexpected = load_result
    return list(missing), list(unexpected)


def _load_weights(torch: object, weights_path: Path, *, allow_unsafe_pickle: bool) -> object:
    if allow_unsafe_pickle:
        return torch.load(weights_path, map_location="cpu")
    try:
        return torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise SystemExit(
            "Safe LeWorldModel weight loading requires a torch version that supports "
            "`torch.load(..., weights_only=True)`. Upgrade torch or rerun with "
            "`--allow-unsafe-pickle` only when the weights file is trusted."
        ) from exc


def build_checkpoint(
    *,
    repo_id: str,
    policy: str,
    stablewm_home: Path,
    cache_dir: Path | None = None,
    revision: str | None = None,
    allow_unsafe_pickle: bool = False,
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

    torch, hf_hub_download, instantiate, omega_conf = _load_optional_build_dependencies()
    asset_dir = (cache_dir or _repo_cache_dir(repo_id)).expanduser()
    config_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            local_dir=str(asset_dir),
            revision=revision,
        )
    )
    weights_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="weights.pt",
            local_dir=str(asset_dir),
            revision=revision,
        )
    )

    config = omega_conf.load(config_path)
    model = instantiate(config)
    weights = _load_weights(torch, weights_path, allow_unsafe_pickle=allow_unsafe_pickle)
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
    summary: dict[str, object] = {
        "created": True,
        "config": str(config_path),
        "output": str(output_path),
        "policy": policy,
        "repo_id": repo_id,
        "weights": str(weights_path),
    }
    if revision is not None:
        summary["revision"] = revision
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--revision",
        default=os.environ.get("LEWORLDMODEL_REVISION"),
        help="Optional Hugging Face git revision, tag, or commit for config.json and weights.pt.",
    )
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
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Allow legacy torch.load pickle deserialization for trusted weights only. "
            "By default, weights are loaded with weights_only=True."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = build_checkpoint(
        repo_id=args.repo_id,
        policy=args.policy,
        stablewm_home=args.stablewm_home,
        cache_dir=args.asset_cache_dir.expanduser() if args.asset_cache_dir else None,
        revision=args.revision,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
