"""Build a LeWorldModel object checkpoint from Hugging Face assets.

Run with the same upstream runtime used by the real smoke:

    uv run --python 3.13 --with "<git stable-worldmodel>" --with "datasets>=2.21"
      --with huggingface_hub --with hydra-core --with omegaconf --with transformers
      --with "opencv-python" --with "imageio" worldforge-build-leworldmodel-checkpoint

The command downloads ``config.json`` and ``weights.pt`` from the model repo,
instantiates the LeWM module, loads the weights, and saves the object checkpoint
where ``stable_worldmodel.policy.AutoCostModel`` expects it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from worldforge.smoke.leworldmodel import DEFAULT_STABLEWM_HOME, _checkpoint_path

DEFAULT_REPO_ID = "quentinll/lewm-pusht"
LEWORLDMODEL_OFFICIAL_REPO_URL = "https://github.com/lucas-maes/le-wm"
LEWORLDMODEL_RUNTIME_API = "stable_worldmodel.policy.AutoCostModel"
LEWORLDMODEL_HF_CONFIG_TARGET = "stable_worldmodel.wm.lewm"
LEWORLDMODEL_HF_BACKBONE_TARGET = "stable_pretraining.backbone.utils.vit_hf"


def _vit_hf_backbone(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs: Any,
) -> object:
    """Build the ViT backbone referenced by the official PushT LeWM config."""

    from transformers import ViTConfig, ViTModel

    size_configs: dict[str, dict[str, int]] = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
        "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
        "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
    }
    if size not in size_configs:
        raise ValueError(f"Invalid ViT size {size!r}; expected one of {sorted(size_configs)}.")

    config_params = {
        **size_configs[size],
        "intermediate_size": size_configs[size]["hidden_size"] * 4,
        "image_size": image_size,
        "patch_size": patch_size,
        **kwargs,
    }
    if pretrained:
        model_name = f"google/vit-{size}-patch{patch_size}-{image_size}"
        model = ViTModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )
    else:
        model = ViTModel(
            ViTConfig(**config_params),
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )
    model.config.interpolate_pos_encoding = True
    return model


def _install_stable_pretraining_backbone_shim() -> None:
    """Expose the single stable-pretraining backbone target used by LeWM configs."""

    stable_pretraining = sys.modules.get("stable_pretraining") or ModuleType("stable_pretraining")
    backbone = sys.modules.get("stable_pretraining.backbone") or ModuleType(
        "stable_pretraining.backbone"
    )
    utils = sys.modules.get("stable_pretraining.backbone.utils") or ModuleType(
        "stable_pretraining.backbone.utils"
    )
    stable_pretraining.__path__ = []  # type: ignore[attr-defined]
    backbone.__path__ = []  # type: ignore[attr-defined]
    utils.vit_hf = _vit_hf_backbone  # type: ignore[attr-defined]
    stable_pretraining.backbone = backbone  # type: ignore[attr-defined]
    backbone.utils = utils  # type: ignore[attr-defined]
    sys.modules["stable_pretraining"] = stable_pretraining
    sys.modules["stable_pretraining.backbone"] = backbone
    sys.modules["stable_pretraining.backbone.utils"] = utils


def _repo_cache_dir(repo_id: str) -> Path:
    safe_repo = repo_id.replace("/", "__")
    return Path("~/.cache/worldforge/leworldmodel").expanduser() / safe_repo


def _load_optional_build_dependencies():
    try:
        # Required by current stable_worldmodel top-level visual-wrapper imports.
        importlib.import_module("cv2")
        importlib.import_module("imageio")
        importlib.import_module("stable_worldmodel")  # probe upstream runtime import envelope
        importlib.import_module("transformers")
        try:
            importlib.import_module("stable_pretraining")  # use upstream package when present
        except ImportError:
            _install_stable_pretraining_backbone_shim()
        import torch
        from huggingface_hub import hf_hub_download
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - exercised by host smoke usage
        missing = exc.name or ""
        if missing == "matplotlib":
            raise SystemExit(
                "Building a LeWorldModel object checkpoint requires matplotlib in current "
                "upstream LeWorldModel runtime environments. Add `--with matplotlib` to the "
                "uv run invocation (see docs/src/operations.md)."
            ) from exc
        if missing == "cv2":
            raise SystemExit(
                "Building a LeWorldModel object checkpoint requires opencv-python because the "
                "current upstream stable_worldmodel package imports cv2 at module load time. "
                'Add `--with "opencv-python"` to the uv run invocation.'
            ) from exc
        if missing == "imageio":
            raise SystemExit(
                "Building a LeWorldModel object checkpoint requires imageio because the current "
                "upstream stable_worldmodel package imports imageio at module load time. "
                "Add `--with imageio` to the uv run invocation."
            ) from exc
        if missing == "transformers":
            raise SystemExit(
                "Building a LeWorldModel object checkpoint requires transformers because the "
                "official LeWM PushT config constructs its ViT encoder through a Hugging Face "
                "ViTModel. Add `--with transformers` to the uv run invocation."
            ) from exc
        raise SystemExit(
            "Building a LeWorldModel object checkpoint requires torch, huggingface_hub, "
            "hydra-core, omegaconf, transformers, opencv-python, imageio, and the upstream "
            "stable-worldmodel LeWM modules. Run the command with the dependency flags "
            "documented in docs/src/operations.md."
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


def _config_target(config: object) -> str:
    getter = getattr(config, "get", None)
    target = getter("_target_") if callable(getter) else None
    if not isinstance(target, str) or not target.strip():
        raise SystemExit("LeWorldModel config.json must contain a non-empty _target_ field.")
    return target.strip()


def _ensure_leworldmodel_target(target: str) -> None:
    normalized = target.lower()
    if "lewm" not in normalized and "jepa" not in normalized:
        raise SystemExit(
            f"LeWorldModel config.json did not describe a LeWM/JEPA model target: {target!r}."
        )


def _checkpoint_provenance() -> dict[str, object]:
    return {
        "model_family": "LeWorldModel (LeWM)",
        "official_code": LEWORLDMODEL_OFFICIAL_REPO_URL,
        "runtime_api": LEWORLDMODEL_RUNTIME_API,
    }


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
            **_checkpoint_provenance(),
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
    target = _config_target(config)
    _ensure_leworldmodel_target(target)
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
        "config_target": target,
        **_checkpoint_provenance(),
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
