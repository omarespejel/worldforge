"""One-command real robotics showcase for LeRobot + LeWorldModel."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from . import lerobot_leworldmodel
from .lerobot_leworldmodel import (
    DEFAULT_DEVICE,
    DEFAULT_LEROBOT_POLICY,
    DEFAULT_LEWORLDMODEL_POLICY,
    DEFAULT_MODE,
)
from .leworldmodel import DEFAULT_STABLEWM_HOME
from .pusht_showcase_inputs import DEFAULT_ACTION_DIM, DEFAULT_HORIZON

DEFAULT_JSON_OUTPUT = Path("/tmp/worldforge-robotics-showcase/real-run.json")
PUSHT_INPUT_MODULE = "worldforge.smoke.pusht_showcase_inputs"


def _env_value(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the polished WorldForge real robotics showcase: LeRobot proposes "
            "PushT actions, LeWorldModel scores checkpoint-native candidates, and "
            "WorldForge ranks and mock-executes the selected action chunk."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default command:\n"
            "  scripts/robotics-showcase\n\n"
            "The lower-level configurable runner remains available as:\n"
            "  scripts/lewm-lerobot-real --help"
        ),
    )
    parser.add_argument(
        "--policy-path",
        default=_env_value("LEROBOT_POLICY_PATH")
        or _env_value("LEROBOT_POLICY")
        or DEFAULT_LEROBOT_POLICY,
        help="LeRobot policy repo id or local checkpoint directory.",
    )
    parser.add_argument(
        "--policy-type",
        default=_env_value("LEROBOT_POLICY_TYPE") or "diffusion",
        help="LeRobot policy type. Defaults to the PushT diffusion policy.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            Path(os.environ["LEWORLDMODEL_CHECKPOINT"]).expanduser()
            if os.environ.get("LEWORLDMODEL_CHECKPOINT")
            else None
        ),
        help="Exact LeWorldModel <policy>_object.ckpt path.",
    )
    parser.add_argument(
        "--lewm-policy",
        default=_env_value("LEWORLDMODEL_POLICY")
        or _env_value("LEWM_POLICY")
        or DEFAULT_LEWORLDMODEL_POLICY,
        help="LeWorldModel policy/checkpoint run name relative to STABLEWM_HOME.",
    )
    parser.add_argument(
        "--stablewm-home",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", DEFAULT_STABLEWM_HOME)).expanduser(),
    )
    parser.add_argument("--lewm-cache-dir", type=Path, default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--lerobot-device", default=_env_value("LEROBOT_DEVICE"))
    parser.add_argument("--lewm-device", default=_env_value("LEWORLDMODEL_DEVICE"))
    parser.add_argument("--lerobot-cache-dir", default=_env_value("LEROBOT_CACHE_DIR"))
    parser.add_argument("--mode", choices=("select_action", "predict_chunk"), default=DEFAULT_MODE)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Write the full summary JSON. Defaults to /tmp to keep the repo clean.",
    )
    parser.add_argument(
        "--no-json-output",
        action="store_true",
        help="Skip writing the default /tmp JSON artifact.",
    )
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Control ANSI colors in the human-readable output.",
    )
    parser.add_argument("--no-color", action="store_const", const="never", dest="color")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--no-execute", action="store_true")
    return parser


def _append_optional_path(argv: list[str], flag: str, value: Path | str | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _forward_args(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--policy-path",
        args.policy_path,
        "--policy-type",
        args.policy_type,
        "--lewm-policy",
        args.lewm_policy,
        "--stablewm-home",
        str(args.stablewm_home),
        "--device",
        args.device,
        "--mode",
        args.mode,
        "--observation-module",
        f"{PUSHT_INPUT_MODULE}:build_observation",
        "--score-info-module",
        f"{PUSHT_INPUT_MODULE}:build_score_info",
        "--translator",
        f"{PUSHT_INPUT_MODULE}:translate_candidates",
        "--candidate-builder",
        f"{PUSHT_INPUT_MODULE}:build_action_candidates",
        "--expected-action-dim",
        str(DEFAULT_ACTION_DIM),
        "--expected-horizon",
        str(DEFAULT_HORIZON),
        "--task",
        "PushT real LeRobot policy plus LeWorldModel score planning",
        "--goal",
        "rank PushT policy action candidates with a LeWorldModel checkpoint",
        "--color",
        args.color,
    ]
    _append_optional_path(forwarded, "--checkpoint", args.checkpoint)
    _append_optional_path(forwarded, "--lewm-cache-dir", args.lewm_cache_dir)
    _append_optional_path(forwarded, "--lerobot-device", args.lerobot_device)
    _append_optional_path(forwarded, "--lewm-device", args.lewm_device)
    _append_optional_path(forwarded, "--lerobot-cache-dir", args.lerobot_cache_dir)
    _append_optional_path(forwarded, "--state-dir", args.state_dir)
    if not args.no_json_output:
        _append_optional_path(forwarded, "--json-output", args.json_output)
    if args.json_only:
        forwarded.append("--json-only")
    if args.health_only:
        forwarded.append("--health-only")
    if args.no_execute:
        forwarded.append("--no-execute")
    return forwarded


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args, extra = parser.parse_known_args(argv)
    forwarded = _forward_args(args)
    forwarded.extend(extra)
    return lerobot_leworldmodel.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
