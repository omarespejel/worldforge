"""Command line interface for the pure-Python WorldForge package."""

from __future__ import annotations

import argparse
import json

from worldforge import Action, WorldForge
from worldforge.eval import EvalSuite


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldforge", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    providers = subparsers.add_parser("providers", help="List registered providers.")
    providers.add_argument("--state-dir", default=".worldforge/state")

    predict = subparsers.add_parser("predict", help="Run a deterministic prediction.")
    predict.add_argument("world_name")
    predict.add_argument("--provider", default="mock")
    predict.add_argument("--x", type=float, required=True)
    predict.add_argument("--y", type=float, required=True)
    predict.add_argument("--z", type=float, required=True)
    predict.add_argument("--steps", type=int, default=1)
    predict.add_argument("--state-dir", default=".worldforge/state")

    evaluate = subparsers.add_parser("eval", help="Run a built-in evaluation suite.")
    evaluate.add_argument("--suite", default="physics")
    evaluate.add_argument("--provider", default="mock")
    evaluate.add_argument("--state-dir", default=".worldforge/state")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    forge = WorldForge(state_dir=args.state_dir)

    if args.command == "providers":
        print(json.dumps([info.to_dict() for info in forge.list_providers()], indent=2))
        return 0

    if args.command == "predict":
        world = forge.create_world(args.world_name, args.provider)
        prediction = world.predict(Action.move_to(args.x, args.y, args.z), steps=args.steps)
        print(
            json.dumps(
                {
                    "provider": prediction.provider,
                    "physics_score": prediction.physics_score,
                    "confidence": prediction.confidence,
                    "world_state": prediction.world_state,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "eval":
        suite = EvalSuite.from_builtin(args.suite)
        report = suite.run_report_data(args.provider, forge=forge)
        print(report.to_markdown())
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
