"""Command line interface for WorldForge."""

from __future__ import annotations

import argparse
import json

from worldforge import Action, WorldForge
from worldforge.evaluation import EvaluationSuite
from worldforge.providers import ProviderError


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldforge", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    providers = subparsers.add_parser("providers", help="List registered providers.")
    providers.add_argument("--state-dir", default=".worldforge/worlds")

    provider = subparsers.add_parser("provider", help="Inspect provider profiles and health.")
    provider_subparsers = provider.add_subparsers(dest="provider_command", required=True)

    provider_list = provider_subparsers.add_parser("list", help="List provider profiles.")
    provider_list.add_argument("--state-dir", default=".worldforge/worlds")
    provider_list.add_argument("--registered-only", action="store_true")
    provider_list.add_argument("--capability")

    provider_info = provider_subparsers.add_parser("info", help="Show provider details.")
    provider_info.add_argument("name")
    provider_info.add_argument("--state-dir", default=".worldforge/worlds")

    provider_health = provider_subparsers.add_parser("health", help="Show provider health.")
    provider_health.add_argument("name", nargs="?")
    provider_health.add_argument("--state-dir", default=".worldforge/worlds")
    provider_health.add_argument("--registered-only", action="store_true")
    provider_health.add_argument("--capability")

    doctor = subparsers.add_parser("doctor", help="Inspect the local WorldForge environment.")
    doctor.add_argument("--state-dir", default=".worldforge/worlds")
    doctor.add_argument("--registered-only", action="store_true")
    doctor.add_argument("--capability")

    predict = subparsers.add_parser("predict", help="Run a deterministic prediction.")
    predict.add_argument("world_name")
    predict.add_argument("--provider", default="mock")
    predict.add_argument("--x", type=float, required=True)
    predict.add_argument("--y", type=float, required=True)
    predict.add_argument("--z", type=float, required=True)
    predict.add_argument("--steps", type=int, default=1)
    predict.add_argument("--state-dir", default=".worldforge/worlds")

    evaluate = subparsers.add_parser("eval", help="Run a built-in evaluation suite.")
    evaluate.add_argument("--suite", default="physics")
    evaluate.add_argument("--provider", default="mock")
    evaluate.add_argument("--state-dir", default=".worldforge/worlds")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    forge = WorldForge(state_dir=args.state_dir)

    try:
        if args.command == "providers":
            _print_json([info.to_dict() for info in forge.list_providers()])
            return 0

        if args.command == "provider":
            if args.provider_command == "list":
                report = forge.doctor(
                    capability=args.capability,
                    registered_only=args.registered_only,
                )
                _print_json([provider.to_dict() for provider in report.providers])
                return 0

            if args.provider_command == "info":
                name = args.name
                payload = {
                    "registered": name in forge.providers(),
                    "profile": forge.provider_profile(name).to_dict(),
                    "health": forge.provider_health(name).to_dict(),
                }
                if name in forge.providers():
                    payload["info"] = forge.provider_info(name).to_dict()
                _print_json(payload)
                return 0

            if args.provider_command == "health":
                if args.name:
                    _print_json(forge.provider_health(args.name).to_dict())
                    return 0
                report = forge.doctor(
                    capability=args.capability,
                    registered_only=args.registered_only,
                )
                _print_json(
                    [
                        {
                            **provider.health.to_dict(),
                            "registered": provider.registered,
                        }
                        for provider in report.providers
                    ]
                )
                return 0

        if args.command == "doctor":
            _print_json(
                forge.doctor(
                    capability=args.capability,
                    registered_only=args.registered_only,
                ).to_dict()
            )
            return 0

        if args.command == "predict":
            world = forge.create_world(args.world_name, args.provider)
            prediction = world.predict(Action.move_to(args.x, args.y, args.z), steps=args.steps)
            _print_json(
                {
                    "provider": prediction.provider,
                    "physics_score": prediction.physics_score,
                    "confidence": prediction.confidence,
                    "world_state": prediction.world_state,
                }
            )
            return 0

        if args.command == "eval":
            suite = EvaluationSuite.from_builtin(args.suite)
            report = suite.run_report(args.provider, forge=forge)
            print(report.to_markdown())
            return 0
    except (ProviderError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
