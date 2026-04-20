"""Typed local-first CLI for WorldForge provider and evaluation workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldforge import Action, GenerationOptions, VideoClip, WorldForge, WorldForgeError
from worldforge.benchmark import ProviderBenchmarkHarness
from worldforge.evaluation import EvaluationSuite
from worldforge.providers import ProviderError
from worldforge.providers.catalog import provider_docs_index

CLI_DESCRIPTION = (
    "Typed local-first CLI for provider diagnostics, prediction, generation, evaluation, "
    "benchmarking, and runnable demos."
)

CLI_EPILOG = """Common commands:
  worldforge examples
  worldforge doctor
  worldforge provider list
  worldforge provider docs
  worldforge provider info mock
  worldforge harness --list
  worldforge predict kitchen --provider mock --x 0.3 --y 0.8 --z 0.0 --steps 2
  worldforge eval --suite planning --provider mock --format json
  worldforge benchmark --provider mock --iterations 5 --format json
"""

EXAMPLE_COMMANDS: tuple[dict[str, str], ...] = (
    {
        "task": "Prediction and evaluation",
        "name": "basic-prediction",
        "surface": "predict, planning, evaluation",
        "requires": "base WorldForge package",
        "command": "uv run python examples/basic_prediction.py",
        "description": (
            "Create a mock world, run a deterministic prediction, plan an object move, and "
            "print a physics evaluation report."
        ),
    },
    {
        "task": "Provider comparison",
        "name": "cross-provider-compare",
        "surface": "provider registry, comparison",
        "requires": "base WorldForge package",
        "command": "uv run python examples/cross_provider_compare.py",
        "description": (
            "Register a second deterministic provider and compare prediction outputs across "
            "provider surfaces."
        ),
    },
    {
        "task": "Score planning",
        "name": "leworldmodel-score-planning",
        "surface": "score provider, planning, persistence",
        "requires": "base WorldForge package; injected deterministic score runtime",
        "command": "uv run worldforge-demo-leworldmodel",
        "description": (
            "Run the packaged LeWorldModel provider-surface demo without downloading "
            "upstream checkpoints."
        ),
    },
    {
        "task": "Policy plus score planning",
        "name": "lerobot-policy-score-planning",
        "surface": "policy provider, score provider, planning, persistence",
        "requires": "base WorldForge package; injected deterministic policy runtime",
        "command": "uv run worldforge-demo-lerobot",
        "description": (
            "Run the packaged LeRobot policy-plus-score planning demo without installing "
            "LeRobot or torch."
        ),
    },
    {
        "task": "Visual harness",
        "name": "theworldharness",
        "surface": "E2E flows, provider diagnostics, benchmark comparison",
        "requires": "Textual through the harness extra",
        "command": "uv run --extra harness worldforge-harness",
        "description": (
            "Open the optional Textual harness for running packaged E2E demos, diagnostics, "
            "and benchmark comparisons as visible provider workflows."
        ),
    },
    {
        "task": "Optional runtime smoke",
        "name": "leworldmodel-real-checkpoint-smoke",
        "surface": "optional runtime smoke",
        "requires": "host-owned stable-worldmodel, torch, datasets, and LeWM checkpoint assets",
        "command": (
            'uv run --python 3.10 --with "stable-worldmodel[train,env] @ '
            'git+https://github.com/galilai-group/stable-worldmodel.git" '
            '--with "datasets>=2.21" worldforge-smoke-leworldmodel'
        ),
        "description": (
            "Exercise the real LeWorldModel checkpoint path from a host environment that owns "
            "the optional runtime and assets."
        ),
    },
)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2))


def _print_examples_markdown() -> None:
    print("# WorldForge Examples")
    current_task = ""
    for example in EXAMPLE_COMMANDS:
        if example["task"] != current_task:
            current_task = example["task"]
            print()
            print(f"## {current_task}")
            print()
            print("| Example | Surface | Requirements | Command |")
            print("| --- | --- | --- | --- |")
        print(
            "| "
            f"`{example['name']}` | "
            f"{example['surface']} | "
            f"{example['requires']} | "
            f"`{example['command']}` |"
        )


def _provider_docs_entries(name: str | None = None) -> tuple[dict[str, str], ...]:
    entries = provider_docs_index()
    if name is None:
        return entries
    return tuple(entry for entry in entries if entry["name"] == name)


def _print_provider_docs_markdown(entries: tuple[dict[str, str], ...]) -> None:
    print("# WorldForge Provider Docs")
    print()
    print("| Provider | Capability surface | Registration | Docs |")
    print("| --- | --- | --- | --- |")
    for entry in entries:
        print(
            "| "
            f"`{entry['name']}` | "
            f"{entry['capabilities']} | "
            f"{entry['registration']} | "
            f"`{entry['docs_path']}` |"
        )


def _add_generation_arguments(parser: argparse.ArgumentParser, *, include_fps: bool = True) -> None:
    parser.add_argument("--image", help="Input image path, URL, or provider-native reference.")
    parser.add_argument("--video", help="Input video path, URL, or provider-native reference.")
    parser.add_argument("--model", help="Provider model identifier.")
    parser.add_argument("--ratio", help="Provider aspect-ratio option.")
    parser.add_argument("--size", help="Provider resolution or size option.")
    if include_fps:
        parser.add_argument("--fps", type=float, help="Requested frames per second.")
    parser.add_argument("--seed", type=int, help="Provider seed when supported.")
    parser.add_argument("--negative-prompt", help="Negative prompt when supported.")
    parser.add_argument(
        "--reference-image",
        action="append",
        default=[],
        help="Reference image path, URL, or data URI. Can be repeated.",
    )


def _build_generation_options(args: argparse.Namespace) -> GenerationOptions:
    return GenerationOptions(
        image=getattr(args, "image", None),
        video=getattr(args, "video", None),
        model=getattr(args, "model", None),
        ratio=getattr(args, "ratio", None),
        size=getattr(args, "size", None),
        fps=getattr(args, "fps", None),
        seed=getattr(args, "seed", None),
        negative_prompt=getattr(args, "negative_prompt", None),
        reference_images=list(getattr(args, "reference_image", [])),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worldforge",
        description=CLI_DESCRIPTION,
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    examples = subparsers.add_parser(
        "examples",
        help="List runnable examples grouped by task.",
    )
    examples.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for the examples index.",
    )

    providers = subparsers.add_parser("providers", help="List registered providers.")
    providers.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )

    provider = subparsers.add_parser("provider", help="Inspect provider profiles and health.")
    provider_subparsers = provider.add_subparsers(dest="provider_command", required=True)

    provider_list = provider_subparsers.add_parser("list", help="List provider profiles.")
    provider_list.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )
    provider_list.add_argument(
        "--registered-only",
        action="store_true",
        help="Show only providers registered for this process.",
    )
    provider_list.add_argument("--capability", help="Filter providers by capability name.")

    provider_info = provider_subparsers.add_parser("info", help="Show provider details.")
    provider_info.add_argument("name", help="Provider name.")
    provider_info.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )

    provider_health = provider_subparsers.add_parser("health", help="Show provider health.")
    provider_health.add_argument("name", nargs="?", help="Optional provider name.")
    provider_health.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )
    provider_health.add_argument(
        "--registered-only",
        action="store_true",
        help="Show only providers registered for this process.",
    )
    provider_health.add_argument("--capability", help="Filter providers by capability name.")

    provider_docs = provider_subparsers.add_parser(
        "docs",
        help="List provider documentation paths.",
    )
    provider_docs.add_argument("name", nargs="?", help="Optional provider name.")
    provider_docs.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for provider docs metadata.",
    )

    doctor = subparsers.add_parser("doctor", help="Inspect the local WorldForge environment.")
    doctor.add_argument("--state-dir", default=".worldforge/worlds", help="World state directory.")
    doctor.add_argument(
        "--registered-only",
        action="store_true",
        help="Show only providers registered for this process.",
    )
    doctor.add_argument("--capability", help="Filter providers by capability name.")

    generate = subparsers.add_parser("generate", help="Generate a clip with a provider.")
    generate.add_argument("prompt", help="Generation prompt.")
    generate.add_argument("--provider", default="mock", help="Provider name.")
    generate.add_argument("--duration", type=float, default=5.0, help="Clip duration in seconds.")
    generate.add_argument("--output", help="Optional path for the generated clip bytes.")
    generate.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )
    _add_generation_arguments(generate)

    transfer = subparsers.add_parser("transfer", help="Transform an input clip with a provider.")
    transfer.add_argument("input", help="Input clip path.")
    transfer.add_argument("--provider", default="mock", help="Provider name.")
    transfer.add_argument("--prompt", default="", help="Transfer prompt.")
    transfer.add_argument("--width", type=int, default=1280, help="Input clip width.")
    transfer.add_argument("--height", type=int, default=720, help="Input clip height.")
    transfer.add_argument("--fps", type=float, default=24.0, help="Input clip frames per second.")
    transfer.add_argument("--duration", type=float, default=5.0, help="Input clip duration.")
    transfer.add_argument("--output", help="Optional path for the transformed clip bytes.")
    transfer.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )
    _add_generation_arguments(transfer, include_fps=False)

    predict = subparsers.add_parser("predict", help="Run a deterministic prediction.")
    predict.add_argument("world_name", help="World name to create or load.")
    predict.add_argument("--provider", default="mock", help="Provider name.")
    predict.add_argument("--x", type=float, required=True, help="Target x coordinate.")
    predict.add_argument("--y", type=float, required=True, help="Target y coordinate.")
    predict.add_argument("--z", type=float, required=True, help="Target z coordinate.")
    predict.add_argument("--steps", type=int, default=1, help="Prediction horizon in steps.")
    predict.add_argument("--state-dir", default=".worldforge/worlds", help="World state directory.")

    evaluate = subparsers.add_parser("eval", help="Run a built-in evaluation suite.")
    evaluate.add_argument(
        "--suite",
        default="physics",
        choices=EvaluationSuite.builtin_names(),
        help="Built-in evaluation suite.",
    )
    evaluate.add_argument(
        "--provider",
        dest="providers",
        action="append",
        default=None,
        help="Provider name to evaluate. Can be repeated.",
    )
    evaluate.add_argument(
        "--format",
        choices=("markdown", "json", "csv"),
        default="markdown",
        help="Evaluation report format.",
    )
    evaluate.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run provider latency and retry benchmarks.",
    )
    benchmark.add_argument(
        "--provider",
        dest="providers",
        action="append",
        default=None,
        help="Provider name to benchmark. Can be repeated.",
    )
    benchmark.add_argument(
        "--operation",
        dest="operations",
        action="append",
        default=None,
        choices=ProviderBenchmarkHarness.benchmarkable_operations,
        help="Operation to benchmark. Can be repeated.",
    )
    benchmark.add_argument("--iterations", type=int, default=5, help="Iterations per operation.")
    benchmark.add_argument("--concurrency", type=int, default=1, help="Concurrent workers.")
    benchmark.add_argument(
        "--format",
        choices=("markdown", "json", "csv"),
        default="markdown",
        help="Benchmark report format.",
    )
    benchmark.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )

    harness = subparsers.add_parser("harness", help="Launch TheWorldHarness TUI.")
    harness.add_argument(
        "--flow",
        choices=("leworldmodel", "lerobot", "diagnostics"),
        default="leworldmodel",
        help="Harness flow to open.",
    )
    harness.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Directory for persisted demo worlds. Defaults to a temporary directory.",
    )
    harness.add_argument(
        "--list",
        action="store_true",
        help="List available harness flows without launching the TUI.",
    )
    harness.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for --list.",
    )
    harness.add_argument("--no-animation", action="store_true", help="Disable step reveal delays.")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "examples":
        if args.format == "json":
            _print_json(EXAMPLE_COMMANDS)
        else:
            _print_examples_markdown()
        return 0

    if args.command == "provider" and args.provider_command == "docs":
        entries = _provider_docs_entries(args.name)
        if not entries:
            parser.exit(2, f"Unknown provider: {args.name}\n")
        if args.format == "json":
            _print_json(entries)
        else:
            _print_provider_docs_markdown(entries)
        return 0

    if args.command == "harness":
        from worldforge.harness.cli import run_from_args

        return run_from_args(
            flow_id=args.flow,
            state_dir=args.state_dir,
            list_only=args.list,
            output_format=args.format,
            animate=not args.no_animation,
        )

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

        if args.command == "generate":
            options = _build_generation_options(args)
            clip = forge.generate(
                args.prompt,
                args.provider,
                duration_seconds=args.duration,
                options=options,
            )
            payload = clip.to_dict()
            if args.output:
                payload["output_path"] = str(clip.save(Path(args.output)))
            _print_json(payload)
            return 0

        if args.command == "transfer":
            options = _build_generation_options(args)
            input_clip = VideoClip.from_file(
                args.input,
                fps=args.fps,
                resolution=(args.width, args.height),
                duration_seconds=args.duration,
            )
            clip = forge.transfer(
                input_clip,
                args.provider,
                width=args.width,
                height=args.height,
                fps=args.fps,
                prompt=args.prompt,
                options=options,
            )
            payload = clip.to_dict()
            if args.output:
                payload["output_path"] = str(clip.save(Path(args.output)))
            _print_json(payload)
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
            providers = args.providers or ["mock"]
            report = suite.run_report(providers, forge=forge)
            if args.format == "json":
                print(report.to_json())
            elif args.format == "csv":
                print(report.to_csv())
            else:
                print(report.to_markdown())
            return 0

        if args.command == "benchmark":
            harness = ProviderBenchmarkHarness(forge=forge)
            providers = args.providers or ["mock"]
            report = harness.run(
                providers,
                operations=args.operations,
                iterations=args.iterations,
                concurrency=args.concurrency,
            )
            if args.format == "json":
                print(report.to_json())
            elif args.format == "csv":
                print(report.to_csv())
            else:
                print(report.to_markdown())
            return 0
    except (ProviderError, WorldForgeError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
