"""CLI for WorldForge provider and evaluation workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from worldforge import Action, GenerationOptions, VideoClip, WorldForge, WorldForgeError
from worldforge.benchmark import ProviderBenchmarkHarness, load_benchmark_budgets
from worldforge.evaluation import EvaluationSuite
from worldforge.models import CAPABILITY_NAMES
from worldforge.providers import ProviderError
from worldforge.providers.catalog import provider_docs_index

CLI_DESCRIPTION = (
    "CLI for WorldForge provider diagnostics, prediction, generation, evaluation, "
    "benchmarking, and runnable demos."
)

CLI_EPILOG = """Common commands:
  worldforge examples
  worldforge doctor
  worldforge world create lab --provider mock
  worldforge world list
  worldforge world history <world-id>
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


def _world_summary(world) -> dict[str, object]:
    return {
        "id": world.id,
        "name": world.name,
        "provider": world.provider,
        "description": world.description,
        "step": world.step,
        "object_count": world.object_count,
        "history_length": world.history_length,
    }


def _world_history_payload(world) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for entry in world.history():
        action = json.loads(entry.action_json) if entry.action_json is not None else None
        entries.append(
            {
                "step": entry.step,
                "summary": entry.summary,
                "action": action,
                "object_count": len(entry.state.get("scene", {}).get("objects", {})),
            }
        )
    return entries


def _print_world_list_markdown(worlds: list[dict[str, object]]) -> None:
    print("# WorldForge Worlds")
    print()
    print("| id | name | provider | step | objects | history |")
    print("| --- | --- | --- | ---: | ---: | ---: |")
    for world in worlds:
        print(
            "| "
            f"`{world['id']}` | "
            f"{world['name']} | "
            f"`{world['provider']}` | "
            f"{world['step']} | "
            f"{world['object_count']} | "
            f"{world['history_length']} |"
        )


def _print_world_summary_markdown(world) -> None:
    print(f"# World {world.id}")
    print()
    print(f"- name: {world.name}")
    print(f"- provider: {world.provider}")
    print(f"- step: {world.step}")
    print(f"- objects: {world.object_count}")
    print(f"- history: {world.history_length}")
    if world.description:
        print(f"- description: {world.description}")


def _print_world_history_markdown(world, entries: list[dict[str, object]]) -> None:
    print(f"# World History: {world.id}")
    print()
    print("| step | summary | action | objects |")
    print("| ---: | --- | --- | ---: |")
    for entry in entries:
        action = entry["action"]
        action_label = ""
        if isinstance(action, dict):
            action_label = str(action.get("type", ""))
        print(
            f"| {entry['step']} | {entry['summary']} | {action_label} | {entry['object_count']} |"
        )


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
    provider_list.add_argument(
        "--capability",
        choices=CAPABILITY_NAMES,
        help="Filter providers by capability name.",
    )

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
    provider_health.add_argument(
        "--capability",
        choices=CAPABILITY_NAMES,
        help="Filter providers by capability name.",
    )

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

    world = subparsers.add_parser("world", help="Manage persisted local JSON worlds.")
    world_subparsers = world.add_subparsers(dest="world_command", required=True)

    world_list = world_subparsers.add_parser("list", help="List persisted worlds.")
    world_list.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )
    world_list.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for persisted world summaries.",
    )

    world_create = world_subparsers.add_parser("create", help="Create and save a world.")
    world_create.add_argument("name", help="World name.")
    world_create.add_argument("--provider", default="mock", help="Provider name.")
    world_create.add_argument(
        "--prompt",
        help="Optional prompt used to seed the world with deterministic checkout-safe objects.",
    )
    world_create.add_argument("--description", default="", help="Optional world description.")
    world_create.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )
    world_create.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for the saved world summary.",
    )

    world_show = world_subparsers.add_parser("show", help="Show a persisted world.")
    world_show.add_argument("world_id", help="World identifier.")
    world_show.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )
    world_show.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for the world.",
    )

    world_history = world_subparsers.add_parser("history", help="Show persisted world history.")
    world_history.add_argument("world_id", help="World identifier.")
    world_history.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )
    world_history.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for history entries.",
    )

    world_export = world_subparsers.add_parser("export", help="Export a persisted world as JSON.")
    world_export.add_argument("world_id", help="World identifier.")
    world_export.add_argument("--output", help="Optional output path for exported JSON.")
    world_export.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )

    world_import = world_subparsers.add_parser(
        "import", help="Import and save exported world JSON."
    )
    world_import.add_argument("input", help="Path to exported world JSON.")
    world_import.add_argument("--new-id", action="store_true", help="Assign a fresh world id.")
    world_import.add_argument("--name", help="Optional replacement world name.")
    world_import.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )
    world_import.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for the imported world summary.",
    )

    world_fork = world_subparsers.add_parser("fork", help="Fork a world from a history entry.")
    world_fork.add_argument("world_id", help="Source world identifier.")
    world_fork.add_argument(
        "--history-index",
        type=int,
        default=0,
        help="History entry index to fork from.",
    )
    world_fork.add_argument("--name", help="Optional forked world name.")
    world_fork.add_argument(
        "--state-dir", default=".worldforge/worlds", help="World state directory."
    )
    world_fork.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for the forked world summary.",
    )

    doctor = subparsers.add_parser("doctor", help="Inspect the local WorldForge environment.")
    doctor.add_argument("--state-dir", default=".worldforge/worlds", help="World state directory.")
    doctor.add_argument(
        "--registered-only",
        action="store_true",
        help="Show only providers registered for this process.",
    )
    doctor.add_argument(
        "--capability",
        choices=CAPABILITY_NAMES,
        help="Filter providers by capability name.",
    )

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
        "--budget-file",
        type=Path,
        help=("Optional JSON budget file. Failing gates exit non-zero after printing the report."),
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


def _cmd_examples(args: argparse.Namespace) -> int:
    if args.format == "json":
        _print_json(EXAMPLE_COMMANDS)
    else:
        _print_examples_markdown()
    return 0


def _cmd_provider_docs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    entries = _provider_docs_entries(args.name)
    if not entries:
        parser.exit(2, f"Unknown provider: {args.name}\n")
    if args.format == "json":
        _print_json(entries)
    else:
        _print_provider_docs_markdown(entries)
    return 0


def _cmd_harness(args: argparse.Namespace) -> int:
    from worldforge.harness.cli import run_from_args

    return run_from_args(
        flow_id=args.flow,
        state_dir=args.state_dir,
        list_only=args.list,
        output_format=args.format,
        animate=not args.no_animation,
    )


def _cmd_providers(args: argparse.Namespace, forge: WorldForge) -> int:
    _print_json([info.to_dict() for info in forge.list_providers()])
    return 0


def _cmd_provider_list(args: argparse.Namespace, forge: WorldForge) -> int:
    report = forge.doctor(
        capability=args.capability,
        registered_only=args.registered_only,
    )
    _print_json([provider.to_dict() for provider in report.providers])
    return 0


def _cmd_provider_info(args: argparse.Namespace, forge: WorldForge) -> int:
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


def _cmd_provider_health(args: argparse.Namespace, forge: WorldForge) -> int:
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


def _cmd_provider(args: argparse.Namespace, forge: WorldForge) -> int | None:
    provider_dispatch = {
        "list": _cmd_provider_list,
        "info": _cmd_provider_info,
        "health": _cmd_provider_health,
    }
    handler = provider_dispatch.get(args.provider_command)
    if handler is None:
        return None
    return handler(args, forge)


def _cmd_world_list(args: argparse.Namespace, forge: WorldForge) -> int:
    worlds = [_world_summary(forge.load_world(world_id)) for world_id in forge.list_worlds()]
    if args.format == "markdown":
        _print_world_list_markdown(worlds)
    else:
        _print_json(worlds)
    return 0


def _cmd_world_create(args: argparse.Namespace, forge: WorldForge) -> int:
    if args.prompt:
        world = forge.create_world_from_prompt(args.prompt, provider=args.provider, name=args.name)
        if args.description:
            world.description = args.description
    else:
        world = forge.create_world(args.name, provider=args.provider, description=args.description)
    forge.save_world(world)
    if args.format == "markdown":
        _print_world_summary_markdown(world)
    else:
        _print_json(_world_summary(world))
    return 0


def _cmd_world_show(args: argparse.Namespace, forge: WorldForge) -> int:
    world = forge.load_world(args.world_id)
    if args.format == "markdown":
        _print_world_summary_markdown(world)
    else:
        _print_json(world.to_dict())
    return 0


def _cmd_world_history(args: argparse.Namespace, forge: WorldForge) -> int:
    world = forge.load_world(args.world_id)
    entries = _world_history_payload(world)
    if args.format == "markdown":
        _print_world_history_markdown(world, entries)
    else:
        _print_json({"world_id": world.id, "history": entries})
    return 0


def _cmd_world_export(args: argparse.Namespace, forge: WorldForge) -> int:
    payload = forge.export_world(args.world_id)
    if args.output:
        target = Path(args.output).expanduser().resolve()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"{payload}\n", encoding="utf-8")
        except OSError as exc:
            raise WorldForgeError(f"Failed to write exported world to {target}: {exc}") from exc
        _print_json({"world_id": args.world_id, "output_path": str(target)})
    else:
        _print_json(json.loads(payload))
    return 0


def _cmd_world_import(args: argparse.Namespace, forge: WorldForge) -> int:
    source = Path(args.input).expanduser().resolve()
    try:
        payload = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorldForgeError(f"Failed to read imported world from {source}: {exc}") from exc
    world = forge.import_world(payload, new_id=args.new_id, name=args.name)
    forge.save_world(world)
    if args.format == "markdown":
        _print_world_summary_markdown(world)
    else:
        summary = _world_summary(world)
        summary["source_path"] = str(source)
        _print_json(summary)
    return 0


def _cmd_world_fork(args: argparse.Namespace, forge: WorldForge) -> int:
    world = forge.fork_world(
        args.world_id,
        history_index=args.history_index,
        name=args.name,
    )
    forge.save_world(world)
    if args.format == "markdown":
        _print_world_summary_markdown(world)
    else:
        summary = _world_summary(world)
        summary["source_world_id"] = args.world_id
        summary["history_index"] = args.history_index
        _print_json(summary)
    return 0


def _cmd_world(args: argparse.Namespace, forge: WorldForge) -> int | None:
    world_dispatch = {
        "list": _cmd_world_list,
        "create": _cmd_world_create,
        "show": _cmd_world_show,
        "history": _cmd_world_history,
        "export": _cmd_world_export,
        "import": _cmd_world_import,
        "fork": _cmd_world_fork,
    }
    handler = world_dispatch.get(args.world_command)
    if handler is None:
        return None
    return handler(args, forge)


def _cmd_doctor(args: argparse.Namespace, forge: WorldForge) -> int:
    _print_json(
        forge.doctor(
            capability=args.capability,
            registered_only=args.registered_only,
        ).to_dict()
    )
    return 0


def _cmd_generate(args: argparse.Namespace, forge: WorldForge) -> int:
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


def _cmd_transfer(args: argparse.Namespace, forge: WorldForge) -> int:
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


def _cmd_predict(args: argparse.Namespace, forge: WorldForge) -> int:
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


def _cmd_eval(args: argparse.Namespace, forge: WorldForge) -> int:
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


def _cmd_benchmark(args: argparse.Namespace, forge: WorldForge) -> int:
    harness = ProviderBenchmarkHarness(forge=forge)
    providers = args.providers or ["mock"]
    report = harness.run(
        providers,
        operations=args.operations,
        iterations=args.iterations,
        concurrency=args.concurrency,
    )
    gate_report = None
    if args.budget_file:
        try:
            budget_payload = json.loads(args.budget_file.expanduser().read_text(encoding="utf-8"))
        except OSError as exc:
            raise WorldForgeError(
                f"Failed to read benchmark budget file {args.budget_file}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise WorldForgeError(f"Benchmark budget file must contain valid JSON: {exc}") from exc
        gate_report = report.evaluate_budgets(load_benchmark_budgets(budget_payload))

    if args.format == "json":
        if gate_report is None:
            print(report.to_json())
        else:
            print(
                json.dumps(
                    {
                        "benchmark": report.to_dict(),
                        "gate": gate_report.to_dict(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    elif args.format == "csv":
        print(gate_report.to_csv() if gate_report is not None else report.to_csv())
    else:
        if gate_report is None:
            print(report.to_markdown())
        else:
            print(report.to_markdown())
            print()
            print(gate_report.to_markdown())
    return 0 if gate_report is None or gate_report.passed else 1


_ForgeHandler = Callable[[argparse.Namespace, WorldForge], "int | None"]

_FORGE_COMMANDS: dict[str, _ForgeHandler] = {
    "providers": _cmd_providers,
    "provider": _cmd_provider,
    "world": _cmd_world,
    "doctor": _cmd_doctor,
    "generate": _cmd_generate,
    "transfer": _cmd_transfer,
    "predict": _cmd_predict,
    "eval": _cmd_eval,
    "benchmark": _cmd_benchmark,
}


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "examples":
        return _cmd_examples(args)

    if args.command == "provider" and args.provider_command == "docs":
        return _cmd_provider_docs(args, parser)

    if args.command == "harness":
        return _cmd_harness(args)

    forge = WorldForge(state_dir=args.state_dir)

    handler = _FORGE_COMMANDS.get(args.command)
    if handler is not None:
        try:
            result = handler(args, forge)
        except (ProviderError, WorldForgeError, ValueError) as exc:
            parser.exit(2, f"{exc}\n")
        if result is not None:
            return result

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
