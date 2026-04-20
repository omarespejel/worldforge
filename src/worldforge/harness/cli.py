"""Command-line entry point for TheWorldHarness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worldforge.harness.flows import available_flows, flow_to_dicts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worldforge-harness",
        description="Launch TheWorldHarness visual E2E integration harness.",
    )
    parser.add_argument(
        "--flow",
        choices=[flow.id for flow in available_flows()],
        default="leworldmodel",
        help="Harness flow to open.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Directory for persisted demo worlds. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available harness flows without launching the TUI.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for --list.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Disable step reveal delays.",
    )
    return parser


def print_flow_index(*, output_format: str = "markdown") -> None:
    """Print available harness flows."""

    if output_format == "json":
        print(json.dumps(flow_to_dicts(), indent=2))
        return

    print("# TheWorldHarness Flows")
    print()
    print("| Flow | Focus | Provider | Command |")
    print("| --- | --- | --- | --- |")
    for flow in available_flows():
        print(f"| `{flow.id}` | {flow.focus} | {flow.provider} | `{flow.command}` |")


def launch_harness(
    *,
    flow_id: str = "leworldmodel",
    state_dir: Path | None = None,
    animate: bool = True,
) -> int:
    """Launch the Textual harness, returning a process exit code."""

    try:
        from worldforge.harness.tui import TheWorldHarnessApp
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("textual"):
            print(
                "TheWorldHarness requires the optional Textual dependency. "
                "Install with `uv run --extra harness worldforge-harness` "
                "or `pip install 'worldforge[harness]'`.",
                file=sys.stderr,
            )
            return 2
        raise

    app = TheWorldHarnessApp(
        initial_flow_id=flow_id,
        state_dir=state_dir,
        step_delay=0.0 if not animate else 0.18,
    )
    app.run()
    return 0


def run_from_args(
    *,
    flow_id: str,
    state_dir: Path | None,
    list_only: bool,
    output_format: str,
    animate: bool,
) -> int:
    """Run harness behavior from an already-parsed command namespace."""

    if list_only:
        print_flow_index(output_format=output_format)
        return 0
    return launch_harness(flow_id=flow_id, state_dir=state_dir, animate=animate)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_from_args(
        flow_id=args.flow,
        state_dir=args.state_dir,
        list_only=args.list,
        output_format=args.format,
        animate=not args.no_animation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
