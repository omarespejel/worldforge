"""Collect safe DimOS probe evidence and write a starter venue_input.json.

This script is intentionally non-actuating. It calls only the safe DimOS MCP
inspection commands already used by live_dimos_bridge.py, then writes an editable
venue input file that can be fed into the trace judge or live bridge.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worldforge import WorldForgeError
from worldforge.models import dump_json

JSON = dict[str, Any]
EXAMPLE_DIR = Path(__file__).resolve().parent
TRACE_APP_PATH = EXAMPLE_DIR / "app.py"
LIVE_BRIDGE_PATH = EXAMPLE_DIR / "live_dimos_bridge.py"
DEFAULT_OUTPUT_DIR = Path(".worldforge/dimos-go2-live-bridge/venue")
DEFAULT_RUN_ID = "dimos-go2-venue-input"


def collect_venue_input(
    *,
    output_dir: Path,
    run_id: str = DEFAULT_RUN_ID,
    goal: str = "inspect the area and move toward the indicated target while avoiding unsafe paths",
    dimos_bin: str = "dimos",
    timeout_seconds: float = 8.0,
    runner: Any | None = None,
) -> JSON:
    """Probe DimOS safely and write editable venue input artifacts."""

    run_id = _require_non_empty(run_id, name="run_id")
    goal = _require_non_empty(goal, name="goal")
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge = _load_module(LIVE_BRIDGE_PATH, "worldforge_dimos_go2_live_bridge_for_collect")
    app = _load_module(TRACE_APP_PATH, "worldforge_dimos_go2_trace_app_for_collect")

    probe_result = bridge.run_probe(
        output_dir=output_dir / "probe",
        run_id=f"{run_id}-probe",
        dimos_bin=dimos_bin,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    probe = _read_json(Path(probe_result["output_dir"]) / "probe.json")
    venue_input = _starter_venue_input(app=app, goal=goal)
    venue_probe = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": "dimos_go2_venue_probe",
        "created_at": _utc_now(),
        "safe_probe_only": True,
        "worldforge_executes_robot": False,
        "dimos_probe": probe,
        "source_commands": [
            "dimos mcp status",
            "dimos mcp list-tools",
            "dimos mcp modules",
        ],
        "source_context": {
            "dimos_readme": (
                "dimos README Agent CLI and MCP section documents mcp list-tools "
                "and mcp call relative_move."
            ),
            "go2_docs": (
                "DimOS Go2 docs describe unitree-go2-agentic with MCP skill access, "
                "Rerun, navigation, costmap, and frontier exploration."
            ),
        },
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": "dimos_go2_collect_venue_input",
        "created_at": _utc_now(),
        "status": "completed",
        "artifact_paths": {
            "venue_probe": "venue_probe.json",
            "venue_input": "venue_input.json",
            "run_manifest": "run_manifest.json",
            "report": "report.md",
            "raw_probe": "probe/probe.json",
        },
        "safe_probe_only": True,
        "worldforge_executes_robot": False,
        "next_steps": [
            (
                "Inspect venue_input.json and adjust observation/candidate feature "
                "values from the venue."
            ),
            "Run live_dimos_bridge.py dry-run-selected --input-json venue_input.json.",
            "Execute only after operator confirmation and the live bridge execution gates.",
        ],
    }

    _write_json(output_dir / "venue_probe.json", venue_probe)
    _write_json(output_dir / "venue_input.json", venue_input)
    _write_json(output_dir / "run_manifest.json", manifest)
    (output_dir / "report.md").write_text(_report(venue_probe=venue_probe), encoding="utf-8")

    return {
        "status": "venue_input_collected",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "artifact_paths": manifest["artifact_paths"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--goal",
        default="inspect the area and move toward the indicated target while avoiding unsafe paths",
    )
    parser.add_argument("--dimos-bin", default="dimos")
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = collect_venue_input(
            output_dir=args.output_dir,
            run_id=args.run_id,
            goal=args.goal,
            dimos_bin=args.dimos_bin,
            timeout_seconds=args.timeout_seconds,
        )
    except WorldForgeError as exc:
        print(json.dumps({"error": {"type": "validation_error", "message": str(exc)}}))
        return 2
    print(dump_json(result))
    return 0


def _starter_venue_input(*, app: Any, goal: str) -> JSON:
    return {
        "schema_version": 1,
        "embodiment": "unitree_go2",
        "host_runtime": "dimos",
        "task": app.sample_task(goal),
        "observation_summary": app.sample_observation_summary(),
        "candidates": app.sample_candidates(),
        "operator_notes": [
            "Replace feature values with current venue observations before live execution.",
            "Keep candidate params inside live_dimos_bridge.py safety limits.",
            "This file is an input to WorldForge scoring, not a robot command log.",
        ],
    }


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise WorldForgeError(f"failed to load module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> JSON:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorldForgeError(f"missing required artifact: {path}.") from exc
    except json.JSONDecodeError as exc:
        raise WorldForgeError(f"artifact must be valid JSON: {path}: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise WorldForgeError(f"artifact must contain a JSON object: {path}.")
    return payload


def _write_json(path: Path, payload: JSON) -> None:
    path.write_text(dump_json(payload) + "\n", encoding="utf-8")


def _report(*, venue_probe: JSON) -> str:
    summary = venue_probe["dimos_probe"]["summary"]
    return (
        "\n".join(
            [
                "# DimOS Go2 Venue Input Collector",
                "",
                f"- run_id: {venue_probe['run_id']}",
                f"- mcp_reachable: {str(summary['mcp_reachable']).lower()}",
                f"- can_list_tools: {str(summary['can_list_tools']).lower()}",
                "- safe_probe_only: true",
                "- worldforge_executes_robot: false",
                "",
                "Generated artifacts:",
                "",
                "- `venue_probe.json`",
                "- `venue_input.json`",
                "- `run_manifest.json`",
                "",
                (
                    "Edit `venue_input.json` with venue observations before any "
                    "live dry run or execution."
                ),
            ]
        )
        + "\n"
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _require_non_empty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorldForgeError(f"{name} must be a non-empty string.")
    return value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
