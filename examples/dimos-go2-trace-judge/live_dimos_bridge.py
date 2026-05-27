"""Host-owned DimOS CLI bridge for the Go2 trace-judge example.

This file is intentionally fork/hackathon-oriented. It does not import DimOS,
open robot sockets, or execute motion commands by default. It shells out to the
documented `dimos mcp` CLI only when asked, captures sanitized evidence, and
requires explicit operator gates before any selected action is executed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worldforge import WorldForgeError
from worldforge.models import _redact_observable_text, dump_json

JSON = dict[str, Any]
TRACE_APP_PATH = Path(__file__).with_name("app.py")
DEFAULT_OUTPUT_DIR = Path(".worldforge/dimos-go2-live-bridge")
DEFAULT_TIMEOUT_SECONDS = 8.0
EXECUTE_ENV_VAR = "WORLDFORGE_DIMOS_ENABLE_EXECUTE"
EXECUTE_CONFIRMATION = "LIVE_DIMOS_GO2_EXECUTE"
MAX_CAPTURE_CHARS = 12_000
SAFE_PARAM_LIMITS: dict[str, dict[str, tuple[float, float]]] = {
    "relative_move": {
        "forward": (-0.5, 0.5),
        "left": (-0.4, 0.4),
        "degrees": (-45.0, 45.0),
    },
    "wait": {"seconds": (0.0, 3.0)},
}
SAFE_PROBE_COMMANDS = (
    ("status", ("mcp", "status")),
    ("list_tools", ("mcp", "list-tools")),
    ("modules", ("mcp", "modules")),
)
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None

    def to_dict(self) -> JSON:
        return {
            "argv": self.argv,
            "exit_code": self.exit_code,
            "stdout": _sanitize_capture(self.stdout),
            "stderr": _sanitize_capture(self.stderr),
            "timed_out": self.timed_out,
            "error": _sanitize_capture(self.error) if self.error else None,
        }


Runner = Callable[[list[str], float], CommandResult]


def run_probe(
    *,
    output_dir: Path,
    run_id: str,
    dimos_bin: str = "dimos",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> JSON:
    """Run safe DimOS MCP inspection commands and write probe evidence."""

    run_id = _require_non_empty(run_id, name="run_id")
    output_dir = _prepare_output_dir(output_dir)
    runner = runner or _run_subprocess
    timeout_seconds = _require_positive_timeout(timeout_seconds)

    command_results: dict[str, JSON] = {}
    for label, args in SAFE_PROBE_COMMANDS:
        command_results[label] = runner([dimos_bin, *args], timeout_seconds).to_dict()

    probe = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": "dimos_go2_live_probe",
        "created_at": _utc_now(),
        "dimos_bin": dimos_bin,
        "timeout_seconds": timeout_seconds,
        "commands": command_results,
        "safe_probe_only": True,
        "worldforge_executes_robot": False,
        "summary": _probe_summary(command_results),
    }
    manifest = _bridge_manifest(
        run_id=run_id,
        operation="probe",
        output_dir=output_dir,
        probe=probe,
        selected_command=None,
        execution_result=None,
    )

    _write_json(output_dir / "probe.json", probe)
    _write_json(output_dir / "bridge_manifest.json", manifest)
    (output_dir / "report.md").write_text(_probe_report(probe), encoding="utf-8")
    return _result_payload(output_dir=output_dir, run_id=run_id, status="probe_completed")


def dry_run_selected(
    *,
    output_dir: Path,
    run_id: str,
    goal: str,
    dimos_bin: str = "dimos",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    with_probe: bool = False,
    runner: Runner | None = None,
) -> JSON:
    """Run the trace judge and write the selected DimOS MCP command without executing it."""

    output_dir = _prepare_output_dir(output_dir)
    run_id = _require_non_empty(run_id, name="run_id")
    trace_result = _run_trace_judge(output_dir=output_dir / "trace_judge", run_id=run_id, goal=goal)
    selected_action = _read_json(output_dir / "trace_judge" / "selected_action.json")
    selected_command = build_selected_mcp_command(
        selected_action=selected_action,
        dimos_bin=dimos_bin,
        will_execute=False,
    )
    probe = None
    if with_probe:
        probe_result = run_probe(
            output_dir=output_dir / "probe",
            run_id=f"{run_id}-probe",
            dimos_bin=dimos_bin,
            timeout_seconds=timeout_seconds,
            runner=runner,
        )
        probe = _read_json(Path(probe_result["output_dir"]) / "probe.json")

    manifest = _bridge_manifest(
        run_id=run_id,
        operation="dry_run_selected",
        output_dir=output_dir,
        probe=probe,
        selected_command=selected_command,
        execution_result=None,
    )
    _write_json(output_dir / "selected_mcp_command.json", selected_command)
    _write_json(output_dir / "bridge_manifest.json", manifest)
    (output_dir / "report.md").write_text(
        _dry_run_report(
            trace_result=trace_result,
            selected_command=selected_command,
            probe=probe,
        ),
        encoding="utf-8",
    )
    return _result_payload(output_dir=output_dir, run_id=run_id, status="dry_run_completed")


def execute_selected(
    *,
    output_dir: Path,
    run_id: str,
    goal: str,
    confirm: str,
    dimos_bin: str = "dimos",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    with_probe: bool = True,
    runner: Runner | None = None,
    environ: dict[str, str] | None = None,
) -> JSON:
    """Execute the selected action through `dimos mcp call` after explicit gates."""

    _require_execution_gates(confirm=confirm, environ=environ or os.environ)
    output_dir = _prepare_output_dir(output_dir)
    runner = runner or _run_subprocess
    dry_run_selected(
        output_dir=output_dir,
        run_id=run_id,
        goal=goal,
        dimos_bin=dimos_bin,
        timeout_seconds=timeout_seconds,
        with_probe=with_probe,
        runner=runner,
    )
    selected_action = _read_json(output_dir / "trace_judge" / "selected_action.json")
    selected_command = build_selected_mcp_command(
        selected_action=selected_action,
        dimos_bin=dimos_bin,
        will_execute=True,
    )
    selected_command["will_execute"] = True
    selected_command["operator_confirmed"] = True
    execution = runner(selected_command["argv"], _require_positive_timeout(timeout_seconds))
    execution_result = {
        "schema_version": 1,
        "run_id": run_id,
        "executed_at": _utc_now(),
        "command": selected_command,
        "result": execution.to_dict(),
        "worldforge_executes_robot": False,
        "executed_by": "host_runtime:dimos-cli",
    }
    probe = None
    probe_path = output_dir / "probe" / "probe.json"
    if probe_path.exists():
        probe = _read_json(probe_path)

    manifest = _bridge_manifest(
        run_id=run_id,
        operation="execute_selected",
        output_dir=output_dir,
        probe=probe,
        selected_command=selected_command,
        execution_result=execution_result,
    )
    _write_json(output_dir / "selected_mcp_command.json", selected_command)
    _write_json(output_dir / "execution_result.json", execution_result)
    _write_json(output_dir / "bridge_manifest.json", manifest)
    (output_dir / "report.md").write_text(
        _execution_report(selected_command=selected_command, execution_result=execution_result),
        encoding="utf-8",
    )
    status = "execute_completed" if execution.exit_code == 0 else "execute_failed"
    return _result_payload(output_dir=output_dir, run_id=run_id, status=status)


def build_selected_mcp_command(
    *,
    selected_action: JSON,
    dimos_bin: str = "dimos",
    will_execute: bool = False,
) -> JSON:
    candidate_id = _require_non_empty(
        selected_action.get("selected_candidate_id"),
        name="selected_action.selected_candidate_id",
    )
    action = _require_non_empty(selected_action.get("action"), name="selected_action.action")
    params = selected_action.get("params")
    if not isinstance(params, dict):
        raise WorldForgeError("selected_action.params must be an object.")
    dimos_bin = _require_non_empty(dimos_bin, name="dimos_bin")
    safe_params = _validate_safe_action_params(action=action, params=params)
    argv = [dimos_bin, "mcp", "call", action, "--json-args", dump_json(safe_params)]
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "action": action,
        "params": safe_params,
        "argv": argv,
        "will_execute": will_execute,
        "operator_confirmed": False,
        "execution_gates": {
            "required_env_var": EXECUTE_ENV_VAR,
            "required_env_value": "1",
            "required_confirmation": EXECUTE_CONFIRMATION,
        },
        "safety_limits": SAFE_PARAM_LIMITS[action],
        "worldforge_executes_robot": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimos-bin", default=os.environ.get("DIMOS_BIN", "dimos"))
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="Inspect the local DimOS MCP server safely.")
    _add_common_options(probe)
    probe.add_argument("--run-id", default="dimos-go2-live-probe")

    dry_run = subparsers.add_parser(
        "dry-run-selected",
        help="Score candidates and render the selected DimOS MCP command without executing it.",
    )
    _add_common_options(dry_run)
    dry_run.add_argument("--run-id", default="dimos-go2-live-dry-run")
    dry_run.add_argument(
        "--goal",
        default="inspect the area and move toward the indicated target",
    )
    dry_run.add_argument("--with-probe", action="store_true")

    execute = subparsers.add_parser(
        "execute-selected",
        help="Execute the selected DimOS MCP command after explicit operator gates.",
    )
    _add_common_options(execute)
    execute.add_argument("--run-id", default="dimos-go2-live-execute")
    execute.add_argument(
        "--goal",
        default="inspect the area and move toward the indicated target",
    )
    execute.add_argument("--confirm", default="")
    execute.add_argument("--skip-probe", action="store_true")
    return parser


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dimos-bin", default=argparse.SUPPRESS)
    parser.add_argument("--timeout-seconds", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=argparse.SUPPRESS)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            result = run_probe(
                output_dir=args.output_dir,
                run_id=args.run_id,
                dimos_bin=args.dimos_bin,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command == "dry-run-selected":
            result = dry_run_selected(
                output_dir=args.output_dir,
                run_id=args.run_id,
                goal=args.goal,
                dimos_bin=args.dimos_bin,
                timeout_seconds=args.timeout_seconds,
                with_probe=args.with_probe,
            )
        elif args.command == "execute-selected":
            result = execute_selected(
                output_dir=args.output_dir,
                run_id=args.run_id,
                goal=args.goal,
                confirm=args.confirm,
                dimos_bin=args.dimos_bin,
                timeout_seconds=args.timeout_seconds,
                with_probe=not args.skip_probe,
            )
        else:  # pragma: no cover - argparse enforces command choices.
            raise WorldForgeError(f"unknown command: {args.command}")
    except WorldForgeError as exc:
        print(json.dumps({"error": {"type": "validation_error", "message": str(exc)}}))
        return 2
    print(dump_json(result))
    return 0


def _run_subprocess(argv: list[str], timeout_seconds: float) -> CommandResult:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            argv=argv,
            exit_code=None,
            stdout="",
            stderr="",
            error=f"command not found: {exc.filename}",
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            argv=argv,
            exit_code=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
            error=f"timed out after {timeout_seconds:.1f}s",
        )
    return CommandResult(
        argv=argv,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _validate_safe_action_params(*, action: str, params: JSON) -> JSON:
    if action not in SAFE_PARAM_LIMITS:
        raise WorldForgeError(f"unsupported live DimOS action: {action}.")
    for name in params:
        if not isinstance(name, str):
            raise WorldForgeError(f"{action} params keys must be strings.")
    limits = SAFE_PARAM_LIMITS[action]
    unexpected = sorted(set(params) - set(limits))
    if unexpected:
        raise WorldForgeError(f"unsupported {action} params: {', '.join(unexpected)}.")
    safe_params: JSON = {}
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise WorldForgeError(f"{action}.{name} must be numeric.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise WorldForgeError(f"{action}.{name} must be finite.")
        lower, upper = limits[name]
        if numeric < lower or numeric > upper:
            raise WorldForgeError(
                f"{action}.{name}={numeric:g} outside safe range [{lower:g}, {upper:g}]."
            )
        safe_params[name] = numeric
    if not safe_params:
        raise WorldForgeError(f"{action} requires at least one safe parameter.")
    return safe_params


def _require_execution_gates(*, confirm: str, environ: dict[str, str]) -> None:
    if environ.get(EXECUTE_ENV_VAR) != "1":
        raise WorldForgeError(f"set {EXECUTE_ENV_VAR}=1 to allow live DimOS execution.")
    if confirm != EXECUTE_CONFIRMATION:
        raise WorldForgeError(
            f"pass --confirm {EXECUTE_CONFIRMATION} to execute live DimOS action."
        )


def _probe_summary(command_results: dict[str, JSON]) -> JSON:
    successful = [name for name, result in command_results.items() if result["exit_code"] == 0]
    failed = [name for name, result in command_results.items() if result["exit_code"] != 0]
    return {
        "commands_successful": successful,
        "commands_failed": failed,
        "mcp_reachable": "status" in successful or "list_tools" in successful,
        "can_list_tools": "list_tools" in successful,
        "safe_for_dry_run": True,
        "safe_for_execute": False,
    }


def _bridge_manifest(
    *,
    run_id: str,
    operation: str,
    output_dir: Path,
    probe: JSON | None,
    selected_command: JSON | None,
    execution_result: JSON | None,
) -> JSON:
    artifact_paths = {
        "bridge_manifest": "bridge_manifest.json",
        "report": "report.md",
    }
    if probe is not None:
        artifact_paths["probe"] = (
            "probe/probe.json" if (output_dir / "probe").exists() else "probe.json"
        )
    if selected_command is not None:
        artifact_paths["selected_mcp_command"] = "selected_mcp_command.json"
        artifact_paths["trace_judge"] = "trace_judge/run_manifest.json"
    if execution_result is not None:
        artifact_paths["execution_result"] = "execution_result.json"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "kind": "dimos_go2_live_bridge",
        "operation": operation,
        "created_at": _utc_now(),
        "artifact_paths": artifact_paths,
        "probe_summary": probe.get("summary") if probe else None,
        "selected_candidate_id": selected_command.get("candidate_id") if selected_command else None,
        "execution_status": _execution_status(execution_result),
        "safety_boundary": {
            "host_owns_robot_networking": True,
            "host_owns_operator_supervision": True,
            "host_owns_emergency_stop": True,
            "worldforge_scores_candidates": True,
            "worldforge_executes_robot": False,
            "dimos_cli_executes_robot_when_gated": execution_result is not None,
        },
        "execution_gates": {
            "env_var": EXECUTE_ENV_VAR,
            "confirmation": EXECUTE_CONFIRMATION,
        },
        "experimental": True,
        "upstream_ready": False,
    }


def _execution_status(execution_result: JSON | None) -> str:
    if execution_result is None:
        return "not_executed"
    result = execution_result.get("result")
    if not isinstance(result, dict):
        return "unknown"
    return "succeeded" if result.get("exit_code") == 0 else "failed"


def _run_trace_judge(*, output_dir: Path, run_id: str, goal: str) -> JSON:
    app = _load_trace_app()
    return app.run_trace_judge(output_dir=output_dir, run_id=run_id, goal=goal)


def _load_trace_app() -> Any:
    spec = importlib.util.spec_from_file_location(
        "worldforge_dimos_go2_trace_judge_example_app",
        TRACE_APP_PATH,
    )
    if spec is None:
        raise WorldForgeError(f"failed to load trace judge module spec for {TRACE_APP_PATH}.")
    if spec.loader is None:
        raise WorldForgeError(f"trace judge module spec has no loader for {TRACE_APP_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _probe_report(probe: JSON) -> str:
    lines = [
        "# DimOS Go2 Live Probe",
        "",
        f"- run_id: {probe['run_id']}",
        f"- mcp_reachable: {str(probe['summary']['mcp_reachable']).lower()}",
        f"- can_list_tools: {str(probe['summary']['can_list_tools']).lower()}",
        "- safe_probe_only: true",
        "- worldforge_executes_robot: false",
        "",
        "| Command | Exit | Timed out |",
        "| --- | ---: | --- |",
    ]
    for name, result in probe["commands"].items():
        lines.append(f"| `{name}` | {result['exit_code']} | {str(result['timed_out']).lower()} |")
    return "\n".join(lines) + "\n"


def _dry_run_report(*, trace_result: JSON, selected_command: JSON, probe: JSON | None) -> str:
    lines = [
        "# DimOS Go2 Live Bridge Dry Run",
        "",
        f"- run_id: {trace_result['run_id']}",
        f"- selected_candidate_id: {trace_result['selected_candidate_id']}",
        f"- action: {selected_command['action']}",
        f"- params: `{dump_json(selected_command['params'])}`",
        "- will_execute: false",
        "- worldforge_executes_robot: false",
        "",
        "## Command",
        "",
        f"`{shlex.join(selected_command['argv'])}`",
    ]
    if probe is not None:
        lines.extend(
            [
                "",
                "## Probe",
                "",
                f"- mcp_reachable: {str(probe['summary']['mcp_reachable']).lower()}",
                f"- can_list_tools: {str(probe['summary']['can_list_tools']).lower()}",
            ]
        )
    return "\n".join(lines) + "\n"


def _execution_report(*, selected_command: JSON, execution_result: JSON) -> str:
    result = execution_result["result"]
    lines = [
        "# DimOS Go2 Live Bridge Execution",
        "",
        f"- action: {selected_command['action']}",
        f"- params: `{dump_json(selected_command['params'])}`",
        f"- exit_code: {result['exit_code']}",
        f"- timed_out: {str(result['timed_out']).lower()}",
        "- worldforge_executes_robot: false",
    ]
    return "\n".join(lines) + "\n"


def _sanitize_capture(value: str | bytes) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    redacted = _redact_observable_text(text)
    redacted = IPV4_PATTERN.sub("<redacted-ipv4>", redacted)
    return redacted[:MAX_CAPTURE_CHARS]


def _result_payload(*, output_dir: Path, run_id: str, status: str) -> JSON:
    return {
        "status": status,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "artifact_paths": {
            "bridge_manifest": "bridge_manifest.json",
            "report": "report.md",
        },
    }


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _require_positive_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WorldForgeError("timeout_seconds must be numeric.")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise WorldForgeError("timeout_seconds must be finite and positive.")
    return timeout


def _require_non_empty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorldForgeError(f"{name} must be a non-empty string.")
    return value.strip()


def _read_json(path: Path) -> JSON:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorldForgeError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorldForgeError(f"{path} must contain valid JSON.") from exc
    if not isinstance(payload, dict):
        raise WorldForgeError(f"{path} must contain a JSON object.")
    return payload


def _write_json(path: Path, payload: JSON) -> None:
    path.write_text(dump_json(payload) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
