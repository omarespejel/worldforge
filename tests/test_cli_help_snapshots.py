from __future__ import annotations

import sys
from textwrap import dedent

import pytest

from worldforge.cli import main

HELP_SNAPSHOTS: dict[tuple[str, ...], str] = {
    ("examples", "--help"): """\
usage: worldforge examples [-h] [--format {markdown,json}]

options:
  -h, --help            show this help message and exit
  --format {markdown,json}
                        Output format for the examples index.
""",
    ("provider", "list", "--help"): (
        "usage: worldforge provider list [-h] [--state-dir STATE_DIR]\n"
        "                                [--registered-only]\n"
        "                                [--capability "
        "{predict,generate,reason,embed,plan,transfer,score,policy}]\n"
        "\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --state-dir STATE_DIR\n"
        "                        World state directory.\n"
        "  --registered-only     Show only providers registered for this process.\n"
        "  --capability {predict,generate,reason,embed,plan,transfer,score,policy}\n"
        "                        Filter providers by capability name.\n"
    ),
    ("provider", "docs", "--help"): """\
usage: worldforge provider docs [-h] [--format {markdown,json}] [name]

positional arguments:
  name                  Optional provider name.

options:
  -h, --help            show this help message and exit
  --format {markdown,json}
                        Output format for provider docs metadata.
""",
    ("harness", "--help"): """\
usage: worldforge harness [-h] [--flow {leworldmodel,lerobot,diagnostics}]
                          [--state-dir STATE_DIR] [--list]
                          [--format {markdown,json}] [--no-animation]

options:
  -h, --help            show this help message and exit
  --flow {leworldmodel,lerobot,diagnostics}
                        Harness flow to open.
  --state-dir STATE_DIR
                        Directory for persisted demo worlds. Defaults to a
                        temporary directory.
  --list                List available harness flows without launching the
                        TUI.
  --format {markdown,json}
                        Output format for --list.
  --no-animation        Disable step reveal delays.
""",
    ("predict", "--help"): """\
usage: worldforge predict [-h] [--provider PROVIDER] --x X --y Y --z Z
                          [--steps STEPS] [--state-dir STATE_DIR]
                          world_name

positional arguments:
  world_name            World name to create or load.

options:
  -h, --help            show this help message and exit
  --provider PROVIDER   Provider name.
  --x X                 Target x coordinate.
  --y Y                 Target y coordinate.
  --z Z                 Target z coordinate.
  --steps STEPS         Prediction horizon in steps.
  --state-dir STATE_DIR
                        World state directory.
""",
    ("eval", "--help"): """\
usage: worldforge eval [-h]
                       [--suite {generation,physics,planning,reasoning,transfer}]
                       [--provider PROVIDERS] [--format {markdown,json,csv}]
                       [--state-dir STATE_DIR]

options:
  -h, --help            show this help message and exit
  --suite {generation,physics,planning,reasoning,transfer}
                        Built-in evaluation suite.
  --provider PROVIDERS  Provider name to evaluate. Can be repeated.
  --format {markdown,json,csv}
                        Evaluation report format.
  --state-dir STATE_DIR
                        World state directory.
""",
    ("benchmark", "--help"): """\
usage: worldforge benchmark [-h] [--provider PROVIDERS]
                            [--operation {predict,reason,generate,transfer}]
                            [--iterations ITERATIONS]
                            [--concurrency CONCURRENCY]
                            [--format {markdown,json,csv}]
                            [--state-dir STATE_DIR]

options:
  -h, --help            show this help message and exit
  --provider PROVIDERS  Provider name to benchmark. Can be repeated.
  --operation {predict,reason,generate,transfer}
                        Operation to benchmark. Can be repeated.
  --iterations ITERATIONS
                        Iterations per operation.
  --concurrency CONCURRENCY
                        Concurrent workers.
  --format {markdown,json,csv}
                        Benchmark report format.
  --state-dir STATE_DIR
                        World state directory.
""",
}


def _help_output(argv: tuple[str, ...], monkeypatch, capsys) -> str:
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(sys, "argv", ["worldforge", *argv])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 0
    return capsys.readouterr().out


def test_top_level_help_lists_command_surface(monkeypatch, capsys) -> None:
    output = _help_output(("--help",), monkeypatch, capsys)

    assert output.startswith("usage: worldforge [-h] command ...")
    assert "Typed local-first CLI for provider diagnostics" in output
    for command in (
        "examples",
        "providers",
        "provider",
        "doctor",
        "generate",
        "transfer",
        "predict",
        "eval",
        "benchmark",
        "harness",
    ):
        assert command in output
    for common_command in (
        "worldforge examples",
        "worldforge provider list",
        "worldforge provider docs",
        "worldforge provider info mock",
        "worldforge harness --list",
        "worldforge eval --suite planning --provider mock --format json",
    ):
        assert common_command in output


@pytest.mark.parametrize("argv", HELP_SNAPSHOTS)
def test_cli_help_snapshots(argv: tuple[str, ...], monkeypatch, capsys) -> None:
    assert _help_output(argv, monkeypatch, capsys) == dedent(HELP_SNAPSHOTS[argv])
