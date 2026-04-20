from __future__ import annotations

import json
import sys

from worldforge.cli import main as worldforge_main
from worldforge.harness.cli import main as harness_main


def test_worldforge_harness_lists_flows_without_textual(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["worldforge", "harness", "--list"])

    assert worldforge_main() == 0
    output = capsys.readouterr().out

    assert output.startswith("# TheWorldHarness Flows")
    assert "leworldmodel" in output
    assert "lerobot" in output


def test_worldforge_harness_lists_json_without_textual(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["worldforge", "harness", "--list", "--format", "json"],
    )

    assert worldforge_main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert [flow["id"] for flow in payload] == ["leworldmodel", "lerobot"]


def test_worldforge_harness_console_entry_lists_flows(capsys) -> None:
    assert harness_main(["--list"]) == 0
    assert "TheWorldHarness Flows" in capsys.readouterr().out
