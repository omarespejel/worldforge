from __future__ import annotations

import json
import sys

from worldforge.cli import main


def test_examples_cli_outputs_human_and_json_indexes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["worldforge", "examples"])
    assert main() == 0
    output = capsys.readouterr().out
    assert output.startswith("# WorldForge Examples")
    assert "worldforge-demo-leworldmodel" in output
    assert "worldforge-demo-lerobot" in output

    monkeypatch.setattr(sys, "argv", ["worldforge", "examples", "--format", "json"])
    assert main() == 0
    examples_payload = json.loads(capsys.readouterr().out)
    example_names = {example["name"] for example in examples_payload}
    assert "leworldmodel-score-planning" in example_names
    assert "lerobot-policy-score-planning" in example_names


def test_doctor_and_provider_info_cli(tmp_path, monkeypatch, capsys) -> None:
    for env_var in (
        "COSMOS_BASE_URL",
        "NVIDIA_API_KEY",
        "RUNWAYML_API_SECRET",
        "RUNWAY_API_SECRET",
        "LEWORLDMODEL_POLICY",
        "LEWM_POLICY",
        "LEWORLDMODEL_CACHE_DIR",
        "LEWORLDMODEL_DEVICE",
        "JEPA_MODEL_PATH",
        "GENIE_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["worldforge", "doctor", "--state-dir", str(tmp_path)],
    )
    assert main() == 0
    doctor_payload = json.loads(capsys.readouterr().out)
    provider_names = {provider["name"] for provider in doctor_payload["providers"]}
    assert {"mock", "cosmos"} <= provider_names
    assert doctor_payload["registered_provider_count"] >= 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["worldforge", "provider", "info", "mock", "--state-dir", str(tmp_path)],
    )
    assert main() == 0
    provider_payload = json.loads(capsys.readouterr().out)
    assert provider_payload["registered"] is True
    assert provider_payload["profile"]["implementation_status"] == "stable"
    assert provider_payload["health"]["healthy"] is True


def test_provider_docs_cli_outputs_markdown_and_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["worldforge", "provider", "docs"])
    assert main() == 0
    output = capsys.readouterr().out
    assert output.startswith("# WorldForge Provider Docs")
    assert "`leworldmodel`" in output
    assert "`docs/src/providers/leworldmodel.md`" in output

    monkeypatch.setattr(
        sys,
        "argv",
        ["worldforge", "provider", "docs", "runway", "--format", "json"],
    )
    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "name": "runway",
            "docs_path": "docs/src/providers/runway.md",
            "capabilities": "generate, transfer",
            "registration": "RUNWAYML_API_SECRET or RUNWAY_API_SECRET",
            "runtime_ownership": "host supplies Runway credentials and persists returned artifacts",
        }
    ]


def test_generate_cli_writes_output_file(tmp_path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "mock-output.bin"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldforge",
            "generate",
            "A cube rolling across a table",
            "--provider",
            "mock",
            "--duration",
            "1",
            "--output",
            str(output_path),
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert output_path.exists()
    assert payload["output_path"] == str(output_path.resolve())


def test_cli_supports_provider_listing_predict_transfer_and_eval(
    tmp_path, monkeypatch, capsys
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"mock-input-video")
    transfer_output_path = tmp_path / "transfer-output.bin"

    monkeypatch.setattr(
        sys,
        "argv",
        ["worldforge", "providers", "--state-dir", str(tmp_path)],
    )
    assert main() == 0
    providers_payload = json.loads(capsys.readouterr().out)
    assert any(provider["name"] == "mock" for provider in providers_payload)

    monkeypatch.setattr(
        sys,
        "argv",
        ["worldforge", "provider", "list", "--state-dir", str(tmp_path)],
    )
    assert main() == 0
    provider_list_payload = json.loads(capsys.readouterr().out)
    assert any(provider["name"] == "mock" for provider in provider_list_payload)

    monkeypatch.setattr(
        sys,
        "argv",
        ["worldforge", "provider", "health", "mock", "--state-dir", str(tmp_path)],
    )
    assert main() == 0
    provider_health_payload = json.loads(capsys.readouterr().out)
    assert provider_health_payload["healthy"] is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldforge",
            "predict",
            "kitchen",
            "--provider",
            "mock",
            "--x",
            "0.3",
            "--y",
            "0.8",
            "--z",
            "0.0",
            "--steps",
            "2",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert main() == 0
    predict_payload = json.loads(capsys.readouterr().out)
    assert predict_payload["provider"] == "mock"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldforge",
            "transfer",
            str(input_path),
            "--provider",
            "mock",
            "--width",
            "320",
            "--height",
            "240",
            "--fps",
            "12",
            "--duration",
            "1",
            "--output",
            str(transfer_output_path),
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert main() == 0
    transfer_payload = json.loads(capsys.readouterr().out)
    assert transfer_output_path.exists()
    assert transfer_payload["output_path"] == str(transfer_output_path.resolve())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldforge",
            "eval",
            "--suite",
            "physics",
            "--provider",
            "mock",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert main() == 0
    eval_output = capsys.readouterr().out
    assert eval_output.startswith("# Evaluation Report")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldforge",
            "eval",
            "--suite",
            "planning",
            "--provider",
            "mock",
            "--format",
            "json",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert main() == 0
    eval_payload = json.loads(capsys.readouterr().out)
    assert eval_payload["suite_id"] == "planning"
    assert len(eval_payload["results"]) == 4

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldforge",
            "eval",
            "--suite",
            "reasoning",
            "--provider",
            "mock",
            "--format",
            "csv",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert main() == 0
    eval_csv = capsys.readouterr().out
    assert eval_csv.startswith("suite_id,suite,provider,scenario,score,passed,metrics_json")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldforge",
            "benchmark",
            "--provider",
            "mock",
            "--operation",
            "generate",
            "--iterations",
            "2",
            "--format",
            "json",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert main() == 0
    benchmark_payload = json.loads(capsys.readouterr().out)
    assert benchmark_payload["results"][0]["provider"] == "mock"
    assert benchmark_payload["results"][0]["operation"] == "generate"
