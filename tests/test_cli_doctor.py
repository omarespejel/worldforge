from __future__ import annotations

import json
import sys

from worldforge.cli import main


def test_doctor_and_provider_info_cli(tmp_path, monkeypatch, capsys) -> None:
    for env_var in ("NVIDIA_API_KEY", "RUNWAY_API_SECRET", "JEPA_MODEL_PATH", "GENIE_API_KEY"):
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
