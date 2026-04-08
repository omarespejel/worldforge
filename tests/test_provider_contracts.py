from __future__ import annotations

from worldforge.providers import CosmosProvider, GenieProvider, JepaProvider, MockProvider
from worldforge.testing import assert_provider_contract


def test_mock_provider_passes_contract_checks() -> None:
    report = assert_provider_contract(MockProvider())

    assert report.configured is True
    assert set(report.exercised_operations) == {
        "predict",
        "reason",
        "embed",
        "generate",
        "transfer",
    }


def test_scaffold_provider_reports_clear_unconfigured_contract(monkeypatch) -> None:
    monkeypatch.delenv("COSMOS_BASE_URL", raising=False)

    report = assert_provider_contract(CosmosProvider())

    assert report.configured is False
    assert report.health.healthy is False
    assert report.exercised_operations == []


def test_configured_stub_remote_providers_pass_contract_checks(monkeypatch) -> None:
    monkeypatch.setenv("JEPA_MODEL_PATH", "/tmp/jepa-model")
    monkeypatch.setenv("GENIE_API_KEY", "genie-test-key")

    jepa_report = assert_provider_contract(JepaProvider())
    assert set(jepa_report.exercised_operations) == {"predict", "reason", "embed"}

    genie_report = assert_provider_contract(GenieProvider())
    assert set(genie_report.exercised_operations) == {"predict", "generate", "reason"}
