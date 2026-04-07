from __future__ import annotations

from worldforge.providers import CosmosProvider, MockProvider
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
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    report = assert_provider_contract(CosmosProvider())

    assert report.configured is False
    assert report.health.healthy is False
    assert report.exercised_operations == []
