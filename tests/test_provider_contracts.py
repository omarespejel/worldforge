from __future__ import annotations

import pytest

from worldforge.providers import (
    CosmosProvider,
    GenieProvider,
    JepaProvider,
    MockProvider,
    ProviderError,
)
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


def test_configured_scaffold_remote_providers_stay_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("JEPA_MODEL_PATH", "/tmp/jepa-model")
    monkeypatch.setenv("GENIE_API_KEY", "genie-test-key")
    monkeypatch.delenv("WORLDFORGE_ENABLE_SCAFFOLD_SURROGATES", raising=False)

    jepa_report = assert_provider_contract(JepaProvider())
    assert jepa_report.configured is True
    assert jepa_report.exercised_operations == []

    genie_report = assert_provider_contract(GenieProvider())
    assert genie_report.configured is True
    assert genie_report.exercised_operations == []


def test_scaffold_surrogate_requires_explicit_local_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("JEPA_MODEL_PATH", "/tmp/jepa-model")
    monkeypatch.delenv("WORLDFORGE_ENABLE_SCAFFOLD_SURROGATES", raising=False)

    with pytest.raises(ProviderError, match="WORLDFORGE_ENABLE_SCAFFOLD_SURROGATES"):
        JepaProvider().embed(text="cube")
