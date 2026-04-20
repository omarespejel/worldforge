from __future__ import annotations

from worldforge.providers.catalog import PROVIDER_CATALOG, create_known_providers


def test_provider_catalog_names_are_unique_and_explicit() -> None:
    names = [entry.name for entry in PROVIDER_CATALOG]

    assert len(names) == len(set(names))
    assert names == [
        "mock",
        "cosmos",
        "runway",
        "leworldmodel",
        "gr00t",
        "lerobot",
        "jepa",
        "genie",
    ]
    assert [entry.name for entry in PROVIDER_CATALOG if entry.always_register] == ["mock"]


def test_provider_catalog_instantiates_known_provider_profiles() -> None:
    providers = create_known_providers()
    profiles = {provider.name: provider.profile() for provider in providers}

    assert profiles["mock"].implementation_status == "stable"
    assert profiles["leworldmodel"].capabilities.score is True
    assert profiles["gr00t"].capabilities.policy is True
    assert profiles["lerobot"].capabilities.policy is True
    assert profiles["jepa"].implementation_status == "scaffold"
    assert profiles["genie"].implementation_status == "scaffold"
