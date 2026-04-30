from __future__ import annotations

from worldforge.testing.runtime_profiles import (
    PROVIDER_RUNTIME_PROFILES_BY_NAME,
    provider_profile_skip_reason,
    pytest_marker_definitions,
    runtime_marker_skip_reason,
)


def test_runtime_markers_require_explicit_opt_in() -> None:
    reason = runtime_marker_skip_reason("live", {"run_live": False})

    assert reason == "requires an explicit --run-live opt-in"
    assert runtime_marker_skip_reason("live", {"run_live": True}) is None


def test_provider_profile_reports_missing_any_env_group() -> None:
    reason = provider_profile_skip_reason("runway", {})

    assert reason == (
        "provider profile 'runway' is not configured: "
        "missing RUNWAYML_API_SECRET or RUNWAY_API_SECRET"
    )


def test_provider_profile_accepts_any_env_alias() -> None:
    reason = provider_profile_skip_reason("runway", {"RUNWAY_API_SECRET": "secret"})

    assert reason is None


def test_provider_profile_reports_unknown_profile() -> None:
    reason = provider_profile_skip_reason("missing-provider", {})

    assert reason == "unknown provider runtime profile: missing-provider"


def test_marker_definitions_cover_runtime_profiles() -> None:
    definitions = "\n".join(pytest_marker_definitions())

    assert "live:" in definitions
    assert "provider_profile(name)" in definitions
    assert set(PROVIDER_RUNTIME_PROFILES_BY_NAME) == {
        "cosmos",
        "runway",
        "leworldmodel",
        "gr00t",
        "lerobot",
    }
