from __future__ import annotations

from importlib.metadata import entry_points

import pytest

from worldforge.testing.runtime_profiles import (
    PROVIDER_RUNTIME_PROFILES_BY_NAME,
    provider_profile_skip_reason,
    pytest_marker_definitions,
    runtime_marker_skip_reason,
)

pytest_plugins = ("pytester",)


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


def test_pytest_plugin_entrypoint_is_registered() -> None:
    pytest_plugins = entry_points(group="pytest11")

    assert any(
        plugin.name == "worldforge-runtime-profiles"
        and plugin.value == "worldforge.testing.pytest_plugin"
        for plugin in pytest_plugins
    )


def test_pytest_plugin_skips_live_tests_by_default(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("""
        import pytest

        @pytest.mark.live
        def test_live_runtime():
            raise AssertionError("should be skipped before test body runs")
    """)

    result = pytester.runpytest("-rs")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*requires an explicit --run-live opt-in*"])


def test_pytest_plugin_runs_configured_provider_profile(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytester.makepyfile("""
        import pytest

        @pytest.mark.live
        @pytest.mark.network
        @pytest.mark.credentialed
        @pytest.mark.provider_profile("runway")
        def test_runway_runtime():
            assert True
    """)
    monkeypatch.setenv("RUNWAY_API_SECRET", "test-secret")

    result = pytester.runpytest(
        "--run-live",
        "--run-network",
        "--run-credentialed",
        "--provider-profile",
        "runway",
    )

    result.assert_outcomes(passed=1)


def test_pytest_plugin_skips_unselected_provider_profile(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("""
        import pytest

        @pytest.mark.live
        @pytest.mark.provider_profile("runway")
        def test_runway_runtime():
            raise AssertionError("should be skipped before test body runs")
    """)

    result = pytester.runpytest("-rs", "--run-live", "--provider-profile", "cosmos")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*selected provider profile is 'cosmos', not 'runway'*"])
