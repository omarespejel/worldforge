from __future__ import annotations

from worldforge import DoctorReport, WorldForge


def test_provider_profiles_and_doctor_report_include_known_scaffolds(tmp_path, monkeypatch) -> None:
    for env_var in (
        "COSMOS_BASE_URL",
        "NVIDIA_API_KEY",
        "RUNWAYML_API_SECRET",
        "RUNWAY_API_SECRET",
        "JEPA_MODEL_PATH",
        "GENIE_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)

    forge = WorldForge(state_dir=tmp_path)

    registered_profiles = {profile.name: profile for profile in forge.list_provider_profiles()}
    assert registered_profiles["mock"].implementation_status == "stable"
    assert registered_profiles["mock"].deterministic is True
    assert registered_profiles["mock"].requires_credentials is False

    builtin_profiles = {profile.name: profile for profile in forge.builtin_provider_profiles()}
    assert {"mock", "cosmos", "runway", "jepa", "genie"} <= set(builtin_profiles)
    assert builtin_profiles["cosmos"].implementation_status == "beta"
    assert builtin_profiles["cosmos"].required_env_vars == ["COSMOS_BASE_URL"]
    assert builtin_profiles["runway"].required_env_vars == [
        "RUNWAYML_API_SECRET",
        "RUNWAY_API_SECRET",
    ]

    report = forge.doctor()
    assert isinstance(report, DoctorReport)

    provider_statuses = {status.profile.name: status for status in report.providers}
    assert provider_statuses["mock"].registered is True
    assert provider_statuses["mock"].health.healthy is True
    assert provider_statuses["cosmos"].registered is False
    assert provider_statuses["cosmos"].health.healthy is False
    assert any("COSMOS_BASE_URL" in issue for issue in report.issues)
