from __future__ import annotations

import httpx
import pytest

from worldforge import (
    Action,
    GenerationOptions,
    ProviderEvent,
    ProviderRequestPolicy,
    RetryPolicy,
    WorldForge,
    WorldForgeError,
)
from worldforge.providers import ProviderError
from worldforge.providers.http_utils import asset_to_uri, parse_size, poll_json_task


def test_http_utils_validate_assets_size_and_polling(tmp_path) -> None:
    image_path = tmp_path / "seed.png"
    image_path.write_bytes(b"png")

    asset_uri = asset_to_uri(str(image_path), default_content_type="image/png")
    assert asset_uri is not None
    assert asset_uri.startswith("data:image/png;base64,")

    with pytest.raises(ProviderError, match="does not exist"):
        asset_to_uri(str(tmp_path / "missing.png"), default_content_type="image/png")

    assert parse_size(GenerationOptions(size="640x360"), fallback=(1280, 720)) == (640, 360)
    assert parse_size(GenerationOptions(ratio="16:9"), fallback=(1280, 720)) == (16, 9)

    with pytest.raises(ProviderError, match="greater than 0"):
        parse_size(GenerationOptions(size="0x360"), fallback=(1280, 720))

    processing_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"status": "PROCESSING"})
        ),
        base_url="http://providers.test",
    )
    with processing_client as client:
        with pytest.raises(ProviderError, match="did not complete before timeout"):
            poll_json_task(
                client,
                path="/tasks/1",
                success_values={"SUCCEEDED"},
                failure_values={"FAILED"},
                poll_interval_seconds=0.0,
                max_polls=1,
                provider_name="mock",
                operation_policy=ProviderRequestPolicy.remote_defaults(
                    request_timeout_seconds=10.0,
                    read_backoff_seconds=0.0,
                ).polling,
            )

    failed_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"status": "FAILED"})
        ),
        base_url="http://providers.test",
    )
    with failed_client as client:
        with pytest.raises(ProviderError, match="task failed with status FAILED"):
            poll_json_task(
                client,
                path="/tasks/2",
                success_values={"SUCCEEDED"},
                failure_values={"FAILED"},
                poll_interval_seconds=0.0,
                max_polls=1,
                provider_name="mock",
                operation_policy=ProviderRequestPolicy.remote_defaults(
                    request_timeout_seconds=10.0,
                    read_backoff_seconds=0.0,
                ).polling,
            )

    with pytest.raises(WorldForgeError, match="max_attempts"):
        RetryPolicy(max_attempts=0)

    default_request_policy = ProviderRequestPolicy.remote_defaults(request_timeout_seconds=12.0)
    assert default_request_policy.request.retry.max_attempts == 1
    assert default_request_policy.download.retry.max_attempts == 3
    assert default_request_policy.to_dict()["health"]["timeout_seconds"] == 10.0

    event = ProviderEvent(
        provider="mock",
        operation="predict",
        phase="success",
        duration_ms=12.5,
        metadata={"steps": 1},
    )
    assert event.to_dict()["metadata"] == {"steps": 1}

    with pytest.raises(WorldForgeError, match="duration_ms"):
        ProviderEvent(provider="mock", operation="predict", phase="success", duration_ms=-1.0)


def test_framework_helpers_and_error_paths(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    world = forge.create_world_from_prompt("empty room", provider="mock")
    assert world.object_count == 1
    assert world.list_objects() == ["cube"]

    prediction = world.predict(Action.move_to(0.1, 0.5, 0.0), steps=1)
    assert prediction.output_world().object_count == world.object_count

    comparison = world.compare(Action.move_to(0.2, 0.5, 0.0), ["mock"], steps=1)
    artifacts = comparison.artifacts()
    assert set(artifacts) == {"json", "markdown", "csv"}

    assert forge.get_provider("mock").name == "mock"
    assert [health.name for health in forge.provider_healths(capability="generate")] == ["mock"]

    clip = forge.generate("a cube rolling across a table", "mock", duration_seconds=1.0)
    saved_clip_path = forge.save_clip(clip, tmp_path / "clip.bin")
    assert saved_clip_path.exists()

    assert forge.compare([prediction]).prediction_count == 1

    with pytest.raises(ValueError, match="Comparison has no predictions"):
        forge.compare([]).best_prediction()

    with pytest.raises(WorldForgeError, match="World name must not be empty"):
        forge.create_world("", "mock")

    with pytest.raises(WorldForgeError, match="Only json export"):
        forge.export_world("missing", format="yaml")

    with pytest.raises(WorldForgeError, match="Only json import"):
        forge.import_world("{}", format="yaml")
