"""Shared helpers for HTTP-backed provider adapters."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from time import sleep

import httpx

from worldforge.models import GenerationOptions, VideoClip

from .base import ProviderError


def asset_to_uri(value: str | None, *, default_content_type: str) -> str | None:
    """Return a URL or data URI suitable for provider APIs."""

    if value is None:
        return None
    if value.startswith(("http://", "https://", "data:")):
        return value

    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ProviderError(f"Local asset path does not exist: {path}")
    if not path.is_file():
        raise ProviderError(f"Local asset path is not a file: {path}")

    content_type = mimetypes.guess_type(path.name)[0] or default_content_type
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{payload}"


def clip_to_data_uri(clip: VideoClip) -> str:
    """Encode a clip as a data URI for video-to-video style APIs."""

    payload = base64.b64encode(clip.blob()).decode("ascii")
    return f"data:{clip.content_type()};base64,{payload}"


def parse_size(options: GenerationOptions | None, *, fallback: tuple[int, int]) -> tuple[int, int]:
    """Resolve output size from typed options."""

    def _validate_dimensions(width: int, height: int) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            raise ProviderError("Output size values must be greater than 0.")
        return width, height

    if options is None:
        return _validate_dimensions(*fallback)
    if options.size:
        try:
            width_text, height_text = options.size.lower().split("x", maxsplit=1)
            return _validate_dimensions(int(width_text), int(height_text))
        except ValueError as exc:
            raise ProviderError(
                f"Invalid size '{options.size}'. Expected format WIDTHxHEIGHT."
            ) from exc
    if options.ratio:
        try:
            width_text, height_text = options.ratio.split(":", maxsplit=1)
            return _validate_dimensions(int(width_text), int(height_text))
        except ValueError as exc:
            raise ProviderError(
                f"Invalid ratio '{options.ratio}'. Expected format WIDTH:HEIGHT."
            ) from exc

    return _validate_dimensions(*fallback)


def poll_json_task(
    client: httpx.Client,
    *,
    path: str,
    status_key: str = "status",
    success_values: set[str],
    failure_values: set[str],
    poll_interval_seconds: float,
    max_polls: int,
    provider_name: str,
) -> dict[str, object]:
    """Poll an HTTP task endpoint until it completes or fails."""

    for _ in range(max_polls):
        response = client.get(path)
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get(status_key, "")).upper()
        if status in success_values:
            return dict(payload)
        if status in failure_values:
            raise ProviderError(f"Provider '{provider_name}' task failed with status {status}.")
        sleep(poll_interval_seconds)
    raise ProviderError(f"Provider '{provider_name}' task did not complete before timeout.")
