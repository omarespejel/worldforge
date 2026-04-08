"""NVIDIA Cosmos provider integration."""

from __future__ import annotations

import base64
import os
from time import perf_counter

import httpx

from worldforge.models import GenerationOptions, ProviderCapabilities, ProviderHealth, VideoClip

from .base import ProviderError, RemoteProvider
from .http_utils import asset_to_uri, parse_size


class CosmosProvider(RemoteProvider):
    """HTTP adapter for self-hosted or managed NVIDIA Cosmos NIM deployments."""

    env_var = "COSMOS_BASE_URL"

    def __init__(
        self,
        name: str = "cosmos",
        *,
        base_url: str | None = None,
        timeout_seconds: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            name=name,
            capabilities=ProviderCapabilities(
                predict=False,
                generate=True,
                reason=False,
                embed=False,
                plan=False,
                transfer=False,
            ),
            is_local=False,
            description="NVIDIA Cosmos NIM adapter for text/image/video-to-world generation.",
            package="worldforge",
            implementation_status="beta",
            deterministic=False,
            supported_modalities=["text", "image", "video"],
            artifact_types=["video"],
            notes=[
                "Targets the documented Cosmos NIM `/v1/infer` API.",
                "Requires a reachable Cosmos deployment via `COSMOS_BASE_URL`.",
                "If `NVIDIA_API_KEY` is set, it is sent as a bearer token.",
            ],
            default_model="Cosmos-Predict1-7B-Text2World",
            supported_models=[
                "Cosmos-Predict1-7B-Text2World",
                "Cosmos-Predict1-7B-Video2World",
            ],
            required_env_vars=["COSMOS_BASE_URL"],
            requires_credentials=False,
        )
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def configured(self) -> bool:
        return bool(self._resolved_base_url())

    def _resolved_base_url(self) -> str | None:
        return self._base_url or os.environ.get("COSMOS_BASE_URL")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        api_key = os.environ.get("NVIDIA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _client(self) -> httpx.Client:
        base_url = self._resolved_base_url()
        if not base_url:
            raise ProviderError(f"Provider '{self.name}' is unavailable: missing COSMOS_BASE_URL.")
        return httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=self._headers(),
            timeout=self._timeout_seconds,
            transport=self._transport,
        )

    def health(self) -> ProviderHealth:
        started = perf_counter()
        base_url = self._resolved_base_url()
        if not base_url:
            return ProviderHealth(
                name=self.name,
                healthy=False,
                latency_ms=max(0.1, (perf_counter() - started) * 1000),
                details="missing COSMOS_BASE_URL",
            )

        try:
            with self._client() as client:
                response = client.get("/v1/health/ready")
                response.raise_for_status()
                payload = response.json()
            status = str(payload.get("status", "unknown"))
            healthy = status.lower() == "ready"
            details = status
        except (httpx.HTTPError, ValueError) as exc:
            healthy = False
            details = str(exc)

        return ProviderHealth(
            name=self.name,
            healthy=healthy,
            latency_ms=max(0.1, (perf_counter() - started) * 1000),
            details=details,
        )

    def generate(
        self,
        prompt: str,
        duration_seconds: float,
        *,
        options: GenerationOptions | None = None,
    ) -> VideoClip:
        self._require_credentials()
        if duration_seconds <= 0.0:
            raise ProviderError("Cosmos duration_seconds must be greater than 0.")

        width, height = parse_size(options, fallback=(1280, 720))
        if width % 8 or height % 8:
            raise ProviderError(
                "Cosmos output size must use width and height that are multiples of 8."
            )

        fps = options.fps if options and options.fps is not None else 24.0
        if fps <= 0.0:
            raise ProviderError("Cosmos fps must be greater than 0.")
        frame_count = max(1, int(round(duration_seconds * fps)))
        body: dict[str, object] = {
            "prompt": prompt,
            "seed": options.seed if options and options.seed is not None else 4,
            "video_params": {
                "height": height,
                "width": width,
                "frames_count": frame_count,
                "frames_per_sec": int(round(fps)),
            },
        }

        if options and options.negative_prompt:
            body["negative_prompt"] = options.negative_prompt

        if options and options.image:
            body["image"] = asset_to_uri(options.image, default_content_type="image/png")
        if options and options.video:
            body["video"] = asset_to_uri(options.video, default_content_type="video/mp4")
        if options:
            body.update(options.extras)

        with self._client() as client:
            response = client.post("/v1/infer", json=body)
            response.raise_for_status()
            payload = response.json()

        b64_video = payload.get("b64_video")
        if not isinstance(b64_video, str) or not b64_video:
            raise ProviderError(f"Provider '{self.name}' did not return a `b64_video` payload.")

        try:
            clip_bytes = base64.b64decode(b64_video, validate=True)
        except (ValueError, TypeError) as exc:
            raise ProviderError(
                f"Provider '{self.name}' returned an invalid base64 video payload."
            ) from exc
        mode = "text2world"
        if options and options.image:
            mode = "image2world"
        if options and options.video:
            mode = "video2world"
        return VideoClip(
            frames=[clip_bytes],
            fps=fps,
            resolution=(width, height),
            duration_seconds=duration_seconds,
            metadata={
                "provider": self.name,
                "prompt": prompt,
                "mode": mode,
                "seed": payload.get("seed"),
                "upsampled_prompt": payload.get("upsampled_prompt"),
                "content_type": "video/mp4",
                "model": options.model if options and options.model else self.default_model,
                "base_url": self._resolved_base_url(),
            },
        )
