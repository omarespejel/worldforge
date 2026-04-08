"""Runway video provider integration."""

from __future__ import annotations

import os
from time import perf_counter

import httpx

from worldforge.models import (
    GenerationOptions,
    ProviderCapabilities,
    ProviderHealth,
    ProviderRequestPolicy,
    VideoClip,
)

from .base import ProviderError, RemoteProvider
from .http_utils import (
    asset_to_uri,
    clip_to_data_uri,
    poll_json_task,
    request_bytes_with_policy,
    request_json_with_policy,
)

_RUNWAY_API_VERSION = "2024-11-06"
_RUNWAY_DEFAULT_RATIO = "1280:720"
_RUNWAY_DEFAULT_DURATION = 5


class RunwayProvider(RemoteProvider):
    """HTTP adapter for Runway's image-to-video and video-to-video APIs."""

    env_var = "RUNWAYML_API_SECRET"

    def __init__(
        self,
        name: str = "runway",
        *,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 6.0,
        max_polls: int = 60,
        request_policy: ProviderRequestPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        resolved_request_policy = request_policy or ProviderRequestPolicy.remote_defaults(
            request_timeout_seconds=timeout_seconds
        )
        super().__init__(
            name=name,
            capabilities=ProviderCapabilities(
                predict=False,
                generate=True,
                reason=False,
                embed=False,
                plan=False,
                transfer=True,
            ),
            is_local=False,
            description="Runway adapter for text/image-to-video and video-to-video generation.",
            package="worldforge",
            implementation_status="beta",
            deterministic=False,
            supported_modalities=["text", "image", "video"],
            artifact_types=["video"],
            notes=[
                "Targets Runway's documented `image_to_video`, `video_to_video`, and `tasks` APIs.",
                "Supports `RUNWAYML_API_SECRET` and the legacy alias `RUNWAY_API_SECRET`.",
                "Downloaded task outputs should be persisted by the caller because URLs expire.",
            ],
            default_model="gen4.5",
            supported_models=["gen4.5", "gen4_turbo", "veo3.1", "veo3.1_fast", "gen4_aleph"],
            required_env_vars=["RUNWAYML_API_SECRET", "RUNWAY_API_SECRET"],
            request_policy=resolved_request_policy,
        )
        self._base_url = (
            base_url or os.environ.get("RUNWAYML_BASE_URL") or "https://api.dev.runwayml.com"
        )
        self._poll_interval_seconds = poll_interval_seconds
        self._max_polls = max_polls
        self._transport = transport

    def configured(self) -> bool:
        return bool(self._api_key())

    def _api_key(self) -> str | None:
        return os.environ.get("RUNWAYML_API_SECRET") or os.environ.get("RUNWAY_API_SECRET")

    def _headers(self) -> dict[str, str]:
        api_key = self._api_key()
        if not api_key:
            raise ProviderError(f"Provider '{self.name}' is unavailable: missing {self.env_var}.")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": _RUNWAY_API_VERSION,
        }

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._base_url.rstrip("/"),
            headers=self._headers(),
            transport=self._transport,
        )

    def health(self) -> ProviderHealth:
        started = perf_counter()
        if not self.configured():
            return ProviderHealth(
                name=self.name,
                healthy=False,
                latency_ms=max(0.1, (perf_counter() - started) * 1000),
                details=f"missing {self.env_var}",
            )

        try:
            request_policy = self._require_request_policy()
            with self._client() as client:
                payload = request_json_with_policy(
                    client,
                    method="GET",
                    url="/v1/organization",
                    provider_name=self.name,
                    operation_name="healthcheck",
                    policy=request_policy.health,
                )
            details = str(payload.get("name") or payload.get("id") or "organization ok")
            healthy = True
        except ProviderError as exc:
            healthy = False
            details = str(exc)

        return ProviderHealth(
            name=self.name,
            healthy=healthy,
            latency_ms=max(0.1, (perf_counter() - started) * 1000),
            details=details,
        )

    def _ratio(
        self,
        width: int | None = None,
        height: int | None = None,
        options: GenerationOptions | None = None,
    ) -> str:
        if options and options.ratio:
            return options.ratio
        if width is not None and height is not None:
            return f"{width}:{height}"
        return _RUNWAY_DEFAULT_RATIO

    def _poll_task(self, client: httpx.Client, task_id: str) -> dict[str, object]:
        request_policy = self._require_request_policy()
        payload = poll_json_task(
            client,
            path=f"/v1/tasks/{task_id}",
            success_values={"SUCCEEDED"},
            failure_values={"FAILED", "CANCELLED"},
            poll_interval_seconds=self._poll_interval_seconds,
            max_polls=self._max_polls,
            provider_name=self.name,
            operation_policy=request_policy.polling,
        )
        outputs = payload.get("output")
        if (
            not isinstance(outputs, list)
            or not outputs
            or not all(isinstance(item, str) and item for item in outputs)
        ):
            raise ProviderError(f"Provider '{self.name}' task {task_id} completed without outputs.")
        return payload

    def _download_output(self, output_url: str) -> bytes:
        request_policy = self._require_request_policy()
        with httpx.Client(transport=self._transport) as client:
            return request_bytes_with_policy(
                client,
                method="GET",
                url=output_url,
                provider_name=self.name,
                operation_name="artifact download",
                policy=request_policy.download,
            )

    def _task_id(self, payload: dict[str, object], *, operation_name: str) -> str:
        task_id = payload.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ProviderError(
                f"Provider '{self.name}' {operation_name} did not return a task id."
            )
        return task_id

    def generate(
        self,
        prompt: str,
        duration_seconds: float,
        *,
        options: GenerationOptions | None = None,
    ) -> VideoClip:
        self._require_credentials()
        if options and options.video:
            raise ProviderError(
                "Runway image_to_video does not accept `options.video`; "
                "use transfer() for video inputs."
            )
        if duration_seconds <= 0.0:
            raise ProviderError("Runway duration_seconds must be greater than 0.")

        duration = max(2, min(10, int(round(duration_seconds or _RUNWAY_DEFAULT_DURATION))))
        ratio = self._ratio(options=options)
        model = options.model if options and options.model else self.default_model
        body: dict[str, object] = {
            "model": model,
            "promptText": prompt,
            "ratio": ratio,
            "duration": duration,
        }
        prompt_image = asset_to_uri(
            options.image if options else None,
            default_content_type="image/png",
        )
        if prompt_image:
            body["promptImage"] = prompt_image
        if options and options.seed is not None:
            body["seed"] = options.seed
        if options and options.extras:
            body.update(options.extras)

        request_policy = self._require_request_policy()
        with self._client() as client:
            payload = request_json_with_policy(
                client,
                method="POST",
                url="/v1/image_to_video",
                provider_name=self.name,
                operation_name="generation request",
                policy=request_policy.request,
                json=body,
            )
            task_id = self._task_id(payload, operation_name="generation request")
            task = self._poll_task(client, task_id)

        output_url = str(task["output"][0])
        clip_bytes = self._download_output(output_url)
        return VideoClip(
            frames=[clip_bytes],
            fps=options.fps if options and options.fps is not None else 24.0,
            resolution=tuple(int(part) for part in ratio.split(":", maxsplit=1)),  # type: ignore[arg-type]
            duration_seconds=float(duration),
            metadata={
                "provider": self.name,
                "prompt": prompt,
                "task_id": task_id,
                "output_url": output_url,
                "content_type": "video/mp4",
                "model": model,
                "mode": "image_to_video" if prompt_image else "text_to_video",
            },
        )

    def transfer(
        self,
        clip: VideoClip,
        *,
        width: int,
        height: int,
        fps: float,
        prompt: str = "",
        options: GenerationOptions | None = None,
    ) -> VideoClip:
        self._require_credentials()
        if width <= 0 or height <= 0:
            raise ProviderError("Runway output width and height must be greater than 0.")
        if fps <= 0.0:
            raise ProviderError("Runway fps must be greater than 0.")
        model = options.model if options and options.model else "gen4_aleph"
        references: list[dict[str, str]] = []
        if options:
            for reference in options.reference_images:
                references.append(
                    {"uri": asset_to_uri(reference, default_content_type="image/png") or reference}
                )

        body: dict[str, object] = {
            "model": model,
            "promptText": prompt or "Re-render the input video while preserving the scene motion.",
            "videoUri": asset_to_uri(
                options.video if options and options.video else clip_to_data_uri(clip),
                default_content_type=clip.content_type(),
            ),
        }
        if references:
            body["references"] = references
        if options and options.seed is not None:
            body["seed"] = options.seed
        if options and options.extras:
            body.update(options.extras)

        request_policy = self._require_request_policy()
        with self._client() as client:
            payload = request_json_with_policy(
                client,
                method="POST",
                url="/v1/video_to_video",
                provider_name=self.name,
                operation_name="transfer request",
                policy=request_policy.request,
                json=body,
            )
            task_id = self._task_id(payload, operation_name="transfer request")
            task = self._poll_task(client, task_id)

        output_url = str(task["output"][0])
        clip_bytes = self._download_output(output_url)
        return VideoClip(
            frames=[clip_bytes],
            fps=fps,
            resolution=(width, height),
            duration_seconds=clip.duration_seconds,
            metadata={
                "provider": self.name,
                "prompt": prompt,
                "task_id": task_id,
                "output_url": output_url,
                "content_type": "video/mp4",
                "model": model,
                "mode": "video_to_video",
            },
        )
