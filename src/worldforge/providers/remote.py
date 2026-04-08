"""Scaffold adapters for providers that are not yet fully implemented."""

from __future__ import annotations

from collections.abc import Callable

from worldforge.models import (
    EmbeddingResult,
    GenerationOptions,
    ProviderCapabilities,
    ProviderEvent,
    ReasoningResult,
    VideoClip,
)

from .base import RemoteProvider
from .mock import MockProvider


class StubRemoteProvider(RemoteProvider):
    """Shared behavior for credential-gated providers still backed by local surrogates."""

    def predict(self, world_state, action, steps):  # type: ignore[no-untyped-def]
        self._require_credentials()
        payload = MockProvider(name=self.name, event_handler=self.event_handler).predict(
            world_state,
            action,
            steps,
        )
        payload.metadata["mode"] = "stub-remote-adapter"
        payload.metadata["credential_env"] = self.env_var
        return payload

    def generate(
        self,
        prompt: str,
        duration_seconds: float,
        *,
        options: GenerationOptions | None = None,
    ) -> VideoClip:
        self._require_credentials()
        clip = MockProvider(name=self.name, event_handler=self.event_handler).generate(
            prompt,
            duration_seconds,
            options=options,
        )
        clip.metadata["mode"] = "stub-remote-adapter"
        clip.metadata["credential_env"] = self.env_var
        return clip

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
        transferred = MockProvider(name=self.name, event_handler=self.event_handler).transfer(
            clip,
            width=width,
            height=height,
            fps=fps,
            prompt=prompt,
            options=options,
        )
        transferred.metadata["mode"] = "stub-remote-adapter"
        transferred.metadata["credential_env"] = self.env_var
        return transferred

    def reason(self, query: str, *, world_state=None) -> ReasoningResult:  # type: ignore[no-untyped-def]
        self._require_credentials()
        result = MockProvider(name=self.name, event_handler=self.event_handler).reason(
            query,
            world_state=world_state,
        )
        result.evidence.append(f"Executed via stub adapter gated by {self.env_var}")
        return result

    def embed(self, *, text: str) -> EmbeddingResult:
        self._require_credentials()
        return MockProvider(name=self.name, event_handler=self.event_handler).embed(text=text)


class JepaProvider(StubRemoteProvider):
    """Python adapter placeholder for JEPA-family models."""

    env_var = "JEPA_MODEL_PATH"

    def __init__(
        self,
        name: str = "jepa",
        *,
        event_handler: Callable[[ProviderEvent], None] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            capabilities=ProviderCapabilities(
                predict=True,
                reason=True,
                embed=True,
                plan=True,
            ),
            is_local=False,
            description="Python adapter surface for JEPA-family models.",
            package="worldforge",
            implementation_status="scaffold",
            deterministic=False,
            supported_modalities=["world_state", "text"],
            artifact_types=["prediction", "reasoning", "embedding"],
            notes=[
                "Credential-gated scaffold adapter.",
                "Current runtime path falls back to deterministic mock behavior after auth checks.",
            ],
            default_model="jepa-scaffold-v1",
            supported_models=["jepa-scaffold-v1"],
            event_handler=event_handler,
        )


class GenieProvider(StubRemoteProvider):
    """Python adapter placeholder for Genie-family models."""

    env_var = "GENIE_API_KEY"

    def __init__(
        self,
        name: str = "genie",
        *,
        event_handler: Callable[[ProviderEvent], None] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            capabilities=ProviderCapabilities(
                predict=True,
                generate=True,
                reason=True,
                plan=True,
            ),
            is_local=False,
            description="Python adapter surface for Genie-family models.",
            package="worldforge",
            implementation_status="scaffold",
            deterministic=False,
            supported_modalities=["world_state", "text", "video"],
            artifact_types=["prediction", "video", "reasoning"],
            notes=[
                "Credential-gated scaffold adapter.",
                "Current runtime path falls back to deterministic mock behavior after auth checks.",
            ],
            default_model="genie-scaffold-v1",
            supported_models=["genie-scaffold-v1"],
            event_handler=event_handler,
        )
