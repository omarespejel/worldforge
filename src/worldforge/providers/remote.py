"""Credential-gated provider adapters."""

from __future__ import annotations

from worldforge.models import EmbeddingResult, ProviderCapabilities, ReasoningResult, VideoClip

from .base import RemoteProvider
from .mock import MockProvider


class StubRemoteProvider(RemoteProvider):
    """Shared behavior for remote adapters that are present but not yet production-complete."""

    def predict(self, world_state, action, steps):  # type: ignore[no-untyped-def]
        self._require_credentials()
        payload = MockProvider(name=self.name).predict(world_state, action, steps)
        payload.metadata["mode"] = "stub-remote-adapter"
        payload.metadata["credential_env"] = self.env_var
        return payload

    def generate(self, prompt: str, duration_seconds: float) -> VideoClip:
        self._require_credentials()
        clip = MockProvider(name=self.name).generate(prompt, duration_seconds)
        clip.metadata["mode"] = "stub-remote-adapter"
        clip.metadata["credential_env"] = self.env_var
        return clip

    def transfer(self, clip: VideoClip, *, width: int, height: int, fps: float) -> VideoClip:
        self._require_credentials()
        transferred = MockProvider(name=self.name).transfer(
            clip, width=width, height=height, fps=fps
        )
        transferred.metadata["mode"] = "stub-remote-adapter"
        transferred.metadata["credential_env"] = self.env_var
        return transferred

    def reason(self, query: str, *, world_state=None) -> ReasoningResult:  # type: ignore[no-untyped-def]
        self._require_credentials()
        result = MockProvider(name=self.name).reason(query, world_state=world_state)
        result.evidence.append(f"Executed via stub adapter gated by {self.env_var}")
        return result

    def embed(self, *, text: str) -> EmbeddingResult:
        self._require_credentials()
        return MockProvider(name=self.name).embed(text=text)


class CosmosProvider(StubRemoteProvider):
    """Python adapter placeholder for NVIDIA Cosmos."""

    env_var = "NVIDIA_API_KEY"

    def __init__(self, name: str = "cosmos") -> None:
        super().__init__(
            name=name,
            capabilities=ProviderCapabilities(
                predict=True,
                generate=True,
                reason=True,
                embed=True,
                plan=True,
                transfer=True,
            ),
            is_local=False,
            description="Python adapter surface for NVIDIA Cosmos.",
        )


class RunwayProvider(StubRemoteProvider):
    """Python adapter placeholder for Runway."""

    env_var = "RUNWAY_API_SECRET"

    def __init__(self, name: str = "runway") -> None:
        super().__init__(
            name=name,
            capabilities=ProviderCapabilities(
                predict=True,
                generate=True,
                plan=True,
                transfer=True,
            ),
            is_local=False,
            description="Python adapter surface for Runway.",
        )


class JepaProvider(StubRemoteProvider):
    """Python adapter placeholder for JEPA-family models."""

    env_var = "JEPA_MODEL_PATH"

    def __init__(self, name: str = "jepa") -> None:
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
        )


class GenieProvider(StubRemoteProvider):
    """Python adapter placeholder for Genie-family models."""

    env_var = "GENIE_API_KEY"

    def __init__(self, name: str = "genie") -> None:
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
        )
