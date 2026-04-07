"""Provider primitives for WorldForge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter

from worldforge.models import (
    Action,
    EmbeddingResult,
    JSONDict,
    ProviderCapabilities,
    ProviderHealth,
    ProviderInfo,
    ReasoningResult,
    VideoClip,
)


class ProviderError(RuntimeError):
    """Raised when a provider cannot satisfy a request."""


@dataclass(slots=True)
class PredictionPayload:
    """Serialized prediction data returned by providers."""

    state: JSONDict
    confidence: float
    physics_score: float
    frames: list[bytes]
    metadata: JSONDict
    latency_ms: float


class BaseProvider:
    """Base class for WorldForge providers."""

    env_var: str | None = None

    def __init__(
        self,
        name: str,
        *,
        capabilities: ProviderCapabilities | None = None,
        is_local: bool = False,
        description: str = "",
    ) -> None:
        self.name = name
        self.capabilities = capabilities or ProviderCapabilities()
        self.is_local = is_local
        self.description = description

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            capabilities=self.capabilities,
            is_local=self.is_local,
            description=self.description,
        )

    def configured(self) -> bool:
        return self.env_var is None or bool(os.environ.get(self.env_var))

    def health(self) -> ProviderHealth:
        started = perf_counter()
        healthy = self.configured()
        details = "configured" if healthy else f"missing {self.env_var}"
        return ProviderHealth(
            name=self.name,
            healthy=healthy,
            latency_ms=max(0.1, (perf_counter() - started) * 1000),
            details=details,
        )

    def predict(self, world_state: JSONDict, action: Action, steps: int) -> PredictionPayload:
        raise NotImplementedError

    def generate(self, prompt: str, duration_seconds: float) -> VideoClip:
        raise ProviderError(f"Provider '{self.name}' does not implement generate().")

    def transfer(self, clip: VideoClip, *, width: int, height: int, fps: float) -> VideoClip:
        raise ProviderError(f"Provider '{self.name}' does not implement transfer().")

    def reason(self, query: str, *, world_state: JSONDict | None = None) -> ReasoningResult:
        raise ProviderError(f"Provider '{self.name}' does not implement reason().")

    def embed(self, *, text: str) -> EmbeddingResult:
        raise ProviderError(f"Provider '{self.name}' does not implement embed().")


class RemoteProvider(BaseProvider):
    """Base class for providers that depend on third-party credentials."""

    def _require_credentials(self) -> None:
        if not self.configured():
            raise ProviderError(f"Provider '{self.name}' is unavailable: missing {self.env_var}.")
