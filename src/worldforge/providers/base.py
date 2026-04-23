"""Provider primitives for WorldForge."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from worldforge.models import (
    Action,
    ActionPolicyResult,
    ActionScoreResult,
    EmbeddingResult,
    GenerationOptions,
    JSONDict,
    ProviderCapabilities,
    ProviderEvent,
    ProviderHealth,
    ProviderInfo,
    ProviderProfile,
    ProviderRequestPolicy,
    ReasoningResult,
    VideoClip,
    WorldForgeError,
    require_finite_number,
    require_probability,
)


class ProviderError(RuntimeError):
    """Raised when a provider cannot satisfy a request."""


@dataclass(slots=True, frozen=True)
class ProviderProfileSpec:
    """Documentation and lifecycle metadata for a :class:`BaseProvider`.

    Groups the descriptive fields that used to live as a dozen optional kwargs on
    ``BaseProvider.__init__``. Every field is optional; concrete adapters declare
    only what's meaningful for their runtime.
    """

    description: str = ""
    package: str = "worldforge"
    implementation_status: str = "experimental"
    deterministic: bool = False
    is_local: bool = False
    supported_modalities: tuple[str, ...] = ()
    artifact_types: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    default_model: str | None = None
    supported_models: tuple[str, ...] = ()
    required_env_vars: tuple[str, ...] = ()
    requires_credentials: bool | None = None


@dataclass(slots=True)
class PredictionPayload:
    """Serialized prediction data returned by providers."""

    state: JSONDict
    confidence: float
    physics_score: float
    frames: list[bytes]
    metadata: JSONDict
    latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.state, dict):
            raise WorldForgeError("PredictionPayload state must be a JSON object.")
        self.confidence = require_probability(
            self.confidence,
            name="PredictionPayload confidence",
        )
        self.physics_score = require_probability(
            self.physics_score,
            name="PredictionPayload physics_score",
        )
        if not isinstance(self.frames, list) or not all(
            isinstance(frame, bytes) for frame in self.frames
        ):
            raise WorldForgeError("PredictionPayload frames must be a list of bytes.")
        if not isinstance(self.metadata, dict):
            raise WorldForgeError("PredictionPayload metadata must be a JSON object.")
        self.latency_ms = require_finite_number(
            self.latency_ms,
            name="PredictionPayload latency_ms",
        )
        if self.latency_ms < 0.0:
            raise WorldForgeError("PredictionPayload latency_ms must be non-negative.")
        self.state = dict(self.state)
        self.frames = list(self.frames)
        self.metadata = dict(self.metadata)


class BaseProvider:
    """Base class for WorldForge providers.

    Subclasses declare identity, profile metadata, and the exact capabilities they implement.
    Capability methods are intentionally fail-closed: if an adapter advertises no implementation,
    calling that surface raises ``ProviderError`` with provider context.
    """

    env_var: str | None = None

    def __init__(
        self,
        name: str,
        *,
        capabilities: ProviderCapabilities | None = None,
        profile: ProviderProfileSpec | None = None,
        request_policy: ProviderRequestPolicy | None = None,
        event_handler: Callable[[ProviderEvent], None] | None = None,
    ) -> None:
        spec = profile or ProviderProfileSpec()
        self.name = name
        self.capabilities = capabilities or ProviderCapabilities()
        self.is_local = spec.is_local
        self.description = spec.description
        self.package = spec.package
        self.implementation_status = spec.implementation_status
        self.deterministic = spec.deterministic
        self.supported_modalities = list(spec.supported_modalities)
        self.artifact_types = list(spec.artifact_types)
        self.notes = list(spec.notes)
        self.default_model = spec.default_model
        self.supported_models = list(spec.supported_models)
        # Use the profile's required_env_vars if given, otherwise fall back to the
        # class-level single `env_var` declaration (kept for simple remote adapters).
        if spec.required_env_vars:
            self.required_env_vars = list(spec.required_env_vars)
        elif self.env_var:
            self.required_env_vars = [self.env_var]
        else:
            self.required_env_vars = []
        self.requires_credentials = (
            spec.requires_credentials
            if spec.requires_credentials is not None
            else self.env_var is not None
        )
        self.request_policy = request_policy
        self.event_handler = event_handler

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            capabilities=self.capabilities,
            is_local=self.is_local,
            description=self.description,
        )

    def profile(self) -> ProviderProfile:
        return ProviderProfile(
            name=self.name,
            capabilities=self.capabilities,
            is_local=self.is_local,
            description=self.description,
            package=self.package,
            implementation_status=self.implementation_status,
            deterministic=self.deterministic,
            requires_credentials=self.requires_credentials,
            credential_env_var=self.env_var,
            required_env_vars=list(self.required_env_vars),
            supported_modalities=list(self.supported_modalities),
            artifact_types=list(self.artifact_types),
            notes=list(self.notes),
            default_model=self.default_model,
            supported_models=list(self.supported_models),
            request_policy=self.request_policy,
        )

    def configured(self) -> bool:
        return self.env_var is None or bool(os.environ.get(self.env_var))

    def health(self) -> ProviderHealth:
        started = perf_counter()
        healthy = self.configured()
        details = "configured" if healthy else f"missing {self.env_var}"
        return self._health(started, details, healthy=healthy)

    def _emit_event(self, event: ProviderEvent) -> None:
        if self.event_handler is not None:
            self.event_handler(event)

    def _emit_operation_event(
        self,
        operation: str,
        *,
        phase: str,
        duration_ms: float,
        message: str = "",
        metadata: JSONDict | None = None,
    ) -> None:
        """Emit a :class:`ProviderEvent` tagged with this provider's ``name``."""

        self._emit_event(
            ProviderEvent(
                provider=self.name,
                operation=operation,
                phase=phase,
                duration_ms=duration_ms,
                message=message,
                metadata=dict(metadata or {}),
            )
        )

    def _health(
        self,
        started: float,
        details: str,
        *,
        healthy: bool,
    ) -> ProviderHealth:
        """Build a :class:`ProviderHealth` using ``started`` as the latency origin.

        The ``max(0.1, ...)`` floor keeps the serialized latency strictly positive
        for fast-returning healthchecks (validators reject ``latency_ms <= 0``).
        """

        return ProviderHealth(
            name=self.name,
            healthy=healthy,
            latency_ms=max(0.1, (perf_counter() - started) * 1000),
            details=details,
        )

    def predict(self, world_state: JSONDict, action: Action, steps: int) -> PredictionPayload:
        raise ProviderError(f"Provider '{self.name}' does not implement predict().")

    def generate(
        self,
        prompt: str,
        duration_seconds: float,
        *,
        options: GenerationOptions | None = None,
    ) -> VideoClip:
        raise ProviderError(f"Provider '{self.name}' does not implement generate().")

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
        raise ProviderError(f"Provider '{self.name}' does not implement transfer().")

    def reason(self, query: str, *, world_state: JSONDict | None = None) -> ReasoningResult:
        raise ProviderError(f"Provider '{self.name}' does not implement reason().")

    def embed(self, *, text: str) -> EmbeddingResult:
        raise ProviderError(f"Provider '{self.name}' does not implement embed().")

    def score_actions(self, *, info: JSONDict, action_candidates: object) -> ActionScoreResult:
        raise ProviderError(f"Provider '{self.name}' does not implement score_actions().")

    def select_actions(self, *, info: JSONDict) -> ActionPolicyResult:
        raise ProviderError(f"Provider '{self.name}' does not implement select_actions().")


class RemoteProvider(BaseProvider):
    """Base class for providers that depend on third-party credentials."""

    def _require_credentials(self) -> None:
        if not self.configured():
            raise ProviderError(f"Provider '{self.name}' is unavailable: missing {self.env_var}.")

    def _require_request_policy(self) -> ProviderRequestPolicy:
        if self.request_policy is None:
            raise ProviderError(f"Provider '{self.name}' does not define a request policy.")
        return self.request_policy
