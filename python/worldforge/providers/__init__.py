"""Provider implementations and registration primitives."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from time import perf_counter
from typing import Any

from worldforge._core import (
    Action,
    BBox,
    EmbeddingResult,
    Position,
    ProviderCapabilities,
    ProviderHealth,
    ProviderInfo,
    ReasoningResult,
    SceneObject,
    VideoClip,
    deterministic_floats,
)


JsonDict = dict[str, Any]


def _frame_bytes(seed: str, index: int) -> bytes:
    return f"{seed}:frame:{index}".encode("utf-8")


class ProviderError(RuntimeError):
    """Raised when a provider cannot satisfy a request."""


@dataclass(slots=True)
class PredictionPayload:
    """Serialized prediction payload returned by providers."""

    state: JsonDict
    confidence: float
    physics_score: float
    frames: list[bytes]
    metadata: JsonDict
    latency_ms: float


class BaseProvider:
    """Base class for pure-Python WorldForge providers."""

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
        healthy = self.configured()
        details = "configured" if healthy else f"missing {self.env_var}"
        return ProviderHealth(
            name=self.name,
            healthy=healthy,
            latency_ms=1.0 if healthy else 0.0,
            details=details,
        )

    def predict(self, world_state: JsonDict, action: Action, steps: int) -> PredictionPayload:
        raise NotImplementedError

    def generate(self, prompt: str, duration_seconds: float) -> VideoClip:
        raise ProviderError(f"Provider '{self.name}' does not implement generate().")

    def transfer(self, clip: VideoClip, *, width: int, height: int, fps: float) -> VideoClip:
        raise ProviderError(f"Provider '{self.name}' does not implement transfer().")

    def reason(self, query: str, *, world_state: JsonDict | None = None) -> ReasoningResult:
        raise ProviderError(f"Provider '{self.name}' does not implement reason().")

    def embed(self, *, text: str) -> EmbeddingResult:
        raise ProviderError(f"Provider '{self.name}' does not implement embed().")


class MockProvider(BaseProvider):
    """Deterministic local provider used for tests and offline workflows."""

    def __init__(self, name: str = "mock") -> None:
        super().__init__(
            name=name,
            capabilities=ProviderCapabilities(
                predict=True,
                generate=True,
                reason=True,
                embed=True,
                plan=True,
                transfer=True,
                verify=True,
            ),
            is_local=True,
            description="Deterministic offline provider for development, testing, and examples.",
        )

    def _updated_world_state(self, world_state: JsonDict, action: Action, steps: int) -> JsonDict:
        updated = deepcopy(world_state)
        scene_objects = updated.setdefault("scene", {}).setdefault("objects", {})
        action_data = action.to_dict()

        if action.kind == "move_to" and scene_objects:
            target = action.parameters["target"]
            first_object_id = next(iter(scene_objects))
            obj = scene_objects[first_object_id]
            obj["pose"]["position"] = dict(target)
            obj.setdefault("metadata", {})["last_action"] = action_data
            obj["metadata"]["moved_by_provider"] = self.name
        elif action.kind == "spawn_object":
            obj = SceneObject(
                name=str(action.parameters["name"]),
                position=Position.from_dict(action.parameters["position"]),
                bbox=BBox.from_dict(action.parameters["bbox"]),
            )
            scene_objects[obj.id] = obj.to_dict()
        elif action.kind == "noop":
            updated.setdefault("metadata", {})["noop"] = True

        updated["step"] = int(updated.get("step", 0)) + max(1, int(steps))
        updated.setdefault("metadata", {})["provider"] = self.name
        updated["metadata"]["last_action"] = action_data
        return updated

    def predict(self, world_state: JsonDict, action: Action, steps: int) -> PredictionPayload:
        started = perf_counter()
        updated_state = self._updated_world_state(world_state, action, steps)
        object_count = len(updated_state.get("scene", {}).get("objects", {}))
        physics_score = max(0.6, min(0.99, 0.72 + (0.03 * object_count)))
        confidence = max(0.65, min(0.99, physics_score - 0.02))
        frame_count = max(1, steps)
        frames = [_frame_bytes(self.name, index) for index in range(frame_count)]
        latency_ms = max(0.1, (perf_counter() - started) * 1000)
        metadata = {
            "provider": self.name,
            "steps": steps,
            "frame_count": frame_count,
            "mode": "deterministic-mock",
        }
        return PredictionPayload(
            state=updated_state,
            confidence=confidence,
            physics_score=physics_score,
            frames=frames,
            metadata=metadata,
            latency_ms=latency_ms,
        )

    def generate(self, prompt: str, duration_seconds: float) -> VideoClip:
        frame_count = max(1, int(round(duration_seconds * 8)))
        return VideoClip(
            frames=[_frame_bytes(prompt, index) for index in range(frame_count)],
            fps=8.0,
            resolution=(640, 360),
            duration_seconds=duration_seconds,
            metadata={"provider": self.name, "prompt": prompt},
        )

    def transfer(self, clip: VideoClip, *, width: int, height: int, fps: float) -> VideoClip:
        return VideoClip(
            frames=list(clip.frames),
            fps=fps,
            resolution=(width, height),
            duration_seconds=clip.duration_seconds,
            metadata={**clip.metadata, "provider": self.name, "transfer": True},
        )

    def reason(self, query: str, *, world_state: JsonDict | None = None) -> ReasoningResult:
        objects = world_state.get("scene", {}).get("objects", {}) if world_state else {}
        answer = f"{len(objects)} object(s) tracked. Query: {query}"
        evidence = [f"Observed object ids: {', '.join(objects) or 'none'}"]
        return ReasoningResult(
            provider=self.name,
            answer=answer,
            confidence=0.81,
            evidence=evidence,
        )

    def embed(self, *, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            provider=self.name,
            model="mock-embedding-v1",
            vector=deterministic_floats(f"{self.name}:{text}", 32),
        )


class RemoteProvider(BaseProvider):
    """Base class for providers that depend on third-party credentials."""

    def predict(self, world_state: JsonDict, action: Action, steps: int) -> PredictionPayload:
        if not self.configured():
            raise ProviderError(f"Provider '{self.name}' is unavailable: missing {self.env_var}.")
        fallback = MockProvider(name=self.name)
        payload = fallback.predict(world_state, action, steps)
        payload.metadata["mode"] = "python-stub"
        payload.metadata["credential_env"] = self.env_var
        return payload

    def generate(self, prompt: str, duration_seconds: float) -> VideoClip:
        if not self.configured():
            raise ProviderError(f"Provider '{self.name}' is unavailable: missing {self.env_var}.")
        clip = MockProvider(name=self.name).generate(prompt, duration_seconds)
        clip.metadata["mode"] = "python-stub"
        clip.metadata["credential_env"] = self.env_var
        return clip

    def transfer(self, clip: VideoClip, *, width: int, height: int, fps: float) -> VideoClip:
        if not self.configured():
            raise ProviderError(f"Provider '{self.name}' is unavailable: missing {self.env_var}.")
        transferred = MockProvider(name=self.name).transfer(clip, width=width, height=height, fps=fps)
        transferred.metadata["mode"] = "python-stub"
        transferred.metadata["credential_env"] = self.env_var
        return transferred

    def reason(self, query: str, *, world_state: JsonDict | None = None) -> ReasoningResult:
        if not self.configured():
            raise ProviderError(f"Provider '{self.name}' is unavailable: missing {self.env_var}.")
        result = MockProvider(name=self.name).reason(query, world_state=world_state)
        result.evidence.append(f"Stubbed remote execution via env var {self.env_var}")
        return result

    def embed(self, *, text: str) -> EmbeddingResult:
        if not self.configured():
            raise ProviderError(f"Provider '{self.name}' is unavailable: missing {self.env_var}.")
        return MockProvider(name=self.name).embed(text=text)


class CosmosProvider(RemoteProvider):
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
            description="Python-first adapter surface for NVIDIA Cosmos APIs.",
        )


class RunwayProvider(RemoteProvider):
    """Python adapter placeholder for Runway world model APIs."""

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
            description="Python-first adapter surface for Runway generation APIs.",
        )


class JepaProvider(RemoteProvider):
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
            description="Python-first adapter surface for JEPA-family local or hosted models.",
        )


class GenieProvider(RemoteProvider):
    """Python adapter placeholder for Genie-style providers."""

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
            description="Python-first adapter surface for Genie-style world models.",
        )


__all__ = [
    "BaseProvider",
    "CosmosProvider",
    "GenieProvider",
    "JepaProvider",
    "MockProvider",
    "PredictionPayload",
    "ProviderError",
    "RunwayProvider",
]
