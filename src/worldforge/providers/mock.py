"""Deterministic local provider used for development and tests."""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter

from worldforge.models import (
    Action,
    BBox,
    EmbeddingResult,
    GenerationOptions,
    JSONDict,
    Position,
    ProviderCapabilities,
    ReasoningResult,
    SceneObject,
    VideoClip,
    deterministic_floats,
)

from .base import BaseProvider, PredictionPayload


def _frame_bytes(seed: str, index: int) -> bytes:
    return f"{seed}:frame:{index}".encode()


class MockProvider(BaseProvider):
    """Deterministic provider for offline development, examples, and tests."""

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
            ),
            is_local=True,
            description=(
                "Deterministic local provider for examples, tests, and framework development."
            ),
            package="worldforge",
            implementation_status="stable",
            deterministic=True,
            supported_modalities=["world_state", "text", "video"],
            artifact_types=["prediction", "video", "reasoning", "embedding", "transfer"],
            notes=[
                "Reference implementation for local development and adapter contract tests.",
            ],
            default_model="mock-deterministic-v1",
            supported_models=["mock-deterministic-v1"],
        )

    def _updated_world_state(self, world_state: JSONDict, action: Action, steps: int) -> JSONDict:
        updated = deepcopy(world_state)
        scene_objects = updated.setdefault("scene", {}).setdefault("objects", {})
        action_data = action.to_dict()

        if action.kind == "move_to" and scene_objects:
            target = action.parameters["target"]
            first_object_id = next(iter(scene_objects))
            scene_object = scene_objects[first_object_id]
            scene_object["pose"]["position"] = dict(target)
            scene_object.setdefault("metadata", {})["last_action"] = action_data
            scene_object["metadata"]["moved_by_provider"] = self.name
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

    def predict(self, world_state: JSONDict, action: Action, steps: int) -> PredictionPayload:
        started = perf_counter()
        updated_state = self._updated_world_state(world_state, action, steps)
        object_count = len(updated_state.get("scene", {}).get("objects", {}))
        physics_score = max(0.6, min(0.99, 0.72 + (0.03 * object_count)))
        confidence = max(0.65, min(0.99, physics_score - 0.02))
        frame_count = max(1, steps)
        return PredictionPayload(
            state=updated_state,
            confidence=confidence,
            physics_score=physics_score,
            frames=[_frame_bytes(self.name, index) for index in range(frame_count)],
            metadata={
                "provider": self.name,
                "steps": steps,
                "frame_count": frame_count,
                "mode": "deterministic-mock",
            },
            latency_ms=max(0.1, (perf_counter() - started) * 1000),
        )

    def generate(
        self,
        prompt: str,
        duration_seconds: float,
        *,
        options: GenerationOptions | None = None,
    ) -> VideoClip:
        frame_count = max(1, int(round(duration_seconds * 8)))
        return VideoClip(
            frames=[_frame_bytes(prompt, index) for index in range(frame_count)],
            fps=8.0,
            resolution=(640, 360),
            duration_seconds=duration_seconds,
            metadata={
                "provider": self.name,
                "prompt": prompt,
                "mode": "deterministic-mock-generate",
                "options": options.to_dict() if options else {},
                "content_type": "application/octet-stream",
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
        return VideoClip(
            frames=list(clip.frames),
            fps=fps,
            resolution=(width, height),
            duration_seconds=clip.duration_seconds,
            metadata={
                **clip.metadata,
                "provider": self.name,
                "transfer": True,
                "prompt": prompt,
                "options": options.to_dict() if options else {},
            },
        )

    def reason(self, query: str, *, world_state: JSONDict | None = None) -> ReasoningResult:
        objects = world_state.get("scene", {}).get("objects", {}) if world_state else {}
        return ReasoningResult(
            provider=self.name,
            answer=f"{len(objects)} object(s) tracked. Query: {query}",
            confidence=0.81,
            evidence=[f"Observed object ids: {', '.join(objects) or 'none'}"],
        )

    def embed(self, *, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            provider=self.name,
            model="mock-embedding-v1",
            vector=deterministic_floats(f"{self.name}:{text}", 32),
        )
