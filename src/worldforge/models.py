"""Core models and serialization helpers for WorldForge."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

JSONDict = dict[str, Any]


def generate_id(prefix: str) -> str:
    """Return an opaque identifier with a stable prefix."""

    return f"{prefix}_{uuid4().hex[:12]}"


def dump_json(payload: Any) -> str:
    """Serialize data with deterministic formatting."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def ensure_directory(path: Path) -> None:
    """Create a directory tree when it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def average(values: Iterable[float]) -> float:
    """Return the arithmetic mean for a non-empty iterable."""

    numbers = list(values)
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def deterministic_floats(seed: str, size: int) -> list[float]:
    """Generate deterministic floats in the range [0, 1)."""

    digest = sha256(seed.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < size:
        block = sha256(digest + counter.to_bytes(4, "big")).digest()
        for index in range(0, len(block), 4):
            if len(values) >= size:
                break
            chunk = int.from_bytes(block[index : index + 4], "big")
            values.append((chunk % 10_000) / 10_000.0)
        counter += 1
    return values


@dataclass(slots=True, frozen=True)
class Position:
    """A 3D position in world coordinates."""

    x: float
    y: float
    z: float

    def to_dict(self) -> JSONDict:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, payload: JSONDict) -> Position:
        return cls(
            x=float(payload["x"]),
            y=float(payload["y"]),
            z=float(payload["z"]),
        )

    def distance_to(self, other: Position) -> float:
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))


@dataclass(slots=True, frozen=True)
class Rotation:
    """A quaternion rotation."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> JSONDict:
        return {"w": self.w, "x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, payload: JSONDict | None) -> Rotation:
        if payload is None:
            return cls()
        return cls(
            w=float(payload.get("w", 1.0)),
            x=float(payload.get("x", 0.0)),
            y=float(payload.get("y", 0.0)),
            z=float(payload.get("z", 0.0)),
        )


@dataclass(slots=True, frozen=True)
class Pose:
    """A 6DoF pose."""

    position: Position
    rotation: Rotation = field(default_factory=Rotation)

    def to_dict(self) -> JSONDict:
        return {"position": self.position.to_dict(), "rotation": self.rotation.to_dict()}

    @classmethod
    def from_dict(cls, payload: JSONDict) -> Pose:
        return cls(
            position=Position.from_dict(payload["position"]),
            rotation=Rotation.from_dict(payload.get("rotation")),
        )


@dataclass(slots=True, frozen=True)
class BBox:
    """Axis-aligned bounding box."""

    min: Position
    max: Position

    def to_dict(self) -> JSONDict:
        return {"min": self.min.to_dict(), "max": self.max.to_dict()}

    @classmethod
    def from_dict(cls, payload: JSONDict) -> BBox:
        return cls(
            min=Position.from_dict(payload["min"]),
            max=Position.from_dict(payload["max"]),
        )


@dataclass(slots=True)
class Action:
    """A structured action applied to a world."""

    kind: str
    parameters: JSONDict = field(default_factory=dict)

    @staticmethod
    def move_to(x: float, y: float, z: float, speed: float = 1.0) -> Action:
        return Action(
            "move_to",
            {
                "target": {"x": float(x), "y": float(y), "z": float(z)},
                "speed": float(speed),
            },
        )

    @staticmethod
    def spawn_object(
        name: str,
        position: Position | None = None,
        bbox: BBox | None = None,
    ) -> Action:
        object_position = position or Position(0.0, 0.5, 0.0)
        object_bbox = bbox or BBox(
            Position(object_position.x - 0.05, object_position.y - 0.05, object_position.z - 0.05),
            Position(object_position.x + 0.05, object_position.y + 0.05, object_position.z + 0.05),
        )
        return Action(
            "spawn_object",
            {
                "name": name,
                "position": object_position.to_dict(),
                "bbox": object_bbox.to_dict(),
            },
        )

    @staticmethod
    def from_dict(payload: JSONDict) -> Action:
        if "type" in payload:
            return Action(str(payload["type"]), dict(payload.get("parameters", {})))
        if len(payload) != 1:
            raise ValueError("Action.from_dict expects {'type': ...} or a single-key mapping.")
        kind, parameters = next(iter(payload.items()))
        return Action(str(kind), dict(parameters))

    def to_dict(self) -> JSONDict:
        return {"type": self.kind, "parameters": dict(self.parameters)}

    def to_json(self) -> str:
        return dump_json(self.to_dict())


@dataclass(slots=True)
class SceneObjectPatch:
    """Partial mutation for a scene object."""

    name: str | None = None
    position: Position | None = None
    graspable: bool | None = None

    def set_name(self, name: str) -> None:
        self.name = name

    def set_position(self, position: Position) -> None:
        self.position = position

    def set_graspable(self, value: bool) -> None:
        self.graspable = bool(value)


@dataclass(slots=True)
class SceneObject:
    """An object tracked in the scene graph."""

    name: str
    position: Position
    bbox: BBox
    id: str = field(default_factory=lambda: generate_id("obj"))
    is_graspable: bool = False
    metadata: JSONDict = field(default_factory=dict)

    @property
    def pose(self) -> Pose:
        return Pose(position=self.position)

    def copy(self) -> SceneObject:
        return SceneObject.from_dict(self.to_dict())

    def apply_patch(self, patch: SceneObjectPatch) -> None:
        if patch.name is not None:
            self.name = patch.name
        if patch.position is not None:
            self.position = patch.position
        if patch.graspable is not None:
            self.is_graspable = patch.graspable

    def to_dict(self) -> JSONDict:
        return {
            "id": self.id,
            "name": self.name,
            "pose": self.pose.to_dict(),
            "bbox": self.bbox.to_dict(),
            "is_graspable": self.is_graspable,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: JSONDict) -> SceneObject:
        pose = (
            Pose.from_dict(payload["pose"])
            if "pose" in payload
            else Pose(Position.from_dict(payload["position"]))
        )
        return cls(
            id=str(payload.get("id") or generate_id("obj")),
            name=str(payload["name"]),
            position=pose.position,
            bbox=BBox.from_dict(payload["bbox"]),
            is_graspable=bool(payload.get("is_graspable", False)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ProviderCapabilities:
    """Boolean capability matrix for a provider."""

    predict: bool = True
    generate: bool = False
    reason: bool = False
    embed: bool = False
    plan: bool = False
    transfer: bool = False

    def to_dict(self) -> JSONDict:
        return {
            "predict": self.predict,
            "generate": self.generate,
            "reason": self.reason,
            "embed": self.embed,
            "plan": self.plan,
            "transfer": self.transfer,
        }

    def supports(self, capability: str) -> bool:
        return bool(getattr(self, capability, False))

    def enabled_names(self) -> list[str]:
        return [name for name, enabled in self.to_dict().items() if enabled]


@dataclass(slots=True)
class ProviderInfo:
    """Public provider metadata."""

    name: str
    capabilities: ProviderCapabilities
    is_local: bool
    description: str = ""

    def to_dict(self) -> JSONDict:
        return {
            "name": self.name,
            "capabilities": self.capabilities.to_dict(),
            "is_local": self.is_local,
            "description": self.description,
        }


@dataclass(slots=True)
class ProviderHealth:
    """Health information for a provider."""

    name: str
    healthy: bool
    latency_ms: float
    details: str = ""

    def to_dict(self) -> JSONDict:
        return {
            "name": self.name,
            "healthy": self.healthy,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


@dataclass(slots=True)
class VideoClip:
    """Generated or transformed video content."""

    frames: list[bytes]
    fps: float
    resolution: tuple[int, int]
    duration_seconds: float
    metadata: JSONDict = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def to_dict(self) -> JSONDict:
        return {
            "frame_count": self.frame_count,
            "fps": self.fps,
            "resolution": list(self.resolution),
            "duration_seconds": self.duration_seconds,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ReasoningResult:
    """A provider answer about a scene or prompt."""

    provider: str
    answer: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EmbeddingResult:
    """Embedding output from a provider."""

    provider: str
    model: str
    vector: list[float]

    @property
    def shape(self) -> list[int]:
        return [len(self.vector)]


@dataclass(slots=True)
class HistoryEntry:
    """A recorded world snapshot."""

    step: int
    state: JSONDict
    summary: str
    action_json: str | None = None

    def to_dict(self) -> JSONDict:
        return {
            "step": self.step,
            "state": self.state,
            "summary": self.summary,
            "action_json": self.action_json,
        }

    @classmethod
    def from_dict(cls, payload: JSONDict) -> HistoryEntry:
        return cls(
            step=int(payload["step"]),
            state=dict(payload["state"]),
            summary=str(payload.get("summary", "")),
            action_json=payload.get("action_json"),
        )
