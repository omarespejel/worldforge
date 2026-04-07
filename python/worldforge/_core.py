"""Core domain types for the pure-Python WorldForge runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from typing import Any, Iterable
from uuid import uuid4


JsonDict = dict[str, Any]


def generate_id(prefix: str) -> str:
    """Return a stable-looking opaque identifier with the given prefix."""

    return f"{prefix}_{uuid4().hex[:12]}"


def json_dumps(payload: Any) -> str:
    """Serialize payloads deterministically for persistence and verification."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def json_hash(payload: Any) -> str:
    """Hash any JSON-serializable payload."""

    return sha256(json_dumps(payload).encode("utf-8")).hexdigest()


def deterministic_floats(seed: str, size: int) -> list[float]:
    """Generate deterministic floats in [0, 1) from a text seed."""

    digest = sha256(seed.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < size:
        block = sha256(digest + counter.to_bytes(4, "big")).digest()
        for idx in range(0, len(block), 4):
            if len(values) >= size:
                break
            chunk = int.from_bytes(block[idx : idx + 4], "big")
            values.append((chunk % 10_000) / 10_000.0)
        counter += 1
    return values


def ensure_directory(path: str) -> None:
    """Create a directory tree if it does not already exist."""

    os.makedirs(path, exist_ok=True)


@dataclass(slots=True)
class Position:
    """A 3D position in world coordinates."""

    x: float
    y: float
    z: float

    def to_dict(self) -> JsonDict:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Position":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data["z"]),
        )

    def distance_to(self, other: "Position") -> float:
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))


@dataclass(slots=True)
class Rotation:
    """A quaternion rotation."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> JsonDict:
        return {"w": self.w, "x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, data: JsonDict | None) -> "Rotation":
        if data is None:
            return cls()
        return cls(
            w=float(data.get("w", 1.0)),
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            z=float(data.get("z", 0.0)),
        )


@dataclass(slots=True)
class Pose:
    """A 6DoF pose."""

    position: Position
    rotation: Rotation = field(default_factory=Rotation)

    def to_dict(self) -> JsonDict:
        return {"position": self.position.to_dict(), "rotation": self.rotation.to_dict()}

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Pose":
        return cls(
            position=Position.from_dict(data["position"]),
            rotation=Rotation.from_dict(data.get("rotation")),
        )


@dataclass(slots=True)
class BBox:
    """Axis-aligned bounding box."""

    min: Position
    max: Position

    def to_dict(self) -> JsonDict:
        return {"min": self.min.to_dict(), "max": self.max.to_dict()}

    @classmethod
    def from_dict(cls, data: JsonDict) -> "BBox":
        return cls(min=Position.from_dict(data["min"]), max=Position.from_dict(data["max"]))


@dataclass(slots=True)
class Action:
    """A structured action applied to a world."""

    kind: str
    parameters: JsonDict = field(default_factory=dict)

    @staticmethod
    def move_to(x: float, y: float, z: float, speed: float = 1.0) -> "Action":
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
    ) -> "Action":
        position = position or Position(0.0, 0.5, 0.0)
        bbox = bbox or BBox(
            Position(position.x - 0.05, position.y - 0.05, position.z - 0.05),
            Position(position.x + 0.05, position.y + 0.05, position.z + 0.05),
        )
        return Action(
            "spawn_object",
            {
                "name": name,
                "position": position.to_dict(),
                "bbox": bbox.to_dict(),
            },
        )

    @staticmethod
    def from_dict(data: JsonDict) -> "Action":
        if "type" in data:
            return Action(str(data["type"]), dict(data.get("parameters", {})))
        if len(data) != 1:
            raise ValueError("Action.from_dict expects either {'type': ...} or a single-key mapping.")
        kind, params = next(iter(data.items()))
        return Action(str(kind), dict(params))

    def to_dict(self) -> JsonDict:
        return {"type": self.kind, "parameters": dict(self.parameters)}

    def to_json(self) -> str:
        return json_dumps(self.to_dict())


@dataclass(slots=True)
class SceneObjectPatch:
    """Partial update for a scene object."""

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
    metadata: JsonDict = field(default_factory=dict)

    @property
    def pose(self) -> Pose:
        return Pose(position=self.position)

    def copy(self) -> "SceneObject":
        return SceneObject.from_dict(self.to_dict())

    def apply_patch(self, patch: SceneObjectPatch) -> None:
        if patch.name is not None:
            self.name = patch.name
        if patch.position is not None:
            self.position = patch.position
        if patch.graspable is not None:
            self.is_graspable = patch.graspable

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "name": self.name,
            "pose": self.pose.to_dict(),
            "bbox": self.bbox.to_dict(),
            "is_graspable": self.is_graspable,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "SceneObject":
        pose = Pose.from_dict(data["pose"]) if "pose" in data else Pose(Position.from_dict(data["position"]))
        return cls(
            id=str(data.get("id") or generate_id("obj")),
            name=str(data["name"]),
            position=pose.position,
            bbox=BBox.from_dict(data["bbox"]),
            is_graspable=bool(data.get("is_graspable", False)),
            metadata=dict(data.get("metadata", {})),
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
    verify: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "predict": self.predict,
            "generate": self.generate,
            "reason": self.reason,
            "embed": self.embed,
            "plan": self.plan,
            "transfer": self.transfer,
            "verify": self.verify,
        }

    def supports(self, capability: str) -> bool:
        return bool(getattr(self, capability, False))

    def enabled_names(self) -> list[str]:
        return [name for name, value in self.to_dict().items() if value]


@dataclass(slots=True)
class ProviderInfo:
    """Public provider metadata."""

    name: str
    capabilities: ProviderCapabilities
    is_local: bool
    description: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "capabilities": self.capabilities.to_dict(),
            "is_local": self.is_local,
            "description": self.description,
        }


@dataclass(slots=True)
class ProviderHealth:
    """A lightweight provider health report."""

    name: str
    healthy: bool
    latency_ms: float
    details: str = ""

    def to_dict(self) -> JsonDict:
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
    metadata: JsonDict = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def to_dict(self) -> JsonDict:
        return {
            "fps": self.fps,
            "resolution": list(self.resolution),
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
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
    """A recorded state transition inside a world."""

    step: int
    state: JsonDict
    summary: str
    action_json: str | None = None

    def to_dict(self) -> JsonDict:
        return {
            "step": self.step,
            "state": self.state,
            "summary": self.summary,
            "action_json": self.action_json,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "HistoryEntry":
        return cls(
            step=int(data["step"]),
            state=dict(data["state"]),
            summary=str(data.get("summary", "")),
            action_json=data.get("action_json"),
        )


def average(values: Iterable[float]) -> float:
    """Return the arithmetic mean for a non-empty iterable."""

    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
