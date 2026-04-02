"""Type stubs for WorldForge Python bindings."""

from typing import Any, Sequence

class Action:
    """Describes an action to apply to a world."""

    @staticmethod
    def move_to(x: float, y: float, z: float) -> Action: ...
    @staticmethod
    def from_dict(data: dict[str, Any]) -> Action: ...
    def to_dict(self) -> dict[str, Any]: ...

class ProviderInfo:
    """Information about a registered provider."""

    name: str
    capabilities: list[str]
    is_local: bool

class Prediction:
    """Result of a prediction call."""

    physics_score: float
    frames: list[bytes]
    world_state: dict[str, Any]
    metadata: dict[str, Any]

class Plan:
    """Result of a planning call."""

    actions: list[Action]
    success_probability: float
    verification_proof: bytes | None

class EvalReport:
    """Result of an evaluation run."""

    scores: dict[str, float]
    suite: str
    provider: str

    def to_markdown(self) -> str: ...
    def to_csv(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

class Comparison:
    """Result of a cross-provider comparison."""

    results: list[ComparisonResult]

    def to_markdown(self) -> str: ...

class ComparisonResult:
    """Single provider result within a comparison."""

    provider: str
    physics_score: float
    latency_ms: float
    metadata: dict[str, Any]

class EvalDimension:
    """Evaluation dimension identifiers."""

    OBJECT_PERMANENCE: str
    GRAVITY_COMPLIANCE: str
    COLLISION_ACCURACY: str
    SPATIAL_CONSISTENCY: str
    TEMPORAL_CONSISTENCY: str
    ACTION_PREDICTION: str
    MATERIAL_UNDERSTANDING: str
    SPATIAL_REASONING: str
    ACTION_SIMULATION_FIDELITY: str
    TRANSITION_SMOOTHNESS: str
    GENERATION_CONSISTENCY: str
    SIMULATIVE_REASONING: str

class EvalSuite:
    """Custom evaluation suite definition."""

    def __init__(
        self,
        name: str,
        dimensions: Sequence[str],
    ) -> None: ...

class World:
    """Represents a simulation world bound to a provider."""

    name: str
    provider: str

    def predict(self, action: Action, steps: int = 1) -> Prediction: ...
    def plan(
        self,
        goal: str,
        planner: str = "cem",
        max_steps: int = 20,
    ) -> Plan: ...
    def compare(
        self,
        action: Action,
        providers: Sequence[str],
        steps: int = 1,
    ) -> Comparison: ...
    def evaluate(self, suite: str | EvalSuite = "physics") -> EvalReport: ...

class WorldForge:
    """Main entry point for the WorldForge SDK."""

    def __init__(
        self,
        state_backend: str | None = None,
        state_db_path: str | None = None,
    ) -> None: ...
    def create_world(self, name: str, provider: str = "mock") -> World: ...
    def list_providers(self) -> list[ProviderInfo]: ...
    def get_provider(self, name: str) -> ProviderInfo: ...
