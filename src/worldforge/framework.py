"""Framework runtime objects for WorldForge."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldforge.models import (
    Action,
    BBox,
    DoctorReport,
    EmbeddingResult,
    HistoryEntry,
    JSONDict,
    Position,
    ProviderDoctorStatus,
    ProviderHealth,
    ProviderInfo,
    ProviderProfile,
    ReasoningResult,
    SceneObject,
    SceneObjectPatch,
    VideoClip,
    average,
    dump_json,
    ensure_directory,
    generate_id,
)
from worldforge.providers import (
    BaseProvider,
    CosmosProvider,
    GenieProvider,
    JepaProvider,
    MockProvider,
    ProviderError,
    RunwayProvider,
)

SCHEMA_VERSION = 1


def _clone_state(state: JSONDict) -> JSONDict:
    return deepcopy(state)


def _normalize_provider_name(provider: str | None, fallback: str) -> str:
    return provider or fallback


def _world_file(state_dir: Path, world_id: str) -> Path:
    return state_dir / f"{world_id}.json"


@dataclass(slots=True)
class Prediction:
    """Result of a world prediction."""

    provider: str
    confidence: float
    physics_score: float
    frames: list[bytes]
    world_state: JSONDict
    metadata: JSONDict
    latency_ms: float
    _forge: WorldForge

    def output_world(self) -> World:
        return World.from_state(self._forge, _clone_state(self.world_state))


class Comparison:
    """Result of comparing multiple predictions."""

    def __init__(self, predictions: Sequence[Prediction]) -> None:
        self.results = list(predictions)

    @property
    def prediction_count(self) -> int:
        return len(self.results)

    def best_prediction(self) -> Prediction:
        if not self.results:
            raise ValueError("Comparison has no predictions.")
        return max(self.results, key=lambda item: (item.physics_score, item.confidence))

    def to_markdown(self) -> str:
        lines = [
            "# WorldForge Comparison",
            "",
            "| provider | physics_score | confidence | latency_ms |",
            "| --- | ---: | ---: | ---: |",
        ]
        for result in self.results:
            lines.append(
                f"| {result.provider} | {result.physics_score:.2f} | "
                f"{result.confidence:.2f} | {result.latency_ms:.2f} |"
            )
        return "\n".join(lines)

    def to_csv(self) -> str:
        rows = ["provider,physics_score,confidence,latency_ms"]
        for result in self.results:
            rows.append(
                f"{result.provider},{result.physics_score:.4f},"
                f"{result.confidence:.4f},{result.latency_ms:.4f}"
            )
        return "\n".join(rows)

    def to_json(self) -> str:
        return dump_json(
            {
                "predictions": [
                    {
                        "provider": result.provider,
                        "physics_score": result.physics_score,
                        "confidence": result.confidence,
                        "latency_ms": result.latency_ms,
                        "metadata": result.metadata,
                    }
                    for result in self.results
                ]
            }
        )

    def artifacts(self) -> dict[str, str]:
        return {
            "json": self.to_json(),
            "markdown": self.to_markdown(),
            "csv": self.to_csv(),
        }


class Plan:
    """Deterministic multi-step execution plan."""

    def __init__(
        self,
        *,
        goal: str,
        planner: str,
        provider: str,
        actions: Sequence[Action],
        predicted_states: Sequence[JSONDict],
        success_probability: float,
    ) -> None:
        self.goal = goal
        self.planner = planner
        self.provider = provider
        self.actions = list(actions)
        self.predicted_states = [_clone_state(state) for state in predicted_states]
        self.success_probability = success_probability

    @property
    def action_count(self) -> int:
        return len(self.actions)

    def to_dict(self) -> JSONDict:
        return {
            "goal": self.goal,
            "planner": self.planner,
            "provider": self.provider,
            "actions": [action.to_dict() for action in self.actions],
            "action_count": self.action_count,
            "success_probability": self.success_probability,
            "predicted_states": self.predicted_states,
        }

    def to_json(self) -> str:
        return dump_json(self.to_dict())


class PlanExecution:
    """Execution result for a plan."""

    def __init__(self, final_world: World, actions_applied: Sequence[Action]) -> None:
        self._final_world = final_world
        self.actions_applied = list(actions_applied)

    def final_world(self) -> World:
        return self._final_world


class World:
    """Mutable world state bound to a provider registry."""

    def __init__(
        self,
        name: str,
        provider: str = "mock",
        *,
        forge: WorldForge | None = None,
        description: str = "",
        world_id: str | None = None,
        metadata: JSONDict | None = None,
    ) -> None:
        self._forge = forge or WorldForge()
        self.id = world_id or generate_id("world")
        self.name = name
        self.provider = provider
        self.description = description
        self.step = 0
        self.metadata: JSONDict = metadata.copy() if metadata else {}
        self.metadata.setdefault("name", self.name)
        self.scene_objects: dict[str, SceneObject] = {}
        self._history: list[HistoryEntry] = []
        self._record_history(summary="world initialized", action=None)

    @classmethod
    def from_state(cls, forge: WorldForge, state: JSONDict) -> World:
        world = cls(
            name=str(state["name"]),
            provider=str(state["provider"]),
            forge=forge,
            description=str(state.get("description", "")),
            world_id=str(state["id"]),
            metadata=dict(state.get("metadata", {})),
        )
        world.step = int(state.get("step", 0))
        world.scene_objects = {
            object_id: SceneObject.from_dict(object_state)
            for object_id, object_state in state.get("scene", {}).get("objects", {}).items()
        }
        world._history = [HistoryEntry.from_dict(entry) for entry in state.get("history", [])] or [
            HistoryEntry(
                step=world.step, state=world._snapshot(), summary="world restored", action_json=None
            )
        ]
        return world

    @property
    def object_count(self) -> int:
        return len(self.scene_objects)

    @property
    def history_length(self) -> int:
        return len(self._history)

    def _snapshot(self) -> JSONDict:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "description": self.description,
            "step": self.step,
            "scene": {
                "objects": {
                    object_id: obj.to_dict() for object_id, obj in self.scene_objects.items()
                }
            },
            "metadata": dict(self.metadata),
        }

    def _apply_state(self, state: JSONDict, *, preserve_history: bool = False) -> None:
        self.id = str(state["id"])
        self.name = str(state["name"])
        self.provider = str(state["provider"])
        self.description = str(state.get("description", ""))
        self.step = int(state.get("step", 0))
        self.metadata = dict(state.get("metadata", {}))
        self.scene_objects = {
            object_id: SceneObject.from_dict(object_state)
            for object_id, object_state in state.get("scene", {}).get("objects", {}).items()
        }
        if not preserve_history:
            self._history = [
                HistoryEntry.from_dict(entry) for entry in state.get("history", [])
            ] or [
                HistoryEntry(
                    step=self.step,
                    state=self._snapshot(),
                    summary="world restored",
                    action_json=None,
                )
            ]

    def _record_history(self, *, summary: str, action: Action | None) -> None:
        self._history.append(
            HistoryEntry(
                step=self.step,
                state=self._snapshot(),
                summary=summary,
                action_json=action.to_json() if action else None,
            )
        )

    def to_dict(self) -> JSONDict:
        state = self._snapshot()
        state["history"] = [entry.to_dict() for entry in self._history]
        return state

    def to_json(self) -> str:
        return dump_json(self.to_dict())

    def add_object(self, obj: SceneObject) -> SceneObject:
        self.scene_objects[obj.id] = obj.copy()
        self.metadata["name"] = self.name
        return self.scene_objects[obj.id]

    def list_objects(self) -> list[str]:
        return [obj.name for obj in self.scene_objects.values()]

    def objects(self) -> list[SceneObject]:
        return [obj.copy() for obj in self.scene_objects.values()]

    def get_object_by_id(self, object_id: str) -> SceneObject | None:
        scene_object = self.scene_objects.get(object_id)
        return scene_object.copy() if scene_object else None

    def update_object_patch(self, object_id: str, patch: SceneObjectPatch) -> SceneObject:
        scene_object = self.scene_objects[object_id]
        scene_object.apply_patch(patch)
        return scene_object.copy()

    def remove_object_by_id(self, object_id: str) -> SceneObject | None:
        removed = self.scene_objects.pop(object_id, None)
        return removed.copy() if removed else None

    def history(self) -> list[HistoryEntry]:
        return [HistoryEntry.from_dict(entry.to_dict()) for entry in self._history]

    def history_state(self, index: int) -> World:
        entry = self._history[index]
        state = _clone_state(entry.state)
        state["history"] = [item.to_dict() for item in self._history[: index + 1]]
        return World.from_state(self._forge, state)

    def restore_history(self, index: int) -> None:
        restored = self.history_state(index)
        self._apply_state(restored.to_dict())

    def _provider(self, provider_name: str | None = None) -> BaseProvider:
        return self._forge._require_provider(_normalize_provider_name(provider_name, self.provider))

    def predict(self, action: Action, steps: int = 1, provider: str | None = None) -> Prediction:
        selected_provider = _normalize_provider_name(provider, self.provider)
        payload = self._provider(selected_provider).predict(self._snapshot(), action, steps)
        next_state = _clone_state(payload.state)
        self._apply_state(next_state, preserve_history=True)
        self.provider = selected_provider
        self.metadata["name"] = self.name
        self._record_history(summary=f"predicted via {selected_provider}", action=action)
        return Prediction(
            provider=selected_provider,
            confidence=payload.confidence,
            physics_score=payload.physics_score,
            frames=list(payload.frames),
            world_state=self._snapshot(),
            metadata=dict(payload.metadata),
            latency_ms=payload.latency_ms,
            _forge=self._forge,
        )

    def compare(self, action: Action, providers: Sequence[str], steps: int = 1) -> Comparison:
        state = self._snapshot()
        predictions: list[Prediction] = []
        for provider_name in providers:
            payload = self._provider(provider_name).predict(state, action, steps)
            predictions.append(
                Prediction(
                    provider=provider_name,
                    confidence=payload.confidence,
                    physics_score=payload.physics_score,
                    frames=list(payload.frames),
                    world_state=_clone_state(payload.state),
                    metadata=dict(payload.metadata),
                    latency_ms=payload.latency_ms,
                    _forge=self._forge,
                )
            )
        return Comparison(predictions)

    def _goal_actions(self, goal: str, goal_json: str | None = None) -> list[Action]:
        if goal_json:
            payload = json.loads(goal_json)
            object_at = payload["condition"]["ObjectAt"]
            position = object_at["position"]
            return [
                Action.move_to(
                    float(position["x"]),
                    float(position["y"]),
                    float(position["z"]),
                )
            ]

        lowered = goal.lower()
        if "spawn" in lowered:
            object_name = "cube"
            for candidate in ("cube", "ball", "block", "mug"):
                if candidate in lowered:
                    object_name = candidate
                    break
            return [Action.spawn_object(object_name)]

        if self.scene_objects:
            primary = next(iter(self.scene_objects.values()))
            target = primary.position
            if "right" in lowered:
                return [Action.move_to(target.x + 1.0, target.y, target.z)]
            if "dishwasher" in lowered:
                return [Action.move_to(target.x + 0.8, target.y, target.z - 0.4)]
            return [Action.move_to(target.x + 0.3, target.y, target.z)]

        return [Action.spawn_object("cube")]

    def plan(
        self,
        goal: str | None = None,
        *,
        goal_json: str | None = None,
        planner: str = "cem",
        max_steps: int = 20,
        provider: str | None = None,
        **_: Any,
    ) -> Plan:
        selected_provider = _normalize_provider_name(provider, self.provider)
        resolved_goal = goal or "goal_json_plan"
        actions = self._goal_actions(resolved_goal, goal_json=goal_json)
        actions = actions[: max(1, min(max_steps, len(actions)))]

        simulated_state = self._snapshot()
        predicted_states: list[JSONDict] = []
        scores: list[float] = []
        for action in actions:
            payload = self._provider(selected_provider).predict(simulated_state, action, 1)
            simulated_state = _clone_state(payload.state)
            predicted_states.append(simulated_state)
            scores.append(payload.physics_score)

        return Plan(
            goal=resolved_goal,
            planner=planner,
            provider=selected_provider,
            actions=actions,
            predicted_states=predicted_states,
            success_probability=max(0.65, min(0.98, average(scores) if scores else 0.7)),
        )

    def execute_plan(self, plan: Plan, *_: Any) -> PlanExecution:
        executed_world = World.from_state(self._forge, self.to_dict())
        for action in plan.actions:
            executed_world.predict(action, steps=1, provider=plan.provider)
        return PlanExecution(executed_world, plan.actions)

    def evaluate(self, suite: str = "physics"):  # type: ignore[no-untyped-def]
        from worldforge.evaluation import EvaluationSuite

        return EvaluationSuite.from_builtin(suite).run_report(
            self.provider,
            world=self,
            forge=self._forge,
        )


class WorldForge:
    """Top-level entry point for provider orchestration and JSON persistence."""

    def __init__(
        self,
        *,
        state_dir: str | Path | None = None,
        auto_register_remote: bool = True,
    ) -> None:
        self.state_dir = Path(state_dir or ".worldforge/worlds").expanduser().resolve()
        ensure_directory(self.state_dir)
        self._providers: dict[str, BaseProvider] = {}
        self.register_provider(MockProvider())
        if auto_register_remote:
            for provider in (CosmosProvider(), RunwayProvider(), JepaProvider(), GenieProvider()):
                if provider.configured():
                    self.register_provider(provider)

    def _require_provider(self, name: str) -> BaseProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ProviderError(f"Provider '{name}' is not registered.") from exc

    def register_provider(self, provider: BaseProvider) -> None:
        self._providers[provider.name] = provider

    def providers(self) -> list[str]:
        return sorted(self._providers)

    def _provider_catalog(self, *, include_known: bool = True) -> dict[str, BaseProvider]:
        catalog: dict[str, BaseProvider] = {}
        if include_known:
            for provider in (
                MockProvider(),
                CosmosProvider(),
                RunwayProvider(),
                JepaProvider(),
                GenieProvider(),
            ):
                catalog[provider.name] = provider
        for provider in self._providers.values():
            catalog[provider.name] = provider
        return catalog

    def list_providers(self) -> list[ProviderInfo]:
        return [self._providers[name].info() for name in self.providers()]

    def list_provider_profiles(self) -> list[ProviderProfile]:
        return [self._providers[name].profile() for name in self.providers()]

    def builtin_provider_profiles(self) -> list[ProviderProfile]:
        catalog = self._provider_catalog(include_known=True)
        return [catalog[name].profile() for name in sorted(catalog)]

    def provider_info(self, name: str) -> ProviderInfo:
        return self._require_provider(name).info()

    def get_provider(self, name: str) -> ProviderInfo:
        return self.provider_info(name)

    def provider_profile(self, name: str) -> ProviderProfile:
        catalog = self._provider_catalog(include_known=True)
        try:
            provider = catalog[name]
        except KeyError as exc:
            raise ProviderError(f"Provider '{name}' is unknown.") from exc
        return provider.profile()

    def provider_health(self, name: str) -> ProviderHealth:
        catalog = self._provider_catalog(include_known=True)
        try:
            provider = catalog[name]
        except KeyError as exc:
            raise ProviderError(f"Provider '{name}' is unknown.") from exc
        return provider.health()

    def provider_healths(self, capability: str | None = None) -> list[ProviderHealth]:
        names = self.providers()
        if capability:
            names = [
                name for name in names if self._providers[name].capabilities.supports(capability)
            ]
        return [self._providers[name].health() for name in names]

    def doctor(
        self,
        capability: str | None = None,
        *,
        registered_only: bool = False,
    ) -> DoctorReport:
        catalog = self._provider_catalog(include_known=not registered_only)
        statuses: list[ProviderDoctorStatus] = []
        issues: list[str] = []

        for name in sorted(catalog):
            provider = catalog[name]
            profile = provider.profile()
            if capability and not profile.capabilities.supports(capability):
                continue
            health = provider.health()
            statuses.append(
                ProviderDoctorStatus(
                    registered=name in self._providers,
                    profile=profile,
                    health=health,
                )
            )
            if not health.healthy:
                if profile.requires_credentials and profile.credential_env_var:
                    issues.append(
                        f"Provider '{name}' is unavailable: missing {profile.credential_env_var}."
                    )
                else:
                    issues.append(f"Provider '{name}' is unhealthy: {health.details}.")

        return DoctorReport(
            state_dir=str(self.state_dir),
            world_count=len(self.list_worlds()),
            providers=statuses,
            issues=issues,
        )

    def create_world(self, name: str, provider: str = "mock", *, description: str = "") -> World:
        self._require_provider(provider)
        return World(name=name, provider=provider, forge=self, description=description)

    def create_world_from_prompt(
        self,
        prompt: str,
        *,
        provider: str = "mock",
        name: str | None = None,
    ) -> World:
        world = self.create_world(name or "prompt-world", provider, description=prompt)
        prompt_lower = prompt.lower()
        if "kitchen" in prompt_lower:
            world.add_object(
                SceneObject(
                    "countertop",
                    Position(0.0, 0.9, 0.0),
                    BBox(Position(-1.0, 0.85, -0.5), Position(1.0, 0.95, 0.5)),
                )
            )
        if "mug" in prompt_lower:
            world.add_object(
                SceneObject(
                    "mug",
                    Position(0.0, 0.8, 0.0),
                    BBox(Position(-0.05, 0.75, -0.05), Position(0.05, 0.85, 0.05)),
                    is_graspable=True,
                )
            )
        if not world.scene_objects:
            world.add_object(
                SceneObject(
                    "cube",
                    Position(0.0, 0.5, 0.0),
                    BBox(Position(-0.05, 0.45, -0.05), Position(0.05, 0.55, 0.05)),
                )
            )
        world._history = []
        world._record_history(summary="world seeded from prompt", action=None)
        return world

    def save_world(self, world: World) -> str:
        path = _world_file(self.state_dir, world.id)
        path.write_text(world.to_json(), encoding="utf-8")
        return world.id

    def load_world(self, world_id: str) -> World:
        path = _world_file(self.state_dir, world_id)
        return World.from_state(self, json.loads(path.read_text(encoding="utf-8")))

    def list_worlds(self) -> list[str]:
        return sorted(path.stem for path in self.state_dir.glob("*.json"))

    def export_world(self, world_id: str, *, format: str = "json") -> str:
        if format != "json":
            raise ValueError("Only json export is currently supported.")
        world = self.load_world(world_id)
        return dump_json({"schema_version": SCHEMA_VERSION, "state": world.to_dict()})

    def import_world(
        self,
        payload: str,
        *,
        format: str = "json",
        new_id: bool = False,
        name: str | None = None,
    ) -> World:
        if format != "json":
            raise ValueError("Only json import is currently supported.")
        data = json.loads(payload)
        state = dict(data["state"]) if "state" in data else dict(data)
        if new_id:
            state["id"] = generate_id("world")
        if name:
            state["name"] = name
            metadata = dict(state.get("metadata", {}))
            metadata["name"] = name
            state["metadata"] = metadata
        return World.from_state(self, state)

    def fork_world(
        self, world_id: str, *, history_index: int = 0, name: str | None = None
    ) -> World:
        fork = self.load_world(world_id).history_state(history_index)
        if name:
            fork.name = name
            fork.metadata["name"] = name
        fork.id = generate_id("world")
        fork._history = []
        fork._record_history(summary="world forked", action=None)
        return fork

    def compare(self, predictions: Iterable[Prediction]) -> Comparison:
        return Comparison(list(predictions))

    def generate(self, prompt: str, provider: str, *, duration_seconds: float = 1.0) -> VideoClip:
        return self._require_provider(provider).generate(prompt, duration_seconds)

    def transfer(
        self,
        clip: VideoClip,
        provider: str,
        *,
        width: int,
        height: int,
        fps: float,
    ) -> VideoClip:
        return self._require_provider(provider).transfer(clip, width=width, height=height, fps=fps)

    def reason(
        self,
        provider: str,
        query: str,
        *,
        world: World | None = None,
    ) -> ReasoningResult:
        world_state = world._snapshot() if world else None
        return self._require_provider(provider).reason(query, world_state=world_state)

    def embed(self, provider: str, *, text: str) -> EmbeddingResult:
        return self._require_provider(provider).embed(text=text)


def list_eval_suites() -> list[str]:
    """Return built-in evaluation suite identifiers."""

    return ["physics"]


def run_eval(suite: str, provider: str, *, forge: WorldForge | None = None):
    """Run a built-in evaluation suite and return scenario-level results."""

    from worldforge.evaluation import EvaluationSuite

    active_forge = forge or WorldForge()
    return EvaluationSuite.from_builtin(suite).run(provider, forge=active_forge)


def plan(world: World, *args: Any, **kwargs: Any) -> Plan:
    """Module-level alias for ``World.plan()``."""

    return world.plan(*args, **kwargs)
