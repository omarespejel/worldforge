# Python SDK

The WorldForge Python SDK provides native bindings via PyO3 for high-performance
access to the full WorldForge API.

## Installation

```bash
pip install worldforge
```

Requires Python 3.10+.

## Quick Example

```python
from worldforge import WorldForge, Action

wf = WorldForge()
world = wf.create_world("kitchen", provider="cosmos")

prediction = world.predict(Action.move_to(0.5, 0.8, 0.0), steps=10)
print(f"Physics score: {prediction.physics_score}")
```

## Core Classes

### WorldForge

The main entry point. Auto-detects providers from environment variables.

```python
wf = WorldForge()
wf = WorldForge(state_backend="sqlite", state_db_path="worlds.db")
```

Methods:
- `create_world(name, provider) -> World`
- `list_providers() -> list[ProviderInfo]`
- `get_provider(name) -> ProviderInfo`

### World

Represents a simulation world bound to a specific provider.

```python
world = wf.create_world("sim", provider="cosmos")
```

Methods:
- `predict(action, steps=1) -> Prediction`
- `plan(goal, planner="cem", max_steps=20) -> Plan`
- `compare(action, providers, steps=1) -> Comparison`
- `evaluate(suite="physics") -> EvalReport`

### Action

Describes an action to apply to the world.

```python
action = Action.move_to(0.5, 0.8, 0.0)
action = Action.from_dict({"push": {"force": 1.0, "direction": [1, 0, 0]}})
```

### Prediction

Result of a prediction call.

Attributes:
- `physics_score: float` — Overall physics fidelity score (0-1).
- `frames: list[bytes]` — Generated video frames.
- `world_state: dict` — Updated world state after prediction.
- `metadata: dict` — Provider-specific metadata.

### Plan

Result of a planning call.

Attributes:
- `actions: list[Action]` — Planned action sequence.
- `success_probability: float` — Estimated probability of success.
- `verification_proof: bytes | None` — Optional ZK proof.

### EvalReport

Result of an evaluation run.

Methods:
- `to_markdown() -> str`
- `to_csv() -> str`
- `to_dict() -> dict`

## Cross-Provider Comparison

```python
comparison = world.compare(
    Action.move_to(0.5, 0.8, 0.0),
    providers=["cosmos", "runway", "sora"],
    steps=10,
)
print(comparison.to_markdown())
```

## State Persistence

```python
wf = WorldForge(state_backend="sqlite", state_db_path="worlds.db")
# Also supports: file, redis, s3, msgpack
```

## Type Stubs

WorldForge ships with PEP 561 type stubs (`py.typed` marker) for full IDE
support including autocompletion and type checking with mypy/pyright.
