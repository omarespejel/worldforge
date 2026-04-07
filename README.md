# WorldForge

WorldForge is an open-source Python framework for building world-model workflows.

It gives you a clean framework surface for prediction, planning, provider orchestration, state persistence, and lightweight evaluation without forcing you into a bespoke runtime stack.

## Why WorldForge

- Python-native architecture aligned with the ML and provider ecosystem
- Clean `src/` layout and typed public API
- Deterministic `MockProvider` for local development, tests, and examples
- Provider registry that can host real adapters and local surrogates behind one interface
- Built-in world persistence and evaluation helpers

## Installation

For application projects:

```bash
uv add worldforge
```

For framework development:

```bash
uv sync --group dev
```

## Quick Start

```python
from worldforge import Action, BBox, Position, SceneObject, WorldForge

forge = WorldForge()
world = forge.create_world("kitchen", provider="mock")

world.add_object(
    SceneObject(
        "red_mug",
        Position(0.0, 0.8, 0.0),
        BBox(Position(-0.05, 0.75, -0.05), Position(0.05, 0.85, 0.05)),
    )
)

prediction = world.predict(Action.move_to(0.3, 0.8, 0.0), steps=2)
print(prediction.provider, prediction.physics_score)

plan = world.plan(goal="move the mug to the right")
print(plan.action_count, plan.success_probability)

doctor = forge.doctor()
print(doctor.healthy_provider_count, doctor.provider_count)
```

## Provider DX

```bash
uv run worldforge doctor
uv run worldforge provider list
uv run worldforge provider info mock
uv run worldforge provider health
```

## Repository Layout

```text
worldforge/
├── src/worldforge/
│   ├── __init__.py
│   ├── cli.py
│   ├── framework.py
│   ├── models.py
│   ├── evaluation/
│   └── providers/
├── tests/
├── examples/
├── docs/
├── scripts/
├── pyproject.toml
└── uv.lock
```

## Development

```bash
make sync
make lint
make test
make test-package
make build
```

Publishing is `uv build` and `uv publish`.

## Current Scope

Implemented in-repo:

- framework runtime and state model
- mock provider
- provider registry and scaffold adapters
- planning and comparison flows
- built-in physics evaluation helpers
- CLI entry point

Not in scope right now:

- a bundled service layer
- claims that scaffold adapters are production-complete integrations
