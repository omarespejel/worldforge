# WorldForge

WorldForge is now a Python-first orchestration library for world model workflows.

This repository previously centered a Rust workspace with PyO3 bindings. That architecture has been removed. The current codebase is pure Python because the provider ecosystem, ML tooling, and operational surface area are overwhelmingly Python-native, and keeping a Rust core was adding integration cost without delivering responsible product leverage.

## What ships today

- Pure-Python package under `python/worldforge/`
- Deterministic `MockProvider` for offline development and tests
- Provider registry with Python adapter placeholders for Cosmos, Runway, JEPA, and Genie
- Stateful `World` objects with JSON persistence and history snapshots
- Deterministic prediction, comparison, planning, evaluation, and verification flows
- Installable CLI via `worldforge`

## What does not ship today

- No Rust crates, Cargo workspace, or PyO3 extension module
- No rebuilt production REST service yet
- No claim that remote provider adapters are production-complete; only the mock provider is fully implemented in-repo

## Install

```bash
python -m pip install -e .
```

Or for users:

```bash
pip install worldforge
```

## Quick start

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

plan = world.plan(goal="move the mug to the right", verify_backend="mock")
print(plan.action_count, plan.success_probability)
```

## CLI

```bash
worldforge providers
worldforge predict kitchen --provider mock --x 0.3 --y 0.8 --z 0.0 --steps 2
worldforge eval --suite physics --provider mock
```

## Architecture

```text
worldforge/
├── python/worldforge/
│   ├── __init__.py          # Public package exports
│   ├── _core.py             # Domain types and serialization helpers
│   ├── _runtime.py          # WorldForge, World, Prediction, Plan
│   ├── cli.py               # Python CLI
│   ├── providers/           # Provider adapters and registry primitives
│   ├── eval/                # Evaluation suites and reports
│   └── verify/              # Verification bundles and verifiers
├── examples/
├── docs/
├── SPECIFICATION.md
└── architecture/ADR.md
```

Design principles:

- Python at the boundary and in the core. The implementation should match the ecosystem it integrates with.
- Honest capability signaling. Only implemented provider behavior is documented as implemented.
- Library-first delivery. Core orchestration is stable before new transport layers are added.
- Simple persistence. JSON snapshots are the default until a real operational need justifies heavier storage.

## Development

```bash
make lint
make test
make test-package
make build
```

The package is intentionally dependency-light. The stdlib handles the current implementation, which keeps editable installs, tests, and contributor setup straightforward.

## Migration note

The `rfcs/` and some `research/` documents describe the retired Rust-based architecture. Treat them as historical context unless they are explicitly updated to the Python-first model.
