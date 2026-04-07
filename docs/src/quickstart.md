# Quick Start

## Install

```bash
uv add worldforge
```

For local development:

```bash
uv sync --group dev
```

## Create a world

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
print(prediction.physics_score)
```

## Plan and evaluate

```python
plan = world.plan(goal="move the mug to the right")
print(plan.action_count, plan.success_probability)

report = world.evaluate("physics")
print(report.to_markdown())
```

## CLI

```bash
uv run worldforge providers
uv run worldforge predict kitchen --provider mock --x 0.3 --y 0.8 --z 0.0 --steps 2
uv run worldforge eval --suite physics --provider mock
```
