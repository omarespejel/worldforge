# Architecture

WorldForge follows a standard `src/` layout:

```text
src/worldforge/
├── __init__.py
├── cli.py
├── framework.py
├── models.py
├── evaluation/
└── providers/
```

## Module responsibilities

### `models.py`

Core domain objects and JSON serialization helpers.

### `framework.py`

Framework runtime:

- `WorldForge`
- `World`
- `Prediction`
- `Comparison`
- `Plan`

### `providers/`

Provider primitives and adapters. The mock provider is the reference implementation.

### `evaluation/`

Evaluation suites, scenario runners, and report rendering.

## Design principles

- Python-native API surface
- typed public models
- simple JSON persistence
- honest provider capability reporting
- deterministic local development path
