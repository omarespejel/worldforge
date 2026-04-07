# Architecture

WorldForge is organized as a small Python package with clear runtime boundaries.

```text
python/worldforge/
├── __init__.py
├── _core.py
├── _runtime.py
├── cli.py
├── providers/
├── eval/
└── verify/
```

## Layers

### `_core.py`

Domain types, serialization helpers, capability descriptors, and other shared primitives.

### `_runtime.py`

Stateful orchestration:

- `WorldForge`
- `World`
- `Prediction`
- `Comparison`
- `Plan`

### `providers/`

Provider interfaces and adapters. The mock provider is the reference implementation. Remote adapters are intentionally scaffold-level until verified.

### `eval/`

Evaluation scenarios, suite runners, and report renderers.

### `verify/`

Proof-shaped artifacts used for verification flows and integration planning.

## Key decisions

- Python-first core: no Rust bridge, no dual build pipeline
- library-first surface: CLI wraps the package, not separate logic
- JSON snapshots: simple persistence and easy debugging
- honest capabilities: only supported operations are marketed as supported
