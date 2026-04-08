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

Framework runtime and persistence:

- `WorldForge`
- `World`
- `Prediction`
- `Comparison`
- `Plan`
- import/export and local JSON world storage

### `providers/`

Provider primitives and adapters. The mock provider is the reference implementation. `cosmos`
and `runway` are live HTTP adapters. `jepa` and `genie` are scaffold adapters.

### `evaluation/`

Evaluation suites, scenario runners, and report rendering.

## Data flow

1. `WorldForge` resolves or registers a provider.
2. `World` snapshots the current state and sends it to the provider.
3. The provider returns a `PredictionPayload` or `VideoClip`.
4. The framework validates and applies the returned state.
5. History, persistence, evaluation, and CLI output are derived from that validated state.

## Invariants

- persisted worlds must contain `id`, `name`, and `provider`
- world `step` is always a non-negative integer
- invalid public inputs fail explicitly instead of being silently coerced
- provider capability metadata must match the implemented surface
- missing local asset files fail before network I/O

## Failure model

- invalid caller input raises `WorldForgeError`
- malformed persisted state raises `WorldStateError`
- provider/runtime integration failures raise `ProviderError`
- remote health checks may fail due to missing credentials, invalid endpoints, or upstream errors

## Design principles

- Python-native API surface
- typed public models
- simple JSON persistence
- honest provider capability reporting
- deterministic local development path
- fail fast on invalid state and invalid inputs
