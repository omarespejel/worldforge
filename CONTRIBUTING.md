# Contributing

## Setup

```bash
uv sync --group dev
```

## Core commands

```bash
make lint
make test
make test-package
make build
```

`make test-package` is the packaging contract check. It builds a wheel with `uv`, installs that wheel into an isolated virtual environment, and runs the root test suite against the installed package.

## Repository layout

- `src/worldforge/models.py`: domain models and serialization helpers
- `src/worldforge/framework.py`: `WorldForge`, `World`, prediction, comparison, and planning
- `src/worldforge/providers/`: provider interfaces and adapters
- `src/worldforge/testing/`: reusable provider contract helpers for adapter packages
- `src/worldforge/evaluation/`: evaluation suites and report rendering
- `tests/`: framework and packaging tests
- `examples/`: runnable examples

## Standards

- keep the public API typed and Pythonic
- use `src/` layout conventions consistently
- do not advertise provider capabilities that are not implemented
- prefer simple, inspectable JSON state over implicit persistence magic
- keep docs aligned with the live package surface

## Adding a provider

1. Add the adapter under `src/worldforge/providers/`.
2. Declare only the capabilities the adapter actually supports.
3. Fail clearly on missing credentials or unsupported flows.
4. Register the provider in `WorldForge` only when auto-detection is safe.
5. Add tests covering registration, health reporting, and a successful runtime path.
6. Run `worldforge.testing.assert_provider_contract()` in adapter tests to validate metadata and capability behavior.
