# Contributing

WorldForge is a pure-Python package. The Rust workspace, Cargo build, and PyO3 bridge have been removed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Development workflow

```bash
make lint
make test
make test-package
```

`make test-package` is the authoritative install-contract check because it builds an isolated virtual environment, installs the package in editable mode, and runs the Python test suite against the installed distribution.

## Repository layout

- `python/worldforge/_core.py`: domain types, serialization helpers, provider metadata
- `python/worldforge/_runtime.py`: `WorldForge`, `World`, prediction/plan/comparison orchestration
- `python/worldforge/providers/`: provider adapter implementations
- `python/worldforge/eval/`: evaluation scenarios, suite runners, report renderers
- `python/worldforge/verify/`: verification bundles and verifier facades
- `python/tests/`: install-contract and behavior tests

## Adding a provider

1. Add a Python adapter in `python/worldforge/providers/`.
2. Be explicit about capability coverage. Do not advertise unsupported operations.
3. Make unavailable credentials fail clearly.
4. Register the provider in `WorldForge` only when the adapter is safe to auto-detect.
5. Add tests that exercise registration, health reporting, and at least one successful flow.

## Standards

- Keep the public API Pythonic and typed.
- Prefer simple data structures and explicit serialization over hidden magic.
- Do not introduce heavyweight dependencies without a concrete operational need.
- Update `README.md`, `SPECIFICATION.md`, and relevant docs when behavior changes.
- If a historical RFC conflicts with the live implementation, the code and live docs win.
