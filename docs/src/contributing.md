# Contributing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Checks

```bash
make lint
make test
make test-package
```

## Provider work

Provider adapters live in `python/worldforge/providers/`.

Rules:

- keep capability declarations honest
- make missing credentials fail clearly
- add tests for registration and at least one working flow
- update README and spec when provider behavior changes
