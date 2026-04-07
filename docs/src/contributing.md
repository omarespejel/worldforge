# Contributing

```bash
uv sync --group dev
make lint
make test
make test-package
```

Key directories:

- `src/worldforge/`
- `tests/`
- `examples/`

Provider work belongs in `src/worldforge/providers/`. Keep adapter capabilities honest and add tests for every new supported path.
