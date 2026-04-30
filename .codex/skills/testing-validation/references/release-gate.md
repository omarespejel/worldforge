# Validation Gates

Focused default:
```bash
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run pytest tests/<target>.py -q
```

Docs/provider changes:
```bash
uv run python scripts/generate_provider_docs.py --check
uv run mkdocs build --strict
```

Public behavior or package changes:
```bash
uv lock --check
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run python scripts/generate_provider_docs.py --check
uv run pytest
uv run --extra harness pytest --cov=src/worldforge --cov-report=term-missing --cov-fail-under=90
bash scripts/test_package.sh
```

Release-quality gate: run the public behavior/package gate above, then run the security audit
fallback below.

Security audit fallback when using raw commands:
```bash
tmp_req="$(mktemp requirements-audit.XXXXXX)"
uv export --frozen --all-groups --no-emit-project --no-hashes -o "$tmp_req" >/dev/null
uvx --from pip-audit pip-audit -r "$tmp_req" --no-deps --disable-pip --progress-spinner off
rm -f "$tmp_req"
```
