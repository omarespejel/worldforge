.PHONY: sync lock format lint docs-check docs-site test test-cov test-package build audit publish check release-check clean

UV ?= uv

sync:
	$(UV) sync --group dev

lock:
	$(UV) lock --check

format:
	$(UV) run ruff format src tests examples scripts

lint:
	$(UV) run ruff check src tests examples scripts
	$(UV) run ruff format --check src tests examples scripts

docs-check:
	$(UV) run python scripts/generate_provider_docs.py --check
	$(UV) run mkdocs build --strict

docs-site:
	$(UV) run mkdocs serve

test:
	$(UV) run pytest

test-cov:
	$(UV) run --extra harness pytest --cov=src/worldforge --cov-report=term-missing --cov-fail-under=90

test-package:
	bash scripts/test_package.sh

build:
	$(UV) build --out-dir dist --clear --no-build-logs

audit:
	tmp_req="$$(mktemp requirements-audit.XXXXXX)"; \
	trap 'rm -f "$$tmp_req"' EXIT; \
	$(UV) export --frozen --all-groups --no-emit-project --no-hashes -o "$$tmp_req" >/dev/null; \
	$(UV) run pip-audit -r "$$tmp_req" --no-deps --disable-pip --progress-spinner off

publish:
	$(UV) publish

check: lock lint docs-check test test-cov test-package build

release-check: check audit

clean:
	rm -rf build dist site .pytest_cache .ruff_cache .worldforge .coverage
	find src tests examples scripts docs -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -path './.git' -prune -o -name '.DS_Store' -type f -delete
