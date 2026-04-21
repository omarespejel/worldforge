.PHONY: sync format lint docs-check test test-cov test-package build publish check clean

UV ?= uv

sync:
	$(UV) sync --group dev

format:
	$(UV) run ruff format src tests examples scripts

lint:
	$(UV) run ruff check src tests examples scripts
	$(UV) run ruff format --check src tests examples scripts

docs-check:
	$(UV) run python scripts/generate_provider_docs.py --check

test:
	$(UV) run pytest

test-cov:
	$(UV) run --extra harness pytest --cov=src/worldforge --cov-report=term-missing --cov-fail-under=90

test-package:
	bash scripts/test_package.sh

build:
	$(UV) build

publish:
	$(UV) publish

check: lint docs-check test test-package

clean:
	rm -rf build dist .pytest_cache .ruff_cache .worldforge .coverage
	find src tests examples scripts -name '__pycache__' -type d -prune -exec rm -rf {} +
