.PHONY: sync format lint test test-package build publish check clean

UV ?= uv

sync:
	$(UV) sync --group dev

format:
	$(UV) run ruff format src tests examples

lint:
	$(UV) run ruff check src tests examples
	$(UV) run ruff format --check src tests examples

test:
	$(UV) run pytest

test-package:
	bash scripts/test_package.sh

build:
	$(UV) build

publish:
	$(UV) publish

check: lint test test-package

clean:
	rm -rf build dist .pytest_cache .ruff_cache .worldforge .coverage
	find src tests examples -name '__pycache__' -type d -prune -exec rm -rf {} +
