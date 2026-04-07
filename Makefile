.PHONY: format lint test test-package build clean check

PYTHON ?= python3

format:
	$(PYTHON) -m compileall -q python

lint:
	$(PYTHON) -m py_compile $$(find python/worldforge -name '*.py' -print)

test:
	PYTHONPATH=python $(PYTHON) -m unittest discover -s python/tests -v

test-package:
	bash scripts/test_python_package.sh

build:
	$(PYTHON) -m pip wheel . --no-deps --wheel-dir dist

check: lint test-package

clean:
	rm -rf build dist .pytest_cache .worldforge .tmp-state
	find python -name '__pycache__' -type d -prune -exec rm -rf {} +
