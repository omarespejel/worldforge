"""Shared helpers for WorldForge Python package tests."""

from __future__ import annotations

import importlib
import os
import unittest
from types import ModuleType

STRICT_IMPORT_ENV = "WORLDFORGE_PYTHON_REQUIRE_PACKAGE"


def strict_install_contract_enabled() -> bool:
    """Return whether the package import contract should fail on missing imports."""
    value = os.environ.get(STRICT_IMPORT_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _module_missing_from_import_error(module_name: str, error: ModuleNotFoundError) -> bool:
    root_name = module_name.split(".", 1)[0]
    return error.name in {module_name, root_name}


def import_optional_module(module_name: str) -> ModuleType | None:
    """Import a module if it exists, otherwise return ``None``.

    Import errors for nested dependencies are propagated so the tests do not
    silently hide real installation failures.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if not _module_missing_from_import_error(module_name, error):
            raise
        return None


def require_installed_module(module_name: str) -> ModuleType:
    """Return an imported module, skipping or failing if it is unavailable."""
    module = import_optional_module(module_name)
    if module is not None:
        return module

    message = (
        f"{module_name} is not installed; run `pip install -e .` "
        f"or set {STRICT_IMPORT_ENV}=1 to enforce the contract"
    )
    if strict_install_contract_enabled():
        raise AssertionError(message)
    raise unittest.SkipTest(message)
