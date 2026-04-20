"""Private provider configuration parsing helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence

from worldforge.models import WorldForgeError, require_positive_int


def env_value(name: str) -> str | None:
    """Return a stripped environment value, treating blank strings as unset."""

    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def first_env_value(names: Sequence[str]) -> str | None:
    """Return the first non-blank environment value from an alias list."""

    for name in names:
        value = env_value(name)
        if value is not None:
            return value
    return None


def optional_non_empty(value: str | None, *, name: str) -> str | None:
    """Normalize an optional string and reject caller-supplied blanks."""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise WorldForgeError(f"{name} must be a non-empty string when provided.")
    return value.strip()


def optional_positive_int(value: int | str | None, *, name: str) -> int | None:
    """Normalize an optional positive integer from direct config or env strings."""

    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = int(value)
        except ValueError:
            raise WorldForgeError(f"{name} must be an integer greater than 0.") from None
    return require_positive_int(value, name=name)


def optional_bool(value: bool | str | None, *, name: str) -> bool | None:
    """Normalize an optional boolean from direct config or common env strings."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise WorldForgeError(f"{name} must be a boolean when provided.")
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise WorldForgeError(f"{name} must be a boolean when provided.")
