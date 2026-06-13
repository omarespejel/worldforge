"""Shared artifact I/O helpers."""

from __future__ import annotations

from pathlib import Path

from worldforge.models import WorldStateError, dump_json


def read_bounded_text_artifact(path: Path, *, max_bytes: int) -> str:
    """Read a UTF-8 artifact with a hard byte cap."""

    try:
        with path.open("rb") as handle:
            raw_payload = handle.read(max_bytes + 1)
    except OSError as exc:
        raise WorldStateError("Artifact could not be read.") from exc
    if len(raw_payload) > max_bytes:
        raise WorldStateError("Artifact exceeds the read limit.")
    try:
        return raw_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorldStateError("Artifact must be UTF-8 text.") from exc


def write_json_artifact(path: Path, payload: object) -> Path:
    """Write a deterministic, finite JSON artifact and return its path."""

    json_text = dump_json(payload, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text, encoding="utf-8")
    return path


__all__ = ["read_bounded_text_artifact", "write_json_artifact"]
