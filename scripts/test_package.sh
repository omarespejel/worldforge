#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/worldforge-package.XXXXXX")"
VENV_DIR="$TMP_DIR/.venv"

cleanup() {
    rm -rf "$TMP_DIR"
}

trap cleanup EXIT

cd "$ROOT_DIR"
uv build --out-dir "$TMP_DIR/dist"
uv venv "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" "$TMP_DIR"/dist/worldforge-*.whl pytest

"$VENV_DIR/bin/python" - <<'PY'
import worldforge
import worldforge.evaluation
import worldforge.providers

assert worldforge.__version__
assert worldforge.WorldForge is not None
PY

"$VENV_DIR/bin/python" -m pytest "$ROOT_DIR/tests" -q
