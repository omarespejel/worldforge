#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$(mktemp -d "${TMPDIR:-/tmp}/worldforge-python.XXXXXX")"

cleanup() {
    find "$ROOT_DIR/python/worldforge" -maxdepth 1 -type f \
        \( -name 'worldforge*.so' -o -name 'worldforge*.pyd' \) \
        -delete
    rm -rf "$VENV_DIR"
}

trap cleanup EXIT

unset PYTHONPATH
export PYTHONNOUSERSITE=1

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR"
"$VENV_DIR/bin/python" -c 'import worldforge; import worldforge.providers; import worldforge.eval; import worldforge.verify'

cd "$ROOT_DIR"
"$VENV_DIR/bin/python" -m unittest discover -s python/tests -v
