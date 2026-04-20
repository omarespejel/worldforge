"""Refresh provider documentation generated from the provider catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROVIDER_INDEX = ROOT / "docs" / "src" / "providers" / "README.md"
START_MARKER = "<!-- provider-catalog:start -->"
END_MARKER = "<!-- provider-catalog:end -->"


def _replace_block(content: str, replacement: str) -> str:
    start = content.index(START_MARKER)
    end = content.index(END_MARKER, start)
    return content[: start + len(START_MARKER)] + "\n" + replacement.rstrip() + "\n" + content[end:]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if generated provider docs are out of date.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from worldforge.providers.catalog import render_provider_catalog_markdown

    args = _parser().parse_args(argv)
    current = PROVIDER_INDEX.read_text(encoding="utf-8")
    updated = _replace_block(current, render_provider_catalog_markdown())

    if args.check:
        if updated != current:
            print(
                f"{PROVIDER_INDEX.relative_to(ROOT)} is out of date; "
                "run `uv run python scripts/generate_provider_docs.py`.",
                file=sys.stderr,
            )
            return 1
        return 0

    PROVIDER_INDEX.write_text(updated, encoding="utf-8")
    print(f"updated {PROVIDER_INDEX.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
