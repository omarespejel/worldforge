from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from worldforge.providers.catalog import PROVIDER_CATALOG, render_provider_catalog_markdown

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_INDEX = ROOT / "docs" / "src" / "providers" / "README.md"
SCRIPT = ROOT / "scripts" / "generate_provider_docs.py"
START_MARKER = "<!-- provider-catalog:start -->"
END_MARKER = "<!-- provider-catalog:end -->"


def _provider_catalog_block() -> str:
    content = PROVIDER_INDEX.read_text(encoding="utf-8")
    start = content.index(START_MARKER) + len(START_MARKER)
    end = content.index(END_MARKER, start)
    return content[start:end].strip()


def test_provider_catalog_docs_are_generated_from_catalog() -> None:
    assert _provider_catalog_block() == render_provider_catalog_markdown()


def test_generate_provider_docs_check_passes() -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_provider_catalog_docs_pages_exist_for_linked_entries() -> None:
    provider_docs_dir = ROOT / "docs" / "src" / "providers"
    for entry in PROVIDER_CATALOG:
        if entry.docs_page is not None:
            assert (provider_docs_dir / entry.docs_page).exists()
        assert entry.runtime_ownership
