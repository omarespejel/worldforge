from __future__ import annotations

import importlib
from pathlib import Path


def test_leworldmodel_uv_commands_are_packaged_console_scripts() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = pyproject_path.read_text()

    assert 'worldforge-demo-leworldmodel = "worldforge.demos.leworldmodel_e2e:main"' in pyproject
    assert 'worldforge-smoke-leworldmodel = "worldforge.smoke.leworldmodel:main"' in pyproject


def test_leworldmodel_console_script_targets_are_importable() -> None:
    targets = [
        "worldforge.demos.leworldmodel_e2e:main",
        "worldforge.smoke.leworldmodel:main",
    ]

    for target in targets:
        module_name, function_name = target.split(":", maxsplit=1)
        module = importlib.import_module(module_name)

        assert callable(getattr(module, function_name))
