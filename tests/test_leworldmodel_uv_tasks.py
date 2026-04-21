from __future__ import annotations

import importlib
from pathlib import Path


def test_leworldmodel_uv_commands_are_packaged_console_scripts() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject_path = root / "pyproject.toml"
    pyproject = pyproject_path.read_text()

    assert 'worldforge-harness = "worldforge.harness.cli:main"' in pyproject
    assert 'worldforge-demo-leworldmodel = "worldforge.demos.leworldmodel_e2e:main"' in pyproject
    assert (
        'worldforge-build-leworldmodel-checkpoint = "worldforge.smoke.leworldmodel_checkpoint:main"'
    ) in pyproject
    assert 'worldforge-smoke-leworldmodel = "worldforge.smoke.leworldmodel:main"' in pyproject
    assert 'lewm-real = "worldforge.smoke.leworldmodel:main"' in pyproject
    task = root / "scripts" / "lewm-real"
    assert task.exists()
    assert task.stat().st_mode & 0o111
    task_text = task.read_text()
    assert "uv run --python 3.10" in task_text
    assert "stable-worldmodel[train,env]" in task_text
    assert 'lewm-real "$@"' in task_text


def test_leworldmodel_console_script_targets_are_importable() -> None:
    targets = [
        "worldforge.harness.cli:main",
        "worldforge.demos.leworldmodel_e2e:main",
        "worldforge.smoke.leworldmodel_checkpoint:main",
        "worldforge.smoke.leworldmodel:main",
        "worldforge.smoke.leworldmodel:main",
    ]

    for target in targets:
        module_name, function_name = target.split(":", maxsplit=1)
        module = importlib.import_module(module_name)

        assert callable(getattr(module, function_name))
