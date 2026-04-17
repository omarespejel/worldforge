from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_leworldmodel.py"
    spec = importlib.util.spec_from_file_location("smoke_leworldmodel", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_script_defaults_to_upstream_stablewm_home(monkeypatch) -> None:
    monkeypatch.delenv("STABLEWM_HOME", raising=False)
    script = _load_script()

    args = script._parser().parse_args([])

    assert args.stablewm_home == Path("~/.stable-wm").expanduser()


def test_smoke_script_honors_stablewm_home_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    script = _load_script()

    args = script._parser().parse_args([])

    assert args.stablewm_home == tmp_path
