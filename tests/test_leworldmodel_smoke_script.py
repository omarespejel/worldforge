from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from worldforge.smoke import leworldmodel


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


def test_checkpoint_path_uses_policy_object_checkpoint_name(tmp_path: Path) -> None:
    assert (
        leworldmodel._checkpoint_path(tmp_path, "pusht/lewm") == tmp_path / "pusht/lewm_object.ckpt"
    )


def test_require_object_checkpoint_reuses_existing_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pusht/lewm_object.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("existing")

    result = leworldmodel._require_object_checkpoint(
        policy="pusht/lewm",
        cache_dir=tmp_path,
    )

    assert result == checkpoint
    assert checkpoint.read_text() == "existing"


def test_require_object_checkpoint_explains_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="LeWorldModel object checkpoint not found"):
        leworldmodel._require_object_checkpoint(policy="pusht/lewm", cache_dir=tmp_path)


def test_build_inputs_uses_expected_tensor_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = ModuleType("torch")
    torch.rand = lambda *shape: {"shape": shape}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    info, action_candidates = leworldmodel._build_inputs(
        batch=2,
        samples=3,
        history=4,
        horizon=5,
        action_dim=6,
        image_size=7,
    )

    assert info["pixels"] == {"shape": (2, 1, 4, 3, 7, 7)}
    assert info["goal"] == {"shape": (2, 1, 4, 3, 7, 7)}
    assert info["action"] == {"shape": (2, 1, 4, 6)}
    assert action_candidates == {"shape": (2, 3, 5, 6)}


def test_build_inputs_rejects_non_rollout_horizon() -> None:
    with pytest.raises(SystemExit, match="horizon must be greater than history"):
        leworldmodel._build_inputs(
            batch=1,
            samples=1,
            history=4,
            horizon=4,
            action_dim=2,
            image_size=8,
        )


def test_smoke_main_prints_provider_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeProvider:
        def __init__(self, *, policy: str, cache_dir: str, device: str) -> None:
            self.policy = policy
            self.cache_dir = cache_dir
            self.device = device

        def score_actions(self, *, info: object, action_candidates: object) -> SimpleNamespace:
            assert info == {"pixels": "pixels"}
            assert action_candidates == ["actions"]
            return SimpleNamespace(to_dict=lambda: {"best_index": 0, "scores": [0.1]})

        def health(self) -> SimpleNamespace:
            return SimpleNamespace(to_dict=lambda: {"healthy": True, "name": "leworldmodel"})

    monkeypatch.setattr(
        leworldmodel,
        "_require_object_checkpoint",
        lambda **_kwargs: tmp_path / "pusht/lewm_object.ckpt",
    )
    monkeypatch.setattr(
        leworldmodel,
        "_build_inputs",
        lambda **_kwargs: ({"pixels": "pixels"}, ["actions"]),
    )
    monkeypatch.setattr(leworldmodel, "LeWorldModelProvider", FakeProvider)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldforge-smoke-leworldmodel",
            "--cache-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ],
    )

    assert leworldmodel.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "checkpoint": str(tmp_path / "pusht/lewm_object.ckpt"),
        "health": {"healthy": True, "name": "leworldmodel"},
        "result": {"best_index": 0, "scores": [0.1]},
    }
