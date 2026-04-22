from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

from worldforge.smoke import pusht_showcase_inputs, robotics_showcase


class FakeTensor:
    def __init__(self, data: object) -> None:
        self.data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape(self.data)

    @staticmethod
    def _shape(value: object) -> tuple[int, ...]:
        if isinstance(value, list):
            if not value:
                return (0,)
            return (len(value), *FakeTensor._shape(value[0]))
        return ()

    def tolist(self) -> object:
        return self.data

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def reshape(self, shape: int) -> FakeTensor:
        if shape != -1:
            raise AssertionError("test fake only implements reshape(-1)")
        flattened: list[float] = []

        def visit(value: object) -> None:
            if isinstance(value, list):
                for child in value:
                    visit(child)
                return
            flattened.append(float(value))  # type: ignore[arg-type]

        visit(self.data)
        return FakeTensor(flattened)

    def numel(self) -> int:
        total = 1
        for part in self.shape:
            total *= part
        return total

    def clamp(self, minimum: float, maximum: float) -> FakeTensor:
        return FakeTensor(self._map(lambda value: max(minimum, min(maximum, value))))

    def repeat_interleave(self, repeats: int, dim: int) -> FakeTensor:
        def repeat_axis(value: object, axis: int) -> object:
            if axis == 0:
                if isinstance(value, list):
                    repeated: list[object] = []
                    for child in value:
                        repeated.extend(copy.deepcopy(child) for _ in range(repeats))
                    return repeated
                return [copy.deepcopy(value) for _ in range(repeats)]
            if not isinstance(value, list):
                raise AssertionError("repeat_interleave axis exceeds tensor rank")
            return [repeat_axis(child, axis - 1) for child in value]

        return FakeTensor(repeat_axis(self.data, dim))

    def unsqueeze(self, dim: int) -> FakeTensor:
        def insert_axis(value: object, axis: int) -> object:
            if axis == 0:
                return [value]
            if not isinstance(value, list):
                raise AssertionError("unsqueeze axis exceeds tensor rank")
            return [insert_axis(child, axis - 1) for child in value]

        return FakeTensor(insert_axis(self.data, dim))

    def repeat(self, *repeats: int) -> FakeTensor:
        def repeat_axes(value: object, axes: tuple[int, ...]) -> object:
            if not axes:
                return value
            if not isinstance(value, list):
                return value
            repeated_children = [repeat_axes(child, axes[1:]) for child in value]
            out: list[object] = []
            for _ in range(axes[0]):
                out.extend(copy.deepcopy(repeated_children))
            return out

        return FakeTensor(repeat_axes(self.data, tuple(repeats)))

    def to(self, _dtype: object) -> FakeTensor:
        return self

    def _map(self, fn: object) -> object:
        def visit(value: object) -> object:
            if isinstance(value, list):
                return [visit(child) for child in value]
            return fn(float(value))  # type: ignore[operator,arg-type]

        return visit(self.data)

    def __iter__(self):
        if not isinstance(self.data, list):
            raise TypeError("scalar fake tensor is not iterable")
        for child in self.data:
            yield FakeTensor(child)

    def __getitem__(self, item: object) -> FakeTensor:
        if not isinstance(self.data, list):
            raise TypeError("scalar fake tensor is not subscriptable")
        return FakeTensor(self.data[item])  # type: ignore[index]

    def __mul__(self, scale: float) -> FakeTensor:
        return FakeTensor(self._map(lambda value: value * scale))

    def __neg__(self) -> FakeTensor:
        return FakeTensor(self._map(lambda value: -value))

    def __float__(self) -> float:
        if isinstance(self.data, list):
            raise TypeError("non-scalar fake tensor cannot be converted to float")
        return float(self.data)  # type: ignore[arg-type]


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def as_tensor(value: object, dtype: object | None = None) -> FakeTensor:
        return FakeTensor(value)

    @staticmethod
    def stack(values: list[FakeTensor], dim: int = 0) -> FakeTensor:
        assert dim == 0
        return FakeTensor([value.tolist() for value in values])


def test_robotics_showcase_forwards_packaged_pusht_defaults(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_low_level_main(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(robotics_showcase.lerobot_leworldmodel, "main", fake_low_level_main)

    assert (
        robotics_showcase.main(
            [
                "--checkpoint",
                str(tmp_path / "pusht/lewm_object.ckpt"),
                "--device",
                "cpu",
                "--no-color",
                "--no-json-output",
            ]
        )
        == 0
    )

    forwarded = captured["argv"]
    assert forwarded[:2] == ["--policy-path", "lerobot/diffusion_pusht"]
    assert "--observation-module" in forwarded
    assert (
        forwarded[forwarded.index("--observation-module") + 1]
        == "worldforge.smoke.pusht_showcase_inputs:build_observation"
    )
    assert (
        forwarded[forwarded.index("--score-info-module") + 1]
        == "worldforge.smoke.pusht_showcase_inputs:build_score_info"
    )
    assert (
        forwarded[forwarded.index("--translator") + 1]
        == "worldforge.smoke.pusht_showcase_inputs:translate_candidates"
    )
    assert (
        forwarded[forwarded.index("--candidate-builder") + 1]
        == "worldforge.smoke.pusht_showcase_inputs:build_action_candidates"
    )
    assert forwarded[forwarded.index("--expected-action-dim") + 1] == "10"
    assert forwarded[forwarded.index("--expected-horizon") + 1] == "4"
    assert "--json-output" not in forwarded


def test_packaged_pusht_bridge_builds_candidates_without_optional_import_at_module_load(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    candidates = pusht_showcase_inputs.build_action_candidates([0.8, -2.0], {}, {})

    assert candidates.shape == (1, 3, 4, 10)
    assert candidates.tolist()[0][0][0] == [0.8, 0.8, 0.8, 0.8, 0.8, -1.0, -1.0, -1.0, -1.0, -1.0]

    plans = pusht_showcase_inputs.translate_candidates(
        [0.8, -2.0],
        {"score_bridge": {"object_id": "block-1"}},
        {},
    )
    assert len(plans) == 3
    assert plans[0][0].parameters["object_id"] == "block-1"
    assert plans[0][0].parameters["target"] == {"x": 0.7, "y": 0.25, "z": 0.0}


def test_robotics_showcase_uses_tmp_json_output_by_default(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_low_level_main(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(robotics_showcase.lerobot_leworldmodel, "main", fake_low_level_main)

    assert robotics_showcase.main(["--checkpoint", "/tmp/pusht/lewm_object.ckpt"]) == 0

    forwarded = captured["argv"]
    assert forwarded[forwarded.index("--json-output") + 1] == str(
        robotics_showcase.DEFAULT_JSON_OUTPUT
    )
