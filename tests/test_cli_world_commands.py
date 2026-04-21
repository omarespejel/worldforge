from __future__ import annotations

import json
import sys

import pytest

from worldforge.cli import main


def _run_world_cli(tmp_path, monkeypatch, capsys, *args: str) -> str:
    monkeypatch.setattr(
        sys,
        "argv",
        ["worldforge", "world", *args, "--state-dir", str(tmp_path)],
    )
    assert main() == 0
    return capsys.readouterr().out


def test_world_cli_edits_persisted_scene_objects(tmp_path, monkeypatch, capsys) -> None:
    created = json.loads(_run_world_cli(tmp_path, monkeypatch, capsys, "create", "lab"))
    world_id = created["id"]

    added = json.loads(
        _run_world_cli(
            tmp_path,
            monkeypatch,
            capsys,
            "add-object",
            world_id,
            "red_mug",
            "--x",
            "0.0",
            "--y",
            "0.8",
            "--z",
            "0.0",
            "--object-id",
            "mug-1",
            "--size",
            "0.2",
            "--graspable",
            "--metadata",
            '{"material": "ceramic"}',
        )
    )
    assert added["object"]["id"] == "mug-1"
    assert added["object"]["is_graspable"] is True
    assert added["object"]["metadata"] == {"material": "ceramic"}
    assert added["object"]["bbox"]["min"]["x"] == pytest.approx(-0.1)
    assert added["object"]["bbox"]["min"]["y"] == pytest.approx(0.7)
    assert added["object"]["bbox"]["min"]["z"] == pytest.approx(-0.1)
    assert added["world"]["object_count"] == 1

    objects = json.loads(_run_world_cli(tmp_path, monkeypatch, capsys, "objects", world_id))
    assert [obj["id"] for obj in objects["objects"]] == ["mug-1"]

    updated = json.loads(
        _run_world_cli(
            tmp_path,
            monkeypatch,
            capsys,
            "update-object",
            world_id,
            "mug-1",
            "--name",
            "coffee_mug",
            "--x",
            "0.25",
            "--y",
            "0.8",
            "--z",
            "0.05",
            "--graspable",
            "false",
        )
    )
    assert updated["object"]["name"] == "coffee_mug"
    assert updated["object"]["position"] == {"x": 0.25, "y": 0.8, "z": 0.05}
    assert updated["object"]["is_graspable"] is False

    removed = json.loads(
        _run_world_cli(tmp_path, monkeypatch, capsys, "remove-object", world_id, "mug-1")
    )
    assert removed["removed_object"]["id"] == "mug-1"
    assert removed["world"]["object_count"] == 0


def test_world_cli_prediction_saves_or_dry_runs_persisted_world(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    created = json.loads(_run_world_cli(tmp_path, monkeypatch, capsys, "create", "lab"))
    world_id = created["id"]
    json.loads(
        _run_world_cli(
            tmp_path,
            monkeypatch,
            capsys,
            "add-object",
            world_id,
            "cube",
            "--object-id",
            "cube-1",
            "--x",
            "0.0",
            "--y",
            "0.5",
            "--z",
            "0.0",
        )
    )

    prediction = json.loads(
        _run_world_cli(
            tmp_path,
            monkeypatch,
            capsys,
            "predict",
            world_id,
            "--object-id",
            "cube-1",
            "--x",
            "0.4",
            "--y",
            "0.5",
            "--z",
            "0.0",
            "--steps",
            "2",
        )
    )
    assert prediction["saved"] is True
    assert prediction["world"]["step"] == 2
    assert prediction["world"]["history_length"] == 2

    saved_objects = json.loads(_run_world_cli(tmp_path, monkeypatch, capsys, "objects", world_id))
    assert saved_objects["objects"][0]["position"] == {"x": 0.4, "y": 0.5, "z": 0.0}

    dry_run = json.loads(
        _run_world_cli(
            tmp_path,
            monkeypatch,
            capsys,
            "predict",
            world_id,
            "--object-id",
            "cube-1",
            "--x",
            "0.9",
            "--y",
            "0.5",
            "--z",
            "0.0",
            "--dry-run",
        )
    )
    assert dry_run["saved"] is False
    assert dry_run["world_state"]["scene"]["objects"]["cube-1"]["pose"]["position"] == {
        "x": 0.9,
        "y": 0.5,
        "z": 0.0,
    }

    persisted_after_dry_run = json.loads(
        _run_world_cli(tmp_path, monkeypatch, capsys, "objects", world_id)
    )
    assert persisted_after_dry_run["objects"][0]["position"] == {"x": 0.4, "y": 0.5, "z": 0.0}


def test_world_cli_rejects_incomplete_object_updates(tmp_path, monkeypatch, capsys) -> None:
    created = json.loads(_run_world_cli(tmp_path, monkeypatch, capsys, "create", "lab"))
    world_id = created["id"]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldforge",
            "world",
            "update-object",
            world_id,
            "missing",
            "--x",
            "1.0",
            "--state-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    assert "Position updates require --x, --y, and --z together." in capsys.readouterr().err
