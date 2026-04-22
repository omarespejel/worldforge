from __future__ import annotations

import json

import pytest

from worldforge import (
    Action,
    BBox,
    Position,
    SceneObject,
    SceneObjectPatch,
    WorldForge,
    WorldForgeError,
    WorldStateError,
)


def test_world_prediction_compare_and_persistence_flow(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    assert "mock" in forge.providers()

    world = forge.create_world("kitchen-counter", "mock")
    world.add_object(
        SceneObject(
            "red_mug",
            Position(0.0, 0.8, 0.0),
            BBox(Position(-0.05, 0.75, -0.05), Position(0.05, 0.85, 0.05)),
        )
    )

    prediction = world.predict(Action.move_to(0.25, 0.8, 0.0, 1.0), steps=4)
    assert prediction.provider == "mock"
    assert prediction.confidence >= 0.0

    comparison = world.compare(Action.move_to(0.3, 0.8, 0.0, 1.0), ["mock"], steps=2)
    assert comparison.best_prediction().provider == "mock"

    world_id = forge.save_world(world)
    assert world_id in forge.list_worlds()
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []

    snapshot_json = forge.export_world(world_id, format="json")
    payload = json.loads(snapshot_json)
    assert payload["schema_version"] == 1
    assert payload["state"]["metadata"]["name"] == "kitchen-counter"

    snapshot_copy = forge.import_world(snapshot_json, format="json", new_id=True, name="copy")
    assert snapshot_copy.id != world_id
    assert snapshot_copy.name == "copy"
    assert snapshot_copy.object_count == 1

    loaded = forge.load_world(world_id)
    assert "red_mug" in loaded.list_objects()

    objects = loaded.objects()
    assert len(objects) == 1
    object_id = objects[0].id
    fetched = loaded.get_object_by_id(object_id)
    assert fetched is not None
    assert fetched.name == "red_mug"

    patch = SceneObjectPatch()
    patch.set_name("coffee_mug")
    patch.set_position(Position(0.1, 0.8, 0.0))
    patch.set_graspable(True)
    updated = loaded.update_object_patch(object_id, patch)
    assert updated.id == object_id
    assert updated.name == "coffee_mug"
    assert updated.is_graspable is True

    removed = loaded.remove_object_by_id(object_id)
    assert removed is not None
    assert removed.id == object_id
    assert loaded.object_count == 0


def test_world_delete_validates_id_and_removes_persisted_file(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    world = forge.create_world("delete-me", provider="mock")
    world_id = forge.save_world(world)

    assert world_id in forge.list_worlds()
    assert forge.delete_world(world_id) == world_id
    assert world_id not in forge.list_worlds()

    with pytest.raises(WorldStateError, match="not present"):
        forge.delete_world(world_id)

    with pytest.raises(WorldForgeError, match="file-safe identifier"):
        forge.delete_world("../outside")


def test_world_delete_can_remove_corrupted_local_json_by_safe_id(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{not valid json", encoding="utf-8")

    assert "broken" in forge.list_worlds()
    assert forge.delete_world("broken") == "broken"
    assert not broken_path.exists()


def test_prompt_seeded_world_history_and_forking(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    seeded = forge.create_world_from_prompt(
        "A kitchen with a mug", provider="mock", name="seeded-kitchen"
    )

    assert seeded.name == "seeded-kitchen"
    assert seeded.description == "A kitchen with a mug"
    assert seeded.object_count >= 2

    prediction = seeded.predict(Action.move_to(0.25, 0.5, 0.0), steps=2)
    assert prediction.provider == "mock"
    assert seeded.history_length >= 2

    history = seeded.history()
    assert history[0].action_json is None
    assert history[-1].action_json is not None

    checkpoint = seeded.history_state(0)
    assert checkpoint.step == 0
    assert checkpoint.history_length == 1

    seeded.restore_history(0)
    assert seeded.step == 0

    saved_id = forge.save_world(seeded)
    forked = forge.fork_world(saved_id, history_index=0, name="seeded-kitchen-branch")
    assert forked.id != saved_id
    assert forked.name == "seeded-kitchen-branch"
    assert forked.history_length == 1


def test_world_rejects_invalid_runtime_inputs(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    world = forge.create_world("validation-world", "mock")

    with pytest.raises(WorldForgeError, match="steps"):
        world.predict(Action.move_to(0.1, 0.2, 0.3), steps=0)

    with pytest.raises(WorldForgeError, match="compare\\(\\) requires at least one provider"):
        world.compare(Action.move_to(0.1, 0.2, 0.3), [], steps=1)

    with pytest.raises(WorldForgeError, match="max_steps"):
        world.plan(goal="spawn cube", max_steps=0)

    with pytest.raises(WorldForgeError, match="History index"):
        world.history_state(1)

    with pytest.raises(WorldForgeError, match="not present in world"):
        world.update_object_patch("missing-object", SceneObjectPatch(name="ghost"))


def test_world_import_and_load_reject_malformed_state(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)

    with pytest.raises(WorldStateError, match="not valid JSON"):
        forge.import_world("{broken json")

    with pytest.raises(WorldStateError, match="missing required keys"):
        forge.import_world(json.dumps({"state": {"name": "invalid"}}))

    with pytest.raises(WorldStateError, match="JSON object"):
        forge.import_world(json.dumps({"state": "not-a-world"}))

    unsafe_state = {
        "id": "../outside",
        "name": "invalid",
        "provider": "mock",
        "scene": {"objects": {}},
        "metadata": {},
        "step": 0,
    }
    with pytest.raises(WorldStateError, match="file-safe identifier"):
        forge.import_world(json.dumps(unsafe_state))

    with pytest.raises(WorldForgeError, match="file-safe identifier"):
        forge.load_world("../outside")

    broken_world_path = tmp_path / "broken.json"
    broken_world_path.write_text('{"state": "not-a-world"}', encoding="utf-8")

    with pytest.raises(WorldStateError, match="World file"):
        forge.load_world("broken")


def test_world_import_rejects_malformed_history_entries(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    valid_history_state = {
        "id": "world_history",
        "name": "history",
        "provider": "mock",
        "scene": {"objects": {}},
        "metadata": {},
        "step": 0,
    }
    base_state = {
        **valid_history_state,
        "step": 1,
        "history": [
            {
                "step": 0,
                "state": valid_history_state,
                "summary": "world initialized",
                "action_json": None,
            }
        ],
    }

    restored = forge.import_world(json.dumps(base_state))
    assert restored.history_length == 1

    malformed_cases = [
        (
            {**base_state, "history": ["not-an-entry"]},
            "history\\[0\\] must be a JSON object",
        ),
        (
            {
                **base_state,
                "history": [{**base_state["history"][0], "step": -1}],
            },
            "HistoryEntry step",
        ),
        (
            {
                **base_state,
                "history": [{**base_state["history"][0], "step": 2}],
            },
            "step must not be greater than current step",
        ),
        (
            {
                **base_state,
                "history": [{**base_state["history"][0], "summary": ""}],
            },
            "summary",
        ),
        (
            {
                **base_state,
                "history": [{**base_state["history"][0], "action_json": "{broken"}],
            },
            "action_json",
        ),
        (
            {
                **base_state,
                "history": [
                    {
                        **base_state["history"][0],
                        "state": {**valid_history_state, "id": "../bad"},
                    }
                ],
            },
            "invalid state",
        ),
    ]

    for payload, message in malformed_cases:
        with pytest.raises(WorldStateError, match=message):
            forge.import_world(json.dumps(payload))
