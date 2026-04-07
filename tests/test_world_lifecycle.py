from __future__ import annotations

import json

from worldforge import Action, BBox, Position, SceneObject, SceneObjectPatch, WorldForge


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
