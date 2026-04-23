from __future__ import annotations

import json

import pytest

from worldforge import (
    Action,
    BBox,
    Position,
    SceneObject,
    StructuredGoal,
    WorldForge,
    WorldForgeError,
    list_eval_suites,
    run_eval,
)
from worldforge.evaluation import EvaluationSuite
from worldforge.providers import MockProvider


def _seed_world(forge: WorldForge):
    world = forge.create_world("eval-world", "mock")
    cube = world.add_object(
        SceneObject(
            "cube",
            Position(0.0, 0.5, 0.0),
            BBox(Position(-0.05, 0.45, -0.05), Position(0.05, 0.55, 0.05)),
        )
    )
    mug = world.add_object(
        SceneObject(
            "mug",
            Position(0.25, 0.8, 0.0),
            BBox(Position(0.2, 0.75, -0.05), Position(0.3, 0.85, 0.05)),
        )
    )
    return world, cube, mug


def test_planning_comparison_and_execution_flow(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    forge.register_provider(MockProvider(name="manual-mock"))

    world = forge.create_world("goal-json-world", "manual-mock")
    ball = SceneObject(
        "ball",
        Position(0.0, 0.5, 0.0),
        BBox(Position(-0.1, 0.4, -0.1), Position(0.1, 0.6, 0.1)),
    )
    ball_id = ball.id
    world.add_object(ball)

    goal_json = json.dumps(
        {
            "type": "condition",
            "condition": {
                "ObjectAt": {
                    "object": ball_id,
                    "position": {"x": 1.0, "y": 0.5, "z": 0.0},
                    "tolerance": 0.05,
                }
            },
        }
    )

    computed_plan = world.plan(
        goal_json=goal_json,
        max_steps=4,
        provider="manual-mock",
        planner="sampling",
    )
    assert computed_plan.action_count > 0
    assert computed_plan.goal_spec == {
        "kind": "object_at",
        "object": {"id": ball_id},
        "position": {"x": 1.0, "y": 0.5, "z": 0.0},
        "tolerance": 0.05,
    }

    plan_json = json.loads(computed_plan.to_json())
    final_state = plan_json["predicted_states"][-1]
    final_ball = next(
        obj for obj in final_state["scene"]["objects"].values() if obj["name"] == "ball"
    )
    assert abs(final_ball["pose"]["position"]["x"] - 1.0) <= 0.15
    assert abs(final_ball["pose"]["position"]["y"] - 0.5) <= 0.15

    execution = world.execute_plan(computed_plan, 1, "manual-mock")
    final_world = execution.final_world()
    final_prediction = final_world.predict(Action.move_to(0.45, 0.5, 0.0), steps=1)
    assert final_prediction.provider == "manual-mock"

    module_plan = world.plan(goal="spawn cube", max_steps=3, provider="manual-mock")
    assert module_plan.action_count >= 1


def test_structured_goal_targets_selected_object_and_validates_inputs(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    world = forge.create_world("structured-goal-world", "mock")
    cube = world.add_object(
        SceneObject(
            "cube",
            Position(0.0, 0.5, 0.0),
            BBox(Position(-0.05, 0.45, -0.05), Position(0.05, 0.55, 0.05)),
        )
    )
    mug = world.add_object(
        SceneObject(
            "mug",
            Position(0.25, 0.8, 0.0),
            BBox(Position(0.2, 0.75, -0.05), Position(0.3, 0.85, 0.05)),
        )
    )

    goal_spec = StructuredGoal.object_at(
        object_id=mug.id,
        object_name="mug",
        position=Position(0.8, 0.8, 0.0),
    )
    computed_plan = world.plan(goal_spec=goal_spec, provider="mock", max_steps=2)
    execution = world.execute_plan(computed_plan, "mock")
    final_world = execution.final_world()

    final_cube = final_world.get_object_by_id(cube.id)
    final_mug = final_world.get_object_by_id(mug.id)
    assert final_cube is not None
    assert final_mug is not None
    assert final_cube.position == cube.position
    assert final_mug.position.x == pytest.approx(0.8)
    assert computed_plan.goal_spec == goal_spec.to_dict()

    with pytest.raises(WorldForgeError, match="requires goal, goal_json, or goal_spec"):
        world.plan()

    ambiguous_world = forge.create_world("ambiguous-goal-world", "mock")
    ambiguous_world.add_object(
        SceneObject(
            "mug",
            Position(0.0, 0.8, 0.0),
            BBox(Position(-0.05, 0.75, -0.05), Position(0.05, 0.85, 0.05)),
        )
    )
    ambiguous_world.add_object(
        SceneObject(
            "mug",
            Position(0.3, 0.8, 0.0),
            BBox(Position(0.25, 0.75, -0.05), Position(0.35, 0.85, 0.05)),
        )
    )

    with pytest.raises(WorldForgeError, match="ambiguous"):
        ambiguous_world.plan(
            goal_spec=StructuredGoal.object_at(
                object_name="mug",
                position=Position(0.6, 0.8, 0.0),
            )
        )


def test_structured_goal_relational_goals_parse_execute_and_validate(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    world, cube, mug = _seed_world(forge)

    near_goal = StructuredGoal.object_near(
        object_id=cube.id,
        object_name=cube.name,
        reference_object_id=mug.id,
        reference_object_name=mug.name,
        offset=Position(0.15, 0.0, 0.0),
        tolerance=0.05,
    )
    near_plan = world.plan(goal_spec=near_goal, provider="mock", max_steps=2)
    near_execution = world.execute_plan(near_plan, "mock")
    near_world = near_execution.final_world()
    final_cube = near_world.get_object_by_id(cube.id)
    final_mug = near_world.get_object_by_id(mug.id)
    assert final_cube is not None
    assert final_mug is not None
    assert final_cube.position.x == pytest.approx(mug.position.x + 0.15)
    assert final_cube.position.y == pytest.approx(mug.position.y)
    assert final_mug.position == mug.position
    assert near_plan.goal_spec == near_goal.to_dict()

    swap_plan = world.plan(
        goal_json=json.dumps(
            {
                "type": "condition",
                "condition": {
                    "SwapObjects": {
                        "first_object": {"id": cube.id, "name": cube.name},
                        "second_object": {"id": mug.id, "name": mug.name},
                        "tolerance": 0.05,
                    }
                },
            }
        ),
        provider="mock",
        max_steps=4,
    )
    assert swap_plan.action_count == 2
    assert swap_plan.goal_spec == {
        "kind": "swap_objects",
        "object": {"id": cube.id, "name": cube.name},
        "reference_object": {"id": mug.id, "name": mug.name},
        "tolerance": 0.05,
    }

    swap_execution = world.execute_plan(swap_plan, "mock")
    swapped_world = swap_execution.final_world()
    swapped_cube = swapped_world.get_object_by_id(cube.id)
    swapped_mug = swapped_world.get_object_by_id(mug.id)
    assert swapped_cube is not None
    assert swapped_mug is not None
    assert swapped_cube.position == mug.position
    assert swapped_mug.position == cube.position

    default_offset_goal = StructuredGoal.object_near(
        object_id=cube.id,
        reference_object_id=mug.id,
    )
    assert default_offset_goal.offset == Position(0.1, 0.0, 0.0)

    with pytest.raises(WorldForgeError, match="distinct primary and reference objects"):
        StructuredGoal.swap_objects(object_id=cube.id, reference_object_id=cube.id)


def test_structured_goal_parser_rejects_invalid_relational_inputs() -> None:
    with pytest.raises(WorldForgeError, match="goal_json must be valid JSON"):
        StructuredGoal.from_json("{broken")

    with pytest.raises(WorldForgeError, match="Structured goals must decode to a JSON object"):
        StructuredGoal.from_dict(["not-a-goal"])  # type: ignore[arg-type]

    with pytest.raises(WorldForgeError, match="field 'reference_object' must be a JSON object"):
        StructuredGoal.from_dict(
            {
                "kind": "object_near",
                "object": {"name": "cube"},
                "reference_object": 42,
            }
        )

    with pytest.raises(WorldForgeError, match="do not accept reference_object selectors"):
        StructuredGoal(
            kind="object_at",
            object_name="cube",
            position=Position(0.1, 0.5, 0.0),
            reference_object_name="mug",
        )

    with pytest.raises(WorldForgeError, match="do not accept object_id"):
        StructuredGoal(kind="spawn_object", object_name="cube", object_id="obj_bad")

    legacy_near = StructuredGoal.from_dict(
        {
            "type": "condition",
            "condition": {
                "ObjectNear": {
                    "object": "cube",
                    "anchor": "mug",
                    "offset": {"x": 0.2, "y": 0.0, "z": 0.0},
                    "tolerance": 0.05,
                }
            },
        }
    )
    assert legacy_near.object_name == "cube"
    assert legacy_near.reference_object_name == "mug"
    assert legacy_near.offset == Position(0.2, 0.0, 0.0)


def test_evaluation_reports_and_eval_helpers(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    forge.register_provider(MockProvider(name="manual-mock"))
    world, _, _ = _seed_world(forge)

    assert list_eval_suites() == ["generation", "physics", "planning", "reasoning", "transfer"]

    suite = EvaluationSuite.from_builtin("physics")
    report = suite.run_report(["mock", "manual-mock"], world=world, forge=forge)
    assert report.suite_id == "physics"
    assert "Physics" in report.suite
    assert {summary.provider for summary in report.provider_summaries} == {"manual-mock", "mock"}
    assert len(report.results) == 4
    assert {result.scenario for result in report.results} == {
        "object-stability",
        "action-response",
    }
    assert all(result.passed for result in report.results)

    artifacts = suite.run_report_artifacts(
        providers=["mock", "manual-mock"],
        world=world,
        forge=forge,
    )
    assert set(artifacts) == {"json", "markdown", "csv"}
    assert json.loads(artifacts["json"])["suite_id"] == "physics"
    assert artifacts["markdown"].startswith("# Evaluation Report")
    assert "metrics_json" in artifacts["csv"]

    results = run_eval("physics", "mock", forge=forge)
    assert len(results) == 2
    assert results[0].provider == "mock"
    assert all(result.passed for result in results)

    generation_results = run_eval("generation", "mock", forge=forge)
    assert len(generation_results) == 2
    assert all(result.passed for result in generation_results)

    transfer_results = run_eval("transfer", "manual-mock", forge=forge)
    assert len(transfer_results) == 2
    assert all(result.passed for result in transfer_results)


def test_planning_and_reasoning_suites_cover_core_workflows(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    world, cube, mug = _seed_world(forge)

    planning_report = world.evaluate("planning")
    assert planning_report.suite_id == "planning"
    assert len(planning_report.results) == 4
    assert {result.scenario for result in planning_report.results} == {
        "object-relocation",
        "object-neighbor-placement",
        "object-swap",
        "object-spawn",
    }
    assert all(result.passed for result in planning_report.results)

    reasoning_report = EvaluationSuite.from_builtin("reasoning").run_report(
        "mock",
        world=world,
        forge=forge,
    )
    assert reasoning_report.suite_id == "reasoning"
    assert len(reasoning_report.results) == 2
    assert all(result.passed for result in reasoning_report.results)
    identity_result = next(
        result for result in reasoning_report.results if result.scenario == "scene-identity"
    )
    assert identity_result.metrics["tracked_object_count"] == 2
    assert identity_result.metrics["matched_object_count"] == len({cube.id, mug.id})


def test_evaluation_suite_validation_errors_are_explicit(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path, auto_register_remote=False)

    with pytest.raises(WorldForgeError, match="Unknown evaluation suite"):
        EvaluationSuite.from_builtin("unknown")

    with pytest.raises(WorldForgeError, match="missing required capabilities: reason"):
        EvaluationSuite.from_builtin("reasoning").run_report("cosmos", forge=forge)

    with pytest.raises(WorldForgeError, match="missing required capabilities: transfer"):
        EvaluationSuite.from_builtin("transfer").run_report("cosmos", forge=forge)
