from __future__ import annotations

import json

from worldforge import (
    Action,
    BBox,
    Position,
    SceneObject,
    WorldForge,
    list_eval_suites,
    plan,
    run_eval,
)
from worldforge.evaluation import EvaluationSuite
from worldforge.providers import MockProvider


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

    module_plan = plan(world, goal="spawn cube", max_steps=3, provider="manual-mock")
    assert module_plan.action_count >= 1


def test_evaluation_reports_and_eval_helpers(tmp_path) -> None:
    forge = WorldForge(state_dir=tmp_path)
    world = forge.create_world("eval-world", "mock")
    world.add_object(
        SceneObject(
            "cube",
            Position(0.0, 0.5, 0.0),
            BBox(Position(-0.05, 0.45, -0.05), Position(0.05, 0.55, 0.05)),
        )
    )

    assert "physics" in list_eval_suites()

    suite = EvaluationSuite.from_builtin("physics")
    report = suite.run_report("mock", world=world, forge=forge)
    assert "Physics" in report.suite
    assert len(report.provider_summaries) >= 1
    assert report.provider_summaries[0].provider == "mock"

    artifacts = suite.run_report_artifacts(providers="mock", world=world, forge=forge)
    assert set(artifacts) == {"json", "markdown", "csv"}
    assert artifacts["markdown"].startswith("# Evaluation Report")

    results = run_eval("physics", "mock", forge=forge)
    assert len(results) >= 1
    assert results[0].provider == "mock"
