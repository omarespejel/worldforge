"""Basic prediction example for the pure-Python WorldForge package."""

from worldforge import Action, BBox, Position, SceneObject, WorldForge


def main() -> None:
    forge = WorldForge()
    world = forge.create_world("kitchen", provider="mock")

    world.add_object(
        SceneObject(
            "red_mug",
            Position(0.0, 0.8, 0.0),
            BBox(Position(-0.05, 0.75, -0.05), Position(0.05, 0.85, 0.05)),
        )
    )

    prediction = world.predict(Action.move_to(0.3, 0.8, 0.0), steps=2)
    print("prediction:", prediction.provider, prediction.physics_score)

    plan = world.plan(goal="move the mug to the right", verify_backend="mock")
    print("plan:", plan.action_count, plan.success_probability)

    report = world.evaluate("physics")
    print(report.to_markdown())


if __name__ == "__main__":
    main()
