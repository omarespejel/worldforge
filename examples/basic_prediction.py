"""Basic prediction example using the WorldForge Python SDK.

This example demonstrates how to create a world and run a single prediction.
The Mock provider is always available — no API keys needed.

Usage:
    pip install worldforge
    python examples/basic_prediction.py
"""

from worldforge import WorldForge, Action


def main() -> None:
    # Initialize WorldForge — auto-detects providers from env vars
    wf = WorldForge()

    # List available providers
    providers = wf.list_providers()
    print(f"Available providers: {[p.name for p in providers]}")

    # Create a world using the mock provider (always available)
    world = wf.create_world("kitchen", provider="mock")
    print("Created world: kitchen (provider: mock)")

    # Define an action: move an object to coordinates (0.5, 0.8, 0.0)
    action = Action.move_to(0.5, 0.8, 0.0)

    # Run a prediction for 10 steps
    prediction = world.predict(action, steps=10)

    # Print results
    print(f"Prediction complete:")
    print(f"  Physics score: {prediction.physics_score:.2f}")
    print(f"  Frames generated: {len(prediction.frames)}")

    # Plan a multi-step task
    plan = world.plan(
        goal="red mug in the dishwasher",
        planner="cem",
        max_steps=20,
    )
    print(f"\nPlan computed:")
    print(f"  Steps: {len(plan.actions)}")
    print(f"  P(success): {plan.success_probability:.2f}")

    # Run an evaluation
    report = world.evaluate(suite="physics")
    print(f"\nEvaluation report:\n{report.to_markdown()}")


if __name__ == "__main__":
    main()
