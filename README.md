# WorldForge

**The orchestration layer for world models.**

WorldForge is a developer toolkit that provides a unified interface for building applications on top of world foundation models (WFMs). It abstracts the differences between providers (NVIDIA Cosmos, Runway GWM, Meta JEPA, Google Genie, and others) behind a single, ergonomic API — letting developers focus on what to build rather than how to integrate.

Think LangChain for world models. Or Vercel for physical AI.

## Why WorldForge Exists

The world model ecosystem in 2026 looks like the LLM ecosystem in early 2023:

- **Foundation models exist** (Cosmos Predict/Transfer/Reason, GWM-1 Worlds/Robotics/Avatars, V-JEPA 2, Genie 3, Marble)
- **Each has its own SDK** (NVIDIA NIM + NGC, Runway Python/Node SDK, Meta research code, Google research preview)
- **No unified abstraction** — every developer writes custom integration code for each provider
- **No orchestration** — composing multi-step workflows (predict → evaluate → plan → verify) requires manual plumbing
- **No persistent state** — world models are stateless; managing scene state across calls is the developer's problem
- **No safety layer** — guardrails are provider-specific and non-portable

WorldForge fills every gap.

## Core Concepts

### Worlds
A `World` is a persistent, stateful environment. It wraps a world model provider and maintains scene state, history, and configuration across inference calls.

### Actions
An `Action` is a standardized representation of something an agent can do. Move, grasp, rotate, navigate — defined once, translated to each provider's format automatically.

### Predictions
A `Prediction` is the result of asking a world model "what happens next?" It contains the predicted future state, confidence scores, and optional video/image output.

### Plans
A `Plan` is a sequence of actions optimized to reach a goal state. WorldForge can use gradient-based planning (for differentiable world models like JEPA) or sampling-based planning (for generative models like Cosmos/GWM).

### Guardrails
A `Guardrail` is a safety constraint. Define forbidden states, energy thresholds, or physical laws. WorldForge checks every prediction against guardrails before returning results. Optional: ZK verification of guardrail compliance for safety-critical applications.

## Quick Example

```python
from worldforge import WorldForge, World, Action
from worldforge.providers import CosmosProvider, RunwayProvider

# Initialize with any provider
wf = WorldForge(provider=CosmosProvider(model="cosmos-predict-2.5"))

# Create a world from a text description
world = wf.create_world("A kitchen counter with a red mug and a plate")

# Predict what happens if we push the mug
prediction = world.predict(
    action=Action.push(target="red_mug", direction="left", force=0.5),
    steps=10
)

# Check physics plausibility
score = prediction.physics_score()  # 0.0 - 1.0

# Plan a sequence of actions to achieve a goal
plan = world.plan(
    goal="The red mug is inside the dishwasher",
    max_steps=20,
    guardrails=["no_collisions", "mug_stays_upright"]
)

# Switch providers seamlessly
world.set_provider(RunwayProvider(model="gwm-1-robotics"))
prediction_2 = world.predict(action=plan.actions[0])

# Compare predictions across providers
comparison = wf.compare([prediction, prediction_2])
```

## Architecture

```
worldforge/
├── worldforge-core/        # Core library (Rust)
│   ├── src/
│   │   ├── world.rs        # World state management
│   │   ├── action.rs       # Action type system
│   │   ├── prediction.rs   # Prediction handling
│   │   ├── plan.rs         # Planning algorithms
│   │   ├── guardrail.rs    # Safety constraints
│   │   ├── scene.rs        # Scene graph representation
│   │   └── provider.rs     # Provider trait
│   └── Cargo.toml
├── worldforge-py/          # Python bindings (PyO3)
│   ├── src/lib.rs
│   └── worldforge/*.py
├── worldforge-providers/   # Provider adapters
│   ├── cosmos/             # NVIDIA Cosmos
│   ├── runway/             # Runway GWM-1
│   ├── jepa/               # Meta JEPA family
│   ├── genie/              # Google Genie
│   └── local/              # Local inference (burn/candle)
├── worldforge-eval/        # Evaluation framework
├── worldforge-verify/      # ZK verification (optional)
├── worldforge-server/      # REST API server
└── worldforge-cli/         # CLI tool
```

## Status

Pre-alpha. Building in public. Star the repo and watch for updates.

## License

Apache 2.0 (core library)
NVIDIA Open Model License (for Cosmos-derived components)

## Links

- [Specification](./SPECIFICATION.md)
- [Architecture Decision Records](./architecture/)
- [Business Plan](./business/)
- [Contributing](./CONTRIBUTING.md)
