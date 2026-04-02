# Architecture

WorldForge is organized as a Cargo workspace with seven crates, each with a
focused responsibility. This page describes the role of each crate and how
they interact.

## Crate Map

```
worldforge/
├── crates/
│   ├── worldforge-core/       # Types, traits, state, scene graph, planning
│   ├── worldforge-providers/  # 11 provider adapters + polling infrastructure
│   ├── worldforge-eval/       # 12 eval dimensions + WR-Arena datasets
│   ├── worldforge-verify/     # ZK verification (STARK, EZKL)
│   ├── worldforge-server/     # REST API (27 endpoints via Axum)
│   ├── worldforge-cli/        # CLI tool (27 commands via Clap)
│   └── worldforge-python/     # PyO3 bindings for Python
├── python/worldforge/         # Python package shim
└── pyproject.toml             # Maturin build config
```

## worldforge-core

The foundation crate. Defines all shared types:

- `Action`, `WorldState`, `Prediction`, `Plan` — core domain types.
- `WorldProvider` trait — the contract every provider must implement.
- `SceneGraph` — spatial representation of objects and relationships.
- `Guardrails` — safety constraints applied before and after predictions.
- `PlanningAlgorithm` — trait for CEM, sampling, MPC, gradient planners.

## worldforge-providers

Implements the `WorldProvider` trait for each supported backend:

- Cloud APIs: Cosmos, Runway, Sora, Veo, PAN, KLING, MiniMax.
- Local models: JEPA, Genie, Marble.
- Testing: Mock (deterministic, always available).
- Shared infrastructure for polling, retries, and rate limiting.

## worldforge-eval

Evaluation framework with 12 scoring dimensions covering physics fidelity,
spatial reasoning, and WR-Arena benchmarks. Supports JSON, Markdown, and CSV
output. Ships with built-in suites: physics, manipulation, spatial, comprehensive.

## worldforge-verify

Zero-knowledge proof generation for guardrail compliance. Supports STARK and
EZKL backends. Produces proofs that can be verified independently.

## worldforge-server

Axum-based REST API exposing 27 endpoints under `/v1/`. Handles world lifecycle,
predictions, planning, evaluation, comparison, and provider health checks.
Generates OpenAPI specs automatically.

## worldforge-cli

Clap-based command-line interface mirroring every server endpoint. Designed for
scripting and CI/CD pipelines.

## worldforge-python

PyO3 bindings that expose core types and the `WorldForge` orchestrator to
Python. Built with Maturin. The `python/worldforge/` directory provides the
package shim that re-exports the native extension.

## Data Flow

1. User submits an action (via Python, Rust, CLI, or REST).
2. `worldforge-core` validates the action against guardrails.
3. `worldforge-providers` dispatches to the selected backend.
4. The provider returns a prediction (frames + metadata).
5. `worldforge-eval` optionally scores the prediction.
6. `worldforge-verify` optionally generates a ZK proof.
7. Results are returned to the caller.
