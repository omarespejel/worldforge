# Introduction

## What is WorldForge?

WorldForge is a unified orchestration layer for world foundation models (WFMs).
It provides a single API surface to predict, plan, evaluate, and verify physical
world simulations across multiple providers including NVIDIA Cosmos, Runway GWM,
Meta JEPA, Google Veo, OpenAI Sora, KLING, MiniMax, PAN, and more.

## The Problem

World foundation models are emerging as a critical tool for robotics, autonomous
driving, and embodied AI. However, each provider exposes a different API, uses
different data formats, and returns results in incompatible structures. Building
on top of multiple WFMs today requires significant glue code and per-provider
expertise.

## The Solution

WorldForge abstracts provider differences behind a clean, unified interface:

- **Predict**: Given a world state and an action, predict the next state.
- **Plan**: Given a goal, synthesize a multi-step action plan.
- **Evaluate**: Score predictions across 12 physics dimensions.
- **Verify**: Generate zero-knowledge proofs of guardrail compliance.

All four operations work identically regardless of the underlying provider.

## Key Features

- **11 providers** with automatic detection from environment variables.
- **27 REST API endpoints** with a built-in server.
- **12 evaluation dimensions** covering physics, spatial reasoning, and WR-Arena.
- **ZK verification** via STARK and EZKL backends.
- **Scene graph** representation of world state.
- **Planning algorithms**: CEM, sampling, MPC, and gradient-based.
- **Python bindings** via PyO3 for seamless integration.
- **CLI tool** with 27 commands for scripting and automation.

## Who Is It For?

- Robotics engineers building manipulation or navigation pipelines.
- Researchers comparing world models across providers.
- Platform teams building simulation-as-a-service.
- Anyone who needs physics-grounded predictions from foundation models.

## Design Principles

1. **Provider-agnostic**: Code once, run on any WFM.
2. **Type-safe**: Strong Rust types propagated to Python via PyO3.
3. **Async-first**: All provider calls are non-blocking.
4. **Observable**: Structured logging, metrics, and health checks.
5. **Extensible**: Adding a new provider is a single trait implementation.

## License

WorldForge is licensed under Apache 2.0.
