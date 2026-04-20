# Introduction

WorldForge is a Python integration layer for physical-AI world-model systems. It provides provider
capability contracts, world state, planning, evaluation, benchmarking, and diagnostics for local
experiments and adapter development.

The project is built around a strict provider boundary:

- predictive models roll state forward through `predict`
- score models rank candidate action sequences through `score`
- embodied policies propose action chunks through `policy`
- video and media systems expose `generate` or `transfer`
- auxiliary models expose `reason` or `embed` only when those operations are implemented directly

This keeps provider semantics honest. A JEPA cost model is not treated as a video generator. A VLA
robot policy is not treated as a predictive dynamics model. A media generation API is not treated
as proof of controllable physical planning.

WorldForge is for Python developers building provider adapters, local physical-AI experiments,
world-model planning loops, evaluation harnesses, and testable prototypes. It is not a hosted
control plane and it does not own checkpoints, robot runtimes, production telemetry, or durable
multi-writer persistence.

Start with:

- [Quick Start](./quickstart.md) for the Python and CLI path.
- [World Model Taxonomy](./world-model-taxonomy.md) for terminology and capability boundaries.
- [Architecture](./architecture.md) for the provider pipeline and planning modes.
- [User And Operator Playbooks](./playbooks.md) for checkout validation, provider diagnostics,
  persistence recovery, optional runtime smokes, benchmarks, and release gates.
- [Provider Authoring Guide](./provider-authoring-guide.md) before adding an adapter.
