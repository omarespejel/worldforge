# Architecture Decision Records

## ADR-001: Rust Core with Python Bindings

**Status:** Accepted

**Context:** World model developers primarily use Python (PyTorch, TensorFlow, JAX). However, world model inference is latency-sensitive, and deployment targets include edge devices, robots, and WASM browsers.

**Decision:** Core library in Rust with Python bindings via PyO3.

**Rationale:**
- Rust provides zero-cost abstractions, no GC pauses, and memory safety
- PyO3 provides seamless Python integration (feels native to Python users)
- Rust compiles to WASM for browser deployment
- Rust compiles to ARM for edge/robot deployment
- Performance overhead < 5ms per WorldForge call (vs 20-50ms in pure Python)
- Type safety catches integration errors at compile time

**Consequences:**
- Higher initial development cost (Rust is harder to write)
- Smaller contributor pool (fewer Rust developers than Python)
- Need to maintain both Rust and Python interfaces
- Mitigated by: auto-generated Python stubs, extensive documentation, good CI

---

## ADR-002: Provider Trait Pattern

**Status:** Accepted

**Context:** Need to support N world model providers with different APIs, capabilities, and data formats.

**Decision:** Use a Rust trait `WorldModelProvider` that all providers implement. Provider adapters are separate crates.

**Rationale:**
- Trait pattern allows compile-time verification of interface compliance
- Separate crates allow independent versioning of provider adapters
- New providers can be added without modifying core library
- Capabilities are introspectable at runtime

**Consequences:**
- Provider adapters must be maintained as APIs evolve
- Some provider-specific features may not map cleanly to the trait

---

## ADR-003: Scene Graph for World State

**Status:** Accepted

**Context:** Need a standardized way to represent world state that works across providers.

**Decision:** Use a scene graph with objects, poses, physics properties, and spatial relationships.

**Rationale:**
- Scene graphs are the industry standard for 3D environments (OpenUSD, GLTF, Omniverse)
- Providers that work with text prompts can serialize scene graphs to text
- Providers that work with 3D inputs can convert scene graphs directly
- Spatial relationships capture semantic structure (not just geometry)

**Consequences:**
- Not all providers can reconstruct a full scene graph from their outputs
- Some information loss when converting between representations
- Mitigated by: optional fields, provider-specific metadata escape hatch

---

## ADR-004: ZK Verification as Optional Module

**Status:** Accepted

**Context:** ZK verification is WorldForge's unique differentiator but adds complexity.

**Decision:** ZK verification is a separate crate (worldforge-verify) that's optional.

**Rationale:**
- Most users don't need ZK verification (only safety-critical applications)
- ZK proof generation is computationally expensive
- Keeping it optional means the core library stays lightweight
- Users who need it can opt in without affecting others
- This allows independent development and release cycles

**Consequences:**
- ZK module has its own dependency tree (Cairo, STARK libraries)
- Need to ensure the interface between core and verify is stable
- Marketing must clearly communicate when ZK verification is needed vs. optional

---

## ADR-005: State Persistence Strategy

**Status:** Accepted

**Context:** World models are stateless (each call is independent). WorldForge needs to provide persistent state across calls.

**Decision:** Pluggable state store with multiple backends (file, SQLite, Redis, S3).

**Rationale:**
- Different deployment scenarios need different persistence backends
- Local development: file or SQLite
- Cloud deployment: Redis or S3
- Edge deployment: file or in-memory
- Pluggable interface allows users to implement custom backends

**Consequences:**
- State serialization overhead (mitigated by MessagePack + LZ4 compression)
- Need to handle state versioning and migration
- Concurrent access to shared state needs coordination (Redis handles this natively)

---

## ADR-006: Evaluation as First-Class Citizen

**Status:** Accepted

**Context:** No standardized evaluation framework exists for world models. This is a unique opportunity.

**Decision:** Ship an evaluation framework (worldforge-eval) as part of the core distribution, not an afterthought.

**Rationale:**
- Evaluation is how WorldForge becomes the reference standard
- Provider-agnostic evaluation drives demand for WorldForge
- Public leaderboards drive community engagement
- Evaluation data is valuable for research and product decisions

**Consequences:**
- Need to maintain evaluation scenarios and ground truth data
- Evaluation results must be reproducible
- Public leaderboard requires hosting and moderation

---

## ADR-007: Open Source Licensing Strategy

**Status:** Accepted

**Decision:**
- Core library + all provider adapters + eval framework: Apache 2.0
- Cloud infrastructure + dashboard: Proprietary
- ZK verification module: Apache 2.0

**Rationale:**
- Apache 2.0 maximizes adoption (permissive, enterprise-friendly)
- Cloud is the monetization layer (standard open-core model)
- ZK verification is open because it needs to be auditable for trust
- This mirrors the LangChain, Hugging Face, and Supabase playbooks

---

## ADR-008: CLI-First Development

**Status:** Accepted

**Context:** Developers experiment with world models interactively before building applications.

**Decision:** Ship a CLI tool (worldforge-cli) that provides full functionality from the terminal.

**Rationale:**
- CLI enables rapid experimentation without writing code
- CLI output can be piped to other tools
- CLI is the fastest way to demo WorldForge
- CLI exercises the full stack (creates a natural integration test)
- Aligns with "Claude Code" / "agentic CLI tool" culture
