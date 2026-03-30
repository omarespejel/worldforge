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

---

## ADR-009: WR-Arena Evaluation Integration

**Status:** Accepted

**Context:** WR-Arena (arXiv 2603.25887, March 2026) introduces a diagnostic benchmark for world foundation models that evaluates 10 models across 4 dimensions. Their evaluation methodology goes beyond visual quality to test instruction-following, temporal smoothness, generation consistency, and planning capability. No existing tool provides a unified way to run these evaluations across providers.

**Decision:** Adopt WR-Arena's 4 evaluation dimensions as first-class metrics in worldforge-eval, alongside our existing physics-based evaluation dimensions.

**New evaluation dimensions:**
1. **ActionSimulationFidelity** — LLM-as-judge scoring (0-3) for instruction following
2. **TransitionSmoothness** — MRS metric via optical flow for temporal quality
3. **GenerationConsistency** — WorldScore-based multi-aspect consistency scoring
4. **SimulativeReasoning** — VLM+WFM planning loop evaluation

**Rationale:**
- WR-Arena is the first systematic benchmark for WFMs. Integrating it positions WorldForge as the reference evaluation platform.
- Their 4 dimensions are complementary to our 8 physics dimensions, creating a comprehensive 12-dimension evaluation framework.
- LLM-as-judge scoring is provider-agnostic and doesn't require ground truth video data.
- The MRS smoothness metric is computationally simple and highly informative for multi-round generation quality.
- WR-Arena datasets (60 action sim, 100 smoothness, 100 consistency instances) provide ready-made benchmarks.

**Consequences:**
- LLM-as-judge evaluation requires multimodal LLM API access (additional cost)
- Optical flow estimation for MRS may require Python sidecar or Rust bindings
- WorldScore integration depends on external models (DROID-SLAM, GroundingDINO, etc.)
- Mitigated by: starting with simpler metrics (Action Fidelity, MRS) before full WorldScore

---

## ADR-010: Expanded Provider Support (API-based video generators)

**Status:** Accepted

**Context:** WR-Arena evaluates 10 world models. WorldForge currently supports 5 (Cosmos, Runway, JEPA, Genie, Marble). The missing models — PAN, KLING, Sora 2, Veo 3, MiniMax — are all commercially available via REST APIs and represent the majority of production video generation usage.

**Decision:** Add provider adapters for PAN, KLING, Sora 2, Veo 3, and MiniMax to worldforge-providers. Defer WAN 2.x (requires multi-GPU local infrastructure).

**Rationale:**
- PAN is the best planning model per WR-Arena. Supporting it makes WorldForge the primary interface for the best WFM planner.
- KLING, Sora 2, Veo 3, MiniMax are the most popular commercial video generators. Supporting them makes WorldForge relevant to the largest user base.
- All 5 new providers use REST APIs with submit/poll/download patterns, which maps cleanly to our async provider trait.
- WAN 2.x requires multi-GPU orchestration (8 GPUs, Ulysses parallelism) which is beyond our initial scope.

**Provider-specific decisions:**
- PAN: Implement stateful multi-round API (first_round/continue endpoints with state_id tracking)
- KLING: JWT authentication (HS256 signing) — add jsonwebtoken dependency
- Sora 2: Direct REST API calls (not OpenAI SDK — keep dependency-free in Rust)
- Veo 3: Direct REST API calls to GenAI endpoint
- MiniMax: Standard REST submit/poll/download

**Consequences:**
- 5 new provider modules to maintain as APIs evolve
- Need shared polling infrastructure for submit/poll patterns
- PAN's stateful API is architecturally different from other providers (server-side state)
- Mitigated by: shared polling module, consistent error handling, per-provider integration tests

---

## ADR-011: VLM-Guided Planning

**Status:** Accepted

**Context:** WR-Arena demonstrates that pairing a VLM (vision-language model) with a WFM in an iterative planning loop produces significantly better plans than either alone — but only with the right WFM (PAN shows +26.7% improvement, while most video generators hurt planning).

**Decision:** Add a `VlmGuided` planner type to worldforge-core that implements the iterative VLM + WFM planning loop pattern.

**Protocol:**
1. VLM proposes K candidate actions given goal + history + current frame
2. WFM generates video segment for each candidate (with best-of-N selection)
3. VLM evaluates resulting frames and selects the best action
4. Repeat until goal achieved or max steps

**Rationale:**
- This is the most practically useful planning pattern from WR-Arena
- It's provider-agnostic: any WFM can be the simulator, any VLM can be the planner
- Fits naturally into WorldForge's existing plan() API as a new PlannerType variant
- Enables WorldForge to demonstrate tangible value (better plans with less effort)

**Consequences:**
- Requires VLM API access (Claude, GPT-4o, Gemini) in addition to WFM providers
- Planning loop is compute-intensive (K * N * steps generations per plan)
- Need careful prompt engineering for VLM action proposal and evaluation
- Mitigated by: configurable K, N, max_steps; cost estimation before planning
