# RFC-0000: WorldForge Master Index & Project Status Assessment

**Status:** Living Document
**Author:** WorldForge Core Team
**Created:** 2026-04-02
**Last Updated:** 2026-04-02

---

## 1. Executive Summary

WorldForge is "The LangChain for world models / physical AI / robotics" -- a Rust
workspace providing a unified orchestration layer across world foundation models
(NVIDIA Cosmos, Runway GWM, Meta V-JEPA, Google Genie/Veo, OpenAI Sora, and others).

The codebase currently stands at approximately 97,000 lines of Rust and Python across
7 crates. The architecture is well-designed and comprehensive. However, the honest
assessment is that the project is in a "sophisticated scaffold" state: types, traits,
APIs, and test harnesses are built out, but almost all provider integrations use
deterministic local surrogates rather than calling real external APIs.

This RFC serves as the master index for all subsequent RFCs needed to bring WorldForge
from scaffold to production.

---

## 2. Project Status Assessment (Brutally Honest)

### 2.1 Crate-by-Crate Audit

#### worldforge-core (20,801 lines, 13 source files)

| Module            | Lines | Status      | Notes                                        |
|-------------------|-------|-------------|----------------------------------------------|
| types.rs          | ~3500 | IMPLEMENTED | Tensor, Frame, VideoClip, Vec3, Pose, etc.   |
| state.rs          | ~2800 | IMPLEMENTED | WorldState with file/JSON, SQLite, Redis, S3 backends. State backends compile but Redis/S3 are feature-gated and untested against real services. |
| world.rs          | ~2400 | IMPLEMENTED | WorldOrchestrator with full lifecycle mgmt    |
| action.rs         | ~2200 | IMPLEMENTED | Action types, ActionTranslator, conditions    |
| prediction.rs     | ~1800 | IMPLEMENTED | Prediction, PredictionConfig, Plan types      |
| provider.rs       | ~1600 | IMPLEMENTED | WorldModelProvider trait (14 methods), ProviderCapabilities, HealthStatus |
| scene.rs          | ~1500 | PARTIAL     | SceneObject, SceneGraph types exist. No real physics engine integration. Collision detection is placeholder math. |
| guardrail.rs      | ~1200 | IMPLEMENTED | Guardrail trait, safety checks, content filters. Works but uses heuristic rules, not ML-based classifiers. |
| proof.rs          | ~900  | PARTIAL     | Proof types defined. No real ZK circuit integration. |
| goal_image.rs     | ~600  | STUB        | Goal image matching -- synthetic distance metrics only. |
| bootstrap.rs      | ~500  | IMPLEMENTED | World bootstrap from text prompts              |
| error.rs          | ~400  | IMPLEMENTED | Comprehensive error types                      |
| async_utils.rs    | ~300  | IMPLEMENTED | Async runtime helpers                          |

**Verdict:** Core types and traits are solid. This is the strongest crate. The
WorldModelProvider trait is well-designed with 14 async methods covering predict,
generate, reason, transfer, embed, plan, health, capabilities, cost estimation,
and batch operations. State management works for file/JSON and SQLite. Redis and
S3 backends exist but need real-world testing.

#### worldforge-providers (19,881 lines, 14 source files)

| Provider   | Lines | Has reqwest/HTTP | Real API Calls | Status         |
|------------|-------|------------------|----------------|----------------|
| cosmos.rs  | 3153  | YES (6 refs)     | PARTIAL        | Has real HTTP request scaffolding for Cosmos Predict/Transfer/Reason/Embed endpoints. Constructs real request bodies, handles auth headers, parses responses. BUT: untested against live NVIDIA API. Response parsing may not match actual API schema. |
| jepa.rs    | 3369  | NO               | NO             | "Deterministic local inference path" -- entirely synthetic. No ONNX/PyTorch integration. |
| genie.rs   | 3036  | NO               | NO             | "Deterministic local surrogate" -- entirely synthetic. No real Genie API. |
| marble.rs  | 1818  | NO               | NO             | "Experimental deterministic local surrogate" -- entirely synthetic. |
| runway.rs  | 1909  | YES (4 refs)     | PARTIAL        | Has HTTP client setup. Likely scaffolded but not tested against real Runway API. |
| kling.rs   | 897   | YES (4 refs)     | SCAFFOLD       | Has reqwest client. Request/response types may not match real API. |
| pan.rs     | 831   | YES (3 refs)     | SCAFFOLD       | Has reqwest client. Untested. |
| veo.rs     | 768   | YES (4 refs)     | SCAFFOLD       | Has reqwest client for Google Veo. Untested. |
| sora.rs    | 676   | YES (3 refs)     | SCAFFOLD       | Has reqwest client for OpenAI Sora. Untested. |
| minimax.rs | 643   | YES (3 refs)     | SCAFFOLD       | Has reqwest client. Untested. |
| mock.rs    | 2781  | NO               | N/A            | MockProvider -- fully implemented, deterministic. This is what everything actually runs against today. |
| polling.rs | 285   | YES (6 refs)     | UTILITY        | Generic async polling for long-running jobs. Well-structured. |
| native_planning.rs | 1619 | NO          | N/A            | Shared planning logic used by all providers. Deterministic. |
| lib.rs     | ~1100 | --               | --             | ProviderRegistry, provider discovery, routing  |

**Verdict:** This is the critical gap. 11 provider adapters exist with correct trait
implementations, proper type definitions, and in some cases (Cosmos, Runway) partial
HTTP scaffolding. But NONE have been validated against real APIs. The Cosmos provider
is the closest to real -- it constructs proper HTTP requests with auth headers and
handles rate limiting, but the request/response schemas are speculative and untested.
Every provider falls back to deterministic synthetic output.

#### worldforge-eval (6,436 lines, 5 source files)

| Module       | Lines | Status  | Notes                                       |
|--------------|-------|---------|---------------------------------------------|
| lib.rs       | 5292  | PARTIAL | Evaluation runner, metrics (SSIM, FID, LPIPS, physics scores) are defined. Metric computation uses simplified formulas, not real perceptual models. |
| wrarena.rs   | 686   | STUB    | WR-Arena integration types. No real arena service exists to connect to. |
| datasets.rs  | 411   | STUB    | Dataset loading types. No real datasets bundled or downloaded. |
| async_utils  | 47    | IMPL    | Async helpers                                |

**Verdict:** The evaluation framework has the right shape -- it defines suites,
metrics, comparison modes, and reporting. But metric calculations are simplified
approximations (not using real SSIM/FID/LPIPS implementations) and there are no
real benchmark datasets. This needs a ground-up rebuild of the metric engine.

#### worldforge-verify (1,581 lines, 1 source file)

| Feature     | Status  | Notes                                         |
|-------------|---------|-----------------------------------------------|
| Proof types | IMPL    | ProofRequest, ProofResult, VerificationResult |
| EZKL backend| STUB    | Types exist but no real EZKL circuit compilation or proving |
| STARK/Cairo | STUB    | Types exist but no real Cairo program integration |
| Mock verifier| IMPL   | Deterministic mock that always "verifies"      |

**Verdict:** The verification layer is almost entirely aspirational. ZK proofs for
world model outputs is a genuinely novel idea but requires significant cryptographic
engineering. The current code defines the interface but does zero real proving.

#### worldforge-server (9,838 lines, 3 source files)

| Feature           | Status      | Notes                                  |
|-------------------|-------------|----------------------------------------|
| HTTP server       | IMPLEMENTED | Custom tokio-based HTTP server (not axum/actix) |
| Route dispatch    | IMPLEMENTED | 27+ routes with pattern matching        |
| World CRUD        | IMPLEMENTED | Create, list, get, fork, import worlds  |
| Prediction routes | IMPLEMENTED | POST /worlds/{id}/predict, etc.         |
| Eval routes       | IMPLEMENTED | POST /eval/run, GET /eval/suites        |
| Provider routes   | IMPLEMENTED | GET /providers, GET /providers/{id}     |
| OpenAPI           | PARTIAL     | Route catalog exists, not full OpenAPI spec |
| Auth              | MISSING     | No authentication or authorization      |
| Rate limiting     | MISSING     | No request rate limiting                |
| CORS              | MISSING     | No CORS headers                         |
| TLS               | MISSING     | No HTTPS support                        |
| Integration tests | IMPLEMENTED | Comprehensive test suite (~40 tests)    |

**Verdict:** The server is surprisingly complete for an MVP. It handles the full
world lifecycle through REST endpoints, has proper error handling, and good test
coverage. However, it's a hand-rolled HTTP server (not using a framework like axum),
lacks all production concerns (auth, TLS, rate limiting, CORS), and would need
significant hardening before exposure to the internet.

#### worldforge-cli (9,327 lines, 2 source files)

| Feature           | Status      | Notes                                  |
|-------------------|-------------|----------------------------------------|
| World management  | IMPLEMENTED | create, list, predict, plan, eval      |
| Provider commands | IMPLEMENTED | list, info, health                     |
| Eval commands     | IMPLEMENTED | run, compare                           |
| Output formatting | IMPLEMENTED | JSON and table output                  |
| Config management | IMPLEMENTED | Config file support                    |

**Verdict:** The CLI is well-built with clap. It mirrors the server API faithfully.
Since it delegates to the same core library, it has the same limitations: all
operations run against mock providers.

#### worldforge-python (11,286 lines Rust + 662 lines Python)

| Feature             | Status      | Notes                                  |
|---------------------|-------------|----------------------------------------|
| PyO3 bindings       | IMPLEMENTED | Full Python class wrappers for core types |
| WorldForge client   | IMPLEMENTED | Python WorldForge class with async support |
| Type conversions    | IMPLEMENTED | Rust <-> Python type marshaling         |
| Provider access     | IMPLEMENTED | Python provider registry and selection  |
| Eval integration    | IMPLEMENTED | Python evaluation runner                |
| Package structure   | IMPLEMENTED | pip-installable with maturin            |
| Test suite          | PARTIAL     | Smoke tests and parity tests exist      |

**Verdict:** The Python SDK is impressively complete at 11k+ lines. It wraps nearly
every Rust type and function. The main limitation is that it inherits all the
limitations of the underlying Rust crates (mock providers, simplified metrics, etc.).

### 2.2 Overall Assessment

```
WHAT EXISTS (done well):
  [x] Unified WorldModelProvider trait (14 async methods)
  [x] 11 provider adapters with correct type signatures
  [x] WorldState with 4 storage backends (file, SQLite, Redis, S3)
  [x] WorldOrchestrator lifecycle management
  [x] REST API server with 27+ routes
  [x] CLI with full command set
  [x] Python SDK with comprehensive bindings
  [x] Evaluation framework structure
  [x] ZK verification interface
  [x] 97,000 lines of well-structured, compiling Rust

WHAT IS MISSING (the hard parts):
  [ ] Real API calls to ANY provider
  [ ] Validated request/response schemas for ANY provider
  [ ] Real evaluation metrics (SSIM, FID, LPIPS)
  [ ] Real benchmark datasets
  [ ] Real ZK proof generation
  [ ] Authentication and authorization
  [ ] TLS/HTTPS
  [ ] Rate limiting
  [ ] CI/CD pipeline
  [ ] Documentation beyond code comments
  [ ] Any form of billing or usage tracking
  [ ] Real physics engine integration
  [ ] Video/frame I/O (encoding, decoding, streaming)
  [ ] Observability (metrics, tracing, logging)
```

**Bottom line:** WorldForge is an excellent architecture document implemented as code.
The type system and trait design are production-quality. The gap is entirely in the
"last mile" -- connecting these well-designed interfaces to real services. This is
not unusual for a project at this stage, but it's important to be clear-eyed about it.

---

## 3. RFC Index

### Phase 1: Core Production Readiness

These RFCs address making the existing codebase work for real.

#### RFC-0001: Provider Integration Protocol

Defines the standard process for bringing a provider from stub/scaffold to real API
integration. Covers: API schema discovery and validation, authentication credential
management, request/response serialization contracts, error mapping from provider-
specific errors to WorldForge errors, integration test methodology (live API tests
with credential fixtures, recorded response fixtures for CI, provider-specific mock
servers), rate limiting and retry strategies, and the acceptance criteria for marking
a provider as "production ready." This RFC is the foundation -- every subsequent
provider RFC follows this protocol.

#### RFC-0007: Real-Time State Management & Persistence

Addresses gaps in the state management layer. The file/JSON and SQLite backends work
but need stress testing. The Redis backend needs testing against real Redis instances
with proper connection pooling, reconnection logic, and pub/sub for state change
notifications. The S3 backend needs testing against real S3 (and S3-compatible stores
like MinIO). This RFC also covers: state versioning and migration, conflict resolution
for concurrent writes, state compaction and garbage collection, snapshot export/import
formats, and performance benchmarks for each backend at scale (10K+ worlds, 1M+ state
transitions).

#### RFC-0013: CI/CD & Release Pipeline

Defines the continuous integration and release infrastructure. Covers: GitHub Actions
workflow for build/test/lint on every PR, integration test tiers (unit -> mock
integration -> live API with credentials), crate publishing workflow to crates.io,
Python wheel builds for manylinux/macOS/Windows via maturin, Docker image builds for
the server, semantic versioning policy, changelog generation, and security scanning
(cargo audit, dependency review). This is a prerequisite for everything else because
without CI, nothing is trustworthy.

#### RFC-0016: Planning System Real Implementation

The planning system currently uses a shared deterministic planner (native_planning.rs)
across all providers. This RFC covers: integrating actual search algorithms (A*, RRT,
MCTS) for action planning, implementing real cost functions based on provider latency
and pricing, goal-conditioned planning with visual goal matching (replacing the stub
goal_image.rs), multi-step plan optimization with rollback, plan caching and reuse,
and integration with provider-specific planning capabilities (e.g., Cosmos Reason 2
for physical reasoning during planning).

#### RFC-0018: Video/Frame I/O Pipeline

WorldForge deals in video frames and tensors, but currently has no real video I/O.
This RFC covers: frame encoding/decoding (H.264, H.265, VP9, AV1) via ffmpeg
bindings or gstreamer, tensor serialization formats (safetensors, NumPy .npy, ONNX
tensor proto), video clip assembly from frame sequences, streaming video I/O for
real-time applications, GPU-accelerated frame processing (CUDA, Metal), image format
support (PNG, JPEG, WebP, EXR for HDR/depth), and memory-efficient handling of large
video buffers. This is critical because every provider ultimately produces or
consumes video/image data.

#### RFC-0019: Security & Auth

The server currently has zero authentication. This RFC covers: API key authentication
for the REST API, JWT token support for service-to-service auth, role-based access
control (RBAC) for multi-tenant deployments, credential storage for provider API keys
(environment variables, secret managers, Vault integration), request signing for
provider API calls, TLS termination (native or via reverse proxy guidance), input
validation and sanitization, and security audit checklist.

#### RFC-0020: Observability & Telemetry

No observability exists today. This RFC covers: structured logging with tracing crate
integration, OpenTelemetry-compatible distributed tracing, Prometheus metrics export
(request latency, provider call duration, error rates, queue depths), health check
endpoints with dependency status, performance profiling hooks, cost tracking per
request, and dashboard templates for Grafana. This is essential for operating
WorldForge in production and for debugging provider integration issues.

### Phase 2: Provider Integrations

Each RFC follows the protocol defined in RFC-0001.

#### RFC-0002: NVIDIA Cosmos Provider

First real provider integration. NVIDIA Cosmos is the most mature target because:
(a) NVIDIA has published API documentation, (b) the existing cosmos.rs already has
HTTP scaffolding, and (c) Cosmos offers the broadest capability set (predict,
generate, reason, transfer, embed). This RFC covers: validating request/response
schemas against the real NVIDIA Cosmos API (build.nvidia.com), implementing the NIM
local deployment path, handling Cosmos-specific features (depth maps, camera control,
multi-view generation), building a recorded fixture test suite, and performance
benchmarking. Acceptance: predict, generate, and reason calls succeed against the
live API with valid credentials.

#### RFC-0003: Runway Gen-4 / GWM Provider

Runway Gen-4 and their General World Model (GWM-1) represent a key integration for
creative and simulation use cases. This RFC covers: reverse-engineering or using the
official Runway API for video generation, implementing the GWM-1 world simulation
capabilities (if API access becomes available), handling Runway's async job model
(submit job, poll for completion), supporting Runway's camera control and motion
brush features, and mapping Runway's output format to WorldForge's Frame/VideoClip
types. Note: Runway API access may require partnership or waitlist approval.

#### RFC-0004: Meta V-JEPA Local Provider

V-JEPA and V-JEPA 2 are Meta's self-supervised video prediction models. Unlike
cloud API providers, V-JEPA runs locally. This RFC covers: loading V-JEPA model
weights (from Hugging Face or direct download), ONNX Runtime or PyTorch/libtorch
integration for inference, GPU memory management for the 300M-1B parameter models,
implementing the predict and embed methods with real model inference, batch inference
optimization, and model quantization options (INT8, FP16). This is the first "local
model" integration and establishes the pattern for other local models.

#### RFC-0005: OpenAI Sora Provider

OpenAI Sora is a video generation model accessible through the OpenAI API. This RFC
covers: implementing the Sora API client using the OpenAI API format, handling Sora's
async generation model, supporting Sora's text-to-video and image-to-video modes,
mapping Sora's output to WorldForge types, implementing cost estimation based on
OpenAI's pricing, and handling the various resolution and duration options. Note:
Sora API availability and pricing may change; this RFC should define a fallback
strategy.

#### RFC-0006: Google Veo Provider

Google Veo (Veo 2 and upcoming Veo 3) is Google's video generation model available
through the Vertex AI API. This RFC covers: implementing the Vertex AI client for
Veo endpoints, handling Google Cloud authentication (service accounts, ADC), supporting
Veo's generation modes, implementing the transfer method for Veo's image-to-video
capabilities, and mapping Veo's output format to WorldForge types. Requires a Google
Cloud project with Vertex AI API enabled.

### Phase 3: Evaluation & Benchmarking

#### RFC-0008: Evaluation Framework v1

The current eval framework has the right structure but uses simplified metric
calculations. This RFC covers: integrating real perceptual metrics (SSIM via the
image crate, FID via a pre-trained InceptionV3 feature extractor, LPIPS via a
pre-trained VGG network), building or adopting a physics evaluation suite (object
permanence tests, gravity compliance, collision detection accuracy), creating a
standard benchmark dataset (curated set of input states + expected outputs for
regression testing), implementing the WR-Arena comparison protocol for head-to-head
provider evaluation, statistical significance testing for metric comparisons,
evaluation report generation (HTML, JSON, CSV), and CI integration for automated
regression detection.

#### RFC-0017: Scene Graph & Physics Integration

The scene graph in scene.rs defines SceneObject and basic spatial relationships but
has no real physics. This RFC covers: integrating a real physics engine (rapier3d for
Rust-native physics, or PhysX bindings for industrial-grade simulation), implementing
scene graph operations (add, remove, reparent, query by volume), spatial indexing for
efficient collision queries, physics-based prediction validation (does the provider's
output obey gravity, conservation of momentum, etc.), and importing/exporting standard
scene formats (glTF, USD, URDF for robotics).

### Phase 4: Developer Experience

#### RFC-0009: Python SDK Production Release

The Python SDK is already comprehensive (11k+ lines) but needs production polish.
This RFC covers: publishing to PyPI with proper versioning, comprehensive Python
documentation with docstrings and type hints, Jupyter notebook examples for common
workflows, async/await support verification (tokio-based async from Python), error
handling and Python-idiomatic exceptions, performance profiling of the Rust-Python
boundary, compatibility testing across Python 3.9-3.13, and a "getting started"
tutorial that walks through world creation -> prediction -> evaluation.

#### RFC-0010: REST API Production Hardening

The server works but needs production hardening. This RFC covers: migrating from
the hand-rolled HTTP server to axum (for ecosystem compatibility, middleware support,
and battle-tested HTTP parsing), implementing proper OpenAPI 3.1 spec generation,
adding request validation with serde, implementing pagination for list endpoints,
adding WebSocket support for streaming predictions, implementing proper CORS handling,
adding request/response compression (gzip, brotli), connection pooling and keep-alive,
graceful shutdown, and load testing with realistic traffic patterns.

#### RFC-0012: Documentation & Developer Portal

No documentation exists beyond code comments and the SPECIFICATION.md. This RFC
covers: API reference documentation (auto-generated from OpenAPI spec), Rust crate
documentation (cargo doc with examples), Python SDK documentation (Sphinx or mkdocs),
architecture guide explaining the crate structure and data flow, provider integration
guide for third-party provider authors, tutorial series (beginner, intermediate,
advanced), example applications (robotics simulation, autonomous driving prediction,
game world generation, video prediction pipeline), and a documentation website
(likely mdBook or Docusaurus).

### Phase 5: Cloud & Monetization

#### RFC-0011: ZK Verification Integration (EZKL + Cairo/STARK)

ZK verification of world model outputs is a unique differentiator but requires
serious cryptographic engineering. This RFC covers: EZKL integration for proving
that a neural network inference produced a specific output (useful for verifying
provider outputs), Cairo/STARK integration for proving state transition correctness,
proof generation performance optimization (proofs must be practical to generate),
on-chain verification contracts (Ethereum, Starknet), proof aggregation for batch
verification, and a clear articulation of what security properties the proofs
actually provide (this is subtle -- proving "the model produced this output" is
different from proving "this output is physically correct").

#### RFC-0014: Cloud Service Architecture

Defines the architecture for running WorldForge as a managed cloud service. Covers:
multi-tenant isolation (separate provider credentials per tenant, resource quotas),
horizontal scaling (stateless server instances behind a load balancer, shared state
via Redis/PostgreSQL), provider credential management (per-tenant API keys stored in
a secret manager), job queue for async predictions (Redis-backed or SQS), result
caching (avoid redundant provider calls for identical inputs), CDN for generated
video/image assets, deployment infrastructure (Kubernetes manifests, Terraform for
cloud resources), and region selection for latency optimization.

#### RFC-0015: Billing & Usage Metering

Defines how to track and bill for WorldForge usage. Covers: usage event collection
(every provider call, storage operation, and eval run is metered), cost attribution
(provider API costs are passed through with markup), metering pipeline (events ->
aggregation -> billing system), integration with Stripe for payment processing,
usage dashboards for customers, rate limiting tied to billing tiers (free, pro,
enterprise), invoice generation, and cost alerts.

---

## 4. Dependency Graph

```
RFC-0013 (CI/CD)
  |
  v
RFC-0001 (Provider Protocol) -----> RFC-0019 (Security)
  |                                      |
  +---> RFC-0002 (Cosmos) ------+        |
  +---> RFC-0003 (Runway) ------+        |
  +---> RFC-0004 (V-JEPA) ------+        |
  +---> RFC-0005 (Sora) --------+        |
  +---> RFC-0006 (Veo) ---------+        |
  |                             |        |
  v                             v        v
RFC-0018 (Video I/O) -------> RFC-0008 (Eval Framework)
  |                             |
  v                             v
RFC-0017 (Scene/Physics)    RFC-0010 (API Hardening) ---> RFC-0020 (Observability)
  |                             |                              |
  v                             v                              v
RFC-0016 (Planning)         RFC-0012 (Docs)               RFC-0014 (Cloud)
  |                             |                              |
  v                             v                              v
RFC-0007 (State Mgmt)      RFC-0009 (Python SDK)          RFC-0015 (Billing)
                                                               |
                                                               v
                                                          RFC-0011 (ZK Verify)
```

Critical path: RFC-0013 -> RFC-0001 -> RFC-0002 -> RFC-0018 -> RFC-0008

### Dependency Details

| RFC   | Depends On           | Reason                                       |
|-------|----------------------|----------------------------------------------|
| 0001  | 0013                 | Need CI before provider integration work      |
| 0002  | 0001                 | Follows provider integration protocol         |
| 0003  | 0001                 | Follows provider integration protocol         |
| 0004  | 0001, 0018           | Needs video I/O for local model inference     |
| 0005  | 0001                 | Follows provider integration protocol         |
| 0006  | 0001                 | Follows provider integration protocol         |
| 0007  | 0013                 | Needs CI for state backend testing            |
| 0008  | 0002 (at least one provider), 0018 | Need real outputs to evaluate       |
| 0009  | 0008, 0010           | SDK should wrap production API                |
| 0010  | 0019, 0020           | Hardening requires auth and observability      |
| 0011  | 0008                 | Need real outputs to verify                   |
| 0012  | 0009, 0010           | Document what's actually production-ready     |
| 0014  | 0010, 0019, 0020     | Cloud service needs hardened API              |
| 0015  | 0014                 | Billing requires cloud infrastructure         |
| 0016  | 0002 (at least one provider) | Real planning needs real predictions   |
| 0017  | 0018                 | Physics validation needs real frame data      |
| 0018  | 0001                 | Video I/O needed for provider integration     |
| 0019  | 0013                 | Security needs CI for testing                 |
| 0020  | 0013                 | Observability needs CI for testing            |

---

## 5. Timeline Estimate

Realistic timeline for a solo developer augmented by AI coding agents.

### Assumptions

- Solo developer working full-time with AI agent assistance (2-3x productivity)
- Provider API access is available (some providers may have waitlists)
- No major architectural redesigns needed (the current design is sound)
- AI agents handle boilerplate, tests, and documentation; human handles
  integration debugging, API schema validation, and architecture decisions

### Phase 1: Core Production Readiness (Weeks 1-6)

| Week | RFC    | Deliverable                                    |
|------|--------|------------------------------------------------|
| 1    | 0013   | CI/CD pipeline: GitHub Actions, cargo test, clippy, maturin builds |
| 2    | 0001   | Provider integration protocol document + template |
| 2-3  | 0018   | Video/frame I/O pipeline (ffmpeg bindings, frame encode/decode) |
| 3-4  | 0019   | API key auth, TLS guidance, input validation   |
| 4-5  | 0020   | tracing integration, Prometheus metrics, health checks |
| 5-6  | 0007   | State backend stress testing, Redis/S3 validation |

### Phase 2: Provider Integrations (Weeks 7-14)

| Week  | RFC    | Deliverable                                   |
|-------|--------|-----------------------------------------------|
| 7-8   | 0002   | NVIDIA Cosmos -- first real provider           |
| 9-10  | 0005   | OpenAI Sora (well-documented API)              |
| 10-11 | 0006   | Google Veo (Vertex AI)                         |
| 11-12 | 0003   | Runway Gen-4 / GWM                             |
| 13-14 | 0004   | Meta V-JEPA local inference                    |

### Phase 3: Evaluation & Quality (Weeks 15-19)

| Week  | RFC    | Deliverable                                   |
|-------|--------|-----------------------------------------------|
| 15-16 | 0008   | Real evaluation metrics + benchmark dataset    |
| 17-18 | 0017   | Physics engine integration (rapier3d)          |
| 18-19 | 0016   | Planning system with real search algorithms    |

### Phase 4: Developer Experience (Weeks 20-23)

| Week  | RFC    | Deliverable                                   |
|-------|--------|-----------------------------------------------|
| 20    | 0010   | Migrate server to axum, OpenAPI spec           |
| 21    | 0009   | Python SDK PyPI release                        |
| 22-23 | 0012   | Documentation site, tutorials, examples        |

### Phase 5: Cloud & Advanced (Weeks 24-30)

| Week  | RFC    | Deliverable                                   |
|-------|--------|-----------------------------------------------|
| 24-26 | 0014   | Cloud service architecture + Kubernetes deploy |
| 27-28 | 0015   | Billing and usage metering                     |
| 29-30 | 0011   | ZK verification (EZKL integration)             |

### Total: ~30 weeks (7-8 months) for a solo developer + AI agents

**Risk factors that could extend this:**
- Provider API access delays (waitlists, partnership requirements)
- API schema changes by providers (these models are all actively evolving)
- ZK verification is research-grade work and could take much longer
- Local model inference (V-JEPA) requires GPU infrastructure for testing
- The "last 20%" of each integration always takes longer than expected

---

## 6. Gap Analysis: SPECIFICATION.md vs Reality

The SPECIFICATION.md is 1,400 lines covering 15 sections. Here is what it specifies
well, what it underspecifies, and what is missing entirely.

### Well-Specified

- Core type system (Section 3): Tensor, Frame, VideoClip types are well-defined
  and accurately implemented.
- Provider abstraction (Section 4): The WorldModelProvider trait matches the spec.
- World state management (Section 5): State backends and lifecycle are well-specified.
- Action system (Section 6): Action types and translation are well-specified.
- Error handling (Section 14): Error types are comprehensive.

### Underspecified

- Provider specifications (Section 13): Lists providers but doesn't specify exact
  API request/response schemas. Each provider needs its own detailed spec mapping
  the WorldForge trait methods to provider-specific API calls.
- Evaluation framework (Section 9): Defines metrics conceptually but doesn't specify
  the exact algorithms or reference implementations to use for SSIM, FID, LPIPS.
- Planning system (Section 8): Describes the planning interface but doesn't specify
  which search algorithms to implement or how to handle the combinatorial explosion
  of multi-step plans.
- ZK verification (Section 11): Describes the concept but lacks detail on circuit
  design, proving system choice, and what exactly is being proved.

### Missing Entirely

- **Video I/O specification:** No mention of how frames are encoded/decoded, what
  codecs are supported, or how video streaming works.
- **Authentication and authorization:** No security model defined.
- **Multi-tenancy:** No specification for isolating tenants in a shared deployment.
- **Billing and metering:** No specification for usage tracking or cost attribution.
- **Deployment architecture:** No specification for how WorldForge runs in production
  (containers, orchestration, scaling).
- **Versioning and compatibility:** No specification for API versioning, backward
  compatibility, or deprecation policy.
- **Provider credential management:** No specification for how provider API keys are
  stored, rotated, or scoped.
- **Streaming/real-time protocol:** No specification for WebSocket or gRPC streaming
  for real-time predictions (critical for robotics use cases).
- **GPU resource management:** No specification for how local model providers (V-JEPA,
  etc.) manage GPU memory, model loading, and concurrent inference.
- **Rate limiting and backpressure:** No specification for how WorldForge handles
  provider rate limits, queues requests, or applies backpressure to clients.
- **Data retention and privacy:** No specification for how generated content, world
  states, and user data are retained, encrypted, or deleted.
- **Offline/edge deployment:** No specification for running WorldForge on edge
  devices or in air-gapped environments (relevant for robotics).

### Specification Action Items

1. Add Section 16: Video & Media I/O
2. Add Section 17: Security Model
3. Add Section 18: Deployment & Operations
4. Add Section 19: Streaming & Real-Time Protocol
5. Add Section 20: GPU & Resource Management
6. Expand Section 13 with per-provider API schema mappings
7. Expand Section 9 with exact metric computation algorithms
8. Expand Section 11 with concrete ZK circuit designs
9. Add Section 21: Versioning & Compatibility Policy
10. Add Section 22: Data Privacy & Retention

---

## 7. Priority Matrix

| RFC   | Impact | Effort | Priority | Rationale                          |
|-------|--------|--------|----------|------------------------------------|
| 0013  | HIGH   | LOW    | P0       | CI/CD is table stakes               |
| 0001  | HIGH   | LOW    | P0       | Protocol doc, enables all providers  |
| 0002  | HIGH   | MED    | P0       | First real provider = proof of concept |
| 0018  | HIGH   | MED    | P0       | Video I/O is fundamental             |
| 0019  | HIGH   | MED    | P1       | Security before any public deployment |
| 0020  | MED    | LOW    | P1       | Observability needed for debugging   |
| 0007  | MED    | MED    | P1       | State backends need validation       |
| 0005  | HIGH   | LOW    | P1       | Sora API is well-documented          |
| 0006  | HIGH   | LOW    | P1       | Veo via Vertex AI is well-documented |
| 0008  | HIGH   | HIGH   | P1       | Real eval is the value proposition   |
| 0003  | MED    | MED    | P2       | Runway may need API partnership      |
| 0004  | MED    | HIGH   | P2       | Local inference is complex           |
| 0010  | MED    | MED    | P2       | API hardening for production         |
| 0016  | MED    | HIGH   | P2       | Real planning is algorithmically hard |
| 0017  | MED    | HIGH   | P2       | Physics integration is complex       |
| 0009  | MED    | LOW    | P2       | Python SDK already mostly works      |
| 0012  | MED    | MED    | P2       | Docs needed for adoption             |
| 0014  | HIGH   | HIGH   | P3       | Cloud service is revenue path        |
| 0015  | HIGH   | HIGH   | P3       | Billing is revenue path              |
| 0011  | LOW    | VERY HIGH | P3    | ZK is novel but not critical path    |

---

## 8. Success Criteria

WorldForge can be considered "production-ready for early adopters" when:

1. At least 3 providers make real API calls and return real results
2. The evaluation framework computes real perceptual metrics
3. The REST API has authentication and TLS
4. The Python SDK is published on PyPI
5. Documentation covers installation, quickstart, and API reference
6. CI/CD runs on every commit with >80% test coverage
7. At least one example application demonstrates end-to-end workflow

WorldForge can be considered "production-ready for enterprises" when all 20 RFCs
are implemented and the cloud service is operational.

---

## Appendix A: Line Count Summary

| Crate              | Lines  | Files | Status              |
|--------------------|--------|-------|---------------------|
| worldforge-core    | 20,801 | 13    | Solid foundation    |
| worldforge-providers| 19,881| 14    | Scaffold + mock     |
| worldforge-python  | 11,286 | 1     | Comprehensive bindings |
| worldforge-server  | 9,838  | 3     | Working MVP         |
| worldforge-cli     | 9,327  | 2     | Working MVP         |
| worldforge-eval    | 6,436  | 5     | Partial             |
| worldforge-verify  | 1,581  | 1     | Mostly stub         |
| Tests & benches    | ~17,850| ~10   | Good coverage       |
| **Total**          | **~97,000** | **~49** | **Architecture: A, Implementation: C+** |

---

## Appendix B: File Inventory

### worldforge-core/src/
- lib.rs -- Crate root, re-exports
- types.rs -- Tensor, Frame, VideoClip, Vec3, Pose, BBox, SimTime, etc.
- state.rs -- WorldState, StateBackend (File, SQLite, Redis, S3)
- world.rs -- WorldOrchestrator, WorldId, WorldConfig
- action.rs -- Action, ActionType, ActionTranslator, Weather, conditions
- prediction.rs -- Prediction, PredictionConfig, Plan, PlanRequest
- provider.rs -- WorldModelProvider trait, ProviderCapabilities, HealthStatus
- scene.rs -- SceneObject, SceneGraph
- guardrail.rs -- Guardrail trait, safety checks
- proof.rs -- ProofRequest, ProofResult, VerificationResult
- goal_image.rs -- Goal image matching (stub)
- bootstrap.rs -- World bootstrap from prompts
- error.rs -- WorldForgeError, Result type
- async_utils.rs -- Async runtime helpers

### worldforge-providers/src/
- lib.rs -- ProviderRegistry, provider discovery
- cosmos.rs -- NVIDIA Cosmos (has HTTP scaffolding)
- runway.rs -- Runway Gen-4/GWM (has HTTP scaffolding)
- jepa.rs -- Meta V-JEPA (deterministic local surrogate)
- genie.rs -- Google Genie (deterministic local surrogate)
- marble.rs -- World Labs Marble (deterministic local surrogate)
- kling.rs -- Kling (has HTTP scaffolding)
- minimax.rs -- MiniMax (has HTTP scaffolding)
- pan.rs -- Pan (has HTTP scaffolding)
- sora.rs -- OpenAI Sora (has HTTP scaffolding)
- veo.rs -- Google Veo (has HTTP scaffolding)
- mock.rs -- MockProvider (fully implemented)
- polling.rs -- Async job polling utility
- native_planning.rs -- Shared deterministic planning logic

### worldforge-eval/src/
- lib.rs -- EvaluationRunner, metrics, suites
- wrarena.rs -- WR-Arena integration (stub)
- datasets.rs -- Dataset loading (stub)
- async_utils.rs -- Async helpers

### worldforge-verify/src/
- lib.rs -- Proof types, EZKL stub, STARK stub, mock verifier

### worldforge-server/src/
- lib.rs -- HTTP server, route dispatch, 27+ handlers
- main.rs -- Entry point

### worldforge-cli/src/
- lib.rs -- CLI commands, output formatting
- main.rs -- Entry point

### worldforge-python/src/
- lib.rs -- PyO3 bindings (11,286 lines)

---

*This is a living document. Update it as RFCs are completed.*
