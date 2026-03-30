# Implementation Plan: WR-Arena Integration

**Version 0.1.0 | March 2026**
**Source:** WR-Arena analysis (WR_ARENA_ANALYSIS.md, PROVIDER_CATALOG.md, EVALUATION_DIMENSIONS.md)

This plan breaks down the work to integrate WR-Arena findings into WorldForge, organized by phase and priority.

---

## Phase 1: New Provider Adapters (API-based)

All Tier 2 providers from the Provider Catalog use REST APIs with submit/poll/download patterns. They share a common architecture that can be generalized.

### Task 1.1: Shared async polling infrastructure

**Crate:** `worldforge-providers`
**File:** `src/polling.rs` (new)

Create a reusable async polling helper for submit/poll/download API patterns:

```rust
pub struct PollingConfig {
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_factor: f32,
    pub max_attempts: u32,
    pub timeout: Duration,
}

pub async fn poll_until_complete<F, G, T>(
    poll_fn: F,
    is_complete: G,
    config: &PollingConfig,
) -> Result<T, WorldForgeError>
```

This is shared by KLING, MiniMax, Sora 2, and Veo 3.

**Estimated scope:** ~100 lines

### Task 1.2: KLING provider adapter

**Crate:** `worldforge-providers`
**File:** `src/kling.rs` (new)

Implement `WorldModelProvider` for KLING:
- JWT authentication (HS256 signing with API key + secret)
- Image-to-video generation via REST API
- Negative prompt quality guard
- Task polling for completion
- Video download and frame extraction

**Env vars:** `KLING_API_KEY`, `KLING_API_SECRET`
**Capabilities:** predict, generate
**Estimated scope:** ~300 lines

### Task 1.3: Sora 2 (OpenAI) provider adapter

**Crate:** `worldforge-providers`
**File:** `src/sora.rs` (new)

Implement `WorldModelProvider` for Sora 2:
- OpenAI REST API (not SDK — we use reqwest directly)
- Video creation with image input
- Polling for completion
- Video download

**Env vars:** `OPENAI_API_KEY`
**Capabilities:** predict, generate
**Estimated scope:** ~250 lines

### Task 1.4: Veo 3 (Google) provider adapter

**Crate:** `worldforge-providers`
**File:** `src/veo.rs` (new)

Implement `WorldModelProvider` for Veo 3:
- Google GenAI REST API
- Video generation with image input
- Operation polling
- Video download from URI

**Env vars:** `GOOGLE_API_KEY`
**Capabilities:** predict, generate
**Estimated scope:** ~250 lines

### Task 1.5: MiniMax/Hailuo provider adapter

**Crate:** `worldforge-providers`
**File:** `src/minimax.rs` (new)

Implement `WorldModelProvider` for MiniMax:
- REST API submit/poll/download
- File ID based retrieval
- Video download

**Env vars:** `MINIMAX_API_KEY`
**Capabilities:** predict, generate
**Estimated scope:** ~250 lines

### Task 1.6: PAN provider adapter

**Crate:** `worldforge-providers`
**File:** `src/pan.rs` (new)

Implement `WorldModelProvider` for PAN:
- Stateful multi-round API (first_round / continue endpoints)
- Server-side state management (state_id, video_id tracking)
- Load-balanced endpoints
- Prompt upsampling support

**Env vars:** `PAN_API_KEY`, `PAN_API_ENDPOINT`
**Capabilities:** predict, generate, plan (via multi-round state)
**Estimated scope:** ~400 lines (most complex due to stateful API)

### Task 1.7: Update auto_detect and lib.rs

**Crate:** `worldforge-providers`
**File:** `src/lib.rs` (modify)

- Add module declarations for new providers
- Add auto_detect entries for each new provider (env var checks)
- Export new provider types

**Estimated scope:** ~80 lines of additions

---

## Phase 2: Evaluation Dimensions

### Task 2.1: Action Simulation Fidelity evaluator

**Crate:** `worldforge-eval`
**Files:** `src/action_fidelity.rs` (new), `src/lib.rs` (modify)

Implement the LLM-as-judge evaluation:
- Frame extraction from VideoClip (N evenly-spaced frames)
- Multimodal LLM API call with rubric prompt
- Score parsing (0-3 integer)
- Agent vs Environment split aggregation
- Register as a built-in evaluation dimension

**Dependencies:** Requires multimodal LLM API access (GPT-4o, Claude, etc.)
**Estimated scope:** ~200 lines

### Task 2.2: Transition Smoothness (MRS) metric

**Crate:** `worldforge-eval`
**Files:** `src/smoothness.rs` (new), `src/lib.rs` (modify)

Implement the MRS smoothness metric:
- Optical flow estimation between consecutive frames
- Velocity magnitude computation
- Acceleration magnitude computation
- MRS formula: `vmag_median * exp(-λ * amag_median)`
- Per-round and boundary analysis
- Register as a built-in evaluation dimension

**Dependencies:** Optical flow estimation (could shell out to Python SEA-RAFT initially, or use a Rust implementation)
**Estimated scope:** ~250 lines

### Task 2.3: Generation Consistency metrics (subset)

**Crate:** `worldforge-eval`
**Files:** `src/consistency.rs` (new), `src/lib.rs` (modify)

Implement a subset of WorldScore metrics:
- Content Alignment (CLIP score) — highest value, moderate complexity
- Style Consistency (Gram matrix distance via VGG features)
- Degradation curve analysis (AP metric)

**Dependencies:** CLIP and VGG model access (Python sidecar or ONNX runtime)
**Estimated scope:** ~300 lines

### Task 2.4: Evaluation dataset loaders

**Crate:** `worldforge-eval`
**Files:** `src/datasets.rs` (new), `src/lib.rs` (modify)

Implement JSON dataset loaders for WR-Arena format:
- Action simulation instances: `{id, image_path, prompt_list}`
- Multi-round instances: `{id, image_path, prompt_list, camera_path, ...}`
- Dataset validation and error reporting
- Serde deserialization into Rust types

**Estimated scope:** ~200 lines

---

## Phase 3: Multi-Round Orchestration

### Task 3.1: Multi-round video generation pipeline

**Crate:** `worldforge-core`
**Files:** `src/prediction.rs` (modify) or `src/multi_round.rs` (new)

Implement the multi-round generation pipeline:
- Sequential prompt execution with frame chaining
- Last-frame extraction from VideoClip
- Provider-specific handling (Cosmos: full video context, PAN: stateful API)
- Frame concatenation with overlap removal
- Per-round segment storage

**Estimated scope:** ~300 lines

### Task 3.2: Best-of-N selection

**Crate:** `worldforge-core`
**Files:** `src/prediction.rs` (modify)

Add best-of-N variant selection:
- Generate N variants for each prediction
- Score each variant (via LLM judge or physics metrics)
- Return best variant with selection metadata

**Estimated scope:** ~150 lines

---

## Phase 4: VLM+WM Planning Loop

### Task 4.1: VLM planning integration

**Crate:** `worldforge-core`
**Files:** `src/prediction.rs` or `src/planning/` (new module)

Implement the iterative VLM + WFM planning pattern:
- VLM proposes K candidate actions
- WFM generates video for each candidate (with best-of-N)
- VLM evaluates results and selects best
- Loop until goal or max steps
- Full history tracking

This could be a new `PlannerType::VlmGuided` variant:
```rust
PlannerType::VlmGuided {
    vlm_provider: String,      // "claude", "gpt-4o", "o3"
    candidates_per_step: u32,  // K=3
    variants_per_candidate: u32, // N for best-of-N
    max_steps: u32,
}
```

**Dependencies:** VLM API access
**Estimated scope:** ~400 lines

---

## Phase 5: SPECIFICATION.md Updates

### Task 5.1: Update Section 4 (Provider Abstraction)

Add provider specs for KLING, Sora 2, Veo 3, MiniMax, PAN to section 4.3.

### Task 5.2: Update Section 9 (Evaluation Framework)

Add WR-Arena evaluation dimensions (ActionSimulationFidelity, TransitionSmoothness, GenerationConsistency, SimulativeReasoning) to section 9.1.

### Task 5.3: Update Section 13 (Provider Specifications)

Add detailed integration specs for each new provider (sections 13.4 through 13.9).

### Task 5.4: Update Section 8 (Planning System)

Add VlmGuided planner type and multi-round planning protocol.

---

## Priority Matrix

| Task | Impact | Effort | Priority | Dependencies |
|------|--------|--------|----------|-------------|
| 1.1 Polling infra | High (enables all API providers) | Low | P0 | None |
| 1.6 PAN adapter | Very High (best planning model) | Medium | P0 | 1.1 |
| 1.2 KLING adapter | High | Low-Med | P1 | 1.1 |
| 1.3 Sora 2 adapter | High | Low-Med | P1 | 1.1 |
| 1.4 Veo 3 adapter | High | Low-Med | P1 | 1.1 |
| 1.5 MiniMax adapter | High | Low-Med | P1 | 1.1 |
| 1.7 Auto-detect update | High | Low | P1 | 1.2-1.6 |
| 2.1 Action fidelity eval | Very High | Medium | P1 | None |
| 2.4 Dataset loaders | High | Low | P1 | None |
| 2.2 MRS smoothness | High | Medium | P2 | None |
| 2.3 Consistency metrics | Medium | High | P2 | External model deps |
| 3.1 Multi-round pipeline | Very High | Medium | P1 | None |
| 3.2 Best-of-N selection | Medium | Low | P2 | None |
| 4.1 VLM planning loop | Very High | High | P2 | 3.1, 3.2 |
| 5.x Spec updates | Medium | Medium | P1 | Parallel with impl |

---

## Estimated Total Scope

| Phase | Lines of code | Files | Status |
|-------|--------------|-------|--------|
| Phase 1: Providers | ~1,630 | 7 new + 1 modified | Not started |
| Phase 2: Evaluation | ~950 | 4 new + 1 modified | Not started |
| Phase 3: Multi-round | ~450 | 1-2 new/modified | Not started |
| Phase 4: VLM planning | ~400 | 1-2 new/modified | Not started |
| Phase 5: Spec updates | ~500 lines of docs | 1 modified | Not started |
| **Total** | **~3,930** | **~15 files** | |

---

## Success Criteria

1. **Provider coverage:** WorldForge supports 11+ world models (current 5 + 6 new)
2. **Evaluation breadth:** 12+ evaluation dimensions (current 8 + 4 new WR-Arena dimensions)
3. **Real benchmarks:** Can load and run WR-Arena datasets through WorldForge eval
4. **Multi-round:** First-class support for multi-round video generation with round chaining
5. **Planning:** VLM+WM planning loop producing measurably better plans than VLM-only
6. **All tests pass:** `cargo test && cargo clippy -- -D warnings`
