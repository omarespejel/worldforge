# WFM Evaluation Dimensions

**Version 0.2.0-draft | March 2026**

This document specifies evaluation dimensions for WorldForge's `worldforge-eval` crate, incorporating findings from WR-Arena (arXiv 2603.25887) alongside WorldForge's existing physics-based evaluation.

---

## 1. Current Evaluation Dimensions (worldforge-eval)

WorldForge currently evaluates:

| Dimension | Metric | Implemented |
|-----------|--------|-------------|
| ObjectPermanence | Object tracking through occlusion | Yes |
| GravityCompliance | Unsupported objects fall correctly | Yes |
| CollisionAccuracy | Contact physics are correct | Yes |
| SpatialConsistency | Scene stable under viewpoint change | Yes |
| TemporalConsistency | Scene stable under time reversal | Yes |
| ActionPredictionAccuracy | Predicted outcome matches physics | Yes |
| MaterialUnderstanding | Material-appropriate behavior | Yes |
| SpatialReasoning | Depth, scale, distance perception | Yes |
| Custom { name } | Registry-based custom metrics | Yes |

---

## 2. New Evaluation Dimensions (from WR-Arena)

### 2.1 Action Simulation Fidelity

**Purpose:** Evaluate whether a WFM can faithfully simulate a described action, going beyond physics to test instruction-following.

**Sub-dimensions:**
- **Agent Simulation:** Can the model simulate agent actions? (robot picks up cup, person walks to door)
- **Environment Simulation:** Can the model simulate environment changes? (shadow moves, water flows, leaves rustle)

**Method:** LLM-as-judge (GPT-4o or equivalent multimodal LLM).

**Protocol:**
1. Generate video from initial image + text prompt describing an action
2. Extract N evenly-spaced frames (N=8 recommended)
3. Send frames + instruction text to multimodal LLM
4. LLM scores on 0-3 rubric

**Scoring rubric:**

| Score | Meaning | Criteria |
|-------|---------|----------|
| 0 | No compliance | Sequence does not follow instruction at all |
| 1 | Partial object/action match | Correct object but wrong action, or correct action but wrong object |
| 2 | Tendency toward goal | Follows instruction, shows movement toward intended outcome |
| 3 | Full compliance | Follows instruction precisely, achieves the goal |

**Evaluation prompt template:**
```
You are given a sequence of frames sampled in chronological order from a video.
Evaluate whether the sequence follows the instruction: "{instruction}".
Use the following scoring criteria:
- 0: The sequence does not follow the instruction at all.
- 1: The sequence includes the correct object but performs the wrong action,
     or the action is correct but on the wrong object.
- 2: The sequence follows the instruction and shows a tendency toward
     the intended goal.
- 3: The sequence follows the instruction precisely and successfully
     achieves the goal.
Return ONLY one integer: 0, 1, 2, or 3. Do not output any other text.
```

**Aggregation:**
- Per-round scores for multi-round scenarios
- Separate Agent vs Environment averages
- Overall mean across all instances

**Implementation notes:**
- Requires multimodal LLM API access (configurable: GPT-4o, Claude, Gemini)
- Frame extraction from VideoClip: evenly-spaced sampling
- Should be registered as `Custom { name: "action_simulation_fidelity" }` or a new first-class enum variant

### 2.2 Transition Smoothness (MRS)

**Purpose:** Evaluate temporal smoothness across multi-round generated videos, especially at round boundaries.

**Method:** Optical flow analysis between consecutive frames.

**Formula:**
```
MRS(V) = median(vmag) * exp(-λ * median(amag))
```

Where:
- `vmag` = per-pixel velocity magnitude from optical flow between consecutive frames
- `amag` = per-pixel acceleration magnitude (change in velocity between consecutive frame pairs)
- `λ` = smoothness penalty weight (default: 1.0)

**Interpretation:**
- Higher MRS = smoother motion
- High velocity + low acceleration = smooth continuous motion (good)
- High velocity + high acceleration = jerky/teleporting motion (bad)
- Low velocity + low acceleration = near-static video (neutral)

**Protocol:**
1. Compute optical flow between every consecutive frame pair (f_i, f_{i+1})
2. From flow fields, compute velocity magnitude per pixel
3. Compute acceleration as change in velocity between consecutive flow fields
4. Take median of velocity magnitudes and median of acceleration magnitudes
5. Apply MRS formula

**Implementation notes:**
- Optical flow estimation: SEA-RAFT (WR-Arena's choice), or RAFT, or Lucas-Kanade
- For Rust implementation: could shell out to Python optical flow, or use a Rust optical flow library
- Per-round and per-instance aggregation: mean, std, min, max
- Particularly important at round boundaries (frames N-1 to N+1 of consecutive rounds)

**Boundary analysis (extension):**
```
MRS_boundary = MRS computed only on frames at round transitions
MRS_intra = MRS computed only on frames within rounds
Boundary_penalty = MRS_boundary / MRS_intra
```

### 2.3 Generation Consistency (WorldScore-based)

**Purpose:** Evaluate whether multi-round generation maintains 3D structure, visual style, and content alignment.

**Sub-metrics (7 aspects):**

| Aspect | Metric | Method | Dependencies |
|--------|--------|--------|-------------|
| Camera Control | Reprojection error | DROID-SLAM | SLAM system |
| Object Control | Detection accuracy | GroundingDINO + SAM2 | Object detection |
| Content Alignment | CLIP score | CLIP embeddings | CLIP model |
| 3D Consistency | Reprojection error | DROID-SLAM | SLAM system |
| Photometric Consistency | AEPE (Average Endpoint Error) | SEA-RAFT optical flow | Flow estimation |
| Style Consistency | Gram matrix distance | VGG features | VGG model |
| Subjective Quality | CLIP-IQA+ and MUSIQ | Quality assessment models | IQA models |

**Composite score:**
```
WorldScore-Static = weighted_mean(camera, object, content, 3d, photometric, style, quality)
```

**Implementation notes:**
- Heavy dependency on external models (SLAM, object detection, CLIP, VGG, IQA)
- Best implemented as a Python sidecar or separate evaluation binary
- Could start with just Content Alignment (CLIP score) and Style Consistency (Gram matrix) as they're simpler
- Per-round scoring enables degradation curve analysis

**Degradation analysis (AP metric):**
```
AP = average over rounds r of: score(r) / score(1)
```
AP < 1.0 indicates degradation. Lower AP = faster degradation.

### 2.4 Simulative Reasoning and Planning

**Purpose:** Evaluate whether a WFM can support goal-directed thought experiments when paired with a VLM planner.

**Method:** VLM + WFM iterative planning loop.

**Protocol:**
1. Given: initial image, goal description
2. VLM proposes K candidate next actions (K=3 recommended)
3. For each candidate: WFM generates video segment
4. Best-of-N selection: generate N variants per candidate, GPT scores each
5. VLM examines resulting frames and selects best action
6. Repeat until goal achieved or max steps

**Scoring per segment:**
- Object permanence (1-5): Are objects preserved across the segment?
- Action following (1-5): Does the generated video match the action description?
- Final check (0-1): Binary penalty for multi-action segments (should be single-step)

**Variants:**
- **Open-ended planning:** Complex robotic manipulation, long action sequences
- **Structured planning:** Constrained environments (tabletop), bounded action count

**Implementation notes:**
- Requires VLM (Claude, GPT-4o, o3) for planning
- Requires WFM for simulation (any WorldForge provider)
- Natural fit for WorldForge's `plan()` API
- Could be implemented as a special PlannerType variant

---

## 3. Proposed Enum Extension

```rust
pub enum EvalDimension {
    // === Existing (physics-based) ===
    ObjectPermanence,
    GravityCompliance,
    CollisionAccuracy,
    SpatialConsistency,
    TemporalConsistency,
    ActionPredictionAccuracy,
    MaterialUnderstanding,
    SpatialReasoning,

    // === New (WR-Arena inspired) ===

    /// Can the model follow action instructions? (GPT-as-judge, 0-3 scale)
    ActionSimulationFidelity {
        /// "agent" or "environment" sub-type
        simulation_type: SimulationType,
    },

    /// Is multi-round video temporally smooth? (MRS metric)
    TransitionSmoothness {
        /// Smoothness penalty weight (default 1.0)
        lambda: f32,
    },

    /// Does multi-round generation maintain consistency? (WorldScore-based)
    GenerationConsistency {
        /// Which aspects to evaluate
        aspects: Vec<ConsistencyAspect>,
    },

    /// Can the model support VLM-guided planning? (planning success rate)
    SimulativeReasoning {
        /// Max planning steps
        max_steps: u32,
        /// Number of candidate actions per step
        candidates_per_step: u32,
    },

    /// Custom evaluation metric (existing)
    Custom { name: String },
}

pub enum SimulationType {
    Agent,
    Environment,
}

pub enum ConsistencyAspect {
    CameraControl,
    ObjectControl,
    ContentAlignment,
    ThreeDConsistency,
    PhotometricConsistency,
    StyleConsistency,
    SubjectiveQuality,
}
```

---

## 4. Dataset Schemas for Evaluation

### 4.1 Action Simulation Dataset

```rust
/// A single action simulation evaluation instance.
pub struct ActionSimulationInstance {
    /// Unique instance identifier (e.g., "agent_000_1")
    pub id: String,
    /// Path to initial frame image
    pub image_path: PathBuf,
    /// Sequential action prompts (typically 3 per instance)
    pub prompt_list: Vec<String>,
    /// Whether this is agent or environment simulation
    pub simulation_type: SimulationType,
}
```

### 4.2 Multi-Round Evaluation Dataset

```rust
/// A multi-round evaluation instance (smoothness or consistency).
pub struct MultiRoundInstance {
    pub id: String,
    pub image_path: PathBuf,
    pub prompt_list: Vec<String>,
    pub camera_path: Vec<CameraMotion>,
    pub visual_movement: String,
    pub visual_style: String,
    pub scene_type: String,
    pub category: String,
    pub scenario: ScenarioMetadata,
}

pub struct ScenarioMetadata {
    pub sid: String,
    pub label: String,
    pub definition: String,
}

pub enum CameraMotion {
    PanLeft,
    PanRight,
    PanUp,
    PanDown,
    ZoomIn,
    ZoomOut,
    TiltUp,
    TiltDown,
    Orbit,
    Static,
}
```

---

## 5. Evaluation Pipeline

```
┌─────────────────┐
│  Load Dataset    │ (JSON → Rust structs)
│  (action sim,    │
│   smoothness,    │
│   consistency)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generate Videos │ → WorldModelProvider.predict() / .generate()
│  (per provider,  │   with multi-round chaining
│   per instance)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Evaluate Dimensions (parallel)          │
│  ┌───────────────┐ ┌──────────────────┐ │
│  │ Action Sim    │ │ Smoothness (MRS) │ │
│  │ (LLM judge)   │ │ (optical flow)   │ │
│  └───────────────┘ └──────────────────┘ │
│  ┌───────────────┐ ┌──────────────────┐ │
│  │ Consistency   │ │ Sim. Reasoning   │ │
│  │ (WorldScore)  │ │ (VLM+WM loop)   │ │
│  └───────────────┘ └──────────────────┘ │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Aggregate &     │ → EvalReport with per-provider, per-dimension scores
│  Report          │ → Markdown, CSV, JSON outputs
│                  │ → Degradation curves for multi-round
└─────────────────┘
```
