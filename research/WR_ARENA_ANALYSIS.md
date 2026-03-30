# WR-Arena Analysis: Lessons for WorldForge

**Source:** [MBZUAI-IFM/WR-Arena](https://github.com/MBZUAI-IFM/WR-Arena) | [arXiv 2603.25887](https://arxiv.org/abs/2603.25887)
**Date:** 2026-03-30

---

## 1. What WR-Arena Is

WR-Arena (World Reasoning Arena) is a diagnostic benchmark from MBZUAI's Institute of Foundation Models that evaluates world foundation models (WFMs) across four complementary dimensions. Unlike visual quality benchmarks (FVD, FID), WR-Arena tests whether a model actually *understands* the world — can it follow instructions, maintain consistency over time, and support goal-directed planning?

The paper covers 10 models and introduces evaluation protocols that are directly relevant to WorldForge's evaluation framework.

---

## 2. Models Covered

### 2.1 World Foundation Models (true WFMs with action-conditioned capabilities)

| Model | Organization | Access | Resolution | Frames | Key Trait |
|-------|-------------|--------|------------|--------|-----------|
| Cosmos-Predict1 14B | NVIDIA | NIM containers, NGC, HuggingFace, API | 1280x704 | 121 | Multi-GPU context parallel (Megatron) |
| Cosmos-Predict2 14B | NVIDIA | NIM containers, NGC, HuggingFace, API | 1280x704 | 93 | Improved temporal coherence |
| V-JEPA 2 | Meta | Open-source weights | — | — | Representation learning, not generation |
| PAN | MBZUAI IFM | API at ifm.mbzuai.ac.ae/pan, HuggingFace | 832x480 | 41 | Stateful multi-round, best at planning |

### 2.2 Video Generation Models (used as WFM proxies)

| Model | Organization | Access | Resolution | Frames | API Type |
|-------|-------------|--------|------------|--------|----------|
| WAN 2.1 I2V-14B | Alibaba | Open-source | 832x480 | 81 | Local (multi-GPU Ulysses parallel) |
| WAN 2.2 I2V-A14B | Alibaba | Open-source | 832x480 | 81 | Local (multi-GPU Ulysses parallel) |
| Gen-3 | Runway | runwayml SDK | 1280x768 | 125 | Task-based submit/poll/download |
| KLING | Kuaishou | REST API (JWT) | 1280x720 | 153 | api-singapore.klingai.com |
| MiniMax/Hailuo | MiniMax | REST API | 1072x720 | 141 | api.minimax.io submit/poll |
| Sora 2 | OpenAI | OpenAI SDK | 1280x720 | 120 | client.videos.create_and_poll |
| Veo 3 | Google | GenAI SDK | 1280x720 | 96 | client.models.generate_videos |

---

## 3. Evaluation Dimensions

### 3.1 Action Simulation Fidelity

**What:** Can the model faithfully simulate a described action?

**Method:** GPT-4o as judge. Extract 8 evenly-spaced frames per round, send to GPT-4o with instruction text.

**Scoring rubric (0-3 integer):**
- **0** — Doesn't follow the instruction at all
- **1** — Correct object but wrong action, or vice versa
- **2** — Follows instruction, shows tendency toward intended goal
- **3** — Follows instruction precisely, achieves the goal

**Split:** Agent simulation (e.g., "robot picks up cup") vs Environment simulation (e.g., "shadow moves across floor"). Environment is significantly harder — no model exceeds 60%.

**Prompt template:**
```
You are given a sequence of frames sampled in chronological order from a video.
Evaluate whether the sequence follows the instruction: "{instruction}".
Use the following scoring criteria:
- 0: The sequence does not follow the instruction at all.
- 1: The sequence includes the correct object but performs the wrong action, or the action is correct but on the wrong object.
- 2: The sequence follows the instruction and shows a tendency toward the intended goal.
- 3: The sequence follows the instruction precisely and successfully achieves the goal.
Return ONLY one integer: 0, 1, 2, or 3. Do not output any other text.
```

### 3.2 Transition Smoothness (MRS Metric)

**What:** Are multi-round generated videos temporally smooth across round boundaries?

**Method:** Optical flow (SEA-RAFT) between consecutive frames to compute per-pixel velocity magnitude (vmag) and acceleration magnitude (amag).

**Formula:**
```
MRS = vmag_median * exp(-λ * amag_median)
```
where λ = 1.0 (default). Higher MRS = smoother motion.

**Interpretation:** High velocity without sudden acceleration changes indicates smooth transitions. Models that "restart" each round from scratch show low MRS at round boundaries.

### 3.3 Generation Consistency (WorldScore)

**What:** Does multi-round generation maintain 3D consistency, style, and object identity?

**Method:** Integration with the WorldScore benchmark framework. Evaluates 7 aspects:

| Aspect | Metric | What It Measures |
|--------|--------|-----------------|
| Camera Control | Reprojection error (DROID-SLAM) | Does the camera go where instructed? |
| Object Control | Detection score (GroundingDINO + SAM2) | Are specified objects present/moved? |
| Content Alignment | CLIP score | Does output match the text prompt? |
| 3D Consistency | Reprojection error (DROID-SLAM) | Is the 3D structure stable across rounds? |
| Photometric Consistency | Optical flow AEPE (SEA-RAFT) | Are pixel intensities consistent? |
| Style Consistency | Gram matrix distance (VGG) | Is the visual style stable? |
| Subjective Quality | CLIP-IQA+ and MUSIQ | How good does it look? |

**Composite:** WorldScore-Static aggregates all 7 into a single score.

### 3.4 Simulative Reasoning and Planning

**What:** Can the model support goal-directed thought experiments?

**Method:** VLM (OpenAI o3) + WFM in an iterative planning loop:
1. VLM proposes 3 candidate next actions given goal + history + current frame
2. For each candidate, generate a video segment (best-of-N with GPT-scored selection)
3. VLM examines resulting last frames and picks the best action
4. Repeat until goal achieved or max steps

**Two variants:**
- **Open-ended:** Agibot-finetuned models, robotic manipulation (15 scenarios)
- **Structured:** Language Table-finetuned, tabletop manipulation (47 cases, max 5-10 actions)

**Scoring per segment:**
- Object permanence (1-5)
- Action following (1-5)
- Final check (0-1, penalizes multi-action segments)

---

## 4. Key Findings

### 4.1 Planning is the Differentiator
- PAN outperforms all others in planning: +26.7% in open-ended, +23.4% in structured planning over VLM-only baselines
- Most video generators actually *hurt* planning when used in the VLM+WM loop — they generate pretty videos that mislead the planner
- Only PAN and (sometimes) Cosmos show consistent planning improvement

### 4.2 Environment Simulation is Hard
- No model exceeds 60% on environment-level interventions (shadows, lighting changes, fluid dynamics)
- Agent simulation averages 11.5% higher than environment simulation
- This suggests current WFMs model agent actions but not physics

### 4.3 Long-Horizon Degrades Universally
- No model sustains above 65% on smoothness or consistency over many rounds
- WAN 2.1 degrades most dramatically: ~90% → ~30% over 9 rounds
- PAN has the flattest degradation curve
- Cosmos models show moderate degradation

### 4.4 Visual Quality ≠ World Understanding
- Commercial video generators (KLING, MiniMax, Gen-3) produce the best-looking videos
- But they fail at planning integration, action fidelity, and consistency
- Action-state aligned training (Cosmos, PAN) matters more than visual fidelity

---

## 5. Datasets

### 5.1 Action Simulation Fidelity (60 instances)

Schema:
```json
{
  "id": "agent_000_1",           // "agent_" or "env_" prefix
  "image_path": "initial_state/000.png",
  "prompt_list": [
    "A man walks toward the toll booth",
    "He reaches into his pocket for change",
    "He hands the coins to the attendant"
  ]
}
```

30 agent-focused + 30 environment-focused instances over 10 unique initial images. Each has 3 sequential prompts describing a multi-step action scenario.

### 5.2 Smoothness Evaluation (100 instances)

Schema:
```json
{
  "id": "scene_001",
  "visual_movement": "dynamic",
  "visual_style": "photorealistic",
  "scene_type": "outdoor",
  "category": "diverse",
  "scenario": {"sid": "S001", "label": "park_scene", "definition": "..."},
  "camera_path": ["pan_right", "pan_right", "zoom_in", ...],
  "content_list": ["A park with oak trees", ...],
  "prompt_list": ["Camera pans right over the park", ...],
  "image_path": "static/photorealistic/outdoor/diverse/001_1.png"
}
```

10-round multi-step prompts with camera motion types. Photorealistic outdoor scenes.

### 5.3 Generation Consistency (100 instances)

Same schema as smoothness but with camera-motion focused prompts. Tests whether models maintain 3D and visual consistency across rounds.

### 5.4 External Dataset Dependencies (not bundled)

- **WorldScore-Dataset:** Initial images for smoothness/consistency evaluations
- **Agibot World Colosseo:** Robotic manipulation scenarios for open-ended planning
- **Language Table:** Tabletop manipulation tasks for structured planning

---

## 6. Provider Integration Patterns (from WR-Arena adapters)

### 6.1 Common Interface

All WR-Arena generators implement:
```python
generate_video(prompt: str, image_path: str) -> List[PIL.Image]
```

### 6.2 Multi-Round Chaining

The core orchestration loop (`generate_videos.py`):
1. Load initial image
2. For each prompt in sequence: generate video, extract last frame as next input
3. Cosmos special case: pass entire video (not just last frame) as context
4. Concatenate frames with overlap removal (drop first frame of subsequent rounds)
5. Save complete video + per-round segments

### 6.3 API Patterns by Provider

**KLING (JWT REST):**
```
POST https://api-singapore.klingai.com/v1/videos/image2video
Authorization: Bearer <JWT from KLING_API_KEY + KLING_API_SECRET>
Body: { model, image (base64), prompt, negative_prompt, cfg_scale, duration, aspect_ratio }
→ Poll task_id → Download video URL
```

**MiniMax (REST):**
```
POST https://api.minimax.io/v1/video_generation
Authorization: Bearer <API_KEY>
Body: { model: "T2V-01", prompt, first_frame_image (base64) }
→ Poll task_id → Get file_id → Download
```

**Sora 2 (OpenAI SDK):**
```python
client = openai.OpenAI()
result = client.videos.create_and_poll(
    model="sora-2",
    prompt=prompt,
    image=base64_image
)
# Download from result.url
```

**Veo 3 (Google GenAI SDK):**
```python
client = genai.Client()
operation = client.models.generate_videos(
    model="veo-3.1-fast-generate-preview",
    prompt=prompt,
    image=genai.types.Image(image_bytes=bytes, mime_type="image/png"),
    config=genai.types.GenerateVideoConfig(...)
)
# Poll operation, download from result.generated_videos[0].video.uri
```

**PAN (Stateful REST):**
```
POST <endpoint>/first_round
Body: { prompt, image_path, state_id }
→ Returns video frames + state_id + video_id

POST <endpoint>/continue
Body: { prompt, state_id, video_id }
→ Returns next round frames (maintains internal state)
```

PAN is unique: it maintains server-side state across rounds, enabling true multi-round simulation without the client needing to pass frames back.

### 6.4 Negative Prompt (shared quality guard)

Used by Cosmos and KLING:
```
"The video captures a series of frames showing ugly scenes, static with no motion,
motion blur, over-saturation, shaky footage, low resolution, grainy texture..."
```

---

## 7. Relevance to WorldForge

### 7.1 Provider Gaps

WorldForge currently has: Cosmos, Runway, JEPA, Genie, Marble, Mock
WR-Arena adds these models we're missing:

| Model | Priority | Rationale |
|-------|----------|-----------|
| **PAN** | Critical | Best planning model, stateful multi-round API, novel architecture |
| **KLING** | High | Popular commercial model, straightforward JWT REST API |
| **Sora 2** | High | OpenAI ecosystem, massive user base |
| **Veo 3** | High | Google ecosystem, strong visual quality |
| **MiniMax** | High | Widely used, simple REST API |
| **WAN 2.1/2.2** | Medium | Open-source, requires multi-GPU infrastructure |

### 7.2 Evaluation Gaps

WorldForge's eval crate has built-in metrics but lacks WR-Arena's dimensions:
- **Action fidelity scoring** (GPT-as-judge with rubric) — not present
- **MRS smoothness metric** (optical flow based) — not present
- **WorldScore integration** (7-aspect consistency) — not present
- **VLM+WM planning evaluation** — not present

### 7.3 Orchestration Gaps

- **Multi-round video generation pipeline** — WorldForge has prediction chaining but not the frame-extraction + overlap-removal pattern
- **Best-of-N selection** — WR-Arena generates N variants and GPT-selects the best; WorldForge's prediction engine should support this
- **Prompt upsampling** — PAN enriches short prompts into detailed descriptions before generation

### 7.4 Dataset Integration

WorldForge should support loading WR-Arena datasets as evaluation inputs. This requires:
- JSON dataset loader for the three WR-Arena schemas
- Image reference resolution (paths to initial frame images)
- Multi-round prompt sequence handling
