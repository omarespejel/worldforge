# World Foundation Model Provider Catalog

**Version 0.2.0-draft | March 2026**

This catalog documents all world foundation models and video generation models that WorldForge should support, organized by integration priority. Sources include direct API documentation, the WR-Arena benchmark (arXiv 2603.25887), and provider documentation.

---

## Tier 1: Currently Implemented

### NVIDIA Cosmos (Predict1/Predict2)

| Property | Value |
|----------|-------|
| **Organization** | NVIDIA |
| **Models** | Cosmos-Predict1-14B, Cosmos-Predict2-14B, Cosmos-Reason2-7B, Cosmos-Embed1, Cosmos-Transfer2.5 |
| **Access** | NIM API, NIM containers, NGC, HuggingFace |
| **Auth** | NVIDIA API Key via `NVIDIA_API_KEY` |
| **Endpoint** | `https://ai.api.nvidia.com/v1/infer` (NIM API) |
| **Resolution** | 1280x704 |
| **Frames** | 121 (Predict1), 93 (Predict2) |
| **License** | Apache 2.0 (code), NVIDIA Open Model License (weights) |
| **WorldForge crate** | `worldforge-providers/src/cosmos.rs` |
| **Status** | Implemented (full-stack: predict, generate, reason, transfer, embed, plan) |

**WR-Arena findings:**
- Cosmos models show moderate long-horizon degradation
- Cosmos-Predict1 performs better on environment simulation than most video generators
- Special case: multi-round generation should pass entire video as context (not just last frame)
- Uses a shared negative prompt for quality guard

**Negative prompt (quality guard):**
```
The video captures a series of frames showing ugly scenes, static with no motion,
motion blur, over-saturation, shaky footage, low resolution, grainy texture...
```

### Runway GWM

| Property | Value |
|----------|-------|
| **Organization** | Runway |
| **Models** | GWM-1 Worlds, GWM-1 Robotics, GWM-1 Avatars, Gen-3, Gen-4.5 |
| **Access** | Python SDK (`runwayml`), Node SDK, REST API |
| **Auth** | API Secret via `RUNWAY_API_SECRET` |
| **Endpoint** | `https://api.runwayml.com` |
| **Resolution** | 1280x768 (Gen-3) |
| **Frames** | 125 (Gen-3) |
| **License** | Proprietary API |
| **WorldForge crate** | `worldforge-providers/src/runway.rs` |
| **Status** | Implemented (predict, generate, transfer, plan; reason via Cosmos fallback) |

**WR-Arena integration pattern (Gen-3):**
```python
import runwayml
client = runwayml.RunwayML()
task = client.image_to_video.create(
    model="gen3a_turbo",
    prompt_image=image_uri,
    prompt_text=prompt
)
# Poll task.id until complete, download video
```

### Meta V-JEPA 2

| Property | Value |
|----------|-------|
| **Organization** | Meta AI |
| **Models** | I-JEPA, V-JEPA, V-JEPA 2, EB-JEPA |
| **Access** | Open-source weights (HuggingFace/GitHub) |
| **Auth** | None (local inference) |
| **Weights** | Safetensors format |
| **License** | CC-BY-NC 4.0 (models), Apache 2.0 (code) |
| **WorldForge crate** | `worldforge-providers/src/jepa.rs` |
| **Status** | Implemented (predict, reason, embed, plan via gradient) |

**WR-Arena findings:**
- V-JEPA 2 is representation-based (not generative) — evaluates differently
- Primary target for ZK verification (deterministic forward pass)

### Google Genie 3

| Property | Value |
|----------|-------|
| **Organization** | Google DeepMind |
| **Models** | Genie 3 |
| **Access** | Research preview |
| **WorldForge crate** | `worldforge-providers/src/genie.rs` |
| **Status** | Implemented (deterministic local surrogate with depth/segmentation) |

### World Labs Marble

| Property | Value |
|----------|-------|
| **Organization** | World Labs |
| **Access** | Experimental |
| **WorldForge crate** | `worldforge-providers/src/marble.rs` |
| **Status** | Implemented (experimental local surrogate with all capabilities) |

---

## Tier 2: High Priority (from WR-Arena)

### MBZUAI PAN

| Property | Value |
|----------|-------|
| **Organization** | MBZUAI Institute of Foundation Models |
| **Models** | PAN (general world model) |
| **Access** | API at `https://ifm.mbzuai.ac.ae/pan/`, HuggingFace: `MBZUAI-IFM` |
| **Resolution** | 832x480 |
| **Frames** | 41 per round |
| **License** | TBD |
| **Priority** | **Critical** — best planning model per WR-Arena |

**Why critical:**
- PAN outperforms all other WFMs in planning: +26.7% open-ended, +23.4% structured
- Flattest long-horizon degradation curve
- Unique stateful multi-round API with server-side state persistence

**API pattern (stateful multi-round):**
```
# Round 1: Initialize
POST <endpoint>/first_round
{
  "prompt": "Robot reaches for red block",
  "image_path": "/path/to/initial.png",
  "state_id": "session-001"
}
→ { "frames": [...], "state_id": "session-001", "video_id": "v001" }

# Round N: Continue (server maintains state)
POST <endpoint>/continue
{
  "prompt": "Robot grasps the block and lifts",
  "state_id": "session-001",
  "video_id": "v001"
}
→ { "frames": [...], "state_id": "session-001", "video_id": "v002" }
```

**Key features:**
- Server-side state: no need to pass frames back and forth
- Load-balanced endpoints
- Prompt upsampling: enriches short prompts to detailed descriptions
- Prepends `"FPS-{fps} "` to prompts

**WorldForge mapping:**
| WorldForge concept | PAN implementation |
|-------------------|-------------------|
| predict() | first_round / continue API |
| plan() | VLM+WM loop (PAN as WM) |
| generate() | first_round with text prompt |
| Multi-round | continue endpoint (stateful) |

**Proposed env vars:** `PAN_API_KEY`, `PAN_API_ENDPOINT`

### Kuaishou KLING

| Property | Value |
|----------|-------|
| **Organization** | Kuaishou Technology |
| **Models** | KLING (video generation) |
| **Access** | REST API |
| **Auth** | JWT from `KLING_API_KEY` + `KLING_API_SECRET` |
| **Endpoint** | `https://api-singapore.klingai.com` |
| **Resolution** | 1280x720 |
| **Frames** | 153 |
| **License** | Proprietary API |
| **Priority** | High — popular commercial model |

**API pattern (JWT REST):**
```
# 1. Generate JWT token
Header: { "alg": "HS256", "typ": "JWT" }
Payload: { "iss": KLING_API_KEY, "exp": now+1800, "iat": now }
Sign with: KLING_API_SECRET

# 2. Submit generation
POST https://api-singapore.klingai.com/v1/videos/image2video
Authorization: Bearer <jwt>
{
  "model": "kling-v1",
  "image": "<base64>",
  "prompt": "...",
  "negative_prompt": "...",
  "cfg_scale": 0.5,
  "duration": "10",
  "aspect_ratio": "16:9"
}
→ { "task_id": "..." }

# 3. Poll for completion
GET /v1/videos/image2video/<task_id>
→ { "status": "completed", "video_url": "..." }

# 4. Download video from URL
```

**WorldForge mapping:**
| WorldForge concept | KLING implementation |
|-------------------|---------------------|
| predict() | image2video with action prompt |
| generate() | image2video with scene prompt |

**Proposed env vars:** `KLING_API_KEY`, `KLING_API_SECRET`

### OpenAI Sora 2

| Property | Value |
|----------|-------|
| **Organization** | OpenAI |
| **Models** | Sora 2 |
| **Access** | OpenAI SDK |
| **Auth** | OpenAI API key via `OPENAI_API_KEY` |
| **Resolution** | 1280x720 |
| **Frames** | 120 |
| **License** | Proprietary API |
| **Priority** | High — OpenAI ecosystem, massive user base |

**API pattern (OpenAI SDK):**
```python
import openai
client = openai.OpenAI()
result = client.videos.create_and_poll(
    model="sora-2",
    prompt=prompt,
    image=base64_image,
    duration=5,
    resolution="1280x720"
)
# result.url contains the generated video
```

**Rust integration:** Use the OpenAI REST API directly with reqwest:
```
POST https://api.openai.com/v1/videos
Authorization: Bearer <OPENAI_API_KEY>
{
  "model": "sora-2",
  "prompt": "...",
  "image": "<base64>",
  "duration": 5
}
→ Poll for completion
```

**WorldForge mapping:**
| WorldForge concept | Sora 2 implementation |
|-------------------|-----------------------|
| predict() | video generation with action prompt |
| generate() | video generation with scene prompt |

**Proposed env var:** `OPENAI_API_KEY`

### Google Veo 3

| Property | Value |
|----------|-------|
| **Organization** | Google DeepMind |
| **Models** | Veo 3.1 (veo-3.1-fast-generate-preview) |
| **Access** | Google GenAI SDK |
| **Auth** | Google API key via `GOOGLE_API_KEY` |
| **Resolution** | 1280x720 |
| **Frames** | 96 |
| **License** | Proprietary API |
| **Priority** | High — Google ecosystem, strong visual quality |

**API pattern (GenAI SDK):**
```python
from google import genai
client = genai.Client(api_key=API_KEY)
operation = client.models.generate_videos(
    model="veo-3.1-fast-generate-preview",
    prompt=prompt,
    image=genai.types.Image(image_bytes=img_bytes, mime_type="image/png"),
    config=genai.types.GenerateVideoConfig(
        person_generation="allow_all",
        aspect_ratio="16:9"
    )
)
# Poll operation until done
# Download from operation.result.generated_videos[0].video.uri
```

**Rust integration:** Use the Vertex AI / GenAI REST API directly:
```
POST https://generativelanguage.googleapis.com/v1beta/models/veo-3.1:generateVideos
Authorization: Bearer <GOOGLE_API_KEY>
{
  "prompt": "...",
  "image": { "bytes": "<base64>", "mimeType": "image/png" },
  "config": { "aspectRatio": "16:9" }
}
→ Returns operation ID → Poll → Download
```

**WorldForge mapping:**
| WorldForge concept | Veo 3 implementation |
|-------------------|-----------------------|
| predict() | video generation with action prompt |
| generate() | video generation with scene prompt |

**Proposed env var:** `GOOGLE_API_KEY`

### MiniMax/Hailuo

| Property | Value |
|----------|-------|
| **Organization** | MiniMax |
| **Models** | T2V-01 (Hailuo) |
| **Access** | REST API |
| **Auth** | API key via `MINIMAX_API_KEY` |
| **Endpoint** | `https://api.minimax.io` |
| **Resolution** | 1072x720 |
| **Frames** | 141 |
| **License** | Proprietary API |
| **Priority** | High — widely used video generator |

**API pattern (REST submit/poll/download):**
```
# 1. Submit
POST https://api.minimax.io/v1/video_generation
Authorization: Bearer <API_KEY>
{
  "model": "T2V-01",
  "prompt": "...",
  "first_frame_image": "<base64>"
}
→ { "task_id": "..." }

# 2. Poll
GET https://api.minimax.io/v1/query/video_generation?task_id=<id>
→ { "status": "Success", "file_id": "..." }

# 3. Download
GET https://api.minimax.io/v1/files/retrieve?file_id=<id>
→ { "download_url": "..." }
```

**WorldForge mapping:**
| WorldForge concept | MiniMax implementation |
|-------------------|-----------------------|
| predict() | video generation with action prompt |
| generate() | video generation with scene prompt |

**Proposed env var:** `MINIMAX_API_KEY`

---

## Tier 3: Future Consideration

### Alibaba WAN 2.1/2.2

| Property | Value |
|----------|-------|
| **Organization** | Alibaba |
| **Models** | WAN 2.1 I2V-14B, WAN 2.2 I2V-A14B |
| **Access** | Open-source (local multi-GPU inference) |
| **Resolution** | 832x480 |
| **Frames** | 81 |
| **License** | Open-source |
| **Priority** | Medium — requires multi-GPU infrastructure |

**Integration notes:**
- Requires 8 GPUs with xfuser/Ulysses sequence parallelism
- Not API-based — local inference only
- WAN 2.1 shows dramatic long-horizon degradation (~90% → ~30% over 9 rounds)
- WAN 2.2 improves on 2.1 but still significant degradation

**Deferred:** Requires multi-GPU orchestration that is beyond WorldForge's initial scope. Could be supported via a Python sidecar process.

---

## Provider Comparison Matrix (WR-Arena Results)

| Capability | Cosmos1 | Cosmos2 | PAN | KLING | MiniMax | Gen-3 | Sora2 | Veo3 | WAN2.1 | WAN2.2 |
|-----------|---------|---------|-----|-------|---------|-------|-------|------|--------|--------|
| Agent Sim Fidelity | Med | Med | Med | High | Med | Med | Med | Med | Low | Med |
| Env Sim Fidelity | Low | Low | Low | Low | Low | Low | Low | Low | Low | Low |
| Smoothness (MRS) | Med | Med | High | Med | Med | Med | Med | Med | Low | Med |
| Consistency | Med | Med | High | Med | Med | Med | Med | Med | Low | Med |
| Planning (+VLM) | Positive | Positive | **Best** | Negative | Negative | Negative | N/A | N/A | N/A | N/A |
| Long-horizon stability | Med | Med | **Best** | Med | Med | Med | Med | Med | **Worst** | Low |
| Visual quality | Med | Med | Low | **Best** | High | High | High | **Best** | Med | Med |

**Key takeaway:** Visual quality and world understanding are inversely correlated. Commercial video generators produce beautiful outputs but fail at planning. PAN has the best understanding despite lower visual fidelity.

---

## Multi-Round Generation Protocol

All providers follow a common multi-round generation pattern:

```
1. Load initial image I₀
2. For round r = 1..N:
   a. Generate video segment V_r from (prompt_r, I_{r-1})
   b. Extract last frame: I_r = last_frame(V_r)
   c. Store V_r
3. Concatenate: V = V_1[all] + V_2[1:] + ... + V_N[1:]
   (Drop first frame of rounds 2+ to avoid duplicate boundary frames)
```

**Provider-specific exceptions:**
- **Cosmos:** Pass entire previous video (not just last frame) as context
- **PAN:** Server maintains state; just send prompt + state_id to `continue` endpoint
- **KLING/MiniMax/Sora/Veo:** Standard last-frame-as-input pattern
