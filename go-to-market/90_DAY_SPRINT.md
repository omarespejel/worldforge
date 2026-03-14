# WorldForge: 90-Day Sprint Plan

## Week-by-Week Execution

---

### WEEK 1: Foundation

**Day 1-2: Setup**
- [ ] Register domains: worldforge.ai, worldforge.dev
- [ ] Create GitHub org: github.com/worldforge-ai (or use AbdelStark/worldforge)
- [ ] Initialize Cargo workspace with all crate stubs
- [ ] Set up CI/CD (GitHub Actions: cargo test, cargo clippy, cargo fmt)
- [ ] Create Discord server for community

**Day 3-5: Core Types**
- [ ] Implement core type system (Tensor, Position, Pose, Frame, VideoClip)
- [ ] Implement WorldState and SceneGraph
- [ ] Implement Action enum with all variants
- [ ] Unit tests for all types (serialization roundtrip, equality, cloning)
- [ ] Property-based tests with proptest

**Day 6-7: Provider Trait**
- [ ] Define WorldModelProvider trait
- [ ] Define ProviderCapabilities struct
- [ ] Define ProviderRegistry
- [ ] Implement MockProvider for testing
- [ ] Write integration test using MockProvider

---

### WEEK 2: First Provider

**Day 8-10: Cosmos Provider**
- [ ] Implement CosmosProvider struct
- [ ] Implement HTTP client for NIM API (Cosmos Predict 2.5)
- [ ] Implement predict() via Cosmos Predict API
- [ ] Implement reason() via Cosmos Reason 2 API
- [ ] Handle authentication (NGC API key)
- [ ] Parse Cosmos video output into WorldForge VideoClip

**Day 11-12: Python Bindings (v0.1)**
- [ ] Set up PyO3 project structure
- [ ] Bind WorldForge, World, Action, Prediction to Python
- [ ] Write minimal Python example
- [ ] Publish to TestPyPI

**Day 13-14: CLI (v0.1)**
- [ ] Implement `worldforge create --prompt "..." --provider cosmos`
- [ ] Implement `worldforge predict --world <id> --action "..." --steps 10`
- [ ] Implement `worldforge providers` (list available)
- [ ] Implement `worldforge health` (check all providers)

---

### WEEK 3: Second Provider + State

**Day 15-17: JEPA Provider (Local)**
- [ ] Implement JepaProvider struct
- [ ] Load V-JEPA weights from safetensors (or use jepa-rs if ready)
- [ ] Implement predict() via local inference
- [ ] Implement physics_score() via energy function
- [ ] Test: same scenario on Cosmos vs JEPA

**Day 18-19: State Persistence**
- [ ] Implement FileStateStore (JSON serialization)
- [ ] Implement SqliteStateStore
- [ ] State history tracking (append-only log)
- [ ] Test: create world, predict, save, reload, predict again

**Day 20-21: Multi-Provider Comparison**
- [ ] Implement predict_multi() (parallel predictions across providers)
- [ ] Implement ComparisonReport (side-by-side metrics)
- [ ] CLI: `worldforge compare --world <id> --action "..." --providers cosmos,jepa`
- [ ] Generate comparison output (text table + optional video side-by-side)

---

### WEEK 4: Launch Preparation

**Day 22-23: Documentation**
- [ ] README with quick start, installation, examples
- [ ] API reference (auto-generated from Rust doc comments)
- [ ] Tutorial 1: "Your First World Model Prediction"
- [ ] Tutorial 2: "Comparing Cosmos vs JEPA on Physics Tasks"
- [ ] CONTRIBUTING.md with development setup instructions

**Day 24-25: Content**
- [ ] Write blog post: "Introducing WorldForge: The Orchestration Layer for World Models"
- [ ] Write X thread (15 tweets) covering the problem, solution, and demo
- [ ] Record 3-minute demo video (CLI: create world, predict, compare providers)

**Day 26: Pre-Launch Seeding**
- [ ] Send preview to 20 people in ML/robotics community for feedback
- [ ] Share in awesome-world-models repo
- [ ] Preview in World Model Weekly newsletter

**Day 27-28: LAUNCH**
- [ ] Push v0.1.0 to GitHub
- [ ] Publish to crates.io and PyPI
- [ ] Post X thread
- [ ] Submit to Hacker News (Show HN)
- [ ] Post to r/MachineLearning, r/robotics, r/rust
- [ ] Post to LinkedIn
- [ ] Send to AMI Labs team (LeBrun, Xie, Rabbat)
- [ ] Send to NVIDIA Cosmos team

---

### WEEK 5-6: Post-Launch Momentum

- [ ] Respond to every GitHub issue within 24 hours
- [ ] Fix bugs reported by early users
- [ ] Add Runway GWM provider (if SDK access granted)
- [ ] Tutorial 3: "Generating Synthetic Data for Robot Training"
- [ ] Tutorial 4: "Building a Physics Evaluation Suite"
- [ ] Guest post on someone else's newsletter/blog
- [ ] First "WorldForge Office Hours" video call

---

### WEEK 7-8: Evaluation Framework

- [ ] Implement EvalSuite with 20 physics scenarios
- [ ] Implement automated scoring (object permanence, gravity, collision)
- [ ] Implement human evaluation web interface (side-by-side voting)
- [ ] Run first comprehensive evaluation: Cosmos vs JEPA on all 20 scenarios
- [ ] Publish results as blog post: "World Model Physics Showdown: Cosmos vs JEPA"
- [ ] Launch public leaderboard on worldmodelarena.com

---

### WEEK 9-10: Guardrails + Planning

- [ ] Implement Guardrail types (no_collisions, stay_upright, boundary, etc.)
- [ ] Implement guardrail pipeline (check every prediction)
- [ ] Implement basic planning (CEM planner)
- [ ] Implement gradient-based planning for JEPA provider
- [ ] Tutorial 5: "Safe Robot Planning with WorldForge Guardrails"
- [ ] CLI: `worldforge plan --world <id> --goal "..." --guardrails no_collisions`

---

### WEEK 11-12: ZK Verification PoC

- [ ] Design ZK circuit for tiny JEPA forward pass
- [ ] Implement proof generation (EZKL or custom STARK)
- [ ] Implement proof verification
- [ ] Benchmark: proof generation time, verification time, proof size
- [ ] Blog post: "I Proved a World Model Forward Pass in Zero Knowledge"
- [ ] This is the "moonshot demo" that differentiates WorldForge from everything else

---

### WEEK 13 (END OF QUARTER): Retrospective + Planning

- [ ] Review metrics: GitHub stars, PyPI downloads, active users, issues/PRs
- [ ] Gather user feedback (survey, interviews, Discord)
- [ ] Plan Q2: Cloud product, Runway provider, enterprise features
- [ ] Begin co-founder search if not already found
- [ ] Begin pre-seed fundraising conversations if metrics justify it

---

## Key Milestones

| Milestone | Target Date | Metric |
|-----------|------------|--------|
| v0.1.0 launch | Week 4 | Cosmos + JEPA providers working |
| 500 GitHub stars | Week 6 | Community traction |
| Evaluation framework live | Week 8 | 20 scenarios, public leaderboard |
| Planning + guardrails | Week 10 | CEM + gradient planners working |
| ZK verification PoC | Week 12 | Proof of tiny JEPA forward pass |
| 2,000 GitHub stars | Week 13 | Strong community signal |
