# WorldForge Business Plan

**Confidential | March 2026**

---

## 1. Executive Summary

WorldForge is the orchestration layer for world foundation models. We provide a unified developer toolkit that lets robotics engineers, AV developers, and physical AI builders work with any world model (NVIDIA Cosmos, Runway GWM, Meta JEPA, Google Genie) through a single API.

We are to world models what LangChain is to LLMs and what Vercel is to web frameworks.

**Market timing:** World models are at the GPT-2-to-GPT-3 inflection point. PitchBook projects the world model market in gaming alone at $276B by 2030. Physical AI raised $38.5B in venture funding in 2025. The tooling layer doesn't exist yet.

**Business model:** Open-source core (Apache 2.0) with cloud offering (managed hosting, caching, dashboards, enterprise support). Revenue from usage-based cloud pricing and enterprise contracts.

**Differentiation:** Only toolkit with (a) native ZK verification for safety-critical deployments, (b) Rust core for edge/embedded deployment, (c) cross-provider evaluation framework.

---

## 2. Market Analysis

### 2.1 Total Addressable Market

| Segment | 2026 | 2030 (projected) |
|---------|------|-------------------|
| World models in gaming | $3B | $276B (PitchBook) |
| Robotics simulation & training | $5B | $50B+ |
| AV synthetic data & testing | $8B | $80B+ |
| Industrial digital twins | $10B | $100B+ |
| Healthcare simulation | $2B | $20B+ |
| **Total developer tooling** (5-10% of above) | **$1.4-2.8B** | **$26-53B** |

### 2.2 Comparable Companies

| Company | What | Revenue | Valuation | Founded |
|---------|------|---------|-----------|---------|
| LangChain | LLM orchestration | ~$30M ARR | ~$3B | 2022 |
| Hugging Face | ML model hub | ~$70M ARR | $4.5B | 2016 |
| Weights & Biases | ML experiment tracking | ~$100M ARR | $5B+ | 2017 |
| Scale AI | Data labeling + evaluation | $1B+ ARR | $14B | 2016 |
| Vercel | Web framework deployment | $200M+ ARR | $3.5B | 2015 |

**Key insight:** Developer tooling companies consistently reach $1B+ valuations when they become the standard layer in their ecosystem. The world model tooling layer is empty.

### 2.3 Competitive Landscape

**Direct competitors:** None. No company is building a unified world model orchestration layer.

**Adjacent players:**
- NVIDIA (Cosmos): Provider, not orchestrator. They'd benefit from WorldForge driving adoption of Cosmos.
- Runway (GWM SDK): Provider SDK, not cross-provider. Focused on their own models.
- Hugging Face: Model hub, not orchestration. Could host WorldForge models.
- RobotHub/LeRobot: Robotics-specific, not world-model-agnostic.

**Potential competitors:**
- LangChain could expand into world models (but they're LLM-focused)
- A new startup could emerge (first-mover advantage matters)
- A provider (NVIDIA, Runway) could build their own orchestration (but this would disadvantage competitors, creating demand for a neutral layer)

---

## 3. Product Strategy

### 3.1 Product Tiers

**Tier 1: Open Source Core (Free)**
- worldforge-core (Rust library)
- worldforge-py (Python bindings)
- All provider adapters
- Evaluation framework
- CLI tool
- Apache 2.0 license

**Tier 2: WorldForge Cloud (Freemium → Paid)**
- Managed inference routing (auto-select best provider per request)
- Response caching (don't re-infer identical requests)
- Dashboard (latency, cost, quality metrics per provider)
- Hosted world state persistence
- API gateway with rate limiting and auth
- Free tier: 100 predictions/month
- Pro tier: $49/month for 10,000 predictions
- Team tier: $199/month for 100,000 predictions

**Tier 3: WorldForge Enterprise (Custom)**
- Self-hosted deployment
- SLA guarantees
- Custom provider adapters
- ZK verification module (verified inference for safety-critical)
- Priority support
- Starting at $5,000/month

### 3.2 Revenue Projections

| Quarter | MRR | ARR | Customers |
|---------|-----|-----|-----------|
| Q3 2026 | $0 | $0 | 0 (building) |
| Q4 2026 | $5K | $60K | 20 (Cloud free/pro) |
| Q1 2027 | $25K | $300K | 100 |
| Q2 2027 | $75K | $900K | 300 + 2 enterprise |
| Q3 2027 | $150K | $1.8M | 600 + 5 enterprise |
| Q4 2027 | $300K | $3.6M | 1,000 + 10 enterprise |

### 3.3 Key Metrics

| Metric | Target (12 months) |
|--------|-------------------|
| GitHub stars (core) | 5,000+ |
| PyPI downloads/month | 50,000+ |
| Active developers | 1,000+ |
| Cloud customers | 500+ |
| Enterprise customers | 10+ |
| ARR | $2M+ |

---

## 4. Go-To-Market Strategy

### 4.1 Phase 1: Open Source Launch (Month 1-3)

**Goal:** Establish WorldForge as the go-to toolkit for world model developers.

**Actions:**
1. Ship worldforge-core with Cosmos + JEPA providers
2. Ship worldforge-py with ergonomic Python API
3. Ship worldforge-cli for quick experimentation
4. Publish 5 tutorials ("Build X with WorldForge")
5. Publish comparison blog posts ("Cosmos vs GWM vs JEPA on 10 physics tests")
6. Submit to Hacker News, r/MachineLearning, r/robotics
7. Present at meetups (Paris AI, London Robotics, online)
8. Get listed in awesome-world-models (your own repo)

**Distribution channels:**
- GitHub (primary)
- PyPI
- X / Twitter
- Hacker News
- Reddit (r/MachineLearning, r/robotics, r/LocalLLaMA)
- World Model Weekly (your newsletter)
- AI/robotics Discord servers

### 4.2 Phase 2: Community & Adoption (Month 3-6)

**Goal:** 1,000+ GitHub stars, 50+ active users, 5+ external contributors.

**Actions:**
1. Add Runway GWM provider (now available via Python SDK)
2. Ship evaluation framework with 50+ physics test scenarios
3. Launch public leaderboard (world-model-arena.com)
4. Start "WorldForge Office Hours" (weekly video call)
5. Sponsor/present at NeurIPS 2026 workshops
6. Reach out to robotics teams at universities (CMU, Stanford, ETH Zurich)
7. Reach out to AMI Labs, Skild AI, Physical Intelligence as design partners
8. Publish "The State of World Models 2026" report (annual, like AI Index)

### 4.3 Phase 3: Cloud Launch (Month 6-9)

**Goal:** Launch WorldForge Cloud, first revenue.

**Actions:**
1. Build cloud infrastructure (managed inference routing, caching, dashboard)
2. Launch free tier (100 predictions/month)
3. Launch Pro tier ($49/month)
4. Content marketing: case studies, benchmarks, ROI calculators
5. Sales outreach to robotics startups and AV companies
6. Partnership with NVIDIA (Cosmos integration showcase)

### 4.4 Phase 4: Enterprise (Month 9-12)

**Goal:** First enterprise customers, $100K+ ARR.

**Actions:**
1. Ship ZK verification module
2. Ship self-hosted deployment option
3. Hire first sales hire (enterprise robotics/AV background)
4. SOC 2 compliance (required for enterprise)
5. Present at industry events (RoboCup, IROS, CES)
6. Case study with at least one well-known robotics company

---

## 5. Funding Strategy

### 5.1 Bootstrap Phase (Month 1-6)

**Funding:** $0. Self-funded with personal savings.
**Costs:** Minimal. Open-source development. Domain names. Basic cloud for website.
**Revenue:** $0. Building open source.
**Supplemental income:** Consulting on world models + agentic engineering ($500-2,000/hour, 5-10 hours/week). This funds operations while building WorldForge.

### 5.2 Pre-Seed (Month 6-9)

**Raise:** $500K-$1M
**Source:** Angel investors in AI/robotics space + small funds
**Milestones to hit before raising:**
- 2,000+ GitHub stars
- 200+ active developers
- 3+ provider integrations working
- Evaluation framework with public leaderboard
- At least 1 design partner (AMI Labs, university lab, or robotics startup)

**Use of funds:**
- 1 additional Rust engineer (6 months)
- Cloud infrastructure (AWS/GCP credits)
- Conference travel
- Legal (incorporation, IP)

### 5.3 Seed Round (Month 12-15)

**Raise:** $3-5M
**Source:** Lux Capital, Playground Global, a]16z, Sequoia Scout, NVIDIA Inception
**Milestones:**
- 5,000+ GitHub stars
- 1,000+ active developers
- Cloud product live with paying customers
- $100K+ ARR
- ZK verification PoC working
- 3+ enterprise design partners

**Use of funds:**
- Team of 5-8 (2 Rust engineers, 1 ML engineer, 1 designer, 1 DevRel, 1 sales)
- GPU compute for evaluation infrastructure
- Marketing and community
- Legal and compliance

### 5.4 Series A (Month 18-24)

**Raise:** $15-30M
**Valuation target:** $100-300M
**Milestones:**
- $2M+ ARR
- 10+ enterprise customers
- ZK verification in production
- Industry standard for world model evaluation
- 10,000+ GitHub stars
- WorldForge mentioned in NVIDIA, Runway, or AMI documentation

---

## 6. Team Plan

### 6.1 Founding Team (Month 1-6)

- **Abdel Bakhta (Founder/CEO):** Architecture, Rust core, provider integrations, ZK verification, community, fundraising
- Need: 1 co-founder with ML/robotics background (ideally from FAIR, CMU, or Stanford robotics)

### 6.2 Ideal Co-Founder Profile

- Strong ML engineering (can implement ViT, JEPA architectures from scratch)
- Robotics experience (understands action spaces, sim-to-real transfer)
- Open-source contributor (credibility in the community)
- Based in Paris or willing to relocate (for AMI Labs proximity)
- Complementary to Abdel: more ML depth, less systems/crypto

Where to find them:
- AMI Labs researchers who want to start something
- FAIR alumni in Paris
- ETH Zurich / EPFL robotics graduates
- ex-Google DeepMind robotics team
- Hugging Face alumni

### 6.3 First Hires (Month 6-12)

1. **Rust Engineer:** Core library, performance, WASM compilation
2. **ML Engineer:** Provider adapters, evaluation framework, model integration
3. **DevRel / Developer Advocate:** Documentation, tutorials, community management, conference presence

### 6.4 Growth Team (Month 12-24)

4. **Cloud Engineer:** Infrastructure, caching, dashboard
5. **Designer:** CLI, dashboard, documentation UX
6. **Enterprise Sales:** Robotics/AV industry background
7. **Security Engineer:** ZK verification module, SOC 2 compliance

---

## 7. Legal & IP

### 7.1 Entity Structure

- **WorldForge SAS** (French simplified stock company) — operating entity
- Registered in Paris (proximity to AMI Labs, French tech ecosystem)
- JEI status (Jeune Entreprise Innovante) for tax benefits
- CIR (Crédit d'Impôt Recherche) for R&D tax credits (up to 30%)

### 7.2 IP Strategy

- Core library: Apache 2.0 (maximizes adoption)
- Cloud offering: Proprietary
- ZK verification module: Apache 2.0 (builds credibility, encourages adoption in safety-critical)
- WorldForge name + logo: Trademark registered in France + EU
- Patents: Consider filing on ZK verification for world model inference (genuinely novel)

### 7.3 Compliance

- GDPR compliant (French/EU company, handling EU data)
- SOC 2 Type II (target for Month 12, required for enterprise customers)
- EU AI Act compliance (WorldForge as a tool for conformity assessment)

---

## 8. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| World model market doesn't grow as fast | Low | High | Diversify across robotics, AV, gaming |
| Provider builds their own orchestration | Medium | High | Be neutral/multi-provider, embed in workflows |
| LangChain expands into world models | Medium | Medium | Ship faster, ZK moat, Rust performance moat |
| Can't find ML co-founder | Medium | High | Start solo, recruit from AMI/FAIR network |
| Open source doesn't gain traction | Medium | High | Marketing, content, conference presence |
| Provider APIs change breaking adapters | High | Low | Adapter abstraction, fast update cycle |
| ZK verification too slow for production | Medium | Medium | Start with offline verification, optimize over time |
| Funding environment deteriorates | Low | Medium | Bootstrap longer, use consulting revenue |

---

## 9. Success Criteria (18 Months)

### Must Hit
- [ ] 5,000+ GitHub stars
- [ ] 3+ providers fully integrated
- [ ] Cloud product live with revenue
- [ ] 500+ active developers
- [ ] Seed round closed ($3-5M)

### Should Hit
- [ ] 10+ enterprise customers
- [ ] ZK verification working on small models
- [ ] Invited to present at major AI conference
- [ ] Referenced in at least one provider's documentation
- [ ] $1M+ ARR

### Aspirational
- [ ] Becomes the default evaluation framework for world models
- [ ] WorldForge Protocol adopted by 2+ providers
- [ ] Partnership or acquisition interest from major player
- [ ] Featured in TechCrunch/Wired/MIT Tech Review
- [ ] $5M+ ARR
