
**WorldForge**

The LangChain of World Models

*Unified orchestration for physical AI foundation models*

**Comprehensive Business Case**

Thesis, Market, Product, Monetization, Funding Strategy

Abdel — Founder

April 2026 | Confidential

# **1\. The Thesis**

**One sentence:** WorldForge is to world models what LangChain is to LLMs. A unified developer toolkit that lets any builder work with any world foundation model through a single API.

## **1.1 The Pattern**

Every major AI paradigm follows the same adoption curve. First, foundation models emerge from research labs. Then, each model ships with its own SDK. Then, developers suffer through incompatible integrations. Finally, someone builds the unifying abstraction layer that becomes the standard. That layer captures enormous value.

| Paradigm | Models | Unifying Layer | Value Captured |
| :---- | :---- | :---- | :---- |
| **LLMs (2022-23)** | GPT-3/4, Claude, LLaMA, Gemini | **LangChain** | $3B valuation, \~$30M ARR, 100K+ GitHub stars |
| **ML Models (2017-)** | BERT, GPT-2, ResNet, ViT | **Hugging Face** | $4.5B valuation, \~$70M ARR, 140K+ stars |
| **ML Ops (2018-)** | TF, PyTorch, JAX training runs, VLLM | **Weights & Biases** | $5B+ valuation, \~$100M ARR |
| **Web Frameworks (2015-)** | Next.js, Remix, Nuxt, SvelteKit | **Vercel** | $3.5B valuation, $200M+ ARR |
| **World Models (2025-)** | Cosmos, GWM, JEPA, Genie, Marble | **WorldForge** | **The opportunity** |

## **1.2 Why Now**

World models are at the GPT-2 to GPT-3 inflection point. The research exists. The models are shipping. The commercial APIs are going live. But the tooling layer is empty.

**1\.** Investment in world models surged from $1.4B in 2024 to $6.9B in 2025 (CB Insights). Companies in the space average a Mosaic score of 722, placing them in the top 3% of all markets globally.

**2\.** Robotics VC hit $40.7B in 2025, up 74% year-over-year, representing 9% of all venture funding worldwide. Morgan Stanley projects a $5T humanoid TAM by 2050\.

**3\.** Every major lab has shipped or is shipping world model APIs: NVIDIA Cosmos (NIM), Runway GWM-1 (Python/Node SDK), Meta V-JEPA 2 (open source), World Labs Marble (commercial), Google Genie 3 (research preview), Tencent Hunyuan (open source).

**4\.** The EU AI Act enters full enforcement on August 2, 2026\. Every autonomous system deployed in Europe needs conformity assessment. WorldForge's guardrail and verification layer addresses this directly.

**5\.** No unified abstraction exists. A developer comparing Cosmos predictions against JEPA predictions must write completely separate integration code, manage different dependency trees, and manually normalize outputs. WorldForge solves this in 3 lines of code.

## **1.3 The Core Insight**

**The developer pain is real:** A robotics engineer at a startup wants to evaluate which world model works best for their task. Today, they must: (1) install NVIDIA's conda environment with 10+ CUDA dependencies, (2) separately install Runway's Python SDK with WebSocket management, (3) clone Meta's research repo and write custom inference scripts, (4) build their own normalization layer to compare outputs. This takes weeks. With WorldForge, it takes an afternoon.

# **2\. Market Opportunity**

## **2.1 The Physical AI Stack**

Physical AI is not one market. It's a stack of markets, each enormous:

| Segment | 2025 Funding | 2026 Est. | 2030 Proj. |
| :---- | :---- | :---- | :---- |
| World model labs | $6.9B (CB Insights) | $15B+ | $50B+ |
| Robotics (total VC) | $40.7B (74% YoY) | $60B+ | $150B+ |
| AV & autonomous systems | $25B+ | $35B+ | $100B+ |
| Industrial digital twins | $10B | $15B | $100B+ |
| Gaming / VR simulation | $3B | $8B | $276B (PitchBook) |
| **Developer tooling layer (5-10%)** | **$4-8B** | **$7-13B** | **$34-68B** |

*WorldForge targets the developer tooling layer: the infrastructure between foundation models and applications. Historically, this layer captures 5-10% of the total market value while requiring 100x less capital than the models themselves.*

## **2.2 Competitive Landscape**

**Direct competitors:** None. As of April 2026, no company is building a unified world model orchestration layer. The space is wide open.

Adjacent players who could expand into this space:

| Player | What they do | Why they won't build WorldForge |
| :---- | :---- | :---- |
| **NVIDIA** | Cosmos models \+ Omniverse simulation \+ Isaac robotics | Provider, not neutral orchestrator. Building WorldForge would compete with their own NIM API. |
| **Runway** | GWM-1 SDK (Worlds, Robotics, Avatars) | Provider SDK, not cross-provider. Would disadvantage competitors. |
| **Hugging Face** | Model hub \+ Transformers library \+ LeRobot | Model hosting, not orchestration. Could host WorldForge models. Partner, not competitor. |
| **LangChain** | LLM orchestration (chains, agents, memory) | LLM-focused DNA. World models require physics, state, spatial reasoning. Different abstraction. |

## **2.3 Recent Proof Points**

Funding rounds in the last 90 days that validate the world model thesis:

| Company | Round | Valuation | Signal |
| :---- | :---- | :---- | :---- |
| **AMI Labs (LeCun)** | $1.03B seed | $3.5B | Largest European seed ever. JEPA world models. Open source. |
| **Ineffable Intelligence** | $1B seed | $4B | RL \+ world models. Largest European seed. |
| **Rhoda AI** | $450M | \~$2B | Video-trained robot world models. |
| **Mind Robotics** | $500M Series A | \~$2B | Rivian spinout. Factory data → robot intelligence. |
| **World Labs (Fei-Fei Li)** | $1B | \~$5B | 3D world generation. Marble product shipping. |

**$4B+ deployed into world model companies in Q1 2026 alone. Every one of these companies is a potential WorldForge customer or design partner.**

# **3\. Product & Monetization**

## **3.1 Product Architecture**

Three layers, each building on the last:

### **Layer 1: Open Source Core (worldforge-core)**

Apache 2.0 licensed Rust library with Python bindings. Unified API across all world model providers. Core abstractions: Worlds (persistent state), Actions (standardized commands), Predictions (normalized outputs), Plans (action sequences toward goals), Guardrails (safety constraints). This is the adoption engine. Every import statement is a brand impression. Every tutorial is marketing.

### **Layer 2: WorldForge Cloud (SaaS)**

Managed infrastructure on top of the open source core. Smart routing (auto-select best provider per request based on task type, latency, cost). Response caching (avoid re-inference for identical requests — saves 30-60% of compute costs). Dashboard (per-provider latency, cost, quality metrics). Hosted world state persistence. API gateway with auth, rate limiting, and usage tracking.

### **Layer 3: WorldForge Enterprise**

Self-hosted deployment for companies that can't send data to third parties. ZK verification module (cryptographic proof that a specific model ran on specific inputs — unique moat). SLA guarantees. Custom provider adapters. EU AI Act conformity assessment integration. SOC 2 compliance.

## **3.2 Monetization Path**

| Tier | Price | Target | What they get |
| :---- | :---- | :---- | :---- |
| **Open Source** | Free | All developers | Core library, all providers, evaluation framework, CLI |
| **Cloud Free** | $0 | Individual devs, students | 100 predictions/month, basic dashboard |
| **Cloud Pro** | $49/mo | Indie devs, small teams | 10K predictions, caching, smart routing, full dashboard |
| **Cloud Team** | $199/mo | Startups, labs | 100K predictions, team management, priority support |
| **Enterprise** | **$5K+/mo** | Robotics companies, AV, industrial | Self-hosted, ZK verification, SLA, custom adapters, EU AI Act compliance |

## **3.3 Revenue Projections**

| Timeline | MRR | ARR | Customers | Milestone |
| :---- | :---- | :---- | :---- | :---- |
| Month 1-6 | $0 | $0 | 0 | Building open source. Consulting revenue ($10-20K/mo) funds operations. |
| Month 7-9 | $5K | $60K | 20 | Cloud launch. First Pro subscribers. First design partners. |
| Month 10-12 | $25K | $300K | 100 | Pre-seed based on traction. 2,000+ stars. |
| Month 13-18 | $75-150K | $0.9-1.8M | 300-600 | Seed round. First enterprise customers. ZK module. |
| Month 19-24 | $300K+ | $3.6M+ | 1,000+ | Series A territory. Industry standard for evaluation. |

## **3.4 Unit Economics**

Cloud tier costs are dominated by inference routing (proxying requests to providers). WorldForge does not run the models — it orchestrates calls to providers, caches results, and adds the abstraction layer. This means:

**1\.** Gross margins of 70-85% on Cloud tier (we're selling software, not compute)

**2\.** Gross margins of 85-95% on Enterprise tier (self-hosted, we provide the software license)

**3\.** CAC is near-zero for Cloud: developers discover WorldForge through open source, tutorials, and word-of-mouth

**4\.** Expansion revenue from Pro → Team → Enterprise as customers scale

# **4\. Moats & Defensibility**

## **4.1 Three Moats**

### **Moat 1: Network Effects (Data)**

Every prediction routed through WorldForge Cloud generates benchmarking data: which provider is fastest, most accurate, cheapest for each task type. This data gets better with scale. New users benefit from the benchmarks generated by existing users. Provider comparisons become more accurate. Smart routing improves. This is the Waze effect: the product gets better for everyone as each user contributes data.

### **Moat 2: ZK Verification (Technical)**

WorldForge is the only toolkit with native cryptographic verification for world model inference. Using STARKs, we can generate a proof that a specific model executed correctly on specific inputs. This is non-trivial to replicate — it requires deep expertise in both ZK proof systems and ML model architectures. For safety-critical deployments (medical robots, autonomous vehicles, industrial automation), verification is not optional. It's the law under the EU AI Act.

This is Abdel Bakhta's unique technical contribution. Co-authored EIP-1559 (cryptographic protocol design), built VeriFlow (STARK-based LLM verification), years of ZK proof expertise at StarkWare. No competing toolkit has this capability.

### **Moat 3: Community (Ecosystem)**

Open source creates a community moat. Once developers build on WorldForge, they don't switch. Their tutorials reference WorldForge. Their libraries depend on WorldForge. Their deployment pipelines use WorldForge. Switching costs compound over time. LangChain demonstrated this: despite widespread criticism of its API design, developers stayed because the ecosystem was built around it.

## **4.2 Why Not a Feature of Something Else?**

The most common objection: "Won't NVIDIA/Runway/Hugging Face just add this?"

**1\.** Providers won't build neutral orchestration. NVIDIA won't build a tool that makes it easy to switch from Cosmos to Runway. Runway won't build a tool that benchmarks their model against competitors.

**2\.** Hugging Face is a model hub, not an orchestration layer. They host models. WorldForge composes workflows across models. These are complementary, not competing.

**3\.** LangChain won't expand into world models because the abstraction is fundamentally different. LLMs are text-in, text-out. World models require state management, physics constraints, spatial reasoning, action spaces, and real-time control. Different primitives, different architecture.

# **5\. Founder-Market Fit**

Why is Abdel Bakhta the right person to build this?

## **5.1 The Ecosystem Pattern**

Abdel has done the exact thing WorldForge requires, as in growing ecosystem of open source developers and rally them around open source projects — mutiple times, few examples:

**1\.** Kakarot (1,004 stars, 85 contributors, spun into its own company): pitched the idea, wrote the spec, recruited first engineers, grew the community from zero. An EVM interpreter that unified Ethereum developers with Starknet.

**2\.** Madara (649 stars, \#1 contributor, 219 commits): same pattern. First commit to independent alliance. Grew from nothing.

**3\.** Starknet developer ecosystem: grew the entire developer community around a novel technology (ZK proofs). Designed developer programs, ran hackathons, managed partnerships.

WorldForge is the same pattern: take a novel technology (world models), build the developer layer, grow the community.


## **5.3 The Production Background**

A decade in payments and banking infrastructure. Mission-critical systems where bugs mean lost money. This matters because WorldForge's enterprise customers are deploying world models in safety-critical contexts: autonomous vehicles, medical robots, industrial automation. They need a toolkit built by someone who understands what "production-grade" means.


More on Abdel:

**If you want to change the world, don't protest. Write code.**

I've been writing production software for 15+ years. Payments, banking, blockchain, cryptography. Industries where security is not an afterthought, mission critical software at scale.  Built and shipped production systems that are used on a daily basis by millions of users, handling sensitive data and high stakes.

I care deeply about freedom tech, and I think it's the only way moving forward, especially as we advance more in the agentic era.

Head of Ecosystem at [StarkWare](https://starkware.co). Former Ethereum Core Developer, Co-author & Technical Champion of [EIP-1559](https://eips.ethereum.org/EIPS/eip-1559). Started [Kakarot](https://github.com/kkrt-labs/kakarot) and [Madara](https://github.com/keep-starknet-strange/madara) from nothing, both now run by independent teams. Bootstrapped and spearheaded the development of dozens of open source community projects.
Practically grew Starknet ecosystem to a vibrant ecosystem of developers over the past years.

Lately most of my time goes to one question: how do you verify what an autonomous system actually did? Yes, I am passionate and obsessed by the topic of AI Safety, on multiple dimensions: technological, philosophical, societal, ethical. 

---

### AI safety

Deployment is moving faster than the verification layer. That's the gap I work on: cryptographic proofs, governance tooling, and agent harness engineering for auditable autonomous systems.

| Project | What it does |
|---------|-------------|
| [claude-md-compiler](https://github.com/AbdelStark/claude-md-compiler) | Compiles CLAUDE.md into a versioned policy lockfile. Enforces it against diffs, hooks, and CI. No LLM in the runtime path. |
| [awesome-ai-safety](https://github.com/AbdelStark/awesome-ai-safety) | Curated list of tools and resources for AI safety. Alignment, interpretability, red teaming, formal verification, ZKML, governance. Things you can actually use, not just papers. |
| [eu-ai-act-toolkit](https://github.com/AbdelStark/eu-ai-act-toolkit) | Open source toolkit for EU AI Act compliance. SDK, CLI, and web app for classifying AI systems and generating compliance docs. |
| [llm-provable-computer](https://github.com/AbdelStark/llm-provable-computer) | Can you prove an LLM produced a specific output without running it again? Exploring verifiable inference with STARKs. |

Writing on this:

| | |
|---|---|
| [Math Is Humanity's Last Bastion Against Skynet](https://hackmd.io/@AbdelStark/math-humanity-last-bastion-skynet) | Why ZK proofs are the foundation for AI safety at scale |
| [Can LLMs Be Provable Computers?](https://hackmd.io/@AbdelStark/llm-provable-computers) | Verifiable AI inference via STARKs |

---

### Machine learning

I implement frontier ML papers in Rust. If you can build it from scratch, you understand it.

| Project | What it does |
|---------|-------------|
| [latent-inspector](https://github.com/AbdelStark/latent-inspector) | Numbers, not vibes. Compare DINOv2, I-JEPA, V-JEPA 2, EUPE representation geometry on the same image — CKA, k-NN overlap, PCA projections, intrinsic dimensionality. In Rust, via ONNX. |
| [jepa-rs](https://github.com/AbdelStark/jepa-rs) | First Rust implementation of JEPA primitives (I-JEPA, V-JEPA, C-JEPA, VICReg, EMA) |
| [gpc_rs](https://github.com/AbdelStark/gpc_rs) | Generative robot policies. Diffusion policy + world model + evaluator, in Rust. |
| [mosaicmem](https://github.com/AbdelStark/mosaicmem) | Geometry-aware spatial memory for camera-controlled video generation |
| [attnres](https://github.com/AbdelStark/attnres) | Attention Residuals (Kimi/MoonshotAI). Softmax attention over all preceding layer outputs. |
| [turboquant](https://github.com/AbdelStark/turboquant) | Google's TurboQuant. Vector quantization of LLM KV caches, in Rust. |
| [jepa-notebooks](https://github.com/AbdelStark/jepa-notebooks) | Interactive notebooks exploring JEPA architectures |

---

### AI Tooling / Apps / Products

| Project | What it does |
|---------|-------------|
| [🇫🇷 parler](https://github.com/AbdelStark/parler) | Multilingual voice intelligence built on Mistral Voxtral model — decision logs from French/English meetings |

---

### Ethereum

Ethereum Core Dev for 4 years. Doing Protocol Engineering and been working on strategic Ethereum protocol upgrades.
Technical Champion & Co-author of [EIP-1559](https://eips.ethereum.org/EIPS/eip-1559), the fee market reform. Shipped to mainnet August 2021.

---

### Applied cryptography — Hellhound (2018)

Back in 2018, I co-founded [Hellhound](https://github.com/Consensys/hellhound) inside ConsenSys R&D. A decentralized blind computation platform: run programs over homomorphically encrypted inputs on a network of nodes, without anyone (including the network operator) ever seeing the data. Privacy by design, end to end.

I was solo on the engineering side. I designed and built the HHVM — a register-based bytecode virtual machine, written from scratch — the Paillier homomorphic encryption pipeline, the Kubernetes/GKE infrastructure, the Ethereum smart contracts for on-chain computation proofs, and the consensus logic for detecting malicious nodes. I was also first author of the [Red Paper](https://github.com/ConsenSys/hellhound/blob/master/hellhound-red-paper.pdf), the formal HHVM specification, and contributed to the vision and strategy alongside my co-founders. Shipped to a [live demo at DevCon4 Prague (2018)](https://youtu.be/mztQHrRXEXs).

We also ran the first Crypto Escape Room at a DevCon. Every system component became a character in a lore universe so non-crypto attendees could learn applied cryptography by playing through it.

---

### Starknet / ZK proofs

Built the open source ecosystem around Starknet. Started Kakarot and Madara, both now have their own teams and communities.

| Project | What it does |
|---------|-------------|
| [Kakarot](https://github.com/kkrt-labs/kakarot) | EVM interpreter written in Cairo. Ethereum compatibility on Starknet via ZK proofs. |
| [Madara](https://github.com/keep-starknet-strange/madara) | Starknet sequencer for sovereign appchains |
| [Raito](https://github.com/starkware-bitcoin/raito) | Bitcoin ZK client in Cairo. Verifies Bitcoin consensus inside a STARK proof. |
| [Askeladd](https://github.com/starkware-bitcoin/askeladd) | Verifiable computation for Nostr Data Vending Machines via STARKs |
| [Cashu ZK Engine](https://github.com/AbdelStark/cashu-zk-engine) | Blind Diffie-Hellmann Key Exchange in Cairo for Cashu ecash |

---

### Bitcoin / freedom tech

| Project | What it does |
|---------|-------------|
| [bitcoin-mcp](https://github.com/AbdelStark/bitcoin-mcp) | Bitcoin & Lightning Network MCP server |
| [nostringer-rs](https://github.com/AbdelStark/nostringer-rs) | Ring signatures (SAG, BLSAG) for Nostr, in Rust |
| [nostr-mcp](https://github.com/AbdelStark/nostr-mcp) | Nostr MCP server |
| [bitcoin-honeybadger](https://github.com/AbdelStark/bitcoin-honeybadger) | Bitcoin Honeybadger |

---

### Writing

| | |
|---|---|
| [Before Fighting Banks, Let's Understand How They Actually Work](https://hackmd.io/@AbdelStark/BeforeFightingBanks) | A cypherpunk's guide to the financial system |
| [Time to Take the Nostr Pill](https://hackmd.io/@AbdelStark/time-to-take-the-nostr-pill) | Why Nostr matters for freedom of speech |
| [Nostr DVMs Meet Verifiable Computation](https://hackmd.io/@AbdelStark/nostr-dvm-verifiable-computation) | STARKs powering trustless Nostr services |
| [Cashu Meets STARKs](https://hackmd.io/@AbdelStark/cashu-starks) | Zero-knowledge proofs for the Cashu protocol |

---

If there's a thread, it's this: the gap between what individuals can verify and what institutions can hide should be closed by math, not by trusting people to behave. Bitcoin showed money doesn't need banks. Ethereum showed computation doesn't need servers. ZK proofs are how we'll know an AI actually did what it claimed.

Billions of autonomous agents are coming. Hospitals, roads, financial systems. The trust model we have assumes a small number of known actors. That assumption is about to break.

Becoming a dad changed my view of the world, my perspectives, my interests. The question shifted from "what's interesting" to "what world will they grow up in." Clarifying and terrifying in equal measure.

There's a video of a Unitree humanoid robot running and playing with kids in New York. The kids don't hesitate, no learned suspicion, no fear of something different. They just play. Kids apply "no enemies" (Thors mantra, from Vinland Saga) by default. And it's not because they're naive, rather because they haven't been taught yet to draw the lines we draw as "adults".

That's the world worth working toward. Not "no AI", but AI where integrity can be made by design, power that's distributed, trust that's verifiable. A world where kids don't have to unlearn anything to feel safe in it.

Math scales. Goodwill doesn't.

---

<p align="center">
  <a href="https://x.com/AbdelStark">X</a> · 
  <a href="https://primal.net/abdel">Nostr</a> · 
  <a href="https://hackmd.io/@AbdelStark">Writing</a>
</p>
