# RFC-0008: Evaluation Framework

- **Status:** Draft
- **Created:** 2026-04-02
- **Authors:** WorldForge Core Team
- **Requires:** RFC-0001 (Core Architecture), RFC-0003 (Provider System)

---

## Abstract

This RFC defines the comprehensive evaluation framework for WorldForge, enabling
systematic assessment of world model providers across 12 evaluation dimensions.
The framework provides built-in datasets, custom dataset support, WR-Arena
integration, scoring methodology, cross-provider comparison reports, leaderboard
generation, regression testing, and human evaluation protocol integration. It
connects to academic benchmarks including PhyWorld and WR-Arena to ensure
scientific rigor and reproducibility.

## Motivation

World models vary dramatically in their ability to simulate physical reality.
A provider that excels at predicting gravity may fail at object permanence. Users
need a principled, reproducible way to:

1. **Select the right provider** for their use case by understanding strengths
   and weaknesses across physics dimensions.
2. **Compare providers** on equal footing with standardized benchmarks.
3. **Detect regressions** when provider updates degrade prediction quality.
4. **Validate custom models** against established academic benchmarks.
5. **Generate evidence** for research papers and engineering decisions.

Without a unified evaluation framework, each team builds ad-hoc tests that are
incomparable across organizations. WorldForge's evaluation framework solves this
by providing a single, extensible, scientifically grounded evaluation system.

## Detailed Design

### 1. Evaluation Dimensions

The framework evaluates providers across 12 dimensions, grouped into three
categories: Physics Understanding, Consistency, and Reasoning.

#### 1.1 Physics Understanding (6 dimensions)

**Dimension 1: Object Permanence**
- Tests whether the model understands that objects continue to exist when
  occluded or out of frame.
- Scenarios: object moves behind barrier, camera pans away and returns,
  object enters container.
- Metrics: presence prediction accuracy, reappearance location error (pixels),
  temporal reappearance accuracy (frames).
- Scoring: weighted combination of binary presence (40%), location error (35%),
  timing error (25%).

**Dimension 2: Gravity**
- Tests understanding of gravitational effects on objects.
- Scenarios: object released from height, projectile motion, objects on
  inclined planes, pendulum motion, stacking stability.
- Metrics: trajectory prediction error (MSE), fall time accuracy, landing
  position error, acceleration consistency.
- Scoring: trajectory MSE normalized against ground truth (50%), physical
  plausibility score (30%), edge case handling (20%).

**Dimension 3: Collisions**
- Tests prediction of collision outcomes between objects.
- Scenarios: elastic collisions, inelastic collisions, multi-body collisions,
  object breakage, ricochet trajectories.
- Metrics: post-collision velocity prediction error, energy conservation
  score, momentum conservation score, deformation prediction accuracy.
- Scoring: velocity error (35%), conservation law adherence (35%),
  qualitative outcome (30%).

**Dimension 4: Material Understanding**
- Tests comprehension of material properties and their effects.
- Scenarios: rigid vs. deformable objects, liquid behavior, material-specific
  friction, transparency/reflectivity, material interactions (e.g., ice melting).
- Metrics: deformation prediction accuracy, material classification accuracy,
  property inference score.
- Scoring: physical property prediction (40%), behavioral prediction (40%),
  material interaction accuracy (20%).

**Dimension 5: Fluid Dynamics**
- Tests understanding of fluid behavior.
- Scenarios: pouring liquids, water flow, splashing, mixing, viscosity
  differences.
- Metrics: flow prediction accuracy, volume conservation, viscosity
  estimation, splash pattern plausibility.
- Scoring: qualitative flow accuracy (50%), volume conservation (30%),
  fine detail accuracy (20%).

**Dimension 6: Thermodynamics**
- Tests understanding of heat transfer and phase changes.
- Scenarios: heating/cooling objects, melting, freezing, steam generation,
  thermal equilibrium.
- Metrics: temperature change direction, phase transition prediction,
  equilibrium prediction accuracy.
- Scoring: directional accuracy (40%), transition timing (30%),
  qualitative behavior (30%).

#### 1.2 Consistency (3 dimensions)

**Dimension 7: Spatial Consistency**
- Tests whether predicted frames maintain consistent spatial relationships.
- Scenarios: multi-object scenes with fixed relative positions, camera
  movement with static objects, object size consistency across frames.
- Metrics: relative position drift, size consistency ratio, geometric
  distortion score, depth ordering accuracy.
- Scoring: position drift (35%), size consistency (30%), depth ordering (20%),
  geometric integrity (15%).

**Dimension 8: Temporal Consistency**
- Tests whether predictions maintain consistent motion and change over time.
- Scenarios: constant velocity motion, acceleration profiles, periodic
  motion, gradual color/lighting changes.
- Metrics: motion smoothness (jerk metric), velocity consistency, temporal
  aliasing score, flickering detection.
- Scoring: motion smoothness (40%), velocity consistency (30%),
  flickering penalty (20%), transition quality (10%).

**Dimension 9: Identity Consistency**
- Tests whether objects maintain their identity across predictions.
- Scenarios: multiple similar objects, objects after occlusion, objects
  after transformation (e.g., rotating), object tracking across frames.
- Metrics: identity swap rate, appearance consistency score, attribute
  preservation accuracy.
- Scoring: identity preservation (50%), appearance consistency (30%),
  attribute accuracy (20%).

#### 1.3 Reasoning (3 dimensions)

**Dimension 10: Action Prediction**
- Tests ability to predict outcomes of specified actions.
- Scenarios: push/pull actions, tool use, stacking/unstacking, pouring,
  throwing at targets.
- Metrics: action outcome accuracy, effect magnitude prediction, side
  effect detection, multi-step action chaining accuracy.
- Scoring: primary effect accuracy (40%), magnitude accuracy (30%),
  side effect detection (20%), chain accuracy (10%).

**Dimension 11: Spatial Reasoning**
- Tests higher-order spatial understanding.
- Scenarios: containment relationships, relative positioning queries,
  path planning feasibility, clearance estimation, reachability.
- Metrics: relationship classification accuracy, distance estimation
  error, feasibility prediction accuracy.
- Scoring: relationship accuracy (40%), distance error (30%),
  feasibility accuracy (30%).

**Dimension 12: Causal Reasoning**
- Tests understanding of cause-and-effect chains.
- Scenarios: Rube Goldberg machines, domino chains, chain reactions,
  intervention prediction (what happens if X is removed).
- Metrics: causal chain prediction accuracy, intervention outcome
  accuracy, counterfactual reasoning score.
- Scoring: chain accuracy (40%), intervention accuracy (35%),
  counterfactual score (25%).

### 2. EvalSuite Architecture

```
EvalSuite
├── EvalDimension (enum, 12 variants)
├── EvalDataset
│   ├── BuiltInDataset (curated scenarios per dimension)
│   ├── CustomDataset (user-provided scenarios)
│   └── WRArenaDataset (loaded from WR-Arena benchmark)
├── EvalRunner
│   ├── SingleProviderRunner
│   └── ComparisonRunner (multi-provider)
├── EvalScorer
│   ├── DimensionScorer (per-dimension scoring logic)
│   └── AggregateScorer (weighted combination)
├── EvalReporter
│   ├── MarkdownReport
│   ├── CsvReport
│   └── HtmlReport
└── EvalHistory
    ├── ResultStore (historical results)
    └── RegressionDetector
```

### 3. Running Evaluations Against Providers

#### 3.1 Basic Evaluation

```rust
use worldforge::eval::{EvalSuite, EvalConfig, EvalDimension};
use worldforge::providers::ProviderConfig;

let provider = ProviderConfig::new("genesis")
    .with_api_key(env::var("GENESIS_API_KEY")?)
    .build()?;

let suite = EvalSuite::builder()
    .dimensions(vec![
        EvalDimension::ObjectPermanence,
        EvalDimension::Gravity,
        EvalDimension::Collisions,
    ])
    .dataset(EvalDataset::BuiltIn)
    .samples_per_scenario(5)  // statistical robustness
    .timeout_per_scenario(Duration::from_secs(30))
    .build()?;

let results = suite.run(&provider).await?;
println!("Overall score: {:.2}", results.aggregate_score());
for (dim, score) in results.dimension_scores() {
    println!("  {}: {:.2} ({})", dim, score.value, score.grade());
}
```

#### 3.2 Cross-Provider Comparison

```rust
let providers = vec![
    ProviderConfig::new("genesis").build()?,
    ProviderConfig::new("cosmos").build()?,
    ProviderConfig::new("wan").build()?,
];

let comparison = EvalSuite::compare(&providers, &suite).await?;

// Generate all report formats
comparison.report_markdown("reports/comparison.md")?;
comparison.report_csv("reports/comparison.csv")?;
comparison.report_html("reports/comparison.html")?;
```

#### 3.3 CLI Interface

```bash
# Run full evaluation
worldforge eval --provider genesis --dimensions all

# Compare providers
worldforge eval --compare genesis,cosmos,wan --output report.html

# Run specific dimension
worldforge eval --provider genesis --dimensions gravity,collisions

# Run with custom dataset
worldforge eval --provider genesis --dataset ./my-scenarios/

# Run WR-Arena benchmark
worldforge eval --provider genesis --dataset wr-arena --split test
```

### 4. Dataset Management

#### 4.1 Built-In Datasets

Each dimension ships with a curated dataset of 50-200 scenarios:

```
datasets/
├── object_permanence/
│   ├── occlusion_simple.json       # 20 scenarios
│   ├── occlusion_complex.json      # 30 scenarios
│   ├── container_insertion.json    # 25 scenarios
│   └── camera_pan.json             # 25 scenarios
├── gravity/
│   ├── free_fall.json              # 30 scenarios
│   ├── projectile.json             # 40 scenarios
│   ├── inclined_plane.json         # 25 scenarios
│   └── pendulum.json               # 20 scenarios
├── collisions/
│   ├── elastic_2body.json          # 35 scenarios
│   ├── inelastic.json              # 30 scenarios
│   └── multi_body.json             # 25 scenarios
...
```

Each scenario is defined as:

```json
{
  "id": "gravity_freefall_001",
  "dimension": "gravity",
  "description": "Ball released from 2m height above ground",
  "initial_state": {
    "image": "gravity/freefall_001_init.png",
    "objects": [
      {"id": "ball", "position": [320, 100], "properties": {"mass": 0.5}}
    ],
    "environment": {"gravity": 9.81}
  },
  "action": "release ball",
  "expected_outcome": {
    "frames": ["gravity/freefall_001_f01.png", "..."],
    "trajectory": [[320, 100], [320, 115], [320, 140], "..."],
    "final_state": {"ball_position": [320, 450]},
    "tolerances": {"position_px": 15, "timing_frames": 2}
  },
  "difficulty": "easy",
  "tags": ["basic_physics", "no_air_resistance"]
}
```

#### 4.2 Custom Datasets

Users can create custom evaluation datasets:

```rust
let custom_dataset = EvalDataset::custom()
    .add_scenario(Scenario {
        id: "my_test_001".into(),
        dimension: EvalDimension::Gravity,
        initial_image: load_image("test_init.png")?,
        action: "drop the red ball".into(),
        expected: ExpectedOutcome::trajectory(vec![...]),
        tolerance: Tolerance::default(),
    })
    .build()?;
```

#### 4.3 WR-Arena Integration

WorldForge integrates directly with the WR-Arena benchmark:

```rust
use worldforge::eval::wr_arena::{WRArenaLoader, WRArenaScorer};

// Load WR-Arena dataset (downloads if not cached)
let wr_dataset = WRArenaLoader::new()
    .split(Split::Test)
    .categories(vec!["physics", "spatial"])
    .cache_dir("~/.worldforge/wr-arena")
    .load()
    .await?;

// Run evaluation with WR-Arena scoring
let results = suite.run_with_scorer(&provider, &wr_dataset, WRArenaScorer::new()).await?;

// Results are compatible with WR-Arena leaderboard format
results.export_wr_arena_format("wr_arena_submission.json")?;
```

The WR-Arena integration supports:
- Automatic dataset download and caching
- Official scoring methodology
- Submission-ready output format
- Category-level and aggregate scores
- ELO-based rating computation

### 5. Scoring Methodology

#### 5.1 Per-Dimension Scoring

Each dimension produces a score in [0.0, 1.0]:

```rust
pub struct DimensionScore {
    pub dimension: EvalDimension,
    pub value: f64,           // 0.0 - 1.0
    pub confidence: f64,      // statistical confidence
    pub num_scenarios: usize,
    pub subscores: HashMap<String, f64>,  // e.g., "trajectory_mse": 0.85
    pub grade: Grade,         // A/B/C/D/F
}

pub enum Grade {
    A,  // 0.9 - 1.0: Excellent
    B,  // 0.75 - 0.9: Good
    C,  // 0.6 - 0.75: Acceptable
    D,  // 0.4 - 0.6: Poor
    F,  // 0.0 - 0.4: Failing
}
```

#### 5.2 Scoring Functions

Each dimension has specialized scoring functions:

**Trajectory-based scoring** (gravity, collisions, action prediction):
```
score = 1.0 - clamp(MSE(predicted, actual) / normalization_factor, 0, 1)
```

**Binary classification scoring** (object permanence, identity consistency):
```
score = (TP + TN) / (TP + TN + FP + FN)
```

**Perceptual similarity scoring** (spatial/temporal consistency):
```
score = mean(LPIPS(frame_i_predicted, frame_i_actual)) for all frames
```

**Structural scoring** (causal reasoning, spatial reasoning):
```
score = graph_edit_distance(predicted_graph, actual_graph) / max_distance
```

#### 5.3 Aggregate Scoring

The aggregate score is a weighted combination with configurable weights:

```rust
pub struct ScoringWeights {
    pub object_permanence: f64,   // default: 1.0
    pub gravity: f64,             // default: 1.0
    pub collisions: f64,          // default: 1.0
    pub material_understanding: f64, // default: 1.0
    pub fluid_dynamics: f64,      // default: 0.8
    pub thermodynamics: f64,      // default: 0.8
    pub spatial_consistency: f64, // default: 1.2
    pub temporal_consistency: f64,// default: 1.2
    pub identity_consistency: f64,// default: 1.0
    pub action_prediction: f64,   // default: 1.5
    pub spatial_reasoning: f64,   // default: 1.0
    pub causal_reasoning: f64,    // default: 1.0
}
```

The aggregate score is: `sum(weight_i * score_i) / sum(weight_i)`.

Users can customize weights for domain-specific evaluation:

```rust
let weights = ScoringWeights::default()
    .with_weight(EvalDimension::Gravity, 3.0)       // robotics focus
    .with_weight(EvalDimension::ActionPrediction, 3.0)
    .with_weight(EvalDimension::FluidDynamics, 0.0); // exclude
```

### 6. Cross-Provider Comparison Reports

#### 6.1 Markdown Report

```markdown
# WorldForge Provider Comparison Report
Generated: 2026-04-02 19:00:00 UTC

## Summary
| Provider | Overall | Grade | Best Dimension | Worst Dimension |
|----------|---------|-------|----------------|-----------------|
| Genesis  | 0.847   | B     | Gravity (0.95) | Causal (0.62)   |
| Cosmos   | 0.791   | B     | Spatial (0.91) | Fluids (0.55)   |
| WAN      | 0.723   | C     | Temporal (0.88)| Material (0.48) |

## Per-Dimension Breakdown
| Dimension           | Genesis | Cosmos | WAN   | Best    |
|---------------------|---------|--------|-------|---------|
| Object Permanence   | 0.89    | 0.85   | 0.78  | Genesis |
| Gravity             | 0.95    | 0.82   | 0.74  | Genesis |
...
```

#### 6.2 CSV Report

Machine-readable format for integration with data pipelines:

```csv
provider,dimension,score,confidence,grade,num_scenarios,timestamp
genesis,object_permanence,0.89,0.95,B,100,2026-04-02T19:00:00Z
genesis,gravity,0.95,0.97,A,115,2026-04-02T19:00:00Z
cosmos,object_permanence,0.85,0.94,B,100,2026-04-02T19:00:00Z
...
```

#### 6.3 HTML Report

Interactive HTML report with:
- Radar charts showing per-dimension scores for each provider
- Bar chart comparisons across providers
- Drill-down into individual scenario results
- Score distribution histograms
- Statistical significance indicators
- Exportable charts (SVG/PNG)

Generated using embedded templates (no external dependencies):

```rust
let html = HtmlReporter::new()
    .title("Q1 2026 Provider Evaluation")
    .theme(Theme::Light)
    .include_charts(true)
    .include_scenario_details(true)
    .render(&comparison_results)?;

std::fs::write("report.html", html)?;
```

### 7. Leaderboard Generation

#### 7.1 Local Leaderboard

```rust
let leaderboard = Leaderboard::new()
    .add_results("genesis", &genesis_results)
    .add_results("cosmos", &cosmos_results)
    .add_results("wan", &wan_results)
    .sort_by(SortCriterion::AggregateScore)
    .build();

leaderboard.print();
// Output:
// Rank | Provider | Score | Grade | Δ vs Previous
// 1    | Genesis  | 0.847 | B     | +0.023
// 2    | Cosmos   | 0.791 | B     | -0.005
// 3    | WAN      | 0.723 | C     | new
```

#### 7.2 ELO Rating System

For head-to-head comparisons, the framework computes ELO ratings:

```rust
let elo = EloCalculator::new()
    .k_factor(32.0)
    .initial_rating(1500.0);

// Compute pairwise ELO from comparison results
let ratings = elo.compute_from_comparisons(&all_results);
for (provider, rating) in ratings.ranked() {
    println!("{}: ELO {:.0}", provider, rating);
}
```

#### 7.3 Public Leaderboard Export

Generate leaderboard data compatible with community leaderboards:

```rust
leaderboard.export_json("leaderboard.json")?;
leaderboard.export_huggingface_format("hf_leaderboard.json")?;
```

### 8. Regression Testing

#### 8.1 Result Storage

All evaluation results are stored for historical comparison:

```rust
let store = ResultStore::new("~/.worldforge/eval_history")?;
store.save(&results)?;  // timestamped, provider-tagged

// Query historical results
let history = store.query()
    .provider("genesis")
    .dimension(EvalDimension::Gravity)
    .since(Utc::now() - Duration::days(30))
    .fetch()?;
```

#### 8.2 Regression Detection

```rust
let detector = RegressionDetector::new()
    .threshold(0.05)          // 5% score drop triggers alert
    .min_samples(3)           // need 3+ historical results
    .significance_level(0.05) // p < 0.05 for statistical test
    .build();

let regressions = detector.check(&current_results, &store)?;
for reg in &regressions {
    eprintln!(
        "REGRESSION: {} on {} dropped from {:.2} to {:.2} (p={:.4})",
        reg.provider, reg.dimension, reg.previous_mean, reg.current_score, reg.p_value
    );
}
```

#### 8.3 CI/CD Integration

```yaml
# .github/workflows/eval.yml
- name: Run WorldForge Evaluation
  run: worldforge eval --provider ${{ matrix.provider }} --output results.json

- name: Check for Regressions
  run: worldforge eval regression-check --input results.json --threshold 0.05
  # Exit code 1 if regression detected
```

The regression checker uses a two-sample t-test to determine statistical
significance, avoiding false positives from normal variance.

### 9. Human Evaluation Protocol

#### 9.1 Protocol Design

For dimensions that benefit from human judgment (especially visual quality
and physical plausibility), the framework includes a human evaluation protocol:

```rust
let human_eval = HumanEvalSession::new()
    .dimension(EvalDimension::SpatialConsistency)
    .num_scenarios(50)
    .evaluators_per_scenario(3)  // inter-rater reliability
    .rating_scale(RatingScale::Likert5)
    .build()?;

// Generate evaluation interface
human_eval.export_interface("human_eval_session/")?;
// Creates a local web interface for human evaluators
```

#### 9.2 Rating Dimensions for Human Evaluation

Each human evaluator rates predictions on:
- **Physical Plausibility** (1-5): Does the prediction look physically correct?
- **Visual Quality** (1-5): Is the prediction visually coherent?
- **Action Accuracy** (1-5): Does the prediction match the described action?
- **Overall Preference** (A/B): In pairwise comparison, which prediction is better?

#### 9.3 Inter-Rater Reliability

The framework computes Krippendorff's alpha to ensure evaluator agreement:

```rust
let reliability = human_eval.compute_reliability(&ratings)?;
assert!(reliability.alpha > 0.67, "Insufficient inter-rater reliability");
```

#### 9.4 Combining Human and Automated Scores

```rust
let combined = CombinedScorer::new()
    .automated_weight(0.7)
    .human_weight(0.3)
    .combine(&automated_results, &human_results)?;
```

### 10. Academic Benchmark Integration

#### 10.1 PhyWorld Integration

PhyWorld is an academic benchmark for physical world understanding:

```rust
use worldforge::eval::phyworld::{PhyWorldLoader, PhyWorldScorer};

let phyworld = PhyWorldLoader::new()
    .version("v2.0")
    .tasks(vec!["mechanics", "optics", "thermodynamics"])
    .load()
    .await?;

let results = suite.run_with_scorer(&provider, &phyworld, PhyWorldScorer::new()).await?;

// Compare against published results
let comparison = PhyWorldScorer::compare_with_published(&results)?;
println!("vs. GPT-4V: {}", comparison.relative_performance("gpt-4v"));
println!("vs. Human: {}", comparison.relative_performance("human"));
```

#### 10.2 WR-Arena Integration

WR-Arena provides ELO-based ranking of world models:

```rust
use worldforge::eval::wr_arena::{WRArena, ArenaMatch};

let arena = WRArena::new().await?;

// Submit provider for arena evaluation
let matches = arena.run_matches(&provider, num_matches: 100).await?;
let elo = arena.compute_elo(&matches)?;

println!("Provider ELO: {:.0}", elo.rating);
println!("95% CI: [{:.0}, {:.0}]", elo.ci_lower, elo.ci_upper);
println!("Rank: {}/{}", elo.rank, elo.total_models);
```

#### 10.3 Custom Benchmark Registration

Researchers can register custom benchmarks:

```rust
let benchmark = BenchmarkSpec::new("my_physics_bench")
    .description("Focus on rigid body dynamics")
    .dimensions(vec![EvalDimension::Gravity, EvalDimension::Collisions])
    .dataset(my_dataset)
    .scorer(my_custom_scorer)
    .register(&eval_registry)?;
```

### 11. Performance Requirements

| Operation                    | Target Latency | Notes                          |
|------------------------------|---------------|--------------------------------|
| Single scenario evaluation   | < 10s         | Including provider API call    |
| Full dimension (100 scenarios)| < 15min      | With parallelism               |
| Full suite (12 dimensions)   | < 2hr         | Sequential dimensions          |
| Report generation            | < 5s          | After results collected        |
| Regression check             | < 1s          | Against stored history         |
| Leaderboard generation       | < 2s          | From cached results            |

### 12. Configuration

```toml
# worldforge-eval.toml
[eval]
parallelism = 4
timeout_per_scenario_secs = 30
samples_per_scenario = 5
cache_predictions = true
cache_dir = "~/.worldforge/eval_cache"

[eval.weights]
object_permanence = 1.0
gravity = 1.0
collisions = 1.0
material_understanding = 1.0
fluid_dynamics = 0.8
thermodynamics = 0.8
spatial_consistency = 1.2
temporal_consistency = 1.2
identity_consistency = 1.0
action_prediction = 1.5
spatial_reasoning = 1.0
causal_reasoning = 1.0

[eval.regression]
threshold = 0.05
min_history_samples = 3
significance_level = 0.05

[eval.reporting]
formats = ["markdown", "csv", "html"]
output_dir = "./eval_reports"
include_charts = true
```

## Implementation Plan

### Phase 1: Core Framework (Weeks 1-3)
- Implement EvalSuite, EvalRunner, DimensionScorer
- Build built-in datasets for 4 core dimensions (object permanence, gravity,
  collisions, action prediction)
- Implement markdown report generation
- Basic CLI integration

### Phase 2: Full Coverage (Weeks 4-6)
- Complete datasets for all 12 dimensions
- Implement CSV and HTML report generation
- Add cross-provider comparison
- Implement leaderboard generation

### Phase 3: Advanced Features (Weeks 7-9)
- ResultStore and regression detection
- WR-Arena integration
- PhyWorld integration
- ELO rating system
- CI/CD integration

### Phase 4: Human Evaluation (Weeks 10-11)
- Human evaluation web interface
- Inter-rater reliability computation
- Combined scoring system
- Documentation and examples

### Phase 5: Polish (Week 12)
- Performance optimization (parallelism, caching)
- Comprehensive test suite
- Public leaderboard export formats
- Tutorial and user guide

## Testing Strategy

### Unit Tests
- Each scoring function tested with known inputs/outputs
- Dataset loading and validation
- Report generation format correctness
- Regression detection with synthetic history

### Integration Tests
- End-to-end evaluation with mock providers
- Cross-provider comparison with deterministic test providers
- WR-Arena format import/export round-trip
- CI/CD integration with GitHub Actions

### Property-Based Tests
- Score always in [0.0, 1.0]
- Aggregate score is monotonic in component scores
- Regression detector has controlled false positive rate
- Report formats are valid (HTML validates, CSV parses)

### Benchmark Tests
- Evaluation throughput (scenarios/second)
- Report generation time
- History query performance with large result stores

## Open Questions

1. **Dimension weights**: Should default weights be equal, or should we
   weight based on research consensus about which dimensions matter most?

2. **Ground truth generation**: For some dimensions (especially causal
   reasoning), generating ground truth is expensive. Should we invest in
   simulation-based ground truth (e.g., using physics engines)?

3. **Video vs. frame evaluation**: Should scoring operate on individual
   frames or on video-level features? Some dimensions (temporal consistency)
   inherently require video-level evaluation.

4. **Provider cost tracking**: Should the evaluation framework track and
   report the API cost of running evaluations? This would help users
   optimize cost-quality tradeoffs.

5. **Versioning of datasets**: How do we handle dataset updates? Changing
   the benchmark makes historical comparisons invalid. Should we version
   datasets independently?

6. **Multi-modal evaluation**: Should we support evaluating providers that
   take text-only input differently from those that accept image+text?

7. **Partial evaluation**: If a provider times out on some scenarios, how
   should we handle the incomplete data in scoring?

8. **Community contributions**: What is the process for community members
   to contribute evaluation scenarios to the built-in datasets?
