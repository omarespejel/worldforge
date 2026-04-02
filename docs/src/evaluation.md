# Evaluation

WorldForge includes a comprehensive evaluation framework with 12 scoring
dimensions for assessing world foundation model predictions.

## Running Evaluations

### CLI

```bash
# Run the physics suite against multiple providers
worldforge eval --suite physics --providers cosmos,runway,jepa \
  --output-markdown report.md --output-csv report.csv

# Run all suites
worldforge eval --suite comprehensive --providers cosmos \
  --output-json report.json
```

### Python

```python
from worldforge import WorldForge

wf = WorldForge()
world = wf.create_world("eval-scene", provider="cosmos")
report = world.evaluate(suite="physics")

print(report.to_markdown())
report.to_csv("results.csv")
```

### REST API

```bash
curl -X POST http://localhost:8080/v1/evals/run \
  -H "Content-Type: application/json" \
  -d '{
    "suite": "physics",
    "providers": ["cosmos", "runway"]
  }'
```

## Evaluation Dimensions

| Dimension                  | Source     | Method                          |
|----------------------------|------------|---------------------------------|
| Object Permanence          | WorldForge | Occlusion tracking              |
| Gravity Compliance         | WorldForge | Unsupported object fall test    |
| Collision Accuracy         | WorldForge | Contact physics validation      |
| Spatial Consistency        | WorldForge | Viewpoint stability             |
| Temporal Consistency       | WorldForge | Time-reversal stability         |
| Action Prediction          | WorldForge | Physics outcome matching        |
| Material Understanding     | WorldForge | Material-specific behavior      |
| Spatial Reasoning          | WorldForge | Depth/scale/distance            |
| Action Simulation Fidelity | WR-Arena   | LLM-as-judge (0-3 scale)       |
| Transition Smoothness      | WR-Arena   | MRS metric (optical flow)       |
| Generation Consistency     | WR-Arena   | WorldScore (7 aspects)          |
| Simulative Reasoning       | WR-Arena   | VLM + WFM planning loop        |

## Built-in Suites

- **physics**: Object permanence, gravity, collisions, material understanding.
- **manipulation**: Action prediction, spatial reasoning, collision accuracy.
- **spatial**: Spatial consistency, spatial reasoning, temporal consistency.
- **comprehensive**: All 12 dimensions.

## Output Formats

Reports can be generated in three formats:

- **JSON**: Machine-readable, includes all raw scores and metadata.
- **Markdown**: Human-readable tables suitable for documentation.
- **CSV**: Spreadsheet-compatible for further analysis.

## Custom Evaluations

You can define custom eval suites by combining dimensions:

```python
from worldforge import EvalSuite, EvalDimension

suite = EvalSuite(
    name="my-suite",
    dimensions=[
        EvalDimension.OBJECT_PERMANENCE,
        EvalDimension.GRAVITY_COMPLIANCE,
        EvalDimension.COLLISION_ACCURACY,
    ],
)
report = world.evaluate(suite=suite)
```
