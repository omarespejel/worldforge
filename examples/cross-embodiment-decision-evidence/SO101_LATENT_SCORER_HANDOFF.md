# SO-101 Latent Scorer Handoff

This note records the host-owned learned-scorer lane for independent referee review. It is not a
claim that the learned scorer adds decision value.

## Boundary

WorldForge owns scorer artifacts and provider wrapping. The external referee owns the held-out
evaluation protocol and the verdict.

Do not claim learned-scorer value unless the independent referee shows all of:

- above-chance performance,
- improvement over proprio or hand-cost baselines,
- no shuffled-label leak,
- useful rank or progress correlation on held-out task/sub-goal items.

## Inputs

Expected local inputs:

- SO-101 LeRobot-style dataset: `data/svla_so101_pickplace`
- DINOv2 side-camera latent cache: `latent_cache.npz`
- External referee module: `referee_headline.py`

The cache contract for this run was:

- encoder: `facebook/dinov2-small`
- token: `CLS`
- normalization: `L2`
- latent shape: `11939 x 384`
- camera: side view

## Reproduce The Handoff

Build or refresh the cache sidecar without re-encoding frames:

```bash
uv run --with numpy --with pyarrow python scripts/build_so101_latent_cache_sidecar.py \
  --cache /path/to/latent_cache.npz \
  --dataset-dir /path/to/svla_so101_pickplace \
  --out .worldforge/so101-latent-cache/latent_cache_meta.json
```

Train scorer artifacts from the existing cache:

```bash
uv run --with numpy --with pyarrow python scripts/train_so101_latent_scorer.py \
  --sidecar .worldforge/so101-latent-cache/latent_cache_meta.json \
  --out-dir .worldforge/so101-goal-history-score-scorer \
  --epochs 120 \
  --batch-size 256 \
  --hidden-dim 128 \
  --horizons 1 \
  --variants goal_score_mlp,goal_proprio_score_mlp,goal_history_score_mlp,goal_history_proprio_score_mlp
```

Run the scorer artifacts through the external referee:

```bash
uv run --with numpy --with pyarrow python scripts/run_so101_referee_smoke.py \
  --dataset-dir /path/to/svla_so101_pickplace \
  --cache /path/to/latent_cache.npz \
  --referee /path/to/referee_headline.py \
  --weights .worldforge/so101-goal-history-score-scorer/so101_latent_scorers.npz \
  --metadata .worldforge/so101-goal-history-score-scorer/so101_latent_scorer_meta.json \
  --out .worldforge/so101-goal-history-score-scorer/claude-referee-goal-history-score-smoke-result.json
```

Success signal: the command prints held-out item count and one row per scorer, then writes the JSON
result under `.worldforge/so101-goal-history-score-scorer/`.

First triage step on failure: check that the cache, sidecar, dataset, and external referee paths
exist and agree on frame ordering.

## Smoke Result

The latest local smoke handoff used `3981` held-out sub-goal decision items across `15` test
episodes.

| Scorer | Top-1 vs demonstrated | Chance | MRR | Rank corr vs proprio truth | Shuffled top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `proprio_baseline` | `0.0570` | `0.1429` | `0.3696` | `1.0000` | `0.1314` |
| `goal_score_mlp_h1` | `0.0151` | `0.1429` | `0.3062` | `0.2812` | `0.1469` |
| `goal_proprio_score_mlp_h1` | `0.0226` | `0.1429` | `0.3227` | `0.3069` | `0.1397` |
| `goal_history_score_mlp_h1` | `0.0183` | `0.1429` | `0.3107` | `0.3638` | `0.1419` |
| `goal_history_proprio_score_mlp_h1` | `0.0219` | `0.1429` | `0.3047` | `0.3609` | `0.1507` |

Decision: this is negative evidence. The learned scorer does not beat the proprio baseline and is
below chance on top-1. Keep the DecisionTrace infrastructure claim, but do not claim learned
world-model planning value from this scorer. The history variants improve rank/progress
correlation over the one-step goal scorers, but still fail the top-1/chance/proprio gate.

Local artifact hashes from that smoke run:

- `so101_latent_scorers.npz`:
  `5876a96ad073c34663a40bcaa218a0d727bc0a5d4fa3a1fa1ff12bed6ecee772`
- `so101_latent_scorer_meta.json`:
  `f28cf2c043362159e79ab8a4d674b7903168adbaa18b8abc55d7c78275a24861`
- `claude-referee-goal-history-score-smoke-result.json`:
  `4612215e1848760e7d5da915d941ec6c5b4ffc54300365670e0b9869b50000c8`

## Next Iteration

The current MLP family is too weak for the task/sub-goal referee. Next scorer-side experiments
should change the model design, not the gate:

- train a real sequence or multi-step scorer instead of concatenating short history features,
- include both side and overhead camera latents when available,
- evaluate ranking/progress correlation before optimizing top-1,
- keep Claude's external referee as the independent judge.
