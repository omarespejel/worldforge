# SO-101 Latent Scorer Handoff

This note records the host-owned learned-scorer lane for independent referee review. It is not a
claim that the learned scorer adds decision value.

## Boundary

WorldForge owns scorer artifacts and provider wrapping. The external referee owns the held-out
evaluation protocol and the verdict.

Do not claim learned-scorer value unless the independent referee shows all of:

- improvement over proprio or hand-cost baselines on fair rank/progress metrics,
- no shuffled-label leak,
- useful held-out task/sub-goal signal under the external referee.

For the current near-duplicate decoy set, exact-match top-1 is diagnostic only. The primary
handoff gate is `fair_beat_rate_clearbad`: the mean demonstrated-action beat rate over
`overshoot`, `reverse`, `random_other`, and `jitter`. `rank_corr_vs_proprio_truth` is calibration
against proprio progress, not an independent success signal.

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

Train residual future-latent scorer artifacts from the existing cache:

```bash
uv run --with numpy --with pyarrow python scripts/train_so101_latent_scorer.py \
  --sidecar .worldforge/so101-latent-cache/latent_cache_meta.json \
  --out-dir .worldforge/so101-latent-scorer \
  --epochs 120 \
  --batch-size 256 \
  --hidden-dim 128 \
  --horizons 1,5,15 \
  --variants vision_mlp,vision_proprio_mlp
```

Run the scorer artifacts through the external referee:

```bash
uv run --with numpy --with pyarrow python scripts/run_so101_referee_smoke.py \
  --dataset-dir /path/to/svla_so101_pickplace \
  --cache /path/to/latent_cache.npz \
  --referee /path/to/referee_headline.py \
  --weights .worldforge/so101-latent-scorer/so101_latent_scorers.npz \
  --metadata .worldforge/so101-latent-scorer/so101_latent_scorer_meta.json \
  --out .worldforge/so101-latent-scorer/claude-referee-smoke-result.json
```

Success signal: the command prints held-out item count and one row per scorer, then writes the JSON
result under `.worldforge/so101-goal-history-score-scorer/`.

First triage step on failure: check that the cache, sidecar, dataset, and external referee paths
exist and agree on frame ordering.

## Smoke Result

The latest local smoke handoff used `3981` held-out sub-goal decision items across `15` test
episodes.

### Residual Future-Latent Family

This is the relevant family for testing whether vision adds decision value, because the provider
predicts a future visual latent and the referee scores distance to the sub-goal latent.

| Scorer | Fair clear-bad beat rate | Near-duplicate beat rate | Top-1 diagnostic | MRR | Calibration vs proprio progress |
| --- | ---: | ---: | ---: | ---: | ---: |
| `proprio_baseline` | `0.6174` | `0.5896` | `0.0570` | `0.3696` | `1.0000` |
| `vision_mlp_h1` | `0.6230` | `0.4479` | `0.0249` | `0.3261` | `0.1470` |
| `vision_mlp_h15` | `0.6124` | `0.5927` | `0.0306` | `0.3520` | `0.1892` |
| `vision_proprio_mlp_h15` | `0.6551` | `0.4585` | `0.0525` | `0.3537` | `0.2421` |
| `residual_ridge_baseline` | `0.6762` | `0.4189` | `0.0723` | `0.3703` | `0.2276` |

Decision: the earlier "learned scorer fails" conclusion was too broad. The proprio-target
goal-cost family is negative evidence for that family, not for vision. The residual future-latent
family shows vision signal on clearly bad decoys, especially overshoot/random, but the current
MLPs still do not beat the residual ridge baseline. Keep the DecisionTrace/evidence-layer claim,
but do not claim learned world-model planning value until the residual scorer clears the fair gate
under the external referee.

Local residual-family artifact hashes from that smoke run:

- `so101_latent_scorers.npz`:
  `34f219c89dfbc4a8cf1589b5d9750dee7976b6d9cca9b6be62ae545e50539f96`
- `so101_latent_scorer_meta.json`:
  `41da756abac69fffa5951e78b3ba640ce2b85cc5d82fa887cdbe4c97479b3410`
- `claude-referee-smoke-result.json`:
  `79cda54992420718d2579446c4c94ba9ec3a7a62d50e39413cc97276bf3280f1`

### Proprio-Target Goal-Cost Family

This family regresses joint-space distance to the goal state. It is useful as an ablation, but it
is not a fair test of whether vision adds decision value because the target is proprio-defined.

| Scorer | Top-1 diagnostic | Chance | MRR | Calibration vs proprio progress | Shuffled top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `proprio_baseline` | `0.0570` | `0.1429` | `0.3696` | `1.0000` | `0.1314` |
| `goal_score_mlp_h1` | `0.0151` | `0.1429` | `0.3062` | `0.2812` | `0.1469` |
| `goal_proprio_score_mlp_h1` | `0.0226` | `0.1429` | `0.3227` | `0.3069` | `0.1397` |
| `goal_history_score_mlp_h1` | `0.0183` | `0.1429` | `0.3107` | `0.3638` | `0.1419` |
| `goal_history_proprio_score_mlp_h1` | `0.0219` | `0.1429` | `0.3047` | `0.3609` | `0.1507` |

Local proprio-target artifact hashes from that smoke run:

- `so101_latent_scorers.npz`:
  `5876a96ad073c34663a40bcaa218a0d727bc0a5d4fa3a1fa1ff12bed6ecee772`
- `so101_latent_scorer_meta.json`:
  `f28cf2c043362159e79ab8a4d674b7903168adbaa18b8abc55d7c78275a24861`
- `claude-referee-goal-history-score-smoke-result.json`:
  `4612215e1848760e7d5da915d941ec6c5b4ffc54300365670e0b9869b50000c8`

## Next Iteration

The scorer-side experiments should improve the residual future-latent path, not weaken the gate:

- make `fair_beat_rate_clearbad` the primary gate and keep top-1 diagnostic,
- keep `rank_corr_vs_proprio_truth` as calibration only,
- align the residual training/evaluation space explicitly,
- add early stopping or best-validation checkpoint selection,
- sweep multi-step horizons to reduce the reverse-decoy blind spot,
- include overhead-camera latents when available,
- keep Claude's external referee as the independent judge.
