# Cross-Embodiment Decision Evidence

This fork-only integration example generates the evidence bundle for
`DecisionTrace v1`.

```bash
uv run python examples/cross-embodiment-decision-evidence/run.py \
  --out .worldforge/cross-embodiment-decision-evidence
```

It runs only checkout-safe replay paths:

- Go2 replay fixture -> `decision-trace-go2.json`
- PimSim export adapter -> `decision-trace-pimsim.json`
- SO-101 replay fixture -> `decision-trace-so101.json`
- Combined report -> `cross-embodiment-report.md`
- Machine summary -> `summary.json`

The generated traces validate against the same schema and explicitly label their
boundaries: hand-cost scoring, analytic/replay outcomes, no live robot command,
and no learned latent scorer claim.
