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

Success signal: the command exits successfully, prints `trace_count=3`, and all five artifacts
listed above exist under the output directory. The generated traces should validate through
`validate_decision_trace()` and the report should include Go2, PimSim, and SO-101 rows.

First triage step on failure: confirm the output files exist, then inspect the command's stdout and
stderr for the first `WorldForgeError`. If an artifact is missing, rerun the same command and keep
the run log with the failing fixture or normalization error.

The generated traces pass the same `validate_decision_trace()` semantic contract.
The packaged JSON Schema is available for external tooling shape checks, while the
Python validator enforces the cross-field arithmetic, claim-boundary, and
shareability invariants. The traces explicitly label their boundaries: hand-cost
scoring, analytic/replay outcomes for every candidate action, no live robot
command, and no learned latent scorer claim. `outcome.kind` is the coarse
measurement class; per-candidate `candidate_outcomes` rows carry the analytic
counterfactual provenance. `outcome_source` records concrete provenance such as
`mock_replay_execution`, which remains an analytic fixture/mock check rather
than sim-measured or real-measured success.

Learned-scorer claims have a stricter gate than the replay evidence bundle:
the scorer must clear an independent held-out referee, beat the proprio or
hand-cost baselines on fair decoy/progress metrics, and avoid shuffled-label
leakage. Exact-match top-1 is diagnostic for near-duplicate decoys, not the
pass/fail gate, and rank-correlation against proprio progress is calibration
rather than an independent success signal.
The SO-101 latent scorer scripts are handoff tools for that referee lane; they
do not grade their own output.
The scorer handoff runbook and latest mixed residual-scorer result are recorded in
`SO101_LATENT_SCORER_HANDOFF.md`.
