---
name: optional-runtime-smokes
description: Use whenever the task touches LeWorldModel, GR00T, LeRobot, stable-worldmodel, torch, CUDA, TensorRT, a checkpoint file, a robot policy, or any host-owned optional runtime path. Trigger on phrases like "run the smoke", "live LeWorldModel", "GR00T server", "LeRobot policy", "torch import", "checkpoint missing", "stablewm-home", "demo vs real inference", "is this real or injected?", or "why does my mac fail". Also trigger any time a demo's wording risks blurring the line between an injected deterministic runtime and real neural inference — that distinction is the whole point of the skill.
---

# Optional Runtime Smokes

The base package ships with `httpx` only. Anything heavier (torch, stable-worldmodel, GR00T client, LeRobot, CUDA, TensorRT, robot SDKs) is **host-owned**: installed by the user, not by WorldForge. This skill keeps that boundary clean and stops checkout-safe demos from being mistaken for real neural inference.

## Fast start

```bash
# Checkout-safe — injected deterministic runtimes, no torch needed
uv run worldforge-demo-leworldmodel
uv run worldforge-demo-lerobot

# What CLI flags exist (do this before attempting a live smoke)
python scripts/smoke_gr00t_policy.py --help
python scripts/smoke_lerobot_policy.py --help

# Live LeWorldModel — host has Python 3.10, torch, and a checkpoint
uv run --python 3.10 \
  --with "stable-worldmodel[train,env] @ git+https://github.com/galilai-group/stable-worldmodel.git" \
  --with "datasets>=2.21" \
  worldforge-smoke-leworldmodel \
    --stablewm-home ~/.stable-wm --policy pusht/lewm --device cpu
```

If a live runtime is missing on the host, run the matching `worldforge-demo-*`
and **state in your output that the run used an injected deterministic runtime,
not real inference**. That single line of honesty is the most important output
this skill produces.

## Why this skill exists

Two failure modes the project explicitly defends against:

1. **Dependency bloat into base.** Adding torch, CUDA, robot SDKs, or checkpoint downloads to `[project.dependencies]` would break installability for users who only want the `httpx`-based provider layer. It's also a one-way ratchet: once added, removing it is a breaking change.
2. **"Demo passed!" misread as "model works"**. The injected demos exercise the *adapter*, *planner*, *event bus*, and *persistence* end-to-end with deterministic in-process stand-ins. They prove the WorldForge wiring is right; they do **not** prove the upstream model checkpoint loads, or that GR00T returns sensible actions, or that LeRobot's policy is safe on hardware.

The skill exists to separate those two questions and label the answer
correctly every time.

## The procedure

1. **Default to the checkout-safe demo** unless the task explicitly requires live inference. Most "does WorldForge handle this provider correctly?" questions are answered by the demo.
2. **For live LeWorldModel**, use the documented `uv run --python 3.10 --with ... worldforge-smoke-leworldmodel` form. Do not edit `[project.dependencies]` to make the import easier.
3. **For GR00T or LeRobot**, the live smoke needs a real observation source and an embodiment-specific `action_translator`. Both are host-owned. Run `--help` first; if either is missing, stop and ask the user to provide them rather than fabricating dummy data.
4. **Capture the run's nature explicitly**: which were used — injected deterministic runtime, real checkpoint inference, remote policy server — and which were exercised — provider events, persistence, reload. This belongs in the user-facing output, not buried in logs.
5. **If the host can't run live**, validate adapter / planner code with the injected demo and state plainly that live inference was not attempted. Do not infer that "demo passed" implies "live would pass".

## Runtime ownership

| Runtime | Lives where | Host must provide | Injected demo equivalent |
| --- | --- | --- | --- |
| LeWorldModel score | `src/worldforge/smoke/leworldmodel.py` | Python 3.10, `stable-worldmodel[train,env]`, `datasets`, checkpoint cache, optional CUDA | `worldforge-demo-leworldmodel` |
| LeRobot policy | `scripts/smoke_lerobot_policy.py` | LeRobot install, policy artifact, `action_translator` for the embodiment, observation source | `worldforge-demo-lerobot` |
| GR00T policy | `scripts/smoke_gr00t_policy.py` | reachable GR00T policy server (often CUDA / TensorRT), observation source, `action_translator` | none — adapter-only injected tests |
| TUI harness | `src/worldforge/harness/tui.py` | `--extra harness` (Textual) | n/a |

## Activation cues

Trigger on:
- "live LeWorldModel", "stable-worldmodel", "stablewm-home", "checkpoint not found"
- "GR00T server", "LeRobot policy", "action translator", "embodiment"
- "torch import fails", "CUDA", "TensorRT", "device cpu/cuda"
- "demo vs real", "is this injected?", "did this actually run the model?"
- mac users hitting CUDA / TensorRT errors

Do **not** trigger for:
- adapter wiring or capability flag work — load `provider-adapter-development`
- benchmark or eval suite output interpretation — load `evaluation-benchmarking`

## Stop and ask the user

- before adding any optional runtime package to base `[project.dependencies]` (forbidden by the project's `<priority_rules>`)
- before fabricating an observation or action translator for a policy provider — these are embodiment-specific and host-owned
- before describing any injected demo as real LeWorldModel / GR00T / LeRobot inference in user-facing output
- before committing checkpoints, datasets, `.stable-wm/` caches, or smoke artifacts (the `.gitignore` covers most of this; do not work around it)

## Patterns

**Do:**
- Keep optional imports lazy, inside the runtime path that needs them.
- Validate tensor / nested-array shapes before returning a provider result.
- Preserve raw policy actions and require host translation before returning executable `Action` objects.
- Document checkpoint / cache / device ownership in `docs/src/providers/<provider>.md`.

**Don't:**
- Add an optional runtime to `[project.dependencies]`.
- Commit downloaded checkpoints, datasets, or per-host caches.
- Describe an injected demo as real inference.
- Assume macOS can start CUDA or TensorRT GR00T servers.

## Troubleshooting

| Symptom | Likely cause | First fix |
| --- | --- | --- |
| `import stable_worldmodel` fails | host runtime not installed | use the checkout-safe demo, or run with the documented `uv --with` command |
| checkpoint not found | host cache lacks `*_object.ckpt` | run `worldforge-build-leworldmodel-checkpoint` if available; do not commit the asset |
| policy provider returns raw arrays only | missing `action_translator` for the embodiment | ask the user for the translator; do not invent one |
| live smoke fails on CUDA / TensorRT | unsupported host (e.g. macOS) | connect to a remote policy server, or run the injected demo and label it as such |

## References

- `src/worldforge/demos/leworldmodel_e2e.py` — injected score demo
- `src/worldforge/demos/lerobot_e2e.py` — injected policy + score demo
- `src/worldforge/smoke/leworldmodel.py` — real checkpoint smoke
- `scripts/smoke_gr00t_policy.py`, `scripts/smoke_lerobot_policy.py` — live policy smoke entry points
- `docs/src/playbooks.md` — optional runtime smoke commands
