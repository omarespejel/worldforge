---
name: optional-runtime-smokes
description: Use when running or editing LeWorldModel, GR00T, LeRobot, stable-worldmodel, torch, checkpoint, CUDA, TensorRT, robot policy, or host-owned optional runtime smoke paths. Also use when a demo must distinguish injected deterministic runtime validation from real neural inference.
prerequisites: uv; optional live runtimes, credentials, checkpoints, CUDA, and robot stacks are host-owned and may be absent.
---

# Optional Runtime Smokes

<purpose>
Exercise optional provider runtimes without contaminating the base package or overstating what checkout-safe demos prove.
</purpose>

<context>
- Checkout-safe demos use injected deterministic runtimes: `worldforge-demo-leworldmodel`, `worldforge-demo-lerobot`.
- Real LeWorldModel smoke requires Python 3.10 plus upstream `stable-worldmodel[train,env]`, datasets, torch, and checkpoint assets.
- GR00T and LeRobot live smokes are host scripts under `scripts/` and require real observations plus action translators.
- The base package must remain installable with only `httpx`.
</context>

<procedure>
1. Start with checkout-safe demos unless the task explicitly requires a live runtime.
2. For live LeWorldModel, use the documented `uv run --python 3.10 --with ... worldforge-smoke-leworldmodel` command; do not edit base deps.
3. For GR00T or LeRobot, run `python scripts/smoke_gr00t_policy.py --help` or `python scripts/smoke_lerobot_policy.py --help` before attempting live execution.
4. Require host-supplied observations and embodiment-specific `action_translator` for policy providers.
5. Capture whether the run used injected deterministic runtime, real checkpoint inference, remote policy server, provider events, persistence, and reload.
6. If the host lacks the runtime, validate adapter/planner code with injected runtime tests and state that live inference was not run.
</procedure>

<commands>
```bash
uv run worldforge-demo-leworldmodel
uv run worldforge-demo-lerobot
uv run --extra harness worldforge-harness --flow diagnostics
uv run --python 3.10 --with "stable-worldmodel[train,env] @ git+https://github.com/galilai-group/stable-worldmodel.git" --with "datasets>=2.21" worldforge-smoke-leworldmodel --stablewm-home ~/.stable-wm --policy pusht/lewm --device cpu
python scripts/smoke_gr00t_policy.py --help
python scripts/smoke_lerobot_policy.py --help
```
</commands>

<patterns>
<do>
- Keep optional imports lazy and inside runtime paths.
- Validate tensor/nested-array shapes before provider result return.
- Preserve raw policy actions and require host translation before executable `Action` output.
- Document checkpoint/cache/device ownership in provider docs.
</do>
<dont>
- Do not add optional runtime packages to `[project.dependencies]`.
- Do not commit downloaded checkpoints, datasets, cache dirs, or smoke artifacts.
- Do not describe injected demos as real LeWorldModel, GR00T, or LeRobot inference.
- Do not assume macOS can start CUDA/TensorRT GR00T servers.
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| `stable_worldmodel` import fails | host runtime not installed | use checkout-safe demo or run with documented `uv --with` command |
| checkpoint not found | host cache lacks `*_object.ckpt` | run/check `worldforge-build-leworldmodel-checkpoint`; do not commit asset |
| policy provider returns raw arrays only | missing action translator | supply host-owned translator for robot/simulator embodiment |
| live smoke fails on CUDA/TensorRT | unsupported host | connect to remote policy server or skip live smoke with limitation |
</troubleshooting>

<references>
- `src/worldforge/demos/leworldmodel_e2e.py`: injected score demo.
- `src/worldforge/demos/lerobot_e2e.py`: injected policy-plus-score demo.
- `src/worldforge/smoke/leworldmodel.py`: real checkpoint smoke.
- `docs/src/playbooks.md`: optional runtime smoke commands.
</references>
