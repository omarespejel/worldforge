---
name: optional-runtime-smokes
description: "Use for LeWorldModel, GR00T, LeRobot, PushT robotics showcase, real-checkpoint smoke scripts, checkpoint building, and any task involving host-owned optional runtime dependencies."
prerequisites: "uv, Python 3.13, optional host runtime packages when live smoke is requested"
---

# Optional Runtime Smokes

<purpose>
Verify optional provider/runtime paths without moving host-owned dependencies or artifacts into the base package.
</purpose>

<context>
Checkout-safe demos inject deterministic runtimes. Live smokes require host-installed runtimes, checkpoints, credentials, and task-specific observation/action translators. WorldForge validates boundaries but does not infer robot transforms, action-space projections, or preprocessing.
</context>

<procedure>
1. Choose deterministic demo or live smoke. Use deterministic demo unless the task explicitly asks for real runtime validation.
2. For LeWorldModel adapter/planner path, start with `uv run worldforge-demo-leworldmodel`.
3. For PushT real robotics status, start with `scripts/robotics-showcase --health-only`; it must not auto-build or download missing checkpoints.
4. For real LeWorldModel, use the documented `uv run --python 3.13 --with "stable-worldmodel[train] @ git+https://github.com/galilai-group/stable-worldmodel.git" --with "datasets>=2.21" ...` form.
5. For GR00T or LeRobot, require host-supplied observations and action translators before executable `Action` objects can be returned.
6. Keep warnings filtered only where scripts already do it; set `WORLDFORGE_SHOW_RUNTIME_WARNINGS=1` when raw third-party stderr is needed.
7. Never commit downloaded assets, checkpoints, datasets, cache directories, or runtime-specific credentials.
</procedure>

<patterns>
<do>
- Keep `stable_worldmodel`, torch, LeRobot, Isaac GR00T, CUDA, TensorRT, datasets, checkpoints, and robot packages out of base dependencies.
- Use `--revision` or `LEWORLDMODEL_REVISION` when pinning Hugging Face asset resolution.
- Preserve `torch.load(..., weights_only=True)` unless a trusted legacy artifact explicitly requires `--allow-unsafe-pickle`.
</do>
<dont>
- Do not pad, project, or reinterpret mismatched action spaces inside WorldForge.
- Do not claim neural inference from deterministic demos.
- Do not auto-register optional providers without required env vars.
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| LeWorldModel import fails | PyPI/source/runtime mismatch | Use GitHub `stable-worldmodel[train]` plus `datasets>=2.21` in the process |
| Missing object checkpoint | Host cache lacks `*_object.ckpt` | Run checkpoint builder with host-owned deps or supply `--cache-dir` |
| GR00T cannot start locally | Unsupported CUDA/TensorRT host | Connect to a remote policy server |
| Policy+score shape mismatch | Task action spaces differ | Stop and require task-aligned translator/candidate bridge |
</troubleshooting>

<references>
- `src/worldforge/smoke/robotics_showcase.py`: PushT showcase entry point.
- `src/worldforge/smoke/lerobot_leworldmodel.py`: policy-plus-score smoke.
- `src/worldforge/smoke/leworldmodel_checkpoint.py`: object-checkpoint builder.
- `docs/src/robotics-showcase.md`: operator workflow.
- `.env.example`: variable names only; never read `.env`.
</references>
