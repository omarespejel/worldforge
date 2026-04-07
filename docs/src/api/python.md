# Python API

WorldForge is a pure-Python package. There is no extension-module bridge in the current architecture.

## Entry points

```python
from worldforge import WorldForge, World, Action
```

## `WorldForge`

Responsibilities:

- register providers
- create and persist worlds
- expose generation, transfer, reasoning, and embedding helpers

Example:

```python
forge = WorldForge(state_dir=".worldforge/state")
world = forge.create_world("kitchen", provider="mock")
```

## `World`

Responsibilities:

- manage scene objects and history
- run predictions
- compare providers
- produce plans
- generate verification bundles

Example:

```python
prediction = world.predict(Action.move_to(0.3, 0.8, 0.0), steps=2)
comparison = world.compare(Action.move_to(0.4, 0.8, 0.0), ["mock"], steps=1)
plan = world.plan(goal="move the mug to the right", verify_backend="mock")
```

## Evaluation

```python
from worldforge.eval import EvalSuite

suite = EvalSuite.from_builtin("physics")
report = suite.run_report_data("mock", world=world, forge=forge)
print(report.to_markdown())
```

## Verification

```python
from worldforge.verify import ZkVerifier

bundle = prediction.prove_inference_bundle()
report = ZkVerifier().verify_inference_bundle(bundle)
print(report.current_verification.valid)
```
