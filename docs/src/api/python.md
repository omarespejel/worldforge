# Python API

## Entry points

```python
from worldforge import WorldForge, World, Action
```

## `WorldForge`

Top-level framework object responsible for:

- provider registration
- world creation and persistence
- generation, transfer, reasoning, and embedding helpers

## `World`

Stateful runtime object responsible for:

- scene object management
- prediction
- comparison
- planning
- evaluation

## Evaluation

```python
from worldforge.evaluation import EvaluationSuite

suite = EvaluationSuite.from_builtin("physics")
report = suite.run_report("mock", forge=forge)
print(report.to_markdown())
```
