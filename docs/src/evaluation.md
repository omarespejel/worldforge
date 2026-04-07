# Evaluation

WorldForge currently ships one built-in suite: `physics`.

## Python

```python
from worldforge import WorldForge
from worldforge.eval import EvalSuite

forge = WorldForge()
world = forge.create_world("eval-world", provider="mock")
suite = EvalSuite.from_builtin("physics")

report = suite.run_report_data("mock", world=world, forge=forge)
print(report.to_markdown())
```

## CLI

```bash
worldforge eval --suite physics --provider mock
```

## Artifacts

Evaluation reports can be rendered as:

- JSON
- Markdown
- CSV

## Current status

The evaluation framework is deterministic and lightweight. It is suitable for package-level regression checks and integration scaffolding, not for making strong scientific claims about model quality.
