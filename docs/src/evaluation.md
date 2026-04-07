# Evaluation

WorldForge currently ships one built-in suite: `physics`.

## Python

```python
from worldforge.evaluation import EvaluationSuite

suite = EvaluationSuite.from_builtin("physics")
report = suite.run_report("mock", forge=forge)
print(report.to_markdown())
```

## CLI

```bash
uv run worldforge eval --suite physics --provider mock
```

## Report formats

- JSON
- Markdown
- CSV
