# REST Guidance

WorldForge does not currently ship a rebuilt REST server.

That is intentional. The repository now treats the Python package as the source of truth, and transport layers should wrap it rather than duplicate orchestration logic.

## Recommended approach

If you need HTTP access today, build a thin Python service around `WorldForge`, for example with FastAPI:

```python
from fastapi import FastAPI
from worldforge import WorldForge

app = FastAPI()
forge = WorldForge()

@app.get("/providers")
def providers() -> list[dict]:
    return [info.to_dict() for info in forge.list_providers()]
```

## Non-goal for this migration

- reintroducing a service surface before the package contract is stable
