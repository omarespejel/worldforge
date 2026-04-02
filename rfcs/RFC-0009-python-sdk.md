# RFC-0009: Python SDK

- **Status:** Draft
- **Created:** 2026-04-02
- **Authors:** WorldForge Core Team
- **Requires:** RFC-0001 (Core Architecture)

---

## Abstract

This RFC defines the plan to make WorldForge's Python SDK production-ready.
The SDK currently consists of ~11,000 lines of PyO3 bindings that expose the
Rust core to Python. This RFC covers PyPI packaging, Pythonic API design,
async support, Jupyter integration, type stubs, documentation, testing, and
version compatibility. The goal is a Python SDK that feels native to Python
developers while leveraging WorldForge's high-performance Rust core.

## Motivation

Python is the dominant language for AI/ML research and application development.
While WorldForge's Rust core provides performance, most users will interact
with WorldForge through Python. The current PyO3 bindings work but have
several shortcomings:

1. **API feels like Rust in Python**: Methods follow Rust conventions (snake_case
   is fine, but builder patterns and Result types feel foreign to Python devs).
2. **No async support**: Python's asyncio is not integrated; all calls block.
3. **No PyPI distribution**: Users must build from source with Rust toolchain.
4. **Missing type stubs**: No IDE autocompletion or type checking support.
5. **No Jupyter support**: No rich display for predictions, no progress bars.
6. **Sparse documentation**: Docstrings are minimal, no tutorials exist.

Competing frameworks like LangChain show that Python developer experience is
critical for adoption. We must match or exceed their ergonomics while providing
the performance advantage of our Rust core.

## Detailed Design

### 1. Current State Assessment

The existing PyO3 bindings (11k lines in `worldforge-python/`) expose:

```
worldforge/
├── __init__.py          # Module initialization
├── _core.pyd/.so        # Compiled Rust extension
├── core.py              # Core types (World, Scene, Object)
├── providers.py         # Provider configuration and access
├── prediction.py        # Prediction engine bindings
├── planning.py          # Planning system bindings
├── eval.py              # Evaluation framework bindings
└── guardrails.py        # Safety guardrails bindings
```

Key issues with current bindings:
- Error handling uses Rust-style `Result` wrappers instead of Python exceptions
- No context managers for resource cleanup
- Synchronous-only API
- Raw bytes returned instead of PIL Images
- No pretty-printing or rich repr

### 2. PyPI Packaging and Distribution

#### 2.1 Build System

We will use `maturin` for building and publishing:

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "worldforge"
version = "0.1.0"
description = "Universal world model framework for AI"
readme = "README.md"
license = { text = "MIT OR Apache-2.0" }
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.20",
    "pillow>=8.0",
]

[project.optional-dependencies]
async = ["anyio>=3.0"]
jupyter = ["ipywidgets>=7.0", "IPython>=7.0"]
eval = ["matplotlib>=3.5", "pandas>=1.3"]
all = ["worldforge[async,jupyter,eval]"]

[tool.maturin]
features = ["python"]
python-source = "python"
module-name = "worldforge._core"
```

#### 2.2 Platform Support

Pre-built wheels for:
- Linux x86_64 (manylinux2014)
- Linux aarch64 (manylinux2014)
- macOS x86_64
- macOS aarch64 (Apple Silicon)
- Windows x86_64

Built via GitHub Actions matrix using `maturin-action`:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    target: [x86_64, aarch64]
    exclude:
      - os: windows-latest
        target: aarch64
```

#### 2.3 Installation

```bash
pip install worldforge              # core only
pip install worldforge[async]       # with async support
pip install worldforge[jupyter]     # with Jupyter integration
pip install worldforge[all]         # everything
```

### 3. Pythonic API Design

#### 3.1 Core Principles

1. **Exceptions over Result types**: Use Python exceptions for errors
2. **Context managers**: For resource management (providers, sessions)
3. **Properties over getters**: Use `@property` instead of `get_x()` methods
4. **PIL Images**: Return PIL Images, not raw bytes
5. **Dict-like access**: Support dictionary-style access where natural
6. **Dataclass-like objects**: Rich repr, equality, serialization
7. **Keyword arguments**: Prefer kwargs over positional args for clarity

#### 3.2 API Surface Redesign

**Before (current Rust-style):**
```python
from worldforge._core import PyProviderConfig, PyPredictionEngine

config = PyProviderConfig.new("genesis")
config.set_api_key("sk-...")
engine = PyPredictionEngine.create(config)
result = engine.predict(image_bytes, "push the ball", 8)
frames = result.get_frames()  # Returns list of byte arrays
```

**After (Pythonic):**
```python
import worldforge as wf

# Simple, Pythonic initialization
engine = wf.Engine("genesis", api_key="sk-...")

# Predict with keyword arguments, returns PIL Images
prediction = engine.predict(
    image="scene.png",        # accepts path, PIL Image, numpy array, URL
    action="push the ball",
    num_frames=8,
)

# Rich access to results
prediction.frames          # list of PIL Images
prediction.frames[0].show() # display first frame
prediction.confidence      # float
prediction.metadata        # dict

# Context manager support
with wf.Engine("genesis", api_key="sk-...") as engine:
    pred = engine.predict(image="scene.png", action="push the ball")
```

#### 3.3 Provider Management

```python
import worldforge as wf

# List available providers
wf.providers.list()
# ['genesis', 'cosmos', 'wan', 'gaia', 'lucid']

# Provider with full configuration
provider = wf.Provider(
    "genesis",
    api_key="sk-...",
    timeout=30,
    max_retries=3,
    rate_limit=10,  # requests per second
)

# Multi-provider setup
engine = wf.Engine(
    providers=["genesis", "cosmos"],
    strategy="best_of",     # or "fastest", "cheapest", "consensus"
)
```

#### 3.4 World and Scene API

```python
# Create and manipulate worlds
world = wf.World("my_simulation")
scene = world.create_scene(
    image="kitchen.png",
    objects={"cup": (100, 200), "plate": (300, 150)},
)

# Natural attribute access
scene.objects["cup"].position  # (100, 200)
scene.objects["cup"].material  # inferred from image

# Predict with scene context
next_scene = engine.predict(scene, action="knock the cup off the table")
next_scene.objects["cup"].position  # (150, 450) - fallen
```

#### 3.5 Evaluation API

```python
import worldforge as wf
from worldforge.eval import EvalSuite, Dimension

# Quick evaluation
results = wf.evaluate("genesis", dimensions="all")
print(results.summary())

# Detailed evaluation
suite = EvalSuite(
    dimensions=[Dimension.GRAVITY, Dimension.COLLISIONS],
    samples_per_scenario=5,
)
results = suite.run("genesis")
results.to_dataframe()  # Returns pandas DataFrame
results.plot()          # Matplotlib visualization

# Compare providers
comparison = suite.compare(["genesis", "cosmos", "wan"])
comparison.to_html("report.html")
```

### 4. Async Support

#### 4.1 Asyncio Integration

All I/O-bound operations have async counterparts:

```python
import asyncio
import worldforge as wf

async def main():
    engine = wf.AsyncEngine("genesis", api_key="sk-...")

    # Single async prediction
    prediction = await engine.predict(
        image="scene.png",
        action="push the ball",
    )

    # Concurrent predictions
    tasks = [
        engine.predict(image="scene.png", action=f"action_{i}")
        for i in range(10)
    ]
    results = await asyncio.gather(*tasks)

    # Async iteration over streaming predictions
    async for frame in engine.predict_stream(image="scene.png", action="push"):
        process_frame(frame)

asyncio.run(main())
```

#### 4.2 Implementation Strategy

The async layer uses `pyo3-asyncio` with `tokio` runtime:

```rust
#[pyfunction]
fn predict<'py>(
    py: Python<'py>,
    engine: &PyEngine,
    image: PyObject,
    action: &str,
) -> PyResult<&'py PyAny> {
    let engine = engine.inner.clone();
    let image = extract_image(py, image)?;
    let action = action.to_string();

    pyo3_asyncio::tokio::future_into_py(py, async move {
        let result = engine.predict(&image, &action).await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PythonPrediction::from(result))
    })
}
```

Both sync and async APIs share the same underlying Rust async runtime.
The sync API simply blocks on the tokio runtime.

### 5. Jupyter Notebook Integration

#### 5.1 Rich Display

```python
# Predictions display as image grids in Jupyter
prediction = engine.predict(image="scene.png", action="push the ball")
prediction  # automatically displays frame grid in notebook

# Worlds display as interactive scenes
world = wf.World("my_sim")
world  # displays current state as image with overlays

# Evaluation results display as charts
results = suite.run("genesis")
results  # displays radar chart of dimension scores
```

#### 5.2 Implementation

Custom `_repr_html_`, `_repr_png_`, and widget integration:

```python
class Prediction:
    def _repr_html_(self):
        """Rich HTML display for Jupyter notebooks."""
        frames_html = "".join(
            f'<img src="data:image/png;base64,{frame_b64}" '
            f'style="width:200px;margin:2px"/>'
            for frame_b64 in self._frames_base64()
        )
        return f"""
        <div style="display:flex;flex-wrap:wrap">
            <div><b>Action:</b> {self.action}</div>
            <div><b>Confidence:</b> {self.confidence:.2f}</div>
            <div>{frames_html}</div>
        </div>
        """

    def animate(self, fps=8):
        """Display prediction as animation in notebook."""
        from IPython.display import HTML
        return HTML(self._to_animation_html(fps))
```

#### 5.3 Progress Bars

Long-running operations show progress:

```python
# Uses tqdm or ipywidgets automatically
results = suite.run("genesis")  # Shows progress bar in notebook
# Evaluating gravity: 100%|████████████| 115/115 [02:30<00:00]
```

### 6. Type Stubs (.pyi Files)

Complete type stubs for IDE support:

```python
# worldforge/__init__.pyi
from typing import Union, Optional, List, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np

ImageInput = Union[str, Path, Image.Image, np.ndarray, bytes]

class Engine:
    def __init__(
        self,
        provider: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None: ...

    def predict(
        self,
        image: ImageInput,
        action: str,
        *,
        num_frames: int = 8,
        guidance_scale: float = 7.5,
    ) -> Prediction: ...

    def __enter__(self) -> "Engine": ...
    def __exit__(self, *args: Any) -> None: ...

class Prediction:
    @property
    def frames(self) -> List[Image.Image]: ...
    @property
    def confidence(self) -> float: ...
    @property
    def metadata(self) -> Dict[str, Any]: ...
    def save(self, path: Union[str, Path], format: str = "png") -> None: ...
    def to_video(self, path: Union[str, Path], fps: int = 8) -> None: ...
    def animate(self, fps: int = 8) -> Any: ...

class AsyncEngine:
    async def predict(
        self,
        image: ImageInput,
        action: str,
        *,
        num_frames: int = 8,
    ) -> Prediction: ...

    async def predict_stream(
        self,
        image: ImageInput,
        action: str,
    ) -> AsyncIterator[Image.Image]: ...
```

Type stubs are auto-generated from Rust docstrings plus manual annotations,
and are included in the PyPI package.

### 7. Documentation

#### 7.1 Docstrings

Every public class and method has comprehensive docstrings:

```python
class Engine:
    """WorldForge prediction engine.

    The Engine is the primary interface for making world model predictions.
    It manages provider connections, handles retries, and provides both
    synchronous and asynchronous prediction APIs.

    Args:
        provider: Name of the world model provider (e.g., "genesis", "cosmos").
        api_key: API key for the provider. If not provided, reads from
            WORLDFORGE_{PROVIDER}_API_KEY environment variable.
        timeout: Maximum time in seconds for a single prediction request.
        max_retries: Number of retry attempts on transient failures.

    Examples:
        Basic usage::

            import worldforge as wf
            engine = wf.Engine("genesis", api_key="sk-...")
            pred = engine.predict("scene.png", "push the ball")
            pred.frames[0].show()

        With context manager::

            with wf.Engine("genesis") as engine:
                pred = engine.predict("scene.png", "push the ball")

    See Also:
        AsyncEngine: Async version of Engine.
        Provider: Low-level provider configuration.
    """
```

#### 7.2 Documentation Structure

```
docs/
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── first-prediction.md
├── guides/
│   ├── providers.md
│   ├── predictions.md
│   ├── planning.md
│   ├── evaluation.md
│   ├── guardrails.md
│   ├── async-usage.md
│   └── jupyter-notebooks.md
├── api-reference/           # Auto-generated from docstrings
│   ├── engine.md
│   ├── prediction.md
│   ├── planning.md
│   └── eval.md
├── tutorials/
│   ├── robot-arm-planning.ipynb
│   ├── video-game-physics.ipynb
│   └── provider-comparison.ipynb
└── migration/
    └── from-langchain.md
```

Built with Sphinx + MyST for Markdown support, hosted on Read the Docs.

### 8. Version Compatibility

#### 8.1 Python Version Support

| Python Version | Support Level | Notes                    |
|----------------|--------------|--------------------------|
| 3.8            | Maintained   | Minimum version, no walrus |
| 3.9            | Maintained   | Dict union syntax avoided |
| 3.10           | Full         | Match statements in internals |
| 3.11           | Full         | Primary development target |
| 3.12           | Full         | Latest stable             |
| 3.13           | Best effort  | Pre-release testing       |

#### 8.2 Compatibility Strategy

- Use `from __future__ import annotations` for modern type hints on 3.8/3.9
- Avoid 3.10+ syntax (match statements, union types with `|`) in public API
- Test matrix covers all supported versions
- Feature detection for optional dependencies (numpy, PIL, etc.)

### 9. Testing Strategy

#### 9.1 Test Structure

```
tests/
├── unit/
│   ├── test_engine.py
│   ├── test_prediction.py
│   ├── test_planning.py
│   ├── test_eval.py
│   ├── test_providers.py
│   └── test_types.py
├── integration/
│   ├── test_provider_integration.py
│   ├── test_async.py
│   ├── test_jupyter.py
│   └── test_end_to_end.py
├── compatibility/
│   ├── test_numpy_compat.py
│   ├── test_pil_compat.py
│   └── test_pandas_compat.py
└── conftest.py              # Shared fixtures, mock providers
```

#### 9.2 Testing Tools

```toml
# tox.ini
[tox]
envlist = py38, py39, py310, py311, py312, lint, typecheck

[testenv]
deps = pytest, pytest-asyncio, pytest-cov, Pillow, numpy
commands = pytest tests/ --cov=worldforge --cov-report=html

[testenv:lint]
deps = ruff, black
commands =
    ruff check python/
    black --check python/

[testenv:typecheck]
deps = mypy, types-Pillow
commands = mypy python/worldforge/ --strict
```

#### 9.3 Mock Providers

Testing does not require real API keys:

```python
# conftest.py
import worldforge as wf
from worldforge.testing import MockProvider

@pytest.fixture
def mock_engine():
    """Engine with deterministic mock provider."""
    provider = MockProvider(
        predict_fn=lambda img, action: generate_test_frames(8),
        latency=0.1,
    )
    return wf.Engine(provider=provider)
```

### 10. Comparison to LangChain Developer Experience

| Aspect              | LangChain        | WorldForge Python SDK |
|---------------------|------------------|----------------------|
| Installation        | `pip install`    | `pip install`        |
| Import style        | `from langchain...` | `import worldforge as wf` |
| Provider setup      | API key + class  | API key + string name |
| Async support       | Full asyncio     | Full asyncio         |
| Jupyter support     | Basic display    | Rich display + animation |
| Type hints          | Partial          | Complete (.pyi stubs) |
| Error messages      | Often cryptic    | Clear, actionable    |
| Docs                | Good but scattered | Structured, with tutorials |
| Streaming           | Callback-based   | Async iterator       |
| Testing utilities   | Limited mocks    | Full mock provider system |

Key differentiators:
- **Performance**: Rust core means 10-100x faster local computation
- **Type safety**: Complete type stubs catch errors before runtime
- **Rich Jupyter**: Animated predictions, interactive exploration
- **Mock system**: First-class testing without API keys

## Implementation Plan

### Phase 1: API Redesign (Weeks 1-2)
- Design final Pythonic API surface
- Create type stubs (.pyi files)
- Implement new Python wrapper layer over existing PyO3 bindings

### Phase 2: Packaging (Weeks 3-4)
- Set up maturin build system
- Configure CI/CD for wheel building (all platforms)
- First PyPI test release

### Phase 3: Async and Jupyter (Weeks 5-6)
- Implement AsyncEngine with pyo3-asyncio
- Add Jupyter rich display (_repr_html_, animations)
- Progress bar integration

### Phase 4: Documentation (Weeks 7-8)
- Write comprehensive docstrings
- Create Sphinx documentation site
- Write 3 tutorial notebooks
- Getting started guide

### Phase 5: Testing and Polish (Weeks 9-10)
- Full test suite with tox
- Mock provider system
- Type checking with mypy strict mode
- Performance benchmarks
- First stable PyPI release (v0.1.0)

## Testing Strategy

- **Unit tests**: Every public method tested with mocked Rust core
- **Integration tests**: End-to-end with mock providers
- **Compatibility tests**: Verify behavior across Python 3.8-3.12
- **Type tests**: mypy strict mode passes
- **Notebook tests**: nbval for testing Jupyter notebooks
- **Performance tests**: Ensure Python overhead < 5% vs direct Rust calls

## Open Questions

1. **Minimum Python version**: Should we support 3.8 (EOL Dec 2024) or
   start at 3.9/3.10? Dropping 3.8 simplifies type hint syntax.

2. **Numpy dependency**: Should numpy be required or optional? It adds
   ~20MB but enables zero-copy array transfer from Rust.

3. **Sync vs async naming**: Should async methods be on a separate class
   (`AsyncEngine`) or use the same class with `await` (`engine.predict()`
   returns awaitable when in async context)?

4. **Image library**: Should we depend on PIL/Pillow or support multiple
   image libraries (PIL, OpenCV, etc.) through adapters?

5. **Notebook widget**: Should we build a custom ipywidget for interactive
   world exploration, or keep it simple with HTML display?

6. **Breaking changes policy**: How do we handle API changes between
   versions? Deprecation warnings for N versions?

7. **Conda packaging**: Should we also distribute via conda-forge for
   users in scientific computing environments?

8. **GPU acceleration**: Should the Python SDK expose GPU configuration
   for providers that support local inference?
