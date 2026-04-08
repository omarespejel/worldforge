from __future__ import annotations

import worldforge
from worldforge.evaluation import EvaluationSuite
from worldforge.providers import MockProvider


def test_top_level_exports_and_subpackages_import() -> None:
    assert worldforge.__version__
    assert worldforge.GenerationOptions is not None
    assert worldforge.WorldForge is not None
    assert worldforge.WorldForgeError is not None
    assert worldforge.WorldStateError is not None
    assert worldforge.SceneObjectPatch is not None
    assert EvaluationSuite is not None
    assert MockProvider is not None
