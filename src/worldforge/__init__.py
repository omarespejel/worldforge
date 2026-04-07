"""WorldForge: a Python framework for world-model workflows."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from worldforge.evaluation import (
    EvalReport,
    EvalResult,
    EvalScenario,
    EvalSuite,
    EvaluationReport,
    EvaluationResult,
    EvaluationScenario,
    EvaluationSuite,
    PhysicsEval,
    PhysicsEvaluationSuite,
    ProviderSummary,
)
from worldforge.framework import (
    Comparison,
    Plan,
    PlanExecution,
    Prediction,
    World,
    WorldForge,
    list_eval_suites,
    plan,
    run_eval,
)
from worldforge.models import (
    Action,
    BBox,
    EmbeddingResult,
    Pose,
    Position,
    ProviderCapabilities,
    ProviderHealth,
    ProviderInfo,
    ReasoningResult,
    Rotation,
    SceneObject,
    SceneObjectPatch,
    VideoClip,
)

try:
    __version__ = version("worldforge")
except PackageNotFoundError:  # pragma: no cover - fallback for editable local imports
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "Action",
    "BBox",
    "Comparison",
    "EmbeddingResult",
    "EvalReport",
    "EvalResult",
    "EvalScenario",
    "EvalSuite",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationScenario",
    "EvaluationSuite",
    "PhysicsEval",
    "PhysicsEvaluationSuite",
    "Plan",
    "PlanExecution",
    "Position",
    "Pose",
    "Prediction",
    "ProviderCapabilities",
    "ProviderHealth",
    "ProviderInfo",
    "ProviderSummary",
    "ReasoningResult",
    "Rotation",
    "SceneObject",
    "SceneObjectPatch",
    "VideoClip",
    "World",
    "WorldForge",
    "list_eval_suites",
    "plan",
    "run_eval",
]
