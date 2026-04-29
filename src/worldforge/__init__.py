"""WorldForge: testable world-model workflows for physical-AI systems.

WorldForge is a Python integration layer that gives world-model providers, score models,
embodied policies, and media generators explicit capability contracts. Planning, evaluation,
benchmarks, diagnostics, local state, and CLI tooling are built on top of those contracts —
optional model runtimes, robot stacks, credentials, and durable storage stay host-owned.

Quick start::

    from worldforge import WorldForge

    forge = WorldForge()
    world = forge.create_world("kitchen", provider="mock")
    plan = world.plan("move the cube to x=0.4")
    execution = world.execute_plan(plan)

The package re-exports the public surface used by adapters, CLI commands, evaluations, and
benchmarks. Every name is also covered by the strict capability matrix on
:class:`ProviderCapabilities`, so calling an unsupported capability raises rather than
returning silently empty results.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from worldforge.benchmark import (
    BenchmarkBudget,
    BenchmarkGateReport,
    BenchmarkGateViolation,
    BenchmarkInputs,
    BenchmarkReport,
    BenchmarkResult,
    ProviderBenchmarkHarness,
    load_benchmark_budgets,
    load_benchmark_inputs,
    run_benchmark,
)
from worldforge.capabilities import (
    Cost,
    Embedder,
    Generator,
    Planner,
    Policy,
    Predictor,
    Reasoner,
    RunnableModel,
    Transferer,
)
from worldforge.evaluation import (
    EvalReport,
    EvalResult,
    EvalScenario,
    EvalSuite,
    EvaluationReport,
    EvaluationResult,
    EvaluationScenario,
    EvaluationSuite,
    GenerationEval,
    GenerationEvaluationSuite,
    PhysicsEval,
    PhysicsEvaluationSuite,
    PlanningEval,
    PlanningEvaluationSuite,
    ProviderSummary,
    ReasoningEval,
    ReasoningEvaluationSuite,
    TransferEval,
    TransferEvaluationSuite,
)
from worldforge.framework import (
    Comparison,
    Plan,
    PlanExecution,
    Prediction,
    World,
    WorldForge,
    list_eval_suites,
    run_eval,
)
from worldforge.models import (
    CAPABILITY_NAMES,
    Action,
    ActionPolicyResult,
    ActionScoreResult,
    BBox,
    DoctorReport,
    EmbeddingResult,
    GenerationOptions,
    Pose,
    Position,
    ProviderCapabilities,
    ProviderDoctorStatus,
    ProviderEvent,
    ProviderHealth,
    ProviderInfo,
    ProviderProfile,
    ProviderRequestPolicy,
    ReasoningResult,
    RequestOperationPolicy,
    RetryPolicy,
    Rotation,
    SceneObject,
    SceneObjectPatch,
    StructuredGoal,
    VideoClip,
    WorldForgeError,
    WorldStateError,
)

try:
    __version__ = version("worldforge-ai")
except PackageNotFoundError:  # pragma: no cover - fallback for editable local imports
    __version__ = "0.0.0"

__all__ = [
    "CAPABILITY_NAMES",
    "Action",
    "ActionPolicyResult",
    "ActionScoreResult",
    "BBox",
    "BenchmarkBudget",
    "BenchmarkGateReport",
    "BenchmarkGateViolation",
    "BenchmarkInputs",
    "BenchmarkReport",
    "BenchmarkResult",
    "Comparison",
    "Cost",
    "DoctorReport",
    "Embedder",
    "EmbeddingResult",
    "EvalReport",
    "EvalResult",
    "EvalScenario",
    "EvalSuite",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationScenario",
    "EvaluationSuite",
    "GenerationEval",
    "GenerationEvaluationSuite",
    "GenerationOptions",
    "Generator",
    "PhysicsEval",
    "PhysicsEvaluationSuite",
    "Plan",
    "PlanExecution",
    "Planner",
    "PlanningEval",
    "PlanningEvaluationSuite",
    "Policy",
    "Pose",
    "Position",
    "Prediction",
    "Predictor",
    "ProviderBenchmarkHarness",
    "ProviderCapabilities",
    "ProviderDoctorStatus",
    "ProviderEvent",
    "ProviderHealth",
    "ProviderInfo",
    "ProviderProfile",
    "ProviderRequestPolicy",
    "ProviderSummary",
    "Reasoner",
    "ReasoningEval",
    "ReasoningEvaluationSuite",
    "ReasoningResult",
    "RequestOperationPolicy",
    "RetryPolicy",
    "Rotation",
    "RunnableModel",
    "SceneObject",
    "SceneObjectPatch",
    "StructuredGoal",
    "TransferEval",
    "TransferEvaluationSuite",
    "Transferer",
    "VideoClip",
    "World",
    "WorldForge",
    "WorldForgeError",
    "WorldStateError",
    "__version__",
    "list_eval_suites",
    "load_benchmark_budgets",
    "load_benchmark_inputs",
    "run_benchmark",
    "run_eval",
]
