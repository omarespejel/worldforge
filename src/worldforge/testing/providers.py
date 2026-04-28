"""Reusable provider contract helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from worldforge.models import (
    Action,
    ActionPolicyResult,
    ActionScoreResult,
    BBox,
    EmbeddingResult,
    JSONDict,
    Position,
    ProviderHealth,
    ProviderProfile,
    ReasoningResult,
    SceneObject,
    VideoClip,
    WorldForgeError,
    dump_json,
)
from worldforge.providers import BaseProvider, PredictionPayload, ProviderError


def sample_contract_action() -> Action:
    """Return a deterministic action for provider contract checks."""

    return Action.move_to(0.25, 0.5, 0.0, speed=1.0)


def sample_contract_world_state() -> JSONDict:
    """Return a minimal world-state payload for provider contract checks."""

    cube = SceneObject(
        "contract-cube",
        Position(0.0, 0.5, 0.0),
        BBox(Position(-0.05, 0.45, -0.05), Position(0.05, 0.55, 0.05)),
        is_graspable=True,
    )
    return {
        "schema_version": 1,
        "id": "world_contract",
        "name": "contract-world",
        "provider": "contract",
        "description": "World state used for provider contract checks.",
        "step": 0,
        "scene": {"objects": {cube.id: cube.to_dict()}},
        "metadata": {"name": "contract-world"},
    }


def sample_contract_policy_info() -> JSONDict:
    """Return a minimal embodied-policy observation payload for contract checks."""

    return {
        "observation": {
            "video": {
                "front": [[[[[0, 0, 0]]]]],
            },
            "state": {
                "eef": [[[0.0, 0.5, 0.0]]],
            },
            "language": {
                "task": [["move the object"]],
            },
        },
        "action_horizon": 1,
    }


@dataclass(slots=True)
class ProviderContractReport:
    """Summary of executed provider contract checks."""

    provider: str
    configured: bool
    profile: ProviderProfile
    health: ProviderHealth
    exercised_operations: list[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return {
            "provider": self.provider,
            "configured": self.configured,
            "profile": self.profile.to_dict(),
            "health": self.health.to_dict(),
            "exercised_operations": list(self.exercised_operations),
        }


def _expect_provider_error[T](operation_name: str, call: Callable[[], T]) -> None:
    try:
        call()
    except ProviderError:
        return
    msg = f"Provider contract expected '{operation_name}' to raise ProviderError when unavailable."
    raise AssertionError(msg)


def _contract_check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _contract_json(value: object, message: str) -> None:
    try:
        dump_json(value)
    except WorldForgeError as exc:
        raise AssertionError(message) from exc


def _validate_prediction(provider: str, payload: PredictionPayload) -> None:
    _contract_check(
        isinstance(payload, PredictionPayload),
        "predict must return PredictionPayload.",
    )
    try:
        from worldforge.framework import _validate_world_state_payload

        _validate_world_state_payload(payload.state, context="Provider contract prediction state")
    except WorldForgeError as exc:
        raise AssertionError("predict returned invalid world state.") from exc
    _contract_check(isinstance(payload.metadata, dict), "predict metadata must be a JSON object.")
    _contract_json(payload.metadata, "predict metadata must be JSON serializable.")
    _contract_check(isinstance(payload.frames, list), "predict frames must be a list.")
    _contract_check(
        all(isinstance(frame, bytes) for frame in payload.frames),
        "predict frames must contain only bytes.",
    )
    _contract_check(
        isinstance(payload.confidence, float) and 0.0 <= payload.confidence <= 1.0,
        "predict confidence must be a probability float.",
    )
    _contract_check(
        isinstance(payload.physics_score, float) and 0.0 <= payload.physics_score <= 1.0,
        "predict physics_score must be a probability float.",
    )
    _contract_check(payload.latency_ms >= 0.0, "predict latency_ms must be non-negative.")
    _contract_check(
        payload.metadata.get("provider") == provider,
        "predict metadata provider must match provider name.",
    )


def _validate_reasoning(provider: str, result: ReasoningResult) -> None:
    _contract_check(isinstance(result, ReasoningResult), "reason must return ReasoningResult.")
    _contract_check(result.provider == provider, "reason provider must match provider name.")
    _contract_check(
        isinstance(result.answer, str) and bool(result.answer),
        "reason answer required.",
    )
    _contract_check(0.0 <= result.confidence <= 1.0, "reason confidence must be a probability.")
    _contract_check(isinstance(result.evidence, list), "reason evidence must be a list.")


def _validate_embedding(provider: str, result: EmbeddingResult) -> None:
    _contract_check(isinstance(result, EmbeddingResult), "embed must return EmbeddingResult.")
    _contract_check(result.provider == provider, "embed provider must match provider name.")
    _contract_check(isinstance(result.model, str) and bool(result.model), "embed model required.")
    _contract_check(isinstance(result.vector, list), "embed vector must be a list.")
    _contract_check(len(result.vector) >= 1, "embed vector must not be empty.")
    _contract_check(
        all(isinstance(value, float) for value in result.vector),
        "embed vector values must be floats.",
    )


def _validate_clip(clip: VideoClip) -> None:
    _contract_check(isinstance(clip, VideoClip), "media operation must return VideoClip.")
    _contract_check(isinstance(clip.frames, list), "VideoClip frames must be a list.")
    _contract_check(
        all(isinstance(frame, bytes) for frame in clip.frames),
        "VideoClip frames must contain only bytes.",
    )
    _contract_check(clip.fps > 0.0, "VideoClip fps must be positive.")
    _contract_check(clip.resolution[0] > 0, "VideoClip width must be positive.")
    _contract_check(clip.resolution[1] > 0, "VideoClip height must be positive.")
    _contract_check(clip.duration_seconds >= 0.0, "VideoClip duration must be non-negative.")
    _contract_check(isinstance(clip.metadata, dict), "VideoClip metadata must be a JSON object.")
    _contract_json(clip.metadata, "VideoClip metadata must be JSON serializable.")


def _validate_action_scores(provider: str, result: ActionScoreResult) -> None:
    _contract_check(isinstance(result, ActionScoreResult), "score must return ActionScoreResult.")
    _contract_check(result.provider == provider, "score provider must match provider name.")
    _contract_check(isinstance(result.scores, list), "score scores must be a list.")
    _contract_check(bool(result.scores), "score scores must not be empty.")
    _contract_check(
        all(isinstance(score, float) for score in result.scores),
        "score values must be floats.",
    )
    _contract_check(isinstance(result.best_index, int), "score best_index must be an integer.")
    _contract_check(
        0 <= result.best_index < len(result.scores),
        "score best_index must point at a score.",
    )
    _contract_check(
        result.best_score == result.scores[result.best_index],
        "score best_score must match scores[best_index].",
    )
    _contract_check(isinstance(result.lower_is_better, bool), "score direction flag must be bool.")
    _contract_check(isinstance(result.metadata, dict), "score metadata must be a JSON object.")


def _validate_action_policy(provider: str, result: ActionPolicyResult) -> None:
    _contract_check(
        isinstance(result, ActionPolicyResult),
        "policy must return ActionPolicyResult.",
    )
    _contract_check(result.provider == provider, "policy provider must match provider name.")
    _contract_check(isinstance(result.actions, list), "policy actions must be a list.")
    _contract_check(bool(result.actions), "policy actions must not be empty.")
    _contract_check(
        all(isinstance(action, Action) for action in result.actions),
        "policy actions must contain only Action objects.",
    )
    _contract_check(
        isinstance(result.raw_actions, dict),
        "policy raw_actions must be a JSON object.",
    )
    _contract_check(
        result.action_horizon is None or result.action_horizon >= 1,
        "policy action_horizon must be positive when provided.",
    )
    _contract_check(isinstance(result.metadata, dict), "policy metadata must be a JSON object.")
    _contract_check(
        isinstance(result.action_candidates, list),
        "policy action_candidates must be a list.",
    )
    _contract_check(bool(result.action_candidates), "policy action_candidates must not be empty.")
    _contract_check(
        all(candidate for candidate in result.action_candidates),
        "policy action candidate plans must not be empty.",
    )


def assert_provider_contract(
    provider: BaseProvider,
    *,
    world_state: JSONDict | None = None,
    action: Action | None = None,
    policy_info: JSONDict | None = None,
    score_info: JSONDict | None = None,
    score_action_candidates: object | None = None,
) -> ProviderContractReport:
    """Assert that a provider obeys the WorldForge adapter contract.

    This helper is intended for provider package tests. It validates metadata,
    health reporting, and either successful execution or clear credential errors
    for every declared capability.
    """

    profile = provider.profile()
    info = provider.info()
    health = provider.health()
    configured = provider.configured()

    _contract_check(profile.name == provider.name, "profile name must match provider name.")
    _contract_check(info.name == provider.name, "provider info name must match provider name.")
    _contract_check(
        info.description == profile.description,
        "provider info description must match profile description.",
    )
    _contract_check(
        info.is_local == profile.is_local,
        "provider info locality must match profile locality.",
    )
    _contract_check(
        sorted(info.capabilities.enabled_names()) == sorted(profile.supported_tasks),
        "provider info capabilities must match profile supported_tasks.",
    )
    _contract_check(health.name == provider.name, "provider health name must match provider name.")
    if health.healthy:
        _contract_check(configured, "healthy provider must report configured=True.")
    if not configured and profile.requires_credentials:
        _contract_check(
            health.healthy is False,
            "credential-gated unconfigured provider must report unhealthy health.",
        )
    if profile.capabilities.plan:
        _contract_check(
            profile.capabilities.predict,
            "provider-level plan capability requires predict capability.",
        )

    sample_state = world_state or sample_contract_world_state()
    sample_action = action or sample_contract_action()
    report = ProviderContractReport(
        provider=provider.name,
        configured=configured,
        profile=profile,
        health=health,
    )
    can_invoke = configured

    if profile.capabilities.predict:
        if can_invoke:
            prediction = provider.predict(sample_state, sample_action, 2)
            _validate_prediction(provider.name, prediction)
            report.exercised_operations.append("predict")
        else:
            _expect_provider_error(
                "predict",
                lambda: provider.predict(sample_state, sample_action, 2),
            )

    if profile.capabilities.reason:
        if can_invoke:
            reasoning = provider.reason(
                "How many objects are in the scene?", world_state=sample_state
            )
            _validate_reasoning(provider.name, reasoning)
            report.exercised_operations.append("reason")
        else:
            _expect_provider_error(
                "reason",
                lambda: provider.reason(
                    "How many objects are in the scene?",
                    world_state=sample_state,
                ),
            )

    if profile.capabilities.embed:
        if can_invoke:
            embedding = provider.embed(text="contract vector")
            _validate_embedding(provider.name, embedding)
            report.exercised_operations.append("embed")
        else:
            _expect_provider_error("embed", lambda: provider.embed(text="contract vector"))

    generated_clip: VideoClip | None = None
    if profile.capabilities.generate:
        if can_invoke:
            generated_clip = provider.generate("contract prompt", duration_seconds=1.0)
            _validate_clip(generated_clip)
            report.exercised_operations.append("generate")
        else:
            _expect_provider_error(
                "generate",
                lambda: provider.generate("contract prompt", duration_seconds=1.0),
            )

    if profile.capabilities.transfer:
        transfer_input = generated_clip or VideoClip(
            frames=[b"contract-frame"],
            fps=8.0,
            resolution=(64, 64),
            duration_seconds=0.125,
            metadata={"provider": provider.name},
        )
        if can_invoke:
            transferred = provider.transfer(transfer_input, width=48, height=48, fps=12.0)
            _validate_clip(transferred)
            report.exercised_operations.append("transfer")
        else:
            _expect_provider_error(
                "transfer",
                lambda: provider.transfer(transfer_input, width=48, height=48, fps=12.0),
            )

    if profile.capabilities.score:
        if can_invoke:
            if score_info is None or score_action_candidates is None:
                raise AssertionError(
                    "Provider contract requires score_info and score_action_candidates for "
                    "configured score providers."
                )
            scores = provider.score_actions(
                info=score_info,
                action_candidates=score_action_candidates,
            )
            _validate_action_scores(provider.name, scores)
            report.exercised_operations.append("score")
        else:
            _expect_provider_error(
                "score",
                lambda: provider.score_actions(
                    info=score_info or {},
                    action_candidates=[]
                    if score_action_candidates is None
                    else score_action_candidates,
                ),
            )

    if profile.capabilities.policy:
        if can_invoke:
            policy = provider.select_actions(info=policy_info or sample_contract_policy_info())
            _validate_action_policy(provider.name, policy)
            report.exercised_operations.append("policy")
        else:
            _expect_provider_error(
                "policy",
                lambda: provider.select_actions(info=policy_info or sample_contract_policy_info()),
            )

    return report
