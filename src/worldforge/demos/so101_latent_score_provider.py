"""SO-101 learned latent score-provider handoff utilities.

The provider reads local scorer artifacts produced by ``scripts/train_so101_latent_scorer.py``.
It keeps numpy optional by importing it only when an instance loads weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldforge.capability_results import ActionScoreResult
from worldforge.models import JSONDict, WorldForgeError, require_json_dict

HISTORY_FRAME_COUNT = 3


class SO101LatentScoreProvider:
    """Score SO-101 candidate actions by predicted latent distance to a goal latent."""

    name = "so101-latent-score-provider"
    profile = None

    def __init__(
        self,
        *,
        weights_path: Path,
        metadata_path: Path,
        cache_path: Path | None = None,
        model_name: str | None = None,
    ) -> None:
        np = _import_numpy()
        self._np = np
        self._weights = np.load(weights_path, allow_pickle=False)
        self._metadata = require_json_dict(
            json.loads(metadata_path.read_text(encoding="utf-8")),
            name="SO-101 latent scorer metadata",
            allow_empty=False,
        )
        self._model_name = model_name or str(self._metadata["selected_model"])
        if self._model_name not in self._metadata["models"]:
            raise WorldForgeError(f"SO-101 latent scorer model not found: {self._model_name!r}.")
        self._model_meta = require_json_dict(
            self._metadata["models"][self._model_name],
            name=f"SO-101 latent scorer metadata.models.{self._model_name}",
            allow_empty=False,
        )
        self._cache = None
        if cache_path is not None:
            self._cache = np.load(cache_path, allow_pickle=False)["latents"].astype(np.float32)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def metadata(self) -> JSONDict:
        return dict(self._metadata)

    def score_actions(self, *, info: JSONDict, action_candidates: object) -> ActionScoreResult:
        candidates = _require_candidate_list(action_candidates)
        target = str(self._model_meta.get("target", "future_latent_residual"))
        current_latent = self._latent_from_info(
            info,
            inline_key="current_latent",
            frame_key="current_frame_index",
        )
        goal_latent = self._latent_from_info(
            info,
            inline_key="goal_latent",
            frame_key="goal_frame_index",
        )
        current_state = _optional_vector(info.get("current_state"), name="current_state")
        history_latents = _optional_history_latents(info.get("history_latents"))
        scores = [
            self._score_candidate(
                candidate,
                current_latent=current_latent,
                goal_latent=goal_latent,
                current_state=current_state,
                history_latents=history_latents,
            )
            for candidate in candidates
        ]
        best_index = min(range(len(scores)), key=scores.__getitem__)
        return ActionScoreResult(
            provider=self.name,
            scores=scores,
            best_index=best_index,
            lower_is_better=True,
            metadata={
                "score_kind": "learned_latent",
                "model_name": self._model_name,
                "variant": self._model_meta["variant"],
                "horizon": self._model_meta["horizon"],
                "target": target,
                "target_scale": self._model_meta.get("target_scale", "unknown"),
                "candidate_count": len(candidates),
                "referee_boundary": (
                    "Scorer output only; independent held-out referee decides whether this "
                    "beats baselines and chance."
                ),
            },
        )

    def predict_latent(
        self,
        *,
        current_latent: object,
        action_delta: object,
        current_state: object | None = None,
    ) -> list[float]:
        """Return the provider's normalized predicted future latent."""

        target = str(self._model_meta.get("target", "future_latent_residual"))
        if target != "future_latent_residual":
            raise WorldForgeError(
                "SO-101 predict_latent is only available for future_latent_residual models."
            )
        current = _required_vector(current_latent, name="current_latent")
        delta = _required_vector(action_delta, name="action_delta")
        state = _optional_vector(current_state, name="current_state")
        return self._predict(
            current_latent=current,
            action_delta=delta,
            current_state=state,
        ).tolist()

    def _score_candidate(
        self,
        candidate: JSONDict,
        *,
        current_latent: Any,
        goal_latent: Any,
        current_state: Any | None,
        history_latents: Any | None,
    ) -> float:
        action_delta = _candidate_action_delta(candidate, current_state=current_state)
        target = str(self._model_meta.get("target", "future_latent_residual"))
        if target == "goal_conditioned_cost":
            return self._predict_goal_cost(
                current_latent=current_latent,
                goal_latent=goal_latent,
                action_delta=action_delta,
                current_state=current_state,
                history_latents=history_latents,
            )
        predicted = self._predict(
            current_latent=current_latent,
            action_delta=action_delta,
            current_state=current_state,
        )
        return float(self._np.linalg.norm(predicted - goal_latent))

    def _predict(
        self,
        *,
        current_latent: Any,
        action_delta: Any,
        current_state: Any | None,
    ) -> Any:
        variant = str(self._model_meta["variant"])
        parts = [current_latent, action_delta]
        if variant == "vision_proprio_mlp":
            if current_state is None:
                raise WorldForgeError(
                    "SO-101 vision_proprio_mlp scorer requires info.current_state."
                )
            parts.append(current_state)
        x = self._np.concatenate(parts).astype(self._np.float32)
        prefix = f"{self._model_name}__"
        x_norm = (x - self._weights[prefix + "x_mean"]) / self._weights[prefix + "x_scale"]
        hidden = self._np.tanh(x_norm @ self._weights[prefix + "w1"] + self._weights[prefix + "b1"])
        residual = hidden @ self._weights[prefix + "w2"] + self._weights[prefix + "b2"]
        predicted = current_latent + residual
        norm = float(self._np.linalg.norm(predicted))
        if norm > 1e-9:
            predicted = predicted / norm
        return predicted.astype(self._np.float32)

    def _predict_goal_cost(
        self,
        *,
        current_latent: Any,
        goal_latent: Any,
        action_delta: Any,
        current_state: Any | None,
        history_latents: Any | None,
    ) -> float:
        variant = str(self._model_meta["variant"])
        if variant in {"goal_history_score_mlp", "goal_history_proprio_score_mlp"}:
            if history_latents is None:
                raise WorldForgeError("SO-101 history goal scorer requires info.history_latents.")
            observation = history_latents.reshape(-1)
        else:
            observation = current_latent
        parts = [observation, goal_latent, action_delta]
        if variant in {"goal_proprio_score_mlp", "goal_history_proprio_score_mlp"}:
            if current_state is None:
                raise WorldForgeError(f"SO-101 {variant} scorer requires info.current_state.")
            parts.append(current_state)
        x = self._np.concatenate(parts).astype(self._np.float32)
        prediction = self._forward(x)
        return float(prediction.reshape(-1)[0])

    def _forward(self, x: Any) -> Any:
        prefix = f"{self._model_name}__"
        x_norm = (x - self._weights[prefix + "x_mean"]) / self._weights[prefix + "x_scale"]
        hidden = self._np.tanh(x_norm @ self._weights[prefix + "w1"] + self._weights[prefix + "b1"])
        return (hidden @ self._weights[prefix + "w2"] + self._weights[prefix + "b2"]).astype(
            self._np.float32
        )

    def _latent_from_info(self, info: JSONDict, *, inline_key: str, frame_key: str) -> Any:
        if inline_key in info:
            return _required_vector(info[inline_key], name=inline_key)
        if frame_key in info:
            if self._cache is None:
                raise WorldForgeError(
                    f"SO-101 latent scorer requires cache_path to resolve info.{frame_key}."
                )
            frame_index = info[frame_key]
            if isinstance(frame_index, bool) or not isinstance(frame_index, int):
                raise WorldForgeError(f"SO-101 latent scorer info.{frame_key} must be an integer.")
            if frame_index < 0 or frame_index >= len(self._cache):
                raise WorldForgeError(f"SO-101 latent scorer info.{frame_key} is out of range.")
            return self._cache[frame_index]
        raise WorldForgeError(
            f"SO-101 latent scorer info must include {inline_key!r} or {frame_key!r}."
        )


def _require_candidate_list(value: object) -> list[JSONDict]:
    if not isinstance(value, list) or not value:
        raise WorldForgeError("SO-101 latent scorer action_candidates must be a non-empty list.")
    return [
        require_json_dict(candidate, name=f"action_candidates[{index}]", allow_empty=False)
        for index, candidate in enumerate(value)
    ]


def _candidate_action_delta(candidate: JSONDict, *, current_state: Any | None) -> Any:
    if "action_delta" in candidate:
        return _required_vector(candidate["action_delta"], name="candidate.action_delta")
    if "action" in candidate:
        if current_state is None:
            raise WorldForgeError(
                "SO-101 latent scorer candidate.action requires info.current_state."
            )
        return _required_vector(candidate["action"], name="candidate.action") - current_state
    raise WorldForgeError("SO-101 latent scorer candidates require action_delta or action.")


def _required_vector(value: object, *, name: str) -> Any:
    np = _import_numpy()
    if not isinstance(value, (list, tuple)):
        raise WorldForgeError(f"SO-101 latent scorer {name} must be a numeric list.")
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise WorldForgeError(f"SO-101 latent scorer {name} must be a non-empty vector.")
    if not np.isfinite(vector).all():
        raise WorldForgeError(f"SO-101 latent scorer {name} must contain finite numbers.")
    return vector


def _optional_vector(value: object, *, name: str) -> Any | None:
    if value is None:
        return None
    return _required_vector(value, name=name)


def _optional_history_latents(value: object) -> Any | None:
    if value is None:
        return None
    np = _import_numpy()
    if not isinstance(value, (list, tuple)):
        raise WorldForgeError("SO-101 latent scorer history_latents must be a numeric list.")
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != HISTORY_FRAME_COUNT or matrix.shape[1] == 0:
        raise WorldForgeError(
            "SO-101 latent scorer history_latents must contain three latent vectors."
        )
    if not np.isfinite(matrix).all():
        raise WorldForgeError("SO-101 latent scorer history_latents must contain finite numbers.")
    return matrix


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on host runtime
        raise WorldForgeError("SO-101 latent scorer requires numpy in the host runtime.") from exc
    return np


__all__ = ["SO101LatentScoreProvider"]
