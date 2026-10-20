"""Normalized environment transition values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class StepResult:
    """Stable result contract shared by teleop, replay, evaluation and RL."""

    observation: Any
    reward: Any = None
    terminated: Any = False
    truncated: Any = False
    info: Mapping[str, Any] = field(default_factory=dict)
    env_ids: Any = None

    @classmethod
    def from_env_result(cls, result: Any, *, env_ids: Any = None) -> "StepResult":
        if isinstance(result, cls):
            return result if env_ids is None else cls(
                result.observation,
                result.reward,
                result.terminated,
                result.truncated,
                result.info,
                env_ids,
            )
        if not isinstance(result, tuple):
            return cls(observation=result, env_ids=env_ids)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, done, info = result
            terminated, truncated = done, False
        else:
            raise ValueError(f"environment step must return 4 or 5 values, got {len(result)}")
        return cls(observation, reward, terminated, truncated, info or {}, env_ids)

    @property
    def done(self) -> Any:
        """Compatibility view for callers that still use the old terminology."""
        return _logical_or(self.terminated, self.truncated)

    def as_tuple(self) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def __iter__(self):
        return iter(self.as_tuple())

    def __getitem__(self, index: int) -> Any:
        return self.as_tuple()[index]


def _logical_or(left: Any, right: Any) -> Any:
    try:
        return left | right
    except TypeError:
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            left_values = list(left) if isinstance(left, (list, tuple)) else [left] * len(right)
            right_values = list(right) if isinstance(right, (list, tuple)) else [right] * len(left_values)
            return [bool(a) or bool(b) for a, b in zip(left_values, right_values)]
        return bool(left) or bool(right)
