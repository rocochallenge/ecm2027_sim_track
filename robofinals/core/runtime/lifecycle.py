"""Explicit host and per-environment episode state machines."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable


class LifecycleError(RuntimeError):
    """Raised when an operation is invalid for the current lifecycle state."""


class HostLifecycle(StrEnum):
    NEW = "new"
    BUILT = "built"
    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"


class EpisodeState(StrEnum):
    RESET_PENDING = "reset_pending"
    ACTIVE = "active"
    TERMINATED = "terminated"
    TRUNCATED = "truncated"


@dataclass
class EpisodeStates:
    """Per-env state storage that supports partial vectorized resets."""

    num_envs: int
    values: list[EpisodeState] = field(init=False)

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be greater than zero")
        self.values = [EpisodeState.RESET_PENDING] * self.num_envs

    def ids(self, env_ids: Iterable[int] | None) -> tuple[int, ...]:
        result = tuple(range(self.num_envs)) if env_ids is None else tuple(int(i) for i in env_ids)
        if any(i < 0 or i >= self.num_envs for i in result):
            raise IndexError(f"env_ids must be in [0, {self.num_envs})")
        return result

    def mark(self, state: EpisodeState, env_ids: Iterable[int] | None = None) -> None:
        for env_id in self.ids(env_ids):
            self.values[env_id] = state

    def state(self, env_id: int) -> EpisodeState:
        return self.values[self.ids((env_id,))[0]]


class LifecycleMachine:
    """Small transition helper shared by runtime and tests."""

    def __init__(self, *, num_envs: int = 1) -> None:
        self.host_state = HostLifecycle.NEW
        self.episodes = EpisodeStates(num_envs)

    def transition(self, target: HostLifecycle) -> None:
        allowed = {
            HostLifecycle.NEW: {HostLifecycle.BUILT, HostLifecycle.CLOSING},
            HostLifecycle.BUILT: {HostLifecycle.RUNNING, HostLifecycle.CLOSING},
            HostLifecycle.RUNNING: {HostLifecycle.CLOSING},
            HostLifecycle.CLOSING: {HostLifecycle.CLOSED},
            HostLifecycle.CLOSED: set(),
        }
        if target not in allowed[self.host_state]:
            raise LifecycleError(f"cannot transition {self.host_state} -> {target}")
        self.host_state = target
