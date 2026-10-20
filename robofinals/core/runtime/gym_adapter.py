"""Gym-compatible view backed by the shared environment lifecycle."""

from __future__ import annotations

from typing import Any

from .environment import EnvironmentLifecycle
from .lifecycle import HostLifecycle


class LifecycleEnvAdapter:
    """Preserve Gym's public tuple API while routing operations through lifecycle."""

    def __init__(self, lifecycle: EnvironmentLifecycle) -> None:
        self.lifecycle = lifecycle
        self.env = lifecycle.env

    @property
    def unwrapped(self) -> Any:
        return self.env

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_started()
        return self.lifecycle.reset(kwargs.pop("env_ids", None), seed=kwargs.pop("seed", None))

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        self._ensure_started()
        return self.lifecycle.step(action).as_tuple()

    def reset_to(self, state: Any, env_ids: Any = None, *, seed: int | None = None, is_relative: bool = False) -> Any:
        self._ensure_started()
        return self.lifecycle.reset_to(state, env_ids=env_ids, seed=seed, is_relative=is_relative)

    def close(self) -> None:
        if self.lifecycle.host_state is not HostLifecycle.CLOSED:
            self.lifecycle.close()

    def _ensure_started(self) -> None:
        if self.lifecycle.host_state is not HostLifecycle.RUNNING:
            self.lifecycle.start()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
