"""Host lifecycle facade shared by teleop, replay, evaluation and RL."""

from __future__ import annotations

from typing import Any, Iterable

from .lifecycle import HostLifecycle, LifecycleError, LifecycleMachine
from .reset import ResetCoordinator
from .results import StepResult
from .sensors import SensorGate


class EnvironmentLifecycle:
    def __init__(self, env: Any, *, state_synchronizer: Any = None, num_envs: int | None = None) -> None:
        count = int(num_envs if num_envs is not None else getattr(env, "num_envs", 1))
        self.env = env
        self.machine = LifecycleMachine(num_envs=count)
        self.resetter = ResetCoordinator(self, state_synchronizer)
        self.sensors = SensorGate(env)

    @property
    def host_state(self) -> HostLifecycle:
        return self.machine.host_state

    @property
    def episode_states(self):
        return tuple(self.machine.episodes.values)

    def build(self) -> None:
        self.machine.transition(HostLifecycle.BUILT)

    def start(self) -> None:
        if self.host_state is HostLifecycle.RUNNING:
            return
        if self.host_state is HostLifecycle.NEW:
            self.build()
        self.machine.transition(HostLifecycle.RUNNING)

    def step(self, action: Any) -> StepResult:
        if self.host_state is not HostLifecycle.RUNNING:
            raise LifecycleError(f"step requires running host, got {self.host_state}")
        result = StepResult.from_env_result(
            self.env.step(action),
            env_ids=tuple(range(self.machine.episodes.num_envs)),
        )
        self._update_episode_states(result)
        return result

    def reset(self, env_ids: Iterable[int] | None = None, *, seed: int | None = None) -> Any:
        if self.host_state is HostLifecycle.NEW:
            self.build()
        return self.resetter.reset(env_ids, seed=seed)

    def reset_to(
        self,
        state: Any,
        env_ids: Iterable[int] | None = None,
        *,
        seed: int | None = None,
        is_relative: bool = False,
    ) -> Any:
        if self.host_state is HostLifecycle.NEW:
            self.build()
        return self.resetter.reset_to(state, env_ids, seed=seed, is_relative=is_relative)

    def idle(self, reason: str | None = None) -> Any:
        del reason
        backend = self.env.backend_runtime
        renderer = getattr(backend, "renderer", None)
        render_idle = getattr(renderer, "render_idle", None)
        if callable(render_idle):
            return render_idle(self.env)
        raise LifecycleError("idle requires an environment with attached backend renderer capability")

    def _update_episode_states(self, result: StepResult) -> None:
        terminated = _truthy_ids(result.terminated, self.machine.episodes.num_envs)
        truncated = _truthy_ids(result.truncated, self.machine.episodes.num_envs)
        if terminated:
            self.resetter.mark_terminated(terminated)
        if truncated:
            self.resetter.mark_truncated(truncated)

    def close(self) -> None:
        if self.host_state is HostLifecycle.CLOSED:
            return
        if self.host_state is not HostLifecycle.CLOSING:
            self.machine.transition(HostLifecycle.CLOSING)
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        self.machine.transition(HostLifecycle.CLOSED)


def _truthy_ids(value: Any, num_envs: int) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, bool):
        return tuple(range(num_envs)) if value else ()
    try:
        raw = value.detach().cpu().tolist()
    except AttributeError:
        try:
            raw = value.tolist()
        except AttributeError:
            raw = value
    if isinstance(raw, (list, tuple)):
        return tuple(index for index, flag in enumerate(raw) if bool(flag))
    return tuple(range(num_envs)) if bool(raw) else ()
