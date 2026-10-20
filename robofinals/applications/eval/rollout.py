from __future__ import annotations

from collections.abc import Callable
from typing import Any

from robofinals.core.runtime.environment import EnvironmentLifecycle
from robofinals.core.runtime.gym_adapter import LifecycleEnvAdapter
from robofinals.core.session.policy_driver import PolicyDriver
from robofinals.core.session import SessionRunner


class RolloutService(LifecycleEnvAdapter):
    def __init__(self, lifecycle: EnvironmentLifecycle) -> None:
        super().__init__(lifecycle)
        self.lifecycle = lifecycle

    def rollout(self, policy: Callable[[Any], Any], steps: int) -> dict[str, Any]:
        context = SessionRunner(self.lifecycle, PolicyDriver(policy, steps)).run()
        completed = max(0, context.frame_index)
        return {"steps": completed, "completed": completed == int(steps)}


class RolloutApplication(LifecycleEnvAdapter):
    """Composition root for online policy rollout evaluation."""

    def __init__(self, lifecycle: EnvironmentLifecycle) -> None:
        super().__init__(lifecycle)
        self.service = RolloutService(lifecycle)

    def rollout(self, policy: Callable[[Any], Any], steps: int) -> dict[str, Any]:
        return self.service.rollout(policy, steps)

    def close(self) -> None:
        self.service.close()
