"""Reusable policy drivers for application services."""

from __future__ import annotations

from typing import Any, Callable

from robofinals.core.runtime import StepResult
from robofinals.core.session import ResetCommand, SessionContext, SessionTransition, StopCommand, StepCommand


class PolicyDriver:
    def __init__(self, policy: Callable[[Any], Any], steps: int) -> None:
        self.policy = policy
        self.steps = max(0, int(steps))
        self._reset_sent = False
        self._steps = 0

    def open(self, _context: SessionContext) -> None:
        self._reset_sent = False
        self._steps = 0

    def next_command(self, context: SessionContext):
        if not self._reset_sent:
            self._reset_sent = True
            return ResetCommand()
        if self._steps >= self.steps:
            return StopCommand("step_limit")
        latest = context.latest_result
        observation = latest.observation if isinstance(latest, StepResult) else None
        self._steps += 1
        return StepCommand(self.policy(observation))

    def on_transition(self, _context: SessionContext, _transition: SessionTransition) -> None:
        return None

    def close(self, _context: SessionContext) -> None:
        return None
