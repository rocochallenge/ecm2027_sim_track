"""Replay drivers that translate versioned trajectories into session commands."""

from __future__ import annotations

from typing import Any

from robofinals.core.session import ResetCommand, ResetToCommand, SessionContext, SessionTransition, StopCommand, StepCommand

from .contracts import ReplayProfile, Trajectory


class ReplayDriver:
    def __init__(self, trajectory: Trajectory, profile: ReplayProfile, decoder: Any = None) -> None:
        self.trajectory = trajectory
        self.profile = ReplayProfile(profile)
        self.decoder = decoder
        self._index = 0
        self._initial_reset_sent = False

    def open(self, _context: SessionContext) -> None:
        self._index = 0
        self._initial_reset_sent = False

    def next_command(self, _context: SessionContext):
        env_ids = self.trajectory.metadata.get("env_ids")
        if not self._initial_reset_sent:
            self._initial_reset_sent = True
            if self.trajectory.initial_state is not None:
                return ResetToCommand(
                    self._decode(self.trajectory.initial_state),
                    env_ids=env_ids,
                    is_relative=False,
                )
            return ResetCommand(env_ids=env_ids)

        if self.profile is ReplayProfile.STATE:
            if self._index >= len(self.trajectory.states):
                return StopCommand("trajectory_complete")
            state = self._decode(self.trajectory.states[self._index])
            self._index += 1
            return ResetToCommand(
                state,
                env_ids=env_ids,
                is_relative=bool(self.trajectory.metadata.get("is_relative", False)),
            )

        if self._index >= len(self.trajectory.actions):
            return StopCommand("trajectory_complete")
        action = self._decode(self.trajectory.actions[self._index])
        self._index += 1
        return StepCommand(action)

    def on_transition(self, _context: SessionContext, _transition: SessionTransition) -> None:
        return None

    def close(self, _context: SessionContext) -> None:
        close = getattr(self.decoder, "close", None)
        if callable(close):
            close()

    def _decode(self, value: Any) -> Any:
        if self.decoder is None:
            return value
        decode = getattr(self.decoder, "decode", None)
        return decode(value) if callable(decode) else self.decoder(value)
