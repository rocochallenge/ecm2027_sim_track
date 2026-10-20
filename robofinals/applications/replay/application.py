from __future__ import annotations

from typing import Any, Iterable

from .contracts import ActionDecoder, ReplayProfile, ReplayReport, Trajectory
from .driver import ReplayDriver
from robofinals.core.runtime.environment import EnvironmentLifecycle
from robofinals.core.session import SessionRunner


class ReplayService:
    def __init__(
        self,
        lifecycle: EnvironmentLifecycle,
        decoder: ActionDecoder | None = None,
        *,
        profile: ReplayProfile = ReplayProfile.ACTION,
        observers: Iterable[Any] = (),
    ) -> None:
        self.lifecycle = lifecycle
        self.decoder = decoder
        self.profile = ReplayProfile(profile)
        self.observers = tuple(observers)

    def replay(self, episode_id: str, trajectory: Trajectory) -> ReplayReport:
        errors: list[str] = []
        try:
            context = SessionRunner(
                self.lifecycle,
                ReplayDriver(trajectory, self.profile, self.decoder),
                observers=self.observers,
                close_runtime=False,
            ).run()
        except Exception as exc:  # report belongs to caller; lifecycle remains inspectable
            errors.append(f"{type(exc).__name__}: {exc}")
            return ReplayReport(episode_id=episode_id, frames=0, completed=False, errors=tuple(errors))
        frames = context.frame_index if self.profile is not ReplayProfile.STATE else context.cycle_index - 1
        return ReplayReport(episode_id=episode_id, frames=frames, completed=True, errors=tuple(errors))


class ReplayApplication:
    """Composition root for trajectory replay."""

    def __init__(
        self,
        lifecycle: EnvironmentLifecycle,
        decoder: ActionDecoder | None = None,
        *,
        profile: ReplayProfile = ReplayProfile.ACTION,
        observers: Iterable[Any] = (),
    ) -> None:
        self.lifecycle = lifecycle
        self.service = ReplayService(
            lifecycle,
            decoder,
            profile=profile,
            observers=observers,
        )

    def replay(self, episode_id: str, trajectory: Trajectory) -> ReplayReport:
        return self.service.replay(episode_id, trajectory)
