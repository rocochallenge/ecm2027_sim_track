"""Versioned trajectory model decoupled from HDF5/LeRobot/MCAP storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from enum import StrEnum


class ReplayProfile(StrEnum):
    ACTION = "action"
    JOINT_TARGET = "joint_target"
    STATE = "state"


@dataclass(frozen=True)
class Trajectory:
    schema_version: int
    actions: tuple[Any, ...] = ()
    initial_state: Any = None
    states: tuple[Any, ...] = ()
    timestamps: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if self.timestamps and len(self.timestamps) not in {len(self.actions), len(self.states)}:
            raise ValueError("timestamps must align with actions or states")


EpisodeSchema = Trajectory


class EpisodeReader(Protocol):
    def read(self, episode_id: str) -> Trajectory: ...


EpisodeSource = EpisodeReader


class EpisodeWriter(Protocol):
    def write(self, episode_id: str, trajectory: Trajectory) -> None: ...


class ActionDecoder(Protocol):
    def decode(self, action: Any) -> Any: ...


class StateDecoder(Protocol):
    def decode(self, state: Any) -> Any: ...


@dataclass(frozen=True)
class DivergenceReport:
    max_position_error: float = 0.0
    max_joint_error: float = 0.0
    contact_mismatches: int = 0


@dataclass(frozen=True)
class ReplayReport:
    episode_id: str
    frames: int
    completed: bool
    divergence: DivergenceReport = field(default_factory=DivergenceReport)
    errors: tuple[str, ...] = ()
