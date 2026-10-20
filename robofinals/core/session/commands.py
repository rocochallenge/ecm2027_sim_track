"""Typed commands exchanged between application drivers and a session runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias


@dataclass(frozen=True)
class StepCommand:
    action: Any


@dataclass(frozen=True)
class ResetCommand:
    env_ids: Any = None
    seed: int | None = None


@dataclass(frozen=True)
class ResetToCommand:
    state: Any
    env_ids: Any = None
    seed: int | None = None
    is_relative: bool = False


@dataclass(frozen=True)
class IdleCommand:
    reason: str | None = None


@dataclass(frozen=True)
class StopCommand:
    reason: str | None = None


@dataclass(frozen=True)
class ApplicationCommand:
    """Extension command handled by a registered application plugin."""

    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("application command name must not be empty")


SessionCommand: TypeAlias = (
    StepCommand | ResetCommand | ResetToCommand | IdleCommand | StopCommand | ApplicationCommand
)
