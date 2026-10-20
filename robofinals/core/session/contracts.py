"""Ports for session applications, runtimes and observers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, MutableMapping, Protocol

from .commands import SessionCommand


class SessionRuntime(Protocol):
    """Environment operations shared by teleop, replay and evaluation."""

    def start(self) -> None: ...

    def reset(self, env_ids: Any = None, *, seed: int | None = None) -> Any: ...

    def reset_to(
        self,
        state: Any,
        env_ids: Any = None,
        *,
        seed: int | None = None,
        is_relative: bool = False,
    ) -> Any: ...

    def step(self, action: Any) -> Any: ...

    def idle(self, reason: str | None = None) -> Any: ...

    def close(self) -> None: ...


@dataclass
class SessionContext:
    runtime: SessionRuntime
    frame_index: int = 0
    cycle_index: int = 0
    latest_result: Any = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionTransition:
    command: SessionCommand
    result: Any
    frame_index: int
    cycle_index: int


class SessionDriver(Protocol):
    """Application policy that produces commands without owning the runtime loop."""

    def open(self, context: SessionContext) -> None: ...

    def next_command(self, context: SessionContext) -> SessionCommand: ...

    def on_transition(self, context: SessionContext, transition: SessionTransition) -> None: ...

    def close(self, context: SessionContext) -> None: ...


class SessionObserver(Protocol):
    """Optional recorder, metrics or presentation plugin."""

    def on_session_start(self, context: SessionContext) -> None: ...

    def on_transition(self, context: SessionContext, transition: SessionTransition) -> None: ...

    def on_session_close(self, context: SessionContext, error: BaseException | None) -> None: ...
