"""Thread-safe command mailbox for device callbacks and the session loop."""

from __future__ import annotations

from collections import deque
from threading import Lock

from robofinals.core.session import SessionCommand


class CommandMailbox:
    """Callbacks publish commands; only the runner consumes and executes them."""

    def __init__(self) -> None:
        self._commands: deque[SessionCommand] = deque()
        self._lock = Lock()

    def publish(self, command: SessionCommand) -> None:
        with self._lock:
            self._commands.append(command)

    def drain_one(self) -> SessionCommand | None:
        with self._lock:
            return self._commands.popleft() if self._commands else None

    def drain_all(self) -> tuple[SessionCommand, ...]:
        with self._lock:
            commands = tuple(self._commands)
            self._commands.clear()
            return commands

    def __len__(self) -> int:
        with self._lock:
            return len(self._commands)
