"""Kit host and idle-loop policy ports."""

from __future__ import annotations

from typing import Any, Protocol


class SimulationHost(Protocol):
    def is_playing(self) -> bool: ...

    def update_idle(self) -> None: ...

    def render(self) -> Any: ...

    def close(self) -> None: ...


class KitLoopPolicy:
    """Conservative policy for idle Kit updates around environment operations."""

    def allow_idle_update(self, host_state: str, *, has_pending_reset: bool = False) -> bool:
        return host_state in {"built", "running"} and not has_pending_reset

    def before_step(self, host_state: str) -> None:
        if host_state != "running":
            raise RuntimeError(f"step requires running host, got {host_state}")

    def after_reset(self, host_state: str) -> bool:
        return host_state in {"built", "running"}
