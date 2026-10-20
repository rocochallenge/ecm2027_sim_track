"""Sensor readiness gate independent of episode termination state."""

from __future__ import annotations

from typing import Any, Iterable


class SensorGate:
    def __init__(self, env: Any) -> None:
        self.env = env

    def ready(self, sensor_names: Iterable[str]) -> bool:
        ready, _ = self.env.backend_runtime.contacts.required_sensors_ready(
            sensor_names, require_sim_playing=True
        )
        return ready

    def assert_ready(self, sensor_names: Iterable[str]) -> None:
        self.env.backend_runtime.contacts.assert_required_sensors_ready(sensor_names)
