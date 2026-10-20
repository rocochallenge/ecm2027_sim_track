"""Compatibility facade for backend-owned contact sensor access."""

from __future__ import annotations

from typing import Any, Iterable

from robofinals.core.backends.contact_common import (
    ContactSensorSample,
    SensorBackendMismatchError,
    SensorNotReadyError,
    SensorRuntimeError,
    SensorStatus,
)


def _contacts(env: Any):
    runtime = getattr(env, "backend_runtime", None)
    if runtime is None:
        raise RuntimeError("environment has no backend_runtime contact provider")
    return runtime.contacts


def sensor_status(env: Any, sensor_name: str | Any) -> SensorStatus:
    return _contacts(env).sensor_status(sensor_name)


def validate_scene_sensor_backends(env: Any) -> list[SensorStatus]:
    return _contacts(env).validate_scene_sensors()


def required_sensors_ready(
    env: Any, sensor_names: Iterable[str], *, require_sim_playing: bool = True
) -> tuple[bool, list[SensorStatus]]:
    return _contacts(env).required_sensors_ready(
        sensor_names, require_sim_playing=require_sim_playing
    )


def assert_required_sensors_ready(env: Any, sensor_names: Iterable[str]) -> None:
    _contacts(env).assert_required_sensors_ready(sensor_names)


def read_contact_sensor(
    env: Any, sensor_name: str | Any, *, strict: bool = False
) -> ContactSensorSample:
    return _contacts(env).read_contact(sensor_name, strict=strict)


def contact_force_magnitude(
    env: Any,
    sensor_name: str | Any,
    *,
    prefer_force_matrix: bool = False,
    strict: bool = False,
):
    return _contacts(env).force_magnitude(
        sensor_name, prefer_force_matrix=prefer_force_matrix, strict=strict
    )


def contact_detected(
    env: Any,
    sensor_name: str | Any,
    *,
    threshold: float = 1.0e-8,
    require_force_matrix: bool = False,
    strict: bool = False,
):
    return _contacts(env).contact_detected(
        sensor_name,
        threshold=threshold,
        require_force_matrix=require_force_matrix,
        strict=strict,
    )
