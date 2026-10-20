# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Backend-neutral sensor lifecycle and contact-data access.

Sensor configs in robofinals intentionally use Isaac Lab's public factory types.
The factory selects the PhysX or Newton implementation when the scene is built.
Application code must therefore consume the public sensor contract and must not
reach into backend views or private data buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable

import torch

from robofinals.utils.isaac_data_compat import as_torch
from .contract import ContactSensorRef


_LOG = logging.getLogger(__name__)
_WARNED_UNAVAILABLE: set[tuple[int, str, str]] = set()


class SensorRuntimeError(RuntimeError):
    """Base error for an invalid sensor runtime contract."""


class SensorBackendMismatchError(SensorRuntimeError):
    """Raised when a factory sensor was built for the wrong physics backend."""


class SensorNotReadyError(SensorRuntimeError):
    """Raised when a required sensor cannot currently be read."""


@dataclass(frozen=True)
class SensorStatus:
    """Backend and lifecycle state for one scene sensor."""

    name: str
    exists: bool
    initialized: bool
    expected_backend: str | None
    actual_backend: str | None
    ready: bool
    reason: str | None = None

    def describe(self) -> str:
        details = [f"sensor={self.name!r}"]
        if self.expected_backend is not None:
            details.append(f"expected_backend={self.expected_backend!r}")
        if self.actual_backend is not None:
            details.append(f"actual_backend={self.actual_backend!r}")
        if self.reason:
            details.append(f"reason={self.reason}")
        return ", ".join(details)


@dataclass(frozen=True)
class ContactSensorSample:
    """Normalized contact data returned by :func:`read_contact_sensor`."""

    status: SensorStatus
    net_forces_w: torch.Tensor
    force_matrix_w: torch.Tensor | None

    @property
    def available(self) -> bool:
        return self.status.ready


def physics_backend_name(env: Any) -> str | None:
    """Return the backend selected by the environment composition root."""

    runtime = getattr(env, "backend_runtime", None)
    kind = getattr(runtime, "kind", None)
    return getattr(kind, "value", str(kind) if kind is not None else None)


def sensor_backend_name(sensor: Any) -> str | None:
    """Return the backend implemented by a sensor instance, when applicable."""

    # Concrete factory implementations live in isaaclab_physx/isaaclab_newton.
    # Prefer that identity because a backend implementation may inherit the
    # base class's ``__backend_name__ = "base"`` marker without overriding it.
    package = type(sensor).__module__.partition(".")[0]
    if package.startswith("isaaclab_"):
        return package.removeprefix("isaaclab_").lower()

    backend = getattr(sensor, "__backend_name__", None)
    backend_name = str(backend).lower() if backend else None
    return None if backend_name == "base" else backend_name


def _backend_matches(expected: str | None, actual: str | None) -> bool:
    if expected is None or actual is None:
        return True
    if expected.endswith("physx") and actual.endswith("physx"):
        return True
    return expected == actual


def _sensor_mapping(env: Any) -> Any:
    return getattr(getattr(env, "scene", None), "sensors", {})


def _resolve_sensor_reference(env: Any, sensor_ref: str | Any) -> tuple[str, Any]:
    if isinstance(sensor_ref, str):
        return sensor_ref, _sensor_mapping(env).get(sensor_ref)
    for name, sensor in _sensor_mapping(env).items():
        if sensor is sensor_ref:
            return str(name), sensor
    return "<unregistered>", sensor_ref


def sensor_status(env: Any, sensor_name: str | Any) -> SensorStatus:
    """Inspect one sensor without triggering a lazy data update."""

    sensor_name, sensor_ref = _resolve_sensor_reference(env, sensor_name)
    sensors = _sensor_mapping(env)
    sensor = sensors.get(sensor_name) if sensor_ref is None else sensor_ref
    expected = physics_backend_name(env)
    if sensor is None:
        return SensorStatus(
            name=sensor_name,
            exists=False,
            initialized=False,
            expected_backend=expected,
            actual_backend=None,
            ready=False,
            reason="missing",
        )

    actual = sensor_backend_name(sensor)
    initialized = bool(getattr(sensor, "is_initialized", False))
    if not _backend_matches(expected, actual):
        return SensorStatus(
            name=sensor_name,
            exists=True,
            initialized=initialized,
            expected_backend=expected,
            actual_backend=actual,
            ready=False,
            reason="backend_mismatch",
        )
    if not initialized:
        return SensorStatus(
            name=sensor_name,
            exists=True,
            initialized=False,
            expected_backend=expected,
            actual_backend=actual,
            ready=False,
            reason="not_initialized",
        )

    return SensorStatus(
        name=sensor_name,
        exists=True,
        initialized=True,
        expected_backend=expected,
        actual_backend=actual,
        ready=True,
    )


def validate_scene_sensor_backends(env: Any) -> list[SensorStatus]:
    """Fail early when a factory sensor does not match the live physics manager."""

    statuses = [sensor_status(env, name) for name in _sensor_mapping(env)]
    mismatches = [status for status in statuses if status.reason == "backend_mismatch"]
    if mismatches:
        detail = "; ".join(status.describe() for status in mismatches)
        raise SensorBackendMismatchError(f"Scene contains sensors for the wrong physics backend: {detail}")
    return statuses


def required_sensors_ready(
    env: Any,
    sensor_names: Iterable[str],
    *,
    require_sim_playing: bool = True,
) -> tuple[bool, list[SensorStatus]]:
    """Return whether device-side code may safely sample its required sensors."""

    names = tuple(sensor_names)
    statuses = [sensor_status(env, name) for name in names]
    if require_sim_playing:
        is_playing = getattr(getattr(env, "sim", None), "is_playing", None)
        if callable(is_playing) and not is_playing():
            return False, statuses
    return all(status.ready for status in statuses), statuses


def assert_required_sensors_ready(env: Any, sensor_names: Iterable[str]) -> None:
    """Validate required device sensors after reset or environment creation."""

    ready, statuses = required_sensors_ready(env, sensor_names, require_sim_playing=True)
    if ready:
        return
    detail = "; ".join(status.describe() for status in statuses if not status.ready)
    if not detail:
        detail = "simulation is not playing"
    raise SensorNotReadyError(f"Required sensors are not ready after physics reset: {detail}")


def _num_envs(env: Any) -> int:
    return max(1, int(getattr(env, "num_envs", 1)))


def _torch_device(env: Any) -> torch.device:
    return torch.device(getattr(env, "device", "cpu"))


def _fallback_contact_forces(env: Any, sensor: Any = None) -> torch.Tensor:
    num_sensors = int(getattr(sensor, "num_sensors", 0) or 0) if sensor is not None else 0
    return torch.zeros(
        (_num_envs(env), max(1, num_sensors), 3),
        dtype=torch.float32,
        device=_torch_device(env),
    )


def _normalize_net_forces(env: Any, raw: Any) -> torch.Tensor:
    forces = as_torch(raw).to(_torch_device(env))
    if forces.ndim == 1 and forces.shape[0] == 3:
        forces = forces.reshape(1, 1, 3)
    elif forces.ndim == 2 and forces.shape[-1] == 3:
        forces = forces.unsqueeze(1)
    if forces.ndim != 3 or forces.shape[-1] != 3:
        raise SensorRuntimeError(
            f"Expected contact net forces with shape (num_envs, num_sensors, 3), got {tuple(forces.shape)}"
        )
    if forces.shape[0] != _num_envs(env):
        raise SensorRuntimeError(
            f"Contact force batch {forces.shape[0]} does not match env.num_envs={_num_envs(env)}"
        )
    return forces


def _normalize_force_matrix(env: Any, raw: Any) -> torch.Tensor:
    forces = as_torch(raw).to(_torch_device(env))
    if forces.ndim == 3 and forces.shape[-1] == 3:
        forces = forces.unsqueeze(2)
    if forces.ndim != 4 or forces.shape[-1] != 3:
        raise SensorRuntimeError(
            "Expected contact force matrix with shape "
            f"(num_envs, num_sensors, num_filters, 3), got {tuple(forces.shape)}"
        )
    if forces.shape[0] != _num_envs(env):
        raise SensorRuntimeError(
            f"Contact force-matrix batch {forces.shape[0]} does not match env.num_envs={_num_envs(env)}"
        )
    return forces


def _unavailable_sample(env: Any, status: SensorStatus, sensor: Any, *, strict: bool) -> ContactSensorSample:
    if strict:
        raise SensorNotReadyError(f"Contact sensor is not readable: {status.describe()}")
    warning_key = (id(env), status.name, status.reason or "unavailable")
    if warning_key not in _WARNED_UNAVAILABLE:
        _WARNED_UNAVAILABLE.add(warning_key)
        _LOG.warning("Using zero contact forces because %s", status.describe())
    return ContactSensorSample(
        status=status,
        net_forces_w=_fallback_contact_forces(env, sensor),
        force_matrix_w=None,
    )


def read_contact_sensor(
    env: Any, sensor_name: str | Any, *, strict: bool = False, status_getter=None
) -> ContactSensorSample:
    """Read and normalize one contact sensor for either PhysX or Newton.

    Unavailable lifecycle states produce a zero-force sample by default. Backend
    computation errors (including CUDA failures) are deliberately not swallowed.
    """

    sensor_name, sensor = _resolve_sensor_reference(env, sensor_name)
    status = (status_getter or sensor_status)(sensor if sensor is not None else sensor_name)
    if not status.ready:
        return _unavailable_sample(env, status, sensor, strict=strict)

    is_playing = getattr(getattr(env, "sim", None), "is_playing", None)
    if callable(is_playing) and not is_playing():
        paused = SensorStatus(
            name=status.name,
            exists=status.exists,
            initialized=status.initialized,
            expected_backend=status.expected_backend,
            actual_backend=status.actual_backend,
            ready=False,
            reason="simulation_not_playing",
        )
        return _unavailable_sample(env, paused, sensor, strict=strict)

    try:
        data = sensor.data
    except (AttributeError, KeyError) as exc:
        failed = SensorStatus(
            name=status.name,
            exists=status.exists,
            initialized=False,
            expected_backend=status.expected_backend,
            actual_backend=status.actual_backend,
            ready=False,
            reason=f"backend_handle_lost:{type(exc).__name__}",
        )
        return _unavailable_sample(env, failed, sensor, strict=strict)

    raw_net = getattr(data, "net_forces_w", None)
    if raw_net is None:
        failed = SensorStatus(
            name=status.name,
            exists=True,
            initialized=True,
            expected_backend=status.expected_backend,
            actual_backend=status.actual_backend,
            ready=False,
            reason="net_forces_unavailable",
        )
        return _unavailable_sample(env, failed, sensor, strict=strict)

    raw_matrix = getattr(data, "force_matrix_w", None)
    return ContactSensorSample(
        status=status,
        net_forces_w=_normalize_net_forces(env, raw_net),
        force_matrix_w=_normalize_force_matrix(env, raw_matrix) if raw_matrix is not None else None,
    )


def contact_force_magnitude(
    env: Any,
    sensor_name: str,
    *,
    prefer_force_matrix: bool = False,
    strict: bool = False,
    reader=None,
) -> torch.Tensor:
    """Return maximum contact-force norm per environment with shape ``(N,)``."""

    sample = reader(sensor_name, strict=strict) if reader else read_contact_sensor(env, sensor_name, strict=strict)
    forces = sample.force_matrix_w if prefer_force_matrix and sample.force_matrix_w is not None else sample.net_forces_w
    magnitudes = torch.linalg.vector_norm(forces, dim=-1)
    while magnitudes.ndim > 1:
        magnitudes = torch.amax(magnitudes, dim=-1)
    return magnitudes


def contact_detected(
    env: Any,
    sensor_name: str,
    *,
    threshold: float = 1.0e-8,
    require_force_matrix: bool = False,
    strict: bool = False,
    reader=None,
) -> torch.Tensor:
    """Return per-environment contact state from normalized sensor data."""

    sample = reader(sensor_name, strict=strict) if reader else read_contact_sensor(env, sensor_name, strict=strict)
    if require_force_matrix:
        if sample.force_matrix_w is None:
            return torch.zeros(_num_envs(env), dtype=torch.bool, device=_torch_device(env))
        forces = sample.force_matrix_w
    else:
        forces = sample.force_matrix_w if sample.force_matrix_w is not None else sample.net_forces_w
    magnitudes = torch.linalg.vector_norm(forces, dim=-1)
    while magnitudes.ndim > 1:
        magnitudes = torch.amax(magnitudes, dim=-1)
    return magnitudes > float(threshold)


def _sensor_ref_name(sensor_name: str | Any) -> str | Any:
    return sensor_name.name if isinstance(sensor_name, ContactSensorRef) else sensor_name


class ContactSensorAccess:
    """Shared sensor data mechanics; providers own backend-specific validation."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def sensor_status(self, sensor_name: str | Any) -> SensorStatus:
        sensor_name = _sensor_ref_name(sensor_name)
        return self._finalize_status(sensor_name, sensor_status(self.env, sensor_name))

    def _finalize_status(self, sensor_name: str | Any, status: SensorStatus) -> SensorStatus:
        del sensor_name
        return status

    def validate_scene_sensors(self) -> list[SensorStatus]:
        statuses = [self.sensor_status(name) for name in _sensor_mapping(self.env)]
        mismatches = [status for status in statuses if status.reason == "backend_mismatch"]
        if mismatches:
            detail = "; ".join(status.describe() for status in mismatches)
            raise SensorBackendMismatchError(
                f"Scene contains sensors for the wrong physics backend: {detail}"
            )
        return statuses

    def required_sensors_ready(
        self, sensor_names: Iterable[str], *, require_sim_playing: bool = True
    ) -> tuple[bool, list[SensorStatus]]:
        statuses = [self.sensor_status(_sensor_ref_name(name)) for name in sensor_names]
        is_playing = getattr(getattr(self.env, "sim", None), "is_playing", None)
        if require_sim_playing and callable(is_playing) and not is_playing():
            return False, statuses
        return all(status.ready for status in statuses), statuses

    def assert_required_sensors_ready(self, sensor_names: Iterable[str]) -> None:
        ready, statuses = self.required_sensors_ready(sensor_names)
        if ready:
            return
        detail = "; ".join(status.describe() for status in statuses if not status.ready)
        raise SensorNotReadyError(
            f"Required sensors are not ready after physics reset: {detail or 'simulation is not playing'}"
        )

    def read_contact(self, sensor_name: str | Any, *, strict: bool = False) -> ContactSensorSample:
        sensor_name = _sensor_ref_name(sensor_name)
        return read_contact_sensor(
            self.env, sensor_name, strict=strict, status_getter=self.sensor_status
        )

    def force_magnitude(
        self,
        sensor_name: str | Any,
        *,
        prefer_force_matrix: bool = False,
        strict: bool = False,
        threshold: float = 1.0e-8,
    ) -> torch.Tensor:
        del threshold
        sensor_name = _sensor_ref_name(sensor_name)
        return contact_force_magnitude(
            self.env,
            sensor_name,
            prefer_force_matrix=prefer_force_matrix,
            strict=strict,
            reader=self.read_contact,
        )

    def contact_detected(
        self,
        sensor_name: str | Any,
        *,
        threshold: float = 1.0e-8,
        require_force_matrix: bool = False,
        strict: bool = False,
    ) -> torch.Tensor:
        sensor_name = _sensor_ref_name(sensor_name)
        return contact_detected(
            self.env,
            sensor_name,
            threshold=threshold,
            require_force_matrix=require_force_matrix,
            strict=strict,
            reader=self.read_contact,
        )
