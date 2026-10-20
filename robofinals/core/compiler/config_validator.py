"""Startup validation for run plans and compiled configuration fragments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robofinals.core.runtime.run_plan import ResolvedRunPlan


class ConfigValidationError(ValueError):
    """Raised when a run plan cannot be compiled safely."""


SUPPORTED_BACKENDS = frozenset({"physx", "newton"})


def validate_run_plan(plan: ResolvedRunPlan) -> None:
    """Validate user-facing fields before any simulator objects are created."""

    errors: list[str] = []
    for field_name in ("scene_name", "robot_name", "task_name"):
        value = getattr(plan.env, field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"env.{field_name} is required")

    backend = plan.launch.physics_backend
    if backend is not None and backend.lower() not in SUPPORTED_BACKENDS:
        errors.append(f"launch.physics_backend={backend!r} is unsupported")
    if plan.launch.num_envs is not None and int(plan.launch.num_envs) < 1:
        errors.append("launch.num_envs must be greater than zero")
    if plan.env.robot_scale <= 0:
        errors.append("env.robot_scale must be greater than zero")
    if plan.application.application not in {"teleop", "replay", "eval", "rl"}:
        errors.append(f"application={plan.application.application!r} is unsupported")
    if errors:
        raise ConfigValidationError("; ".join(errors))


def validate_plugin_ids(plugins: Mapping[str, Any]) -> None:
    """Reject duplicate/empty plugin identifiers in a compiled registry."""

    if any(not isinstance(key, str) or not key.strip() for key in plugins):
        raise ConfigValidationError("plugin identifiers must be non-empty strings")
