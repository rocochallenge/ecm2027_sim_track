"""Isaac composition helpers for legacy-shaped command line entry points."""

from __future__ import annotations

from typing import Any

from .environment_factory import EnvironmentFactory
from .run_plan import ConfigResolver
from robofinals.core.runtime import LifecycleEnvAdapter


def lifecycle_env_from_config(
    config: Any,
    *,
    env_name: str,
    application: str,
    task_name: str,
    robot_name: str = "runtime",
    scene_name: str = "runtime",
    execute_mode: str | None = None,
) -> LifecycleEnvAdapter:
    """Wrap a parser-produced Isaac config in the canonical lifecycle runtime."""

    plan = ConfigResolver.resolve(
        {
            "application": application,
            "execute_mode": execute_mode,
            "task_name": task_name or env_name,
            "robot_name": robot_name or "runtime",
            "scene_name": scene_name or "runtime",
            "physics_backend": getattr(config, "physics_backend", None),
            "device": getattr(getattr(config, "sim", None), "device", "cuda:0"),
            "num_envs": getattr(getattr(config, "scene", None), "num_envs", 1),
            "headless_mode": getattr(config, "headless_mode", False),
            "env_name": env_name,
        }
    )
    return LifecycleEnvAdapter(EnvironmentFactory().create_from_config(plan, config, env_name=env_name))
