"""Runtime settings passed through the environment composition boundary.

The IsaacLab arena constructors still have a zero-argument construction API.
Until that upstream API accepts the compiled plan directly, this module keeps
the settings in a context-local value rather than a process-global ``Context``
singleton.  The composition root installs it once per environment build.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeSettings:
    scene_name: str | None = None
    scene_backend: str | None = None
    robot_name: str | None = None
    task_name: str | None = None
    task_backend: str | None = None
    execute_mode: Any = None
    device: str | None = None
    ep_meta: dict[str, Any] = field(default_factory=dict)
    robot_scale: float = 1.0
    first_person_view: bool = False
    enable_cameras: bool = False
    usd_simplify: bool = False
    object_init_offset: list[float] = field(default_factory=lambda: [0.0, 0.0])
    max_scene_retry: int = 5
    max_object_placement_retry: int = 3
    seed: int | None = None
    sources: list[str] | None = None
    object_projects: list[str] | None = None
    headless_mode: bool = False
    initial_state: dict[str, Any] | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    replay_cfgs: dict[str, Any] | None = field(default_factory=dict)
    resample_objects_placement_on_reset: bool | None = None
    resample_robot_placement_on_reset: bool | None = None
    num_envs: int | None = 1
    use_fabric: bool | None = None
    physics_backend: str | None = None
    add_camera_to_observation: bool = False
    defer_camera_sensors: bool = False
    test_fixture_path: str | None = None
    test_fixture_type: str | None = None
    test_object_paths: list[str] | None = None
    init_robot_base_pos: list[float] | None = None
    init_robot_base_ori: list[float] | None = None
    enable_global_illumination: bool = True
    enable_full_local_scene: bool = False
    newton_deformable_solver: dict[str, Any] | None = None


_SETTINGS: ContextVar[RuntimeSettings | None] = ContextVar("robofinals_runtime_settings", default=None)


def get_runtime_settings() -> RuntimeSettings:
    """Return settings for the current environment composition context."""

    settings = _SETTINGS.get()
    if settings is None:
        settings = RuntimeSettings()
        _SETTINGS.set(settings)
    return settings


def set_runtime_settings(settings: RuntimeSettings) -> RuntimeSettings:
    """Install settings for the current environment composition context."""

    _SETTINGS.set(settings)
    return settings


__all__ = ["RuntimeSettings", "get_runtime_settings", "set_runtime_settings"]
