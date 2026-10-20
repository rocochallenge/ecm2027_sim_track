"""Typed, immutable configuration contracts for application entry points."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from robofinals.core.backends.common import BackendKind, resolve_backend_kind


_ALIASES = {
    "task": "task_name",
    "robot": "robot_name",
    "layout": "scene_name",
    "headless": "headless_mode",
}


@dataclass(frozen=True)
class LaunchSpec:
    physics_backend: str | None = None
    device: str = "cuda:0"
    num_envs: int | None = None
    headless_mode: bool = False
    enable_cameras: bool = False
    use_fabric: bool | None = None

    @property
    def resolved_backend(self) -> BackendKind:
        return resolve_backend_kind(self.physics_backend)


@dataclass(frozen=True)
class EnvSpec:
    scene_backend: str
    task_backend: str
    scene_name: str
    robot_name: str
    task_name: str
    robot_scale: float = 1.0
    gripper_closed_gap: float = 0.0
    seed: int | None = None
    sources: tuple[str, ...] | None = None
    object_projects: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ApplicationSpec:
    application: str = "teleop"
    execute_mode: str = "TELEOP"
    rl_name: str | None = None
    replay_cfgs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedRunPlan:
    launch: LaunchSpec
    env: EnvSpec
    application: ApplicationSpec
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_parse_kwargs(self) -> dict[str, Any]:
        """Convert the plan to the legacy parser's keyword contract."""
        values: dict[str, Any] = {
            **asdict(self.launch),
            **asdict(self.env),
            **asdict(self.application),
            **dict(self.extra),
        }
        values["sources"] = list(self.env.sources) if self.env.sources is not None else None
        values["object_projects"] = (
            list(self.env.object_projects) if self.env.object_projects is not None else None
        )
        return values

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot used for reproducibility."""
        return _json_safe(
            {
                "launch": asdict(self.launch),
                "env": asdict(self.env),
                "application": asdict(self.application),
                "extra": dict(self.extra),
            }
        )

    def write_snapshot(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


class ConfigResolver:
    """Resolve defaults, YAML and CLI values with CLI taking precedence."""

    @classmethod
    def from_yaml(cls, path: str | Path, cli: Mapping[str, Any] | None = None) -> ResolvedRunPlan:
        import yaml

        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(values, Mapping):
            raise TypeError("run configuration YAML must contain a mapping")
        return cls.resolve(values, cli)

    @classmethod
    def resolve(
        cls,
        yaml_values: Mapping[str, Any] | None = None,
        cli_values: Mapping[str, Any] | None = None,
    ) -> ResolvedRunPlan:
        merged: dict[str, Any] = {}
        for source in (yaml_values or {}, cli_values or {}):
            for key, value in source.items():
                if key in {"launch", "env", "application"} and isinstance(value, Mapping):
                    nested = value
                    for nested_key, nested_value in nested.items():
                        if nested_value is not None:
                            merged[_ALIASES.get(nested_key, nested_key)] = nested_value
                elif value is not None:
                    merged[_ALIASES.get(key, key)] = value

        application = str(merged.get("application", "teleop")).lower()
        raw_mode = merged.get("execute_mode")
        if hasattr(raw_mode, "name"):
            raw_mode = raw_mode.name
        mode = str(raw_mode or _default_execute_mode(application)).upper()
        launch = LaunchSpec(
            physics_backend=(str(merged["physics_backend"]).lower() if merged.get("physics_backend") is not None else None),
            device=str(merged.get("device", "cuda:0")),
            num_envs=merged.get("num_envs"),
            headless_mode=bool(merged.get("headless_mode", False)),
            enable_cameras=bool(merged.get("enable_cameras", False)),
            use_fabric=merged.get("use_fabric"),
        )
        env = EnvSpec(
            scene_backend=str(merged.get("scene_backend", "robocasa")),
            task_backend=str(merged.get("task_backend", "robocasa")),
            scene_name=str(merged.get("scene_name", "")),
            robot_name=str(merged.get("robot_name", "")),
            task_name=str(merged.get("task_name", "")),
            robot_scale=float(merged.get("robot_scale", 1.0)),
            gripper_closed_gap=float(merged.get("gripper_closed_gap", 0.0)),
            seed=merged.get("seed"),
            sources=_as_tuple(merged.get("sources")),
            object_projects=_as_tuple(merged.get("object_projects")),
        )
        app = ApplicationSpec(
            application=application,
            execute_mode=mode,
            rl_name=merged.get("rl_name"),
            replay_cfgs=merged.get("replay_cfgs"),
        )
        known = set(_ALIASES) | {
            "application", "execute_mode", "physics_backend", "device", "num_envs", "headless_mode",
            "enable_cameras", "use_fabric", "scene_backend", "task_backend", "scene_name", "robot_name",
            "task_name", "robot_scale", "gripper_closed_gap", "seed", "sources", "object_projects", "rl_name",
            "replay_cfgs",
        }
        extra = {key: value for key, value in merged.items() if key not in known}
        return ResolvedRunPlan(launch=launch, env=env, application=app, extra=extra)


def _default_execute_mode(application: str) -> str:
    return {
        "teleop": "TELEOP",
        "replay": "REPLAY_TELEOP",
        "eval": "EVAL",
        "rl": "TRAIN",
    }.get(application, "TELEOP")


def _as_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value
