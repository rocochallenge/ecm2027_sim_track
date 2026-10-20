from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class AssetRequirement:
    asset_id: str
    required: bool = True
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    joints: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    action_schema: Mapping[str, object] = field(default_factory=dict)
    observation_schema: Mapping[str, object] = field(default_factory=dict)
    sensors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    assets: tuple[AssetRequirement, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    required_sensors: tuple[str, ...] = ()
    required_capabilities: frozenset[str] = frozenset()
