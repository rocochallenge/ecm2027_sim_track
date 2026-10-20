"""Immutable teleoperation-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class TeleopSpec:
    device_name: str
    robot_name: str
    sensitivity: float = 1.0
    relative_control: bool = False
    headless: bool = False
    step_hz: float = 50.0
    enable_recording: bool = False
    enable_cameras: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.device_name.strip():
            raise ValueError("device_name must not be empty")
        if not self.robot_name.strip():
            raise ValueError("robot_name must not be empty")
        if self.sensitivity <= 0:
            raise ValueError("sensitivity must be greater than zero")
        if self.step_hz <= 0:
            raise ValueError("step_hz must be greater than zero")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TeleopSpec":
        """Build the immutable teleop contract from YAML/CLI-shaped values."""

        data = dict(values)
        known = {
            "device_name",
            "robot_name",
            "sensitivity",
            "relative_control",
            "headless",
            "step_hz",
            "enable_recording",
            "enable_cameras",
            "options",
        }
        options = dict(data.pop("options", {}))
        options.update({key: value for key, value in data.items() if key not in known})
        data["options"] = MappingProxyType(options)
        return cls(**{key: data[key] for key in known if key in data})

    @classmethod
    def from_args(cls, args: Any) -> "TeleopSpec":
        """Translate the legacy argparse namespace at the composition boundary."""

        device_name = getattr(args, "teleop_device", None) or "keyboard"
        robot_name = getattr(args, "robot", None) or "runtime"
        return cls.from_mapping(
            {
                "device_name": str(device_name),
                "robot_name": str(robot_name),
                "sensitivity": float(getattr(args, "sensitivity", 1.0) or 1.0),
                "relative_control": bool(getattr(args, "relative_control", False)),
                "headless": bool(getattr(args, "headless", False)),
                "step_hz": float(getattr(args, "step_hz", 50.0) or 50.0),
                "enable_recording": bool(
                    getattr(args, "record", False) or getattr(args, "record_lerobot", False)
                ),
                "enable_cameras": bool(getattr(args, "enable_cameras", False)),
                "options": {
                    "physics_backend": getattr(args, "physics_backend", None),
                    "teleop_auto_start": bool(getattr(args, "teleop_auto_start", False)),
                    "teleop_auto_commit": bool(getattr(args, "teleop_auto_commit", False)),
                },
            }
        )
