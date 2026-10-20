"""Pluggable construction of legacy IsaacLab teleoperation devices."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class DeviceContext:
    env: Any
    args: Any
    callbacks: Mapping[str, Callable[[], None]]


@dataclass
class TeleopDeviceBinding:
    interface: Any
    active: bool
    callbacks: Mapping[str, Callable[[], None]]


class DeviceFactory(Protocol):
    def __call__(self, context: DeviceContext) -> TeleopDeviceBinding: ...


class DeviceRegistry:
    """Explicit registry; concrete devices are imported only when selected."""

    def __init__(self) -> None:
        self._factories: dict[str, DeviceFactory] = {}

    def register(self, name: str, factory: DeviceFactory, *, replace: bool = False) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("device name must not be empty")
        if key in self._factories and not replace:
            raise ValueError(f"device {name!r} is already registered")
        self._factories[key] = factory

    def create(self, name: str, context: DeviceContext) -> TeleopDeviceBinding:
        key = name.strip().lower()
        try:
            factory = self._factories[key]
        except KeyError as exc:
            supported = ", ".join(sorted(self._factories))
            raise ValueError(f"Unknown teleop device {name!r}; registered devices: {supported}") from exc
        return factory(context)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def register_embodiment_devices(self, names: Any) -> None:
        """Register IsaacLab/Arena-provided device names through one adapter."""

        for name in tuple(names or ()):
            if str(name).strip().lower() not in self._factories:
                self.register(str(name), _arena_factory)


def _binding(
    interface: Any,
    context: DeviceContext,
    *,
    active: bool,
    register_callbacks: bool = True,
) -> TeleopDeviceBinding:
    if register_callbacks:
        for key, callback in context.callbacks.items():
            add_callback = getattr(interface, "add_callback", None)
            if callable(add_callback):
                try:
                    add_callback(key, callback)
                except (ValueError, TypeError):
                    # IsaacLab devices reject unknown keys; the device remains usable.
                    pass
    return TeleopDeviceBinding(interface, active, context.callbacks)


def _arena_factory(context: DeviceContext) -> TeleopDeviceBinding:
    from isaaclab.devices.teleop_device_factory import create_teleop_device
    from robofinals.utils.teleop_utils import get_joint_pos_offset

    env_cfg = context.env.cfg
    args = context.args
    interface = create_teleop_device(
        context.env,
        args.teleop_device,
        env_cfg.teleop_devices.devices,
        dict(context.callbacks),
    )
    if hasattr(interface, "init_internal_state"):
        interface.init_internal_state(
            get_joint_pos_offset(context.env),
            context.env.scene.articulations["robot"].data.joint_names,
        )
    return _binding(interface, context, active=False, register_callbacks=False)


def _keyboard_factory(context: DeviceContext) -> TeleopDeviceBinding:
    from robofinals.core.devices import GalbotKeyboard, KEYCONTROLLER_MAP

    args = context.args
    robot_family = args.robot.lower().split("-")[0]
    if robot_family == "galbot":
        interface = GalbotKeyboard(
            context.env,
            pos_sensitivity=0.02 * args.sensitivity,
            rot_sensitivity=0.08 * args.sensitivity,
        )
    else:
        device_type = KEYCONTROLLER_MAP[args.teleop_device.lower() + "-" + robot_family]
        interface = device_type(
            pos_sensitivity=0.05 * args.sensitivity,
            rot_sensitivity=0.05 * args.sensitivity,
            base_sensitivity=0.5 * args.sensitivity,
            base_yaw_sensitivity=0.8 * args.sensitivity,
        )
        interface.env = context.env
    return _binding(interface, context, active=False)


def _spacemouse_factory(context: DeviceContext) -> TeleopDeviceBinding:
    from isaaclab.devices import Se3SpaceMouse

    args = context.args
    interface = Se3SpaceMouse(
        context.env,
        pos_sensitivity=0.1 * args.sensitivity,
        rot_sensitivity=0.2 * args.sensitivity,
    )
    interface.env = context.env
    return _binding(interface, context, active=False)












def _vr_factory(context: DeviceContext) -> TeleopDeviceBinding:
    import numpy as np
    from multiprocessing import shared_memory

    from robofinals.core.devices import VRController

    args = context.args
    image_size = (720, 1280)
    shm = shared_memory.SharedMemory(create=True, size=image_size[0] * image_size[1] * 3 * np.uint8().itemsize)
    device_type = {
        "vr-controller": VRController,
    }[args.teleop_device.lower()]
    interface = device_type(
        context.env,
        img_shape=image_size,
        shm_name=shm.name,
        relative_control=args.relative_control,
    )
    setattr(interface, "_robofinals_shared_memory", shm)
    return _binding(interface, context, active=True)




def build_default_device_registry() -> DeviceRegistry:
    registry = DeviceRegistry()
    registry.register("keyboard", _keyboard_factory)
    registry.register("spacemouse", _spacemouse_factory)
    registry.register("vr-controller", _vr_factory)
    return registry
