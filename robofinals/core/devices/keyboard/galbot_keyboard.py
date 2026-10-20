"""Keyboard teleoperation for Galbot's dual absolute-pose arm interface."""

from __future__ import annotations

import weakref
from collections.abc import Callable

import carb
import numpy as np
import omni
import torch
from isaaclab.devices.device_base import DeviceBase
from scipy.spatial.transform import Rotation


class GalbotKeyboard(DeviceBase):
    """Keyboard teleoperation for one Galbot hand at a time.

    DiffIK absolute mode consumes poses in the robot base frame. Capturing the
    current end-effector poses on the first action prevents a jump when
    teleoperation starts.
    """

    # The session gate starts closed.  Continue polling this device so the B
    # callback can open it, but do not compute IK until that callback arrives.
    poll_when_inactive = True

    # Do NOT preserve started state across resets - always require B press after reset
    preserve_started_on_reset = False

    def __init__(self, env, pos_sensitivity: float = 0.02, rot_sensitivity: float = 0.08):
        self.env = env
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        self._callbacks: dict[str, Callable[[], None]] = {}
        self.started = False
        self._active_arm = "right"
        self._close_gripper = {"left": False, "right": False}
        self._pressed_keys: set[str] = set()
        self._poses: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._pose_capture_failed = False  # Track if we've already warned about capture failure

        # Print controls on initialization
        self._print_controls()

    def _print_controls(self):
        """Print keyboard control instructions."""
        print("\n" + "="*60)
        print("🎮 Galbot Keyboard Teleop Controls")
        print("="*60)
        print("\n📍 Position Control (Left Hand):")
        print("  W/S - Forward/Backward")
        print("  A/D - Left/Right")
        print("  Q/E - Up/Down")
        print("\n🔄 Rotation Control (Right Hand):")
        print("  U/J - Roll (X-axis)")
        print("  I/K - Pitch (Y-axis)")
        print("  O/L - Yaw (Z-axis)")
        print("\n🤖 Robot Control:")
        print("  Tab - Switch between Left/Right arm")
        print("  G   - Toggle gripper (open/close)")
        print("  B   - Start teleoperation")
        print("="*60 + "\n")

    def __del__(self):
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __str__(self) -> str:
        return (
            "Galbot dual-hand keyboard controller\n"
            "  B: start teleoperation; Tab: switch active hand; G: toggle active gripper\n"
            "  W/S: forward/backward; A/D: left/right; Q/E: up/down\n"
            "  U/J: rotate X; I/K: rotate Y; O/L: rotate Z\n"
        )

    def add_callback(self, key: str, func: Callable[[], None]):
        self._callbacks[key] = func

    def reset(self):
        # Reset device state completely - user needs to press B again to restart
        self.started = False
        self._pressed_keys.clear()
        self._poses = None
        self._pose_capture_failed = False
        print("[galbot_keyboard] Device reset. Press B to restart teleoperation.")

    @staticmethod
    def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
        return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)

    def _capture_current_poses(self) -> None:
        # Prefer ee_frame (base-relative TCP). Fall back to arm_link7 body poses.
        try:
            if not hasattr(self.env, "scene") or self.env.scene is None:
                if not self._pose_capture_failed:
                    print("[galbot_keyboard] Warning: env.scene not available, waiting for scene to be ready...")
                    self._pose_capture_failed = True
                return

            ee_frame = None
            if hasattr(self.env.scene, "sensors") and "ee_frame" in self.env.scene.sensors:
                ee_frame = self.env.scene.sensors["ee_frame"]
            elif "ee_frame" in getattr(self.env.scene, "keys", lambda: [])():
                ee_frame = self.env.scene["ee_frame"]
            else:
                try:
                    ee_frame = self.env.scene["ee_frame"]
                except (KeyError, TypeError, AttributeError):
                    ee_frame = None

            self._poses = {}
            if ee_frame is not None and hasattr(ee_frame.data, "target_pos_source"):
                pos_b = ee_frame.data.target_pos_source[0].detach().cpu().numpy()
                quat_wxyz = ee_frame.data.target_quat_source[0].detach().cpu().numpy()
                for index, side in enumerate(("left", "right")):
                    self._poses[side] = (
                        pos_b[index].copy(),
                        self._quat_wxyz_to_xyzw(quat_wxyz[index]),
                    )
            else:
                if not hasattr(self.env.scene, "articulations") or "robot" not in self.env.scene.articulations:
                    if not self._pose_capture_failed:
                        print("[galbot_keyboard] Warning: robot articulation not found in scene")
                        self._pose_capture_failed = True
                    self._poses = None
                    return
                robot = self.env.scene.articulations["robot"]
                body_ids = robot.find_bodies(["left_arm_link7", "right_arm_link7"])[0]
                root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
                root_quat_wxyz = robot.data.root_quat_w[0].detach().cpu().numpy()
                root_rot = Rotation.from_quat(self._quat_wxyz_to_xyzw(root_quat_wxyz))
                # Match DiffIK body_offset: pos + 180deg about Y in link frame.
                offset_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                offset_rot = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])  # xyzw identity
                for index, side in enumerate(("left", "right")):
                    body_pos = robot.data.body_pos_w[0, body_ids[index]].detach().cpu().numpy()
                    body_quat_wxyz = robot.data.body_quat_w[0, body_ids[index]].detach().cpu().numpy()
                    body_rot = Rotation.from_quat(self._quat_wxyz_to_xyzw(body_quat_wxyz))
                    tcp_pos_w = body_pos + body_rot.apply(offset_pos)
                    tcp_rot_w = body_rot * offset_rot
                    pos_b = root_rot.inv().apply(tcp_pos_w - root_pos)
                    quat_b = (root_rot.inv() * tcp_rot_w).as_quat()
                    self._poses[side] = (pos_b.copy(), quat_b.copy())

            self._pose_capture_failed = False
        except (AttributeError, KeyError, RuntimeError, IndexError, ValueError) as e:
            if not self._pose_capture_failed:
                print(f"[galbot_keyboard] Warning: Failed to capture EE poses: {e}")
                self._pose_capture_failed = True
            self._poses = None

    def advance(self) -> torch.Tensor:
        if not self.started:
            return False
        if self._poses is None:
            self._capture_current_poses()
        # If pose capture failed (e.g., after scene reset), return False to wait
        if self._poses is None:
            return False
        position, orientation = self._poses[self._active_arm]
        delta_pos = sum((_POSITION_KEYS[key] for key in self._pressed_keys if key in _POSITION_KEYS), start=np.zeros(3))
        delta_rot = sum((_ROTATION_KEYS[key] for key in self._pressed_keys if key in _ROTATION_KEYS), start=np.zeros(3))
        position = position + delta_pos * self.pos_sensitivity
        orientation = (
            Rotation.from_quat(orientation)
            * Rotation.from_euler("XYZ", delta_rot * self.rot_sensitivity)
        ).as_quat()
        self._poses[self._active_arm] = (position, orientation)

        action = {}
        for side in ("left", "right"):
            pos, quat_xyzw = self._poses[side]
            action[f"{side}_arm_abs"] = torch.tensor(
                np.concatenate([pos, quat_xyzw]), device=self.env.device, dtype=torch.float32
            )
            action[f"{side}_gripper"] = torch.tensor(
                1.0 if self._close_gripper[side] else -1.0, device=self.env.device
            )
        return self.env.cfg.isaaclab_arena_env.embodiment.preprocess_device_action(action, self)

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        input_value = getattr(event, "input", "")
        key = str(getattr(input_value, "name", input_value)).upper()
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed_keys.discard(key)
            return True
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if key == "TAB":
            self._active_arm = "left" if self._active_arm == "right" else "right"
            print(f"[teleop] Galbot keyboard active hand: {self._active_arm}", flush=True)
        elif key == "G":
            self._close_gripper[self._active_arm] = not self._close_gripper[self._active_arm]
            state = "closed" if self._close_gripper[self._active_arm] else "open"
            print(f"[teleop] Galbot {self._active_arm} gripper: {state}", flush=True)
        elif key in _POSITION_KEYS or key in _ROTATION_KEYS:
            self._pressed_keys.add(key)
        if key in self._callbacks:
            self._callbacks[key]()
        return True


_POSITION_KEYS = {
    # With the configured base yaw, local +X points from the robot to the table.
    "W": np.array([1.0, 0.0, 0.0]), "S": np.array([-1.0, 0.0, 0.0]),
    "A": np.array([0.0, 1.0, 0.0]), "D": np.array([0.0, -1.0, 0.0]),
    "Q": np.array([0.0, 0.0, 1.0]), "E": np.array([0.0, 0.0, -1.0]),
}
_ROTATION_KEYS = {
    # 使用数字小键盘区域，避免冲突
    "U": np.array([1.0, 0.0, 0.0]), "J": np.array([-1.0, 0.0, 0.0]),  # X轴旋转 (roll)
    "I": np.array([0.0, 1.0, 0.0]), "K": np.array([0.0, -1.0, 0.0]),  # Y轴旋转 (pitch)
    "O": np.array([0.0, 0.0, 1.0]), "L": np.array([0.0, 0.0, -1.0]),  # Z轴旋转 (yaw)
}
