# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from robofinals.core.checks.base_checker import BaseChecker


class GripperBlacklistCollisionChecker(BaseChecker):
    """Detect any contact between the robot and blacklisted scene objects.

    Each blacklisted object must already have a ContactSensor registered under
    the name ``{object_name}_contact`` in the Isaac Lab scene.

    Detection uses ``net_forces_w`` from each object's contact sensor, which is
    synchronous (current-frame) and does not require the ``contact_queues``
    two-frame handshake used by ``check_contact``.  Any contact force magnitude
    exceeding ``force_threshold`` on a blacklisted object is counted as a
    collision event.  A 50-frame cooldown prevents a single sustained contact
    from inflating the counter.

    Args:
        warning_on_screen: Surface warning text to the renderer.
        blacklist_object_names: Names of objects the gripper must not touch.
            Each name must have a corresponding ``{name}_contact`` sensor in
            the scene.
        force_threshold: Minimum net force magnitude (N) to count as a
            collision.  Default 0.5 N filters sensor noise.
        warmup_steps: Number of initial steps to skip before checking.
            PhysX emits large settlement forces (thousands of N) in the first
            few steps after reset while objects come to rest.  Default 10
            safely clears all initialization impulses (consistent with the
            warmup window used by TrajectorySmoothnessChecker).
        warning_display_frames: How many frames the collision warning text
            stays visible on screen after a collision event.  Independent of
            the de-bounce cooldown that prevents double-counting sustained
            contacts.  Default 50.
    """

    type = "gripper_blacklist_collision"

    # De-bounce window: a single sustained contact counts as one event.
    _COOLDOWN_FRAMES: int = 50

    def __init__(
        self,
        warning_on_screen: bool = False,
        blacklist_object_names: list[str] | None = None,
        force_threshold: float = 0.5,
        warmup_steps: int = 10,
        warning_display_frames: int = 50,
    ):
        super().__init__(warning_on_screen)
        self._blacklist: list[str] = blacklist_object_names or []
        self._force_threshold = force_threshold
        self._warmup_steps = max(1, int(warmup_steps))
        self._warning_display_frames = max(1, int(warning_display_frames))
        self._init_state()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        self._collision_count: int = 0
        # De-bounce cooldown: prevents a sustained contact from inflating the counter.
        self._cooldown: int = 0
        # Display cooldown: counts down after a collision so the warning stays
        # visible for warning_display_frames frames, independent of de-bounce.
        self._display_cooldown: int = 0
        self._last_warning: str = ""
        self._step_count: int = 0
        # Each entry: {"step": int, "objects": [str], "forces_N": [float]}
        self._collision_log: list[dict] = []

    def reset(self) -> None:
        self._init_state()

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    def _check(self, env) -> dict:
        return self._check_collision(env)

    def _check_collision(self, env) -> dict:
        self._step_count += 1

        # Skip the first warmup_steps steps: PhysX emits large settlement
        # forces while the scene comes to rest after reset, producing false
        # positives that are unrelated to robot behaviour.
        if self._step_count <= self._warmup_steps:
            return {
                "success": True,
                "warning_text": "",
                "metrics": {
                    "gripper_blacklist_collision_times": self._collision_count,
                    "gripper_blacklist_collision_log": self._collision_log,
                    "success": True,
                },
            }

        # --- De-bounce cooldown: prevents a sustained contact counting as multiple events ---
        if 0 < self._cooldown < self._COOLDOWN_FRAMES:
            self._cooldown += 1
        else:
            self._cooldown = 0

        # --- Display cooldown: keeps the warning text visible for warning_display_frames ---
        if self._display_cooldown > 0:
            self._display_cooldown -= 1
            if self._display_cooldown == 0:
                self._last_warning = ""

        collision_detected = False
        # List of (obj_name, force_N) for objects with force > threshold this frame
        hit_objects: list[tuple[str, float]] = []

        step = getattr(env, "common_step_counter", None)

        for obj_name in self._blacklist:
            sensor_key = f"{obj_name}_contact"
            if sensor_key not in env.scene.sensors:
                # Sensor not registered — skip silently (object absent from scene).
                continue

            # Use the backend-neutral runtime adapter. It also handles a
            # sensor invalidated during a stop/reset transition.
            force_mag = env.backend_runtime.contacts.force_magnitude(sensor_key)  # (num_envs,)

            f = force_mag[0].item()
            if f > self._force_threshold:
                collision_detected = True
                hit_objects.append((obj_name, f))

        if collision_detected and self._cooldown == 0:
            self._collision_count += 1
            self._cooldown += 1
            objects_str = " | ".join(
                f"<<{name}>> force={f:.2f}N" for name, f in hit_objects
            )
            self._last_warning = (
                f"collision [WARN] "
                f"[step={step}, count=#{self._collision_count}]: {objects_str}"
            )
            # Reset the display cooldown so the warning stays visible for the
            # configured number of frames.
            self._display_cooldown = self._warning_display_frames
            print(f"[COLLISION][gripper_blacklist] {self._last_warning}", flush=True)
            self._collision_log.append({
                "step": step,
                "objects": [name for name, _ in hit_objects],
                "forces_N": [round(f, 3) for _, f in hit_objects],
            })

        success = self._collision_count == 0

        return {
            "success": success,
            "warning_text": self._last_warning,
            "metrics": {
                "gripper_blacklist_collision_times": self._collision_count,
                "gripper_blacklist_collision_log": self._collision_log,
                "success": success,
            },
        }
