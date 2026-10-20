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


class FactoryObjDropChecker(BaseChecker):
    """Detect whether a grasped object is dropped outside all valid placement zones.

    An object is considered **grasped** once its Z position rises more than
    ``lift_threshold`` (0.10 m) above its resting Z at the end of warmup.
    This avoids reliance on EE-to-object distance, which can be unreliable when
    the ee_frame target points are far from the actual fingertip.

    Detection fires when all of the following are true:
        - **Was lifted**: the object was picked up at some point (Z rise > 0.10 m).
        - **Stable**: linear velocity magnitude < ``stable_vel_threshold``.
        - **Out of zone**: current position is outside all configured ``valid_zones``.

    A ``cooldown_frames``-long cooldown prevents a single sustained out-of-zone
    state from inflating the counter.

    The first ``warmup_steps`` environment steps are always skipped so that
    PhysX settlement impulses and initial object placement do not trigger false
    positives.

    Configuration example (factory_task1)::

        "factory_obj_drop": {
            "warning_on_screen": True,
            "watch_objects": {
                "Part001":    [{"x_range": [2.576, 2.840], "y_range": [33.200, 33.550], "z_range": [0.62, 0.92]}],
                "Part023_03": [{"x_range": [2.480, 2.790], "y_range": [32.450, 32.750], "z_range": [0.62, 0.92]}],
            },
            "stable_vel_threshold": 0.3,
            "gripper_far_threshold": 0.15,
            "warmup_steps": 20,
            "cooldown_frames": 50,
        }

    Configuration example (factory_task2)::

        "factory_obj_drop": {
            "warning_on_screen": True,
            "watch_objects": {
                "phillips_screwdriver":    [{"x_range": [12.72, 13.83], "y_range": [18.37, 18.63], "z_range": [0.30, 1.65]}],
                "phillips_screwdriver_02": [{"x_range": [12.72, 13.83], "y_range": [18.37, 18.63], "z_range": [0.30, 1.65]}],
            },
            "stable_vel_threshold": 0.3,
            "gripper_far_threshold": 0.15,
            "warmup_steps": 20,
            "cooldown_frames": 50,
        }

    Args:
        warning_on_screen:      Surface warning text to the renderer.
        watch_objects:          Dict mapping object name → list of valid-zone dicts.
                                Each zone dict must have ``x_range``, ``y_range``,
                                ``z_range`` keys (each a [min, max] list).
                                An object is considered "not dropped" if it falls
                                inside *any* of its zones.
        stable_vel_threshold:   Linear velocity (m/s) below which the object is
                                considered stationary.  Default 0.3.
        gripper_far_threshold:  Retained for API compatibility; currently unused.
                                The checker uses Z-lift detection instead of
                                EE-to-object distance.  Default 0.15.
        warmup_steps:           Steps to skip after reset.  Default 20.
        cooldown_frames:        Frames to suppress re-detection after a drop event
                                so one sustained drop counts as one event.  Default 50.
    """

    type = "factory_obj_drop"

    def __init__(
        self,
        warning_on_screen: bool = False,
        watch_objects: dict | None = None,
        stable_vel_threshold: float = 0.3,
        gripper_far_threshold: float = 0.15,
        warmup_steps: int = 20,
        cooldown_frames: int = 50,
    ):
        super().__init__(warning_on_screen)
        # watch_objects: {obj_name: [zone_dict, ...]}
        self._watch_objects: dict[str, list[dict]] = watch_objects or {}
        self._stable_vel_threshold = stable_vel_threshold
        self._gripper_far_threshold = gripper_far_threshold
        # Minimum Z rise above initial resting position to be considered "lifted".
        # 0.10 m is a conservative threshold: even a slight pick-up will exceed it.
        self._lift_threshold: float = 0.10
        self._warmup_steps = max(1, int(warmup_steps))
        self._cooldown_frames = max(1, int(cooldown_frames))
        self._init_state()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        self._step_count: int = 0
        self._drop_count: int = 0
        self._last_warning: str = ""
        # Per-object cooldown counter: {obj_name: remaining_frames}
        self._cooldowns: dict[str, int] = {}
        # Drop log: list of {step, object, pos, reason}
        self._drop_log: list[dict] = []
        # Per-object "was ever lifted" flag.
        # We detect "grasped" by observing the object rise significantly above
        # its initial (warmup-end) Z position.  This is more reliable than using
        # EE distance because the ee_frame target points can be centimetres away
        # from the actual fingertip even when firmly grasping an object.
        self._was_grasped: dict[str, bool] = {}
        # Initial Z recorded at end of warmup for each object, used as baseline.
        self._init_z: dict[str, float] = {}
        # Latest known world position for each watched object: {obj_name: [x, y, z]}
        self._last_pos: dict[str, list[float]] = {}
        # Objects whose initial position is clearly out of the scene (|z| > 5 m).
        # These are skipped entirely — their HDF5 initial_state data is invalid.
        self._invalid_init: dict[str, bool] = {}

    def reset(self) -> None:
        self._init_state()

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    def _check(self, env) -> dict:
        self._step_count += 1

        if self._step_count <= self._warmup_steps:
            return self._build_result()

        # Tick down all active cooldowns.
        for name in list(self._cooldowns):
            self._cooldowns[name] -= 1
            if self._cooldowns[name] <= 0:
                del self._cooldowns[name]

        step = getattr(env, "common_step_counter", self._step_count)

        for obj_name, zones in self._watch_objects.items():
            # Retrieve object from rigid_objects; skip silently if not present.
            rigid_objects = getattr(env.scene, "rigid_objects", {})
            if obj_name not in rigid_objects:
                continue

            obj = rigid_objects[obj_name]
            pos = obj.data.body_com_pos_w[:, 0, :]   # (num_envs, 3)
            vel = obj.data.body_com_vel_w[:, 0, :3]  # (num_envs, 3)

            obj_z = pos[0, 2].item()

            # Always update the latest known position for final-state reporting.
            self._last_pos[obj_name] = [
                round(pos[0, 0].item(), 4),
                round(pos[0, 1].item(), 4),
                round(obj_z, 4),
            ]

            # --- Record initial resting Z at end of warmup (first real step) ---
            if obj_name not in self._init_z:
                self._init_z[obj_name] = obj_z
                # Flag objects whose initial state is clearly invalid (e.g. launched
                # out of the scene by a bad HDF5 initial_state).  A resting object on
                # any workbench/shelf in the scene is always within a few metres of the
                # floor; anything above 5 m means the recorded pose is garbage.
                _invalid = abs(obj_z) > 5.0
                self._invalid_init[obj_name] = _invalid
                tag = " [INVALID_INIT - will be ignored]" if _invalid else ""
                print(
                    f"[DROP_INIT] {obj_name} init_pos="
                    f"[{pos[0, 0].item():.4f}, {pos[0, 1].item():.4f}, {obj_z:.4f}]{tag}",
                    flush=True,
                )

            # Skip objects whose initial position was detected as invalid.
            if self._invalid_init.get(obj_name, False):
                continue

            # --- Grasp detection: object is considered "ever grasped" when it
            #     rises more than lift_threshold above its initial resting Z.
            #     This is robust to large EE-to-object offsets in ee_frame. ---
            if not self._was_grasped.get(obj_name, False):
                if obj_z > self._init_z[obj_name] + self._lift_threshold:
                    self._was_grasped[obj_name] = True
                    print(
                        f"[DROP] {obj_name} was_grasped=True at step={step} "
                        f"z={obj_z:.3f} (init_z={self._init_z[obj_name]:.3f})",
                        flush=True,
                    )

            # --- Periodic position log every 50 steps ---
            if self._step_count % 50 == 0:
                print(
                    f"[DROP_POS][{obj_name}] step={step} "
                    f"pos=[{pos[0, 0].item():.4f}, {pos[0, 1].item():.4f}, {obj_z:.4f}] "
                    f"speed={torch.norm(vel[0]).item():.4f} "
                    f"was_grasped={self._was_grasped.get(obj_name, False)}",
                    flush=True,
                )

            # Only evaluate drop for objects that have been lifted at some point.
            if not self._was_grasped.get(obj_name, False):
                continue

            # Skip if still in cooldown from a previous drop event.
            if obj_name in self._cooldowns:
                continue

            # --- Stable / settled check ---
            # We cannot reliably use EE-to-object distance because the ee_frame
            # target points can be far from the fingertip even when grasping.
            # Instead we consider an object "released and settled" when:
            #   1. Its linear velocity is below the stable threshold.
            #   2. Its Z has descended back near the initial resting height,
            #      meaning it has landed (not still being carried aloft).
            #      We allow up to lift_threshold above init_z — objects placed
            #      into containers sit slightly higher than their original floor
            #      position, so we give a 1× lift_threshold margin.
            speed = torch.norm(vel[0]).item()
            is_stable = speed < self._stable_vel_threshold

            init_z = self._init_z.get(obj_name, obj_z)
            has_landed = obj_z < init_z + self._lift_threshold

            if not (is_stable and has_landed):
                # Object is still being carried (too high) or moving — skip.
                continue

            # --- Valid-zone check ---
            obj_x = pos[0, 0].item()
            obj_y = pos[0, 1].item()
            # obj_z is already defined above from pos[0, 2]

            in_valid_zone = False
            for zone in zones:
                x_ok = zone["x_range"][0] <= obj_x <= zone["x_range"][1]
                y_ok = zone["y_range"][0] <= obj_y <= zone["y_range"][1]
                z_ok = zone["z_range"][0] <= obj_z <= zone["z_range"][1]
                if x_ok and y_ok and z_ok:
                    in_valid_zone = True
                    break

            if not in_valid_zone:
                self._drop_count += 1
                self._cooldowns[obj_name] = self._cooldown_frames

                pos_str = f"[{obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f}]"
                self._last_warning = (
                    f"factory_obj_drop Warning "
                    f"[step={step}, count=#{self._drop_count}]: "
                    f"<<{obj_name}>> released outside valid zone at {pos_str}"
                )
                print(f"[DROP][factory_obj_drop] {self._last_warning}", flush=True)

                self._drop_log.append({
                    "step": step,
                    "object": obj_name,
                    "pos": [round(obj_x, 4), round(obj_y, 4), round(obj_z, 4)],
                    "reason": "not_in_valid_zone",
                })

        return self._build_result()

    def _build_result(self) -> dict:
        success = self._drop_count == 0

        # Build per-object final placement summary.
        final_states: list[dict] = []
        # Also accumulate compact per-object tokens for the real-time status line.
        obj_tokens: list[str] = []

        for obj_name, zones in self._watch_objects.items():
            last_pos = self._last_pos.get(obj_name)
            was_grasped = self._was_grasped.get(obj_name, False)

            if last_pos is None:
                # Object not yet seen in scene (env not stepped or not present).
                final_states.append({
                    "object": obj_name,
                    "was_grasped": was_grasped,
                    "final_pos": None,
                    "valid_zones": zones,
                    "in_valid_zone": None,
                })
                obj_tokens.append(f"{obj_name}[G=?,ok=?]")
                continue

            obj_x, obj_y, obj_z = last_pos
            in_valid_zone = False
            matched_zone: dict | None = None
            for zone in zones:
                x_ok = zone["x_range"][0] <= obj_x <= zone["x_range"][1]
                y_ok = zone["y_range"][0] <= obj_y <= zone["y_range"][1]
                z_ok = zone["z_range"][0] <= obj_z <= zone["z_range"][1]
                if x_ok and y_ok and z_ok:
                    in_valid_zone = True
                    matched_zone = zone
                    break

            entry: dict = {
                "object": obj_name,
                "was_grasped": was_grasped,
                "final_pos": last_pos,
                "valid_zones": zones,
                "in_valid_zone": in_valid_zone,
            }
            if matched_zone is not None:
                entry["matched_zone"] = matched_zone
            final_states.append(entry)

            g_flag = "T" if was_grasped else "F"
            ok_flag = "T" if in_valid_zone else ("F" if was_grasped else "?")
            obj_tokens.append(f"{obj_name}[G={g_flag},ok={ok_flag}]")

        # Real-time status line shown every frame when warning_on_screen=True.
        # If a drop event warning is active (_last_warning non-empty), show that
        # instead so the alert is not masked by the status line.
        if self._last_warning:
            warning_text = self._last_warning
        else:
            status = " ".join(obj_tokens)
            warning_text = f"drop | cnt={self._drop_count} {status}"

        return {
            "success": success,
            "warning_text": warning_text,
            "metrics": {
                "factory_obj_drop_count": self._drop_count,
                "factory_obj_drop_log": self._drop_log,
                "final_placement": final_states,
                "success": success,
            },
        }
