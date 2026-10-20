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

import math

import torch

from robofinals.core.checks.base_checker import BaseChecker


class PathEfficiencyChecker(BaseChecker):
    """
    Path efficiency checker that accumulates motion distances over an episode.

    Tracks up to five independent path length metrics. Each can be toggled independently:

    End-effector (EE) metrics:
        ee_cartesian_path_length   Accumulated L2 displacement of EE position (m).
        ee_orientation_path_length Accumulated angular displacement of EE orientation (rad),
                                   computed as 2 * acos(|q1 · q2|) between consecutive frames.

    Arm metrics:
        joint_path_length          Accumulated L2 norm of arm joint angle changes (rad).

    Mobile base metrics (only when base_link_name is set; for mobile manipulators like X7S in Task2):
        base_translation_path_length  Accumulated XY displacement of the robot base (m).
        base_yaw_path_length          Accumulated absolute yaw change of the base (rad),
                                      with wrap-around handling across the ±π boundary.
        When base_link_name is None, base metrics use articulation root (often fixed); when set,
        the given body link pose is used so that chassis motion is recorded correctly.

    Optional baseline comparison:
        For each tracked metric a corresponding baseline parameter can be provided.
        When set, an extra ratio output is added:
            <metric>_ratio = actual / baseline  (1.0 = identical, >1 = longer, <1 = shorter)
        Example: ee_cartesian_path_length_ratio = actual_ee_path / baseline_ee_path

    All metrics are statistics-only; success always returns True.

    Adapted from RoboEval metric_rollout.py for Isaac Lab's tensor-parallel architecture.

    Args:
        ee_link_name:              End-effector body link name. X7S right arm: "right_hand_link".
        track_orientation:         Track EE orientation path. Default True.
        track_joint:               Track arm joint angle path. Default True.
        track_base_translation:    Track base XY translation. Default False.
        track_base_rotation:       Track base yaw rotation. Default False.
        base_link_name:            Optional body name for the mobile base (e.g. "base_link" or "world").
                                   When set, base path is computed from this body's pose; when None,
                                   root pose is used (may be fixed for some robots).
        baseline_ee_path:          Optional reference ee_cartesian_path_length (m). Enables
                                   ee_cartesian_path_length_ratio output when set.
        baseline_orientation_path: Optional reference ee_orientation_path_length (rad).
        baseline_joint_path:       Optional reference joint_path_length (rad).
        baseline_base_translation: Optional reference base_translation_path_length (m).
        baseline_base_rotation:    Optional reference base_yaw_path_length (rad).
    """

    type = "path_efficiency"

    def __init__(
        self,
        warning_on_screen=False,
        ee_link_name="right_hand_link",
        track_orientation=True,
        track_joint=True,
        track_base_translation=False,
        track_base_rotation=False,
        base_link_name=None,
        baseline_ee_path=None,
        baseline_orientation_path=None,
        baseline_joint_path=None,
        baseline_base_translation=None,
        baseline_base_rotation=None,
    ):
        super().__init__(warning_on_screen)
        self.ee_link_name = ee_link_name
        self.track_orientation = track_orientation
        self.track_joint = track_joint
        self.track_base_translation = track_base_translation
        self.track_base_rotation = track_base_rotation
        self.base_link_name = base_link_name

        self.baseline_ee_path = baseline_ee_path
        self.baseline_orientation_path = baseline_orientation_path
        self.baseline_joint_path = baseline_joint_path
        self.baseline_base_translation = baseline_base_translation
        self.baseline_base_rotation = baseline_base_rotation

        self._is_initialized = False
        self._ee_idx = None
        self._base_link_idx = None

    def _init_buffers(self, num_envs, num_joints, device):
        # EE Cartesian path
        self._prev_ee_pos = torch.zeros((num_envs, 3), device=device)
        self._ee_cart_path = torch.zeros(num_envs, device=device)

        # EE orientation path (quaternion angular distance)
        if self.track_orientation:
            self._prev_ee_quat = torch.zeros((num_envs, 4), device=device)
            self._ee_orient_path = torch.zeros(num_envs, device=device)

        # Arm joint path
        if self.track_joint:
            self._prev_joint_pos = torch.zeros((num_envs, num_joints), device=device)
            self._joint_path = torch.zeros(num_envs, device=device)

        # Mobile base XY translation path
        if self.track_base_translation:
            self._prev_base_xy = torch.zeros((num_envs, 2), device=device)
            self._base_trans_path = torch.zeros(num_envs, device=device)

        # Mobile base yaw rotation path
        if self.track_base_rotation:
            self._prev_base_yaw = torch.zeros(num_envs, device=device)
            self._base_yaw_path = torch.zeros(num_envs, device=device)

        self._step_counts = torch.zeros(num_envs, device=device)
        self._is_initialized = True

    @staticmethod
    def _quat_to_yaw(quat_wxyz):
        """
        Convert quaternion (w, x, y, z) to yaw (rotation around Z axis).

        Isaac Lab quaternions use (w, x, y, z) convention.
        """
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)  # (num_envs,)

    def reset(self, env_ids=None):
        if not self._is_initialized:
            return
        if env_ids is None:
            self._prev_ee_pos.zero_()
            self._ee_cart_path.zero_()
            if self.track_orientation:
                self._prev_ee_quat.zero_()
                self._ee_orient_path.zero_()
            if self.track_joint:
                self._prev_joint_pos.zero_()
                self._joint_path.zero_()
            if self.track_base_translation:
                self._prev_base_xy.zero_()
                self._base_trans_path.zero_()
            if self.track_base_rotation:
                self._prev_base_yaw.zero_()
                self._base_yaw_path.zero_()
            self._step_counts.zero_()
        else:
            self._prev_ee_pos[env_ids] = 0
            self._ee_cart_path[env_ids] = 0
            if self.track_orientation:
                self._prev_ee_quat[env_ids] = 0
                self._ee_orient_path[env_ids] = 0
            if self.track_joint:
                self._prev_joint_pos[env_ids] = 0
                self._joint_path[env_ids] = 0
            if self.track_base_translation:
                self._prev_base_xy[env_ids] = 0
                self._base_trans_path[env_ids] = 0
            if self.track_base_rotation:
                self._prev_base_yaw[env_ids] = 0
                self._base_yaw_path[env_ids] = 0
            self._step_counts[env_ids] = 0

    def _get_ee_idx(self, robot):
        if self._ee_idx is None:
            ids, _ = robot.find_bodies(self.ee_link_name)
            if len(ids) == 0:
                raise ValueError(
                    f"[PathEfficiencyChecker] Cannot find body '{self.ee_link_name}'. "
                    f"Available: {robot.data.body_names}"
                )
            self._ee_idx = ids[0]
        return self._ee_idx

    def _get_base_link_idx(self, robot):
        if self._base_link_idx is None:
            if not self.base_link_name:
                return None
            ids, _ = robot.find_bodies(self.base_link_name)
            if len(ids) == 0:
                raise ValueError(
                    f"[PathEfficiencyChecker] Cannot find base body '{self.base_link_name}'. "
                    f"Available: {robot.data.body_names}"
                )
            self._base_link_idx = ids[0]
        return self._base_link_idx

    def _check(self, env):
        robot = env.scene["robot"]
        num_envs = env.num_envs
        device = robot.device

        cur_joint_pos = robot.data.joint_pos  # (num_envs, num_joints)
        num_joints = cur_joint_pos.shape[1]

        if not self._is_initialized:
            self._init_buffers(num_envs, num_joints, device)

        ee_idx = self._get_ee_idx(robot)
        ee_state = robot.data.body_state_w[:, ee_idx, :]
        cur_ee_pos = ee_state[:, 0:3]   # (num_envs, 3)
        cur_ee_quat = ee_state[:, 3:7]  # (num_envs, 4) in (w, x, y, z) convention

        valid = self._step_counts > 0

        # --- EE Cartesian path ---
        ee_cart_delta = torch.norm(cur_ee_pos - self._prev_ee_pos, dim=-1)  # (num_envs,)
        self._ee_cart_path[valid] += ee_cart_delta[valid]

        # --- EE orientation path ---
        if self.track_orientation:
            # Angular distance between consecutive quaternions: 2 * acos(|q1 · q2|)
            dot = (cur_ee_quat * self._prev_ee_quat).sum(dim=-1).clamp(-1.0, 1.0)
            orient_delta = 2.0 * torch.acos(dot.abs())
            self._ee_orient_path[valid] += orient_delta[valid]

        # --- Joint path ---
        if self.track_joint:
            joint_delta = torch.norm(cur_joint_pos - self._prev_joint_pos, dim=-1)
            self._joint_path[valid] += joint_delta[valid]

        # --- Mobile base: use base_link body pose when base_link_name is set, else root ---
        base_idx = self._get_base_link_idx(robot) if self.base_link_name else None
        if self.track_base_translation:
            if base_idx is not None:
                base_state = robot.data.body_state_w[:, base_idx, :]
                cur_base_xy = base_state[:, 0:2]  # (num_envs, 2)
            else:
                cur_base_xy = robot.data.root_pos_w[:, 0:2]  # (num_envs, 2)
            base_trans_delta = torch.norm(cur_base_xy - self._prev_base_xy, dim=-1)
            self._base_trans_path[valid] += base_trans_delta[valid]

        if self.track_base_rotation:
            if base_idx is not None:
                base_state = robot.data.body_state_w[:, base_idx, :]
                cur_base_quat = base_state[:, 3:7]  # (num_envs, 4) wxyz
            else:
                cur_base_quat = robot.data.root_quat_w  # (num_envs, 4) in (w, x, y, z)
            cur_yaw = self._quat_to_yaw(cur_base_quat)  # (num_envs,)
            # Wrap-around-aware yaw difference: map to [-pi, pi]
            yaw_diff = cur_yaw - self._prev_base_yaw
            yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
            self._base_yaw_path[valid] += yaw_diff.abs()[valid]

        # Update previous state
        self._prev_ee_pos = cur_ee_pos.clone()
        if self.track_orientation:
            self._prev_ee_quat = cur_ee_quat.clone()
        if self.track_joint:
            self._prev_joint_pos = cur_joint_pos.clone()
        if self.track_base_translation:
            if base_idx is not None:
                self._prev_base_xy = robot.data.body_state_w[:, base_idx, 0:2].clone()
            else:
                self._prev_base_xy = robot.data.root_pos_w[:, 0:2].clone()
        if self.track_base_rotation:
            if base_idx is not None:
                cur_base_quat = robot.data.body_state_w[:, base_idx, 3:7]
            else:
                cur_base_quat = robot.data.root_quat_w
            self._prev_base_yaw = self._quat_to_yaw(cur_base_quat).clone()

        self._step_counts += 1

        # --- Build metrics output for env[0] ---
        ee_cart_val = round(self._ee_cart_path[0].item(), 4)
        metrics = {
            "ee_cartesian_path_length": ee_cart_val,
        }

        if self.track_orientation:
            orient_val = round(self._ee_orient_path[0].item(), 4)
            metrics["ee_orientation_path_length"] = orient_val

        if self.track_joint:
            joint_val = round(self._joint_path[0].item(), 4)
            metrics["joint_path_length"] = joint_val

        if self.track_base_translation:
            base_t_val = round(self._base_trans_path[0].item(), 4)
            metrics["base_translation_path_length"] = base_t_val

        if self.track_base_rotation:
            base_r_val = round(self._base_yaw_path[0].item(), 4)
            metrics["base_yaw_path_length"] = base_r_val

        # --- Optional baseline ratio outputs ---
        # ratio = actual / baseline; 1.0 = same as reference, >1 = longer (less efficient).
        def _ratio(actual, baseline, key):
            if baseline is not None and baseline > 0:
                metrics[key] = round(actual / baseline, 4)

        _ratio(ee_cart_val, self.baseline_ee_path, "ee_cartesian_path_length_ratio")

        if self.track_orientation:
            _ratio(metrics.get("ee_orientation_path_length", 0), self.baseline_orientation_path, "ee_orientation_path_length_ratio")

        if self.track_joint:
            _ratio(metrics.get("joint_path_length", 0), self.baseline_joint_path, "joint_path_length_ratio")

        if self.track_base_translation:
            _ratio(metrics.get("base_translation_path_length", 0), self.baseline_base_translation, "base_translation_path_length_ratio")

        if self.track_base_rotation:
            _ratio(metrics.get("base_yaw_path_length", 0), self.baseline_base_rotation, "base_yaw_path_length_ratio")

        # Build a compact real-time status line so operators can monitor path
        # accumulation on screen each frame when warning_on_screen=True.
        parts = [f"path | ee={ee_cart_val:.3f}m"]
        if self.track_orientation:
            parts.append(f"ori={metrics.get('ee_orientation_path_length', 0):.2f}rad")
        if self.track_joint:
            parts.append(f"jnt={metrics.get('joint_path_length', 0):.2f}rad")
        if self.track_base_translation:
            parts.append(f"base={metrics.get('base_translation_path_length', 0):.3f}m")
        if self.track_base_rotation:
            parts.append(f"yaw={metrics.get('base_yaw_path_length', 0):.2f}rad")
        warning_text = " ".join(parts)

        return {
            "success": True,
            "warning_text": warning_text,
            "metrics": metrics,
        }
