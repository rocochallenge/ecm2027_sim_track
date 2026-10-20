# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from robofinals.utils.math_utils.transform_utils import convert_quat, make_pose

from .wbc_policy.config.configs import BaseConfig
from .wbc_policy.policy.g1_wbc_upperbody_controller import G1BodyIKSolverSettings, G1WBCUpperbodyController
from .wbc_policy.utils.g1 import instantiate_g1_robot_model

USE_P_CONTROL = False

# Fixed IK-model -> simulator wrist-frame correction:
# from the current target-vs-actual marker observation, the simulator wrist
# frame differs from the IK/URDF wrist frame by an opposite signed +/-90 deg
# offset around the local Z axis on the two mirrored hands.
_LEFT_IK_WRIST_FRAME_COMPENSATION = R.from_euler("z", 90.0, degrees=True).as_matrix().astype(np.float64)
_RIGHT_IK_WRIST_FRAME_COMPENSATION = R.from_euler("z", -90.0, degrees=True).as_matrix().astype(np.float64)


class G1TactileAction(ActionTerm):

    cfg: "G1TactileActionCfg"

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: "G1TactileActionCfg", env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)

        # Avoid indexing across all joints for efficiency
        if (self._num_joints == self._asset.num_joints and
                not self.cfg.preserve_order):
            self._joint_ids = slice(None)

        self._processed_actions = torch.zeros([self.num_envs, self._num_joints], device=self.device)
        self._target_robot_joints_mujoco = None
        self._wrist_joint_debug_enabled = False
        self._wrist_joint_debug_interval = 10
        self._wrist_joint_debug_frame = 0
        self._left_wrist_roll_local_idx = self._get_local_joint_idx("left_wrist_roll_joint")
        self._left_wrist_pitch_local_idx = self._get_local_joint_idx("left_wrist_pitch_joint")
        self._left_wrist_yaw_local_idx = self._get_local_joint_idx("left_wrist_yaw_joint")
        self._right_wrist_roll_local_idx = self._get_local_joint_idx("right_wrist_roll_joint")
        self._right_wrist_pitch_local_idx = self._get_local_joint_idx("right_wrist_pitch_joint")
        self._right_wrist_yaw_local_idx = self._get_local_joint_idx("right_wrist_yaw_joint")
        self._left_wrist_roll_asset_idx = self._get_asset_joint_idx(self._left_wrist_roll_local_idx)
        self._left_wrist_pitch_asset_idx = self._get_asset_joint_idx(self._left_wrist_pitch_local_idx)
        self._left_wrist_yaw_asset_idx = self._get_asset_joint_idx(self._left_wrist_yaw_local_idx)
        self._right_wrist_roll_asset_idx = self._get_asset_joint_idx(self._right_wrist_roll_local_idx)
        self._right_wrist_pitch_asset_idx = self._get_asset_joint_idx(self._right_wrist_pitch_local_idx)
        self._right_wrist_yaw_asset_idx = self._get_asset_joint_idx(self._right_wrist_yaw_local_idx)

        use_waist = any(name.startswith("waist_") for name in self._joint_names)
        config = BaseConfig()
        config.enable_waist = use_waist

        waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
        self.robot_model = instantiate_g1_robot_model(waist_location=waist_location)

        self.current_upper_body_pose = self.robot_model.get_initial_upper_body_pose()
        active_joint_groups = ["arms", "waist"] if use_waist else ["arms"]

        self.upperbody_controller = G1WBCUpperbodyController(
            robot_model=self.robot_model,
            body_active_joint_groups=active_joint_groups,
            control_hands=False,
            body_ik_solver_settings=self.cfg.ik_solver_settings,
        )
        ik_joint_to_idx = dict(self.upperbody_controller.body.joint_to_dof_index)
        self._joint_to_ik_indices = [ik_joint_to_idx.get(name) for name in self._joint_names]
        self._can_reorder_ik_output = all(idx is not None for idx in self._joint_to_ik_indices)
        self._left_wrist_yaw_ik_idx = ik_joint_to_idx.get("left_wrist_yaw_joint")
        self._right_wrist_yaw_ik_idx = ik_joint_to_idx.get("right_wrist_yaw_joint")
        self._cached_ee_targets: list[dict[str, np.ndarray] | None] = [None] * self.num_envs
        self._cached_joint_targets: list[np.ndarray | None] = [None] * self.num_envs
        self._target_pos_reuse_tol = 1.0e-5
        self._target_rot_reuse_tol_rad = np.deg2rad(0.1)
        self._frozen_joint_target: torch.Tensor | None = None
        self._last_invalid_ik_warn_ts = 0.0

    # Properties.
    @property
    def num_joints(self) -> int:
        """Get the number of joints."""
        return self._num_joints

    @property
    def target_robot_joints_mujoco(self) -> torch.Tensor:
        """Get the target robot joints mujoco tensor."""
        return self._target_robot_joints_mujoco

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (based on number of tasks and pose dimension)."""
        return 14

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    # """
    # Operations.
    # """
    def save_check_point(self):
        return self.upperbody_controller.body_ik_solver.save_configuration_q()

    def load_check_point(self, q):
        self.upperbody_controller.body_ik_solver.load_configuration_q(q)

    def compute_upperbody_joint_positions(self, body_data):
        if self.upperbody_controller.in_warmup:
            for _ in range(50):
                target_robot_joints = self.upperbody_controller.inverse_kinematics(body_data, only_body=True)

            self.upperbody_controller.in_warmup = False
        else:
            target_robot_joints = self.upperbody_controller.inverse_kinematics(body_data, only_body=True)

        return target_robot_joints

    @staticmethod
    def _rotation_distance_rad(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
        rel = rot_a.T @ rot_b
        cos_theta = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.arccos(cos_theta))

    def _can_reuse_cached_target(
        self,
        env_idx: int,
        left_pos: np.ndarray,
        left_rot: np.ndarray,
        right_pos: np.ndarray,
        right_rot: np.ndarray,
    ) -> bool:
        cached_target = self._cached_joint_targets[env_idx]
        cached_pose = self._cached_ee_targets[env_idx]
        if cached_target is None or cached_pose is None:
            return False
        if np.max(np.abs(cached_pose["left_pos"] - left_pos)) > self._target_pos_reuse_tol:
            return False
        if np.max(np.abs(cached_pose["right_pos"] - right_pos)) > self._target_pos_reuse_tol:
            return False
        if self._rotation_distance_rad(cached_pose["left_rot"], left_rot) > self._target_rot_reuse_tol_rad:
            return False
        if self._rotation_distance_rad(cached_pose["right_rot"], right_rot) > self._target_rot_reuse_tol_rad:
            return False
        return True

    def _warn_invalid_ik_target(
        self,
        env_idx: int,
        reason: str,
        *,
        left_pos: np.ndarray | None = None,
        right_pos: np.ndarray | None = None,
    ) -> None:
        now = time.time()
        if (now - self._last_invalid_ik_warn_ts) < 1.0:
            return
        self._last_invalid_ik_warn_ts = now
        extra = []
        if left_pos is not None:
            extra.append(f"left_pos={np.array2string(np.asarray(left_pos), precision=4)}")
        if right_pos is not None:
            extra.append(f"right_pos={np.array2string(np.asarray(right_pos), precision=4)}")
        extra_msg = f" ({', '.join(extra)})" if extra else ""
        print(
            f"[G1TactileAction] Ignoring invalid IK target for env {env_idx}: {reason}. "
            f"Reusing previous stable joint target{extra_msg}."
        )

    def _get_fallback_joint_target(self, env_idx: int) -> np.ndarray:
        cached_target = self._cached_joint_targets[env_idx]
        if cached_target is not None:
            cached_target = np.asarray(cached_target, dtype=np.float32).reshape(-1)
            if cached_target.shape == (self._num_joints,) and np.all(np.isfinite(cached_target)):
                return cached_target.copy()

        try:
            current_joint_target = (
                self._asset.data.joint_pos[env_idx, self._joint_ids].detach().cpu().numpy().astype(np.float32, copy=True)
            )
            if current_joint_target.shape == (self._num_joints,) and np.all(np.isfinite(current_joint_target)):
                return current_joint_target
        except Exception:
            pass

        return np.zeros(self._num_joints, dtype=np.float32)

    def _clip_joint_target_to_limits(self, env_idx: int, joint_target: np.ndarray) -> np.ndarray:
        try:
            joint_limits = self._asset.data.joint_pos_limits[env_idx, self._joint_ids].detach().cpu().numpy()
            lower = np.asarray(joint_limits[:, 0], dtype=np.float32)
            upper = np.asarray(joint_limits[:, 1], dtype=np.float32)
            return np.clip(np.asarray(joint_target, dtype=np.float32), lower, upper)
        except Exception:
            return np.asarray(joint_target, dtype=np.float32)

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.
        """
        self._raw_actions[:] = actions[:, :self.action_dim]

        if getattr(self._env.cfg, "_hold_current_arm_joints", False):
            if self._frozen_joint_target is None:
                self._frozen_joint_target = self._asset.data.joint_pos[:, self._joint_ids].detach().clone()
            self._processed_actions = self._frozen_joint_target.detach().clone()
            return
        self._frozen_joint_target = None

        if actions.dim() == 2:
            left_arm_pos = actions[:, :3]
            left_arm_quat = actions[:, 3:7]
            right_arm_pos = actions[:, 7:10]
            right_arm_quat = actions[:, 10:14]
        else:
            left_arm_pos = actions[:3].unsqueeze(0)
            left_arm_quat = actions[3:7].unsqueeze(0)
            right_arm_pos = actions[7:10].unsqueeze(0)
            right_arm_quat = actions[10:14].unsqueeze(0)

        if left_arm_quat.dim() == 2:
            left_arm_quat_xyzw = torch.zeros_like(left_arm_quat)
            right_arm_quat_xyzw = torch.zeros_like(right_arm_quat)
            for i in range(left_arm_quat.shape[0]):
                left_arm_quat_xyzw[i] = convert_quat(left_arm_quat[i], to="xyzw")
                right_arm_quat_xyzw[i] = convert_quat(right_arm_quat[i], to="xyzw")
        else:
            left_arm_quat_xyzw = convert_quat(left_arm_quat, to="xyzw")
            right_arm_quat_xyzw = convert_quat(right_arm_quat, to="xyzw")

        left_arm_quat_xyzw = left_arm_quat_xyzw.clone()
        right_arm_quat_xyzw = right_arm_quat_xyzw.clone()
        for _, quat_tensor in (("left", left_arm_quat_xyzw), ("right", right_arm_quat_xyzw)):
            if quat_tensor.dim() == 2:
                quat_norms = torch.linalg.norm(quat_tensor, dim=1)
                invalid_mask = (~torch.isfinite(quat_norms)) | (quat_norms < 1e-6)
                if torch.any(invalid_mask):
                    quat_tensor[invalid_mask] = torch.tensor(
                        [0.0, 0.0, 0.0, 1.0], device=quat_tensor.device, dtype=quat_tensor.dtype
                    )
            else:
                quat_norm = torch.linalg.norm(quat_tensor)
                if (not torch.isfinite(quat_norm)) or quat_norm.item() < 1e-6:
                    quat_tensor[:] = torch.tensor(
                        [0.0, 0.0, 0.0, 1.0], device=quat_tensor.device, dtype=quat_tensor.dtype
                    )

        left_rotmat = R.from_quat(left_arm_quat_xyzw.cpu().numpy()).as_matrix()
        right_rotmat = R.from_quat(right_arm_quat_xyzw.cpu().numpy()).as_matrix()

        num_envs = left_rotmat.shape[0] if left_rotmat.ndim == 3 else 1
        all_target_joints = []
        for env_idx in range(num_envs):
            if left_rotmat.ndim == 3:
                env_left_rotmat = left_rotmat[env_idx]
                env_right_rotmat = right_rotmat[env_idx]
                env_left_pos = left_arm_pos[env_idx]
                env_right_pos = right_arm_pos[env_idx]
            else:
                env_left_rotmat = left_rotmat
                env_right_rotmat = right_rotmat
                env_left_pos = left_arm_pos[0]
                env_right_pos = right_arm_pos[0]

            env_left_pos_np = env_left_pos.detach().cpu().numpy().astype(np.float64, copy=False)
            env_right_pos_np = env_right_pos.detach().cpu().numpy().astype(np.float64, copy=False)
            env_left_rotmat_ik = env_left_rotmat @ _LEFT_IK_WRIST_FRAME_COMPENSATION
            env_right_rotmat_ik = env_right_rotmat @ _RIGHT_IK_WRIST_FRAME_COMPENSATION
            if self._can_reuse_cached_target(
                env_idx,
                env_left_pos_np,
                env_left_rotmat_ik,
                env_right_pos_np,
                env_right_rotmat_ik,
            ):
                all_target_joints.append(self._cached_joint_targets[env_idx].copy())
                continue

            left_arm_pose = make_pose(env_left_pos, torch.tensor(env_left_rotmat_ik, device=self.device))
            right_arm_pose = make_pose(env_right_pos, torch.tensor(env_right_rotmat_ik, device=self.device))

            body_data = {
                "left_wrist_yaw_link": left_arm_pose.cpu().numpy(),
                "right_wrist_yaw_link": right_arm_pose.cpu().numpy(),
            }
            try:
                ik_target_joints = np.asarray(self.compute_upperbody_joint_positions(body_data), dtype=np.float32)
            except Exception as exc:
                self._warn_invalid_ik_target(
                    env_idx,
                    f"IK solver failed ({exc})",
                    left_pos=env_left_pos_np,
                    right_pos=env_right_pos_np,
                )
                all_target_joints.append(self._get_fallback_joint_target(env_idx))
                continue

            if not np.all(np.isfinite(ik_target_joints)):
                self._warn_invalid_ik_target(
                    env_idx,
                    "IK solver returned non-finite joint values",
                    left_pos=env_left_pos_np,
                    right_pos=env_right_pos_np,
                )
                all_target_joints.append(self._get_fallback_joint_target(env_idx))
                continue

            if self.cfg.negate_left_wrist_yaw and self._left_wrist_yaw_ik_idx is not None:
                ik_target_joints[self._left_wrist_yaw_ik_idx] *= -1

            if self._can_reorder_ik_output:
                target_robot_joints_mujoco = np.asarray(
                    [ik_target_joints[idx] for idx in self._joint_to_ik_indices], dtype=np.float32
                )
            else:
                target_robot_joints_mujoco = ik_target_joints[: self._num_joints]
            target_robot_joints_mujoco = self._clip_joint_target_to_limits(env_idx, target_robot_joints_mujoco)
            if not np.all(np.isfinite(target_robot_joints_mujoco)):
                self._warn_invalid_ik_target(
                    env_idx,
                    "post-processed joint target contains non-finite values",
                    left_pos=env_left_pos_np,
                    right_pos=env_right_pos_np,
                )
                all_target_joints.append(self._get_fallback_joint_target(env_idx))
                continue
            self._cached_ee_targets[env_idx] = {
                "left_pos": env_left_pos_np.copy(),
                "left_rot": env_left_rotmat_ik.copy(),
                "right_pos": env_right_pos_np.copy(),
                "right_rot": env_right_rotmat_ik.copy(),
            }
            self._cached_joint_targets[env_idx] = target_robot_joints_mujoco.copy()
            all_target_joints.append(target_robot_joints_mujoco)

        if num_envs > 1:
            self._processed_actions = torch.tensor(
                np.array(all_target_joints),
                device=self.device,
                dtype=torch.float32
            )
        else:
            self._processed_actions = torch.tensor(
                all_target_joints[0],
                device=self.device,
                dtype=torch.float32
            ).unsqueeze(0)

    def apply_actions(self):
        """Apply the computed joint positions based on the WBC solution."""
        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)
        if self._wrist_joint_debug_enabled:
            self._wrist_joint_debug_frame += 1
            if self._wrist_joint_debug_frame % self._wrist_joint_debug_interval == 0:
                self._print_wrist_joint_debug()

    def _get_local_joint_idx(self, joint_name: str) -> int | None:
        try:
            return self._joint_names.index(joint_name)
        except ValueError:
            return None

    def _get_asset_joint_idx(self, local_idx: int | None) -> int | None:
        if local_idx is None:
            return None
        if isinstance(self._joint_ids, slice):
            start = 0 if self._joint_ids.start is None else self._joint_ids.start
            step = 1 if self._joint_ids.step is None else self._joint_ids.step
            return start + local_idx * step
        return int(self._joint_ids[local_idx])

    def _format_single_wrist_joint_debug(self, side: str, local_idx: int | None, asset_idx: int | None) -> str:
        if local_idx is None or asset_idx is None:
            return f"{side} unavailable"

        target = float(self._processed_actions[0, local_idx].item())
        current = float(self._asset.data.joint_pos[0, asset_idx].item())
        lower = float(self._asset.data.joint_pos_limits[0, asset_idx, 0].item())
        upper = float(self._asset.data.joint_pos_limits[0, asset_idx, 1].item())
        at_limit = target <= lower + 1e-4 or target >= upper - 1e-4
        return (
            f"{side} tgt={target:+.4f}rad({np.degrees(target):+6.1f}deg) "
            f"cur={current:+.4f}rad({np.degrees(current):+6.1f}deg) "
            f"lim=[{lower:+.4f},{upper:+.4f}]rad "
            f"at_lim={at_limit}"
        )

    def _print_wrist_joint_debug(self):
        left_text = self._format_single_wrist_joint_debug(
            "L_yaw", self._left_wrist_yaw_local_idx, self._left_wrist_yaw_asset_idx
        )
        right_text = self._format_single_wrist_joint_debug(
            "R_yaw", self._right_wrist_yaw_local_idx, self._right_wrist_yaw_asset_idx
        )
        print(f"[WristJoint dbg #{self._wrist_joint_debug_frame}] {left_text} | {right_text}")

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.
        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        env_ids_ = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        self._raw_actions[env_ids_] = 0.0
        self._processed_actions[env_ids_] = 0.0
        for env_id in env_ids_:
            self._cached_ee_targets[env_id] = None
            self._cached_joint_targets[env_id] = None
        self.upperbody_controller.body_ik_solver.reset()


@configclass
class G1TactileActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = G1TactileAction
    """Specifies the action term class type for G1 WBC action."""

    preserve_order: bool = True
    joint_names: list[str] = MISSING
    ik_solver_settings: G1BodyIKSolverSettings | None = None
    negate_left_wrist_yaw: bool = False
