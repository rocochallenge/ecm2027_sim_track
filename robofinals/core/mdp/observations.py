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

from __future__ import annotations

import re
from typing import Iterable, List

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationData
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerData
from isaaclab.utils.math import subtract_frame_transforms

from robofinals.utils.isaac_data_compat import as_torch, first_target_pos_w, first_target_quat_w


def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    object_data: ArticulationData = env.scene["object"].data

    return as_torch(object_data.root_pos_w) - first_target_pos_w(ee_tf_data)


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    p = as_torch(ee_tf_data.target_pos_w)
    if p.ndim < 3 or p.shape[1] <= 1:
        return p.new_zeros((env.num_envs, 0))
    fingertips_pos = p[:, 1:, :] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = as_torch(ee_tf_data.target_pos_w) - env.scene.env_origins.unsqueeze(1)

    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = as_torch(ee_tf_data.target_quat_w)
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat


def ee_pose(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The pose of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.

    Returns:
        The pose of the end-effector in the environment frame.
        The pose is a tensor of shape (num_envs, 7) where the last dimension is [x, y, z, qw, qx, qy, qz].
    """
    return torch.cat([ee_pos(env), ee_quat(env, make_quat_unique)], dim=-1)


def get_target_qpos(
    env: ManagerBasedRLEnv,
    action_name: str = 'arm_action'
) -> torch.Tensor:
    """The last input action to the environment.

       The name of the action term for which the action is required. If None, the
       entire action tensor is returned.
       """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.scene['robot']._data.joint_pos_target


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target_pos_w = as_torch(ee_frame.data.target_pos_w)
    return target_pos_w - env.scene.env_origins[:, None, 0:3]


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target_quat_w = as_torch(ee_frame.data.target_quat_w)
    return target_quat_w


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    jp = as_torch(robot.data.joint_pos)
    finger_joint_1 = jp[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * jp[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def get_eef_base(env: ManagerBasedRLEnv, base_link_name: str = 'dummy_link', eef_link_name: str = 'hand_link') -> torch.Tensor:
    robot = env.scene.articulations['robot']
    hand_ids, _ = robot.find_bodies(eef_link_name)
    base_link_ids, _ = robot.find_bodies(base_link_name)
    hand_idx = hand_ids[0]
    base_link_idx = base_link_ids[0]
    blp = as_torch(robot.data.body_link_pose_w)
    hand_pose_w = blp[0, hand_idx, :]
    base_link_pose_w = blp[0, base_link_idx, :]
    hand_pos_base, hand_quat_base = subtract_frame_transforms(
        base_link_pose_w[:3].unsqueeze(0),   # base position in world
        base_link_pose_w[3:].unsqueeze(0),   # base quaternion [w,x,y,z] in world
        hand_pose_w[:3].unsqueeze(0),   # left hand position in world
        hand_pose_w[3:].unsqueeze(0)    # left hand quaternion in world
    )

    eef_base = torch.cat([hand_pos_base, hand_quat_base], dim=-1)
    return eef_base


#
# Generic helpers expected by some Arena embodiments (e.g. GR1T2)
# These functions override the ones from isaaclab_tasks to fix bugs
#

def get_eef_pos(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    """Return the position of an end-effector link in env frame.

    This is a generic version of :func:`get_left_eef_pos` / :func:`get_right_eef_pos`
    that selects the link by name. It is used by Arena configs such as GR1T2.
    """
    body_pos_w = as_torch(env.scene["robot"].data.body_pos_w)
    body_names: List[str] = env.scene["robot"].data.body_names
    try:
        link_idx = body_names.index(link_name)
    except ValueError:
        raise ValueError(f"Link name '{link_name}' not found in robot body names: {body_names}")

    # Positions are stored in world frame; subtract env origins to get env frame.
    eef_pos = body_pos_w[:, link_idx] - env.scene.env_origins
    return eef_pos


def get_eef_quat(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    """Return the orientation (quaternion) of an end-effector link in world frame."""
    body_quat_w = as_torch(env.scene["robot"].data.body_quat_w)
    body_names: List[str] = env.scene["robot"].data.body_names
    try:
        link_idx = body_names.index(link_name)
    except ValueError:
        raise ValueError(f"Link name '{link_name}' not found in robot body names: {body_names}")

    eef_quat = body_quat_w[:, link_idx]
    return eef_quat


def _match_joint_indices(joint_names: Iterable[str], patterns: Iterable[str], device: torch.device) -> torch.Tensor:
    """Return indices of joints whose names match any of the provided patterns.

    Patterns can be exact names or regexes (e.g. ``\"R_.*\"``).

    Args:
        joint_names: List of joint names to search in.
        patterns: Patterns to match against joint names.
        device: The device to create the tensor on.

    Returns:
        Tensor of indices matching the patterns.
    """
    indices: List[int] = []
    for i, name in enumerate(joint_names):
        for pat in patterns:
            # Treat pattern as a regular expression.
            if re.fullmatch(pat, name):
                indices.append(i)
                break
    if not indices:
        raise ValueError(f"No joints matched patterns {list(patterns)} in joint names {list(joint_names)}")
    return torch.tensor(indices, dtype=torch.long, device=device)


def get_robot_joint_state(env: ManagerBasedRLEnv, joint_names: Iterable[str]) -> torch.Tensor:
    """Return joint positions for selected robot joints.

    Args:
        env: The RL environment.
        joint_names: Iterable of joint name patterns. Each element can be:
            - an exact joint name, or
            - a regex pattern like ``\"R_.*\"`` or ``\"L_.*\"``.

    Returns:
        Tensor of shape ``(num_envs, num_selected_joints)`` with joint positions.
    """
    robot_joint_names = env.scene["robot"].data.joint_names
    jp = as_torch(env.scene["robot"].data.joint_pos)
    device = jp.device
    indices = _match_joint_indices(robot_joint_names, joint_names, device)
    joint_pos = jp[:, indices]
    return joint_pos
