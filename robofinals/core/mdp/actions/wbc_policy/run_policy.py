# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import isaaclab.utils.math as math_utils

from isaaclab.assets import ArticulationData
from robofinals.utils.isaac_data_compat import as_torch, sim_quat_raw_to_xyzw_np


def _as_numpy(data):
    """Convert torch tensors / array-likes to numpy arrays safely."""
    if isinstance(data, np.ndarray):
        return data
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    try:
        return np.asarray(data)
    except Exception:
        return as_torch(data).detach().cpu().numpy()


def _sim_quat_raw_to_policy_wxyz(data) -> np.ndarray:
    """Convert IsaacLab runtime quaternion buffers to Homie policy (w, x, y, z)."""
    quat_xyzw = sim_quat_raw_to_xyzw_np(_as_numpy(data))
    return quat_xyzw[..., [3, 0, 1, 2]].astype(np.float32, copy=False)


def convert_sim_joint_to_wbc_joint(sim_joint_data: np.ndarray, sim_joint_names: list, wbc_joints_order: dict):
    """Convert sim joint observations to WBC joint observations."""
    num_joints = len(wbc_joints_order)
    num_envs = sim_joint_data.shape[0]
    wbc_joint_data = np.zeros((num_envs, num_joints))

    # Check if sim_joint_data is a numpy array, if not, convert from torch tensor to numpy
    if not isinstance(sim_joint_data, np.ndarray):
        sim_joint_data = _as_numpy(sim_joint_data)

    for sim_joint_name in sim_joint_names:
        sim_joint_index = sim_joint_names.index(sim_joint_name)
        if sim_joint_name not in wbc_joints_order:
            continue
        wbc_joint_index = wbc_joints_order[sim_joint_name]
        wbc_joint_data[:, wbc_joint_index] = sim_joint_data[:, sim_joint_index]
    return wbc_joint_data


def prepare_observations(num_envs: int, robot_data: ArticulationData, wbc_joints_order: dict):
    """Prepare observations for the policy."""
    # Get robot joint observations
    sim_joint_pos = _as_numpy(robot_data.joint_pos)
    sim_joint_vel = _as_numpy(robot_data.joint_vel)
    wbc_num_joints = len(wbc_joints_order)

    # Convert joints data from Lab's order to GR00T's order saved in config yaml
    wbc_joint_pos = np.zeros((num_envs, wbc_num_joints))
    wbc_joint_vel = np.zeros((num_envs, wbc_num_joints))
    wbc_joint_acc = np.zeros((num_envs, wbc_num_joints))
    wbc_joint_pos = convert_sim_joint_to_wbc_joint(sim_joint_pos, robot_data.joint_names, wbc_joints_order)
    wbc_joint_vel = convert_sim_joint_to_wbc_joint(sim_joint_vel, robot_data.joint_names, wbc_joints_order)

    # Prepare obs dict for WBC policy input to G1DecoupledWholeBodyPolicy class
    assert wbc_joint_pos.shape == wbc_joint_vel.shape == wbc_joint_acc.shape == (num_envs, wbc_num_joints)

    root_link_pos_w = _as_numpy(robot_data.root_link_pos_w)
    root_link_quat_w = _sim_quat_raw_to_policy_wxyz(robot_data.root_link_quat_w)
    base_pose_w = np.concatenate((root_link_pos_w, root_link_quat_w), axis=1)
    base_lin_vel_b = _as_numpy(robot_data.root_link_lin_vel_b)
    base_ang_vel_b = _as_numpy(robot_data.root_link_ang_vel_b)

    base_vel_b = np.concatenate((base_lin_vel_b, base_ang_vel_b), axis=1)
    # torso link in world frame
    body_link_state_w = as_torch(robot_data.body_link_state_w)
    torso_link_pose_w = body_link_state_w[:, robot_data.body_names.index("torso_link"), :]
    torso_link_quat_sim = torso_link_pose_w[:, 3:7]
    torso_link_quat_w = _sim_quat_raw_to_policy_wxyz(torso_link_quat_sim)
    torso_link_ang_vel_w = torso_link_pose_w[:, -3:]

    torso_link_ang_vel_b = math_utils.quat_apply_inverse(torso_link_quat_sim, torso_link_ang_vel_w)

    # Prepare obs tmers
    wbc_obs = {
        "q": wbc_joint_pos,
        "dq": wbc_joint_vel,
        "ddq": np.zeros((num_envs, wbc_num_joints)),     # Not used by Standing Waist Height Policy
        "tau_est": np.zeros((num_envs, wbc_num_joints)),     # Not used by Standing Waist Height Policy
        "floating_base_pose": base_pose_w,  # wrt world frame, used to project gravity vector to local frame
        "floating_base_vel": base_vel_b,  # wrt body frame
        "floating_base_acc": np.zeros((num_envs, 6)),     # Not used by Standing Waist Height Policy
        "torso_quat": torso_link_quat_w,
        "torso_ang_vel": _as_numpy(torso_link_ang_vel_b),
    }
    return wbc_obs


def postprocess_actions(wbc_action: dict, robot_data: ArticulationData, wbc_g1_joints_order: dict, device: torch.device):
    """Postprocess actions for the policy."""
    num_envs = wbc_action["q"].shape[0]
    num_joints = len(robot_data.joint_names)
    processed_actions = torch.zeros((num_envs, num_joints), device=device)
    wbc_joints_pos_action = torch.from_numpy(wbc_action["q"])
    # Convert wbc gr00t joints order to Lab joints order
    for wbc_joint_name, wbc_joint_index in wbc_g1_joints_order.items():
        if wbc_joint_name not in robot_data.joint_names:
            continue
        sim_joint_index = robot_data.joint_names.index(wbc_joint_name)
        processed_actions[:, sim_joint_index] = wbc_joints_pos_action[:, wbc_joint_index]
    return processed_actions
