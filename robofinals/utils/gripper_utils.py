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


def _as_like_tensor(value, reference: torch.Tensor) -> torch.Tensor:
    reference_device = reference.device if torch.is_tensor(reference) else None
    reference_dtype = reference.dtype if torch.is_tensor(reference) and isinstance(reference.dtype, torch.dtype) else torch.float32
    if torch.is_tensor(value):
        return value.to(device=reference_device, dtype=reference_dtype)
    return torch.as_tensor(value, device=reference_device, dtype=reference_dtype)


def dynamic_reset_gripper_effort_limit_sim(env, teleop_device):
    need_to_set = []
    if teleop_device == "bi-so101leader":
        need_to_set = [env.scene.articulations['left_arm'], env.scene.articulations['right_arm']]
    elif teleop_device in ["so101leader", "keyboard"]:
        need_to_set = [env.scene['robot']]
    for arm in need_to_set:
        write_gripper_effort_limit_sim(env, arm)
    return


def write_gripper_effort_limit_sim(env, env_arm):
    gripper_pos = env_arm.data.body_link_pos_w[:, -1]  # [num_envs, 3]
    gripper_pos = _as_like_tensor(gripper_pos, gripper_pos)
    num_envs = gripper_pos.shape[0]

    object_positions = []
    object_masses = []
    object_names = []

    for name, obj in env.scene._rigid_objects.items():
        pos = obj.data.body_link_pos_w[:, 0]  # [num_envs, 3]
        pos = _as_like_tensor(pos, gripper_pos)
        object_positions.append(pos)
        object_masses.append(_as_like_tensor(obj.data.default_mass, gripper_pos))
        object_names.append(name)

    if not object_positions:
        return

    object_positions = torch.stack(object_positions)  # [num_objects, num_envs, 3]
    object_masses = torch.stack(object_masses)  # [num_objects, num_envs, 1]

    distances = torch.sqrt(torch.sum((object_positions - gripper_pos.unsqueeze(0)) ** 2, dim=2))

    min_distances, min_indices = torch.min(distances, dim=0)  # [num_envs]

    target_masses = object_masses[min_indices.cpu(), 0, 0]  # [num_envs]

    joint_effort_limits = _as_like_tensor(env_arm._data.joint_effort_limits, gripper_pos)
    target_effort_limits = (target_masses / 0.15).to(joint_effort_limits)

    current_effort_limit_sim = joint_effort_limits[:, -1]  # [num_envs]
    need_update = torch.abs(target_effort_limits - current_effort_limit_sim) > 0.1

    if torch.any(need_update):
        new_limits = current_effort_limit_sim.clone()
        new_limits[need_update] = target_effort_limits[need_update]

        env_arm.write_joint_effort_limit_to_sim(
            limits=new_limits.unsqueeze(-1),
            joint_ids=[5]
        )


def get_task_type(task: str, task_type: str | None = None) -> str:
    """
    Make sure the task type is in the supported teleop devices.
    """
    if task_type is not None:
        return task_type
    if "BiArm" in task:
        return "bi-so101leader"
    else:
        return "so101leader"


def delete_attribute(obj, attr_name):
    if hasattr(obj, attr_name):
        delattr(obj, attr_name)
