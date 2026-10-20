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

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg  # noqa: F401
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from .observations import *  # noqa: F401, F403
from .actions import *  # noqa: F401, F403


_reset_scene_to_default = reset_scene_to_default  # noqa: F405

# override the default reset_scene_to_default


def reset_scene_to_default(env, env_ids, reset_joint_targets: bool = False):
    omni_deformables = _pop_omniphysics_deformables(env)
    try:
        _reset_scene_to_default(env, env_ids, reset_joint_targets=reset_joint_targets)
    finally:
        _restore_deformables(env, omni_deformables)
    _reset_omniphysics_deformables(omni_deformables, env_ids)
    env.cfg.isaaclab_arena_env.orchestrator._reset_internal(env, env_ids)
    if hasattr(env.cfg.isaaclab_arena_env.embodiment, 'reset_robot_cfg_state'):
        env.cfg.isaaclab_arena_env.embodiment.reset_robot_cfg_state()


def _pop_omniphysics_deformables(env):
    deformables = getattr(env.scene, "_deformable_objects", None)
    if not isinstance(deformables, dict):
        return {}
    omni_deformables = {}
    for name, deformable in list(deformables.items()):
        if getattr(deformable, "runtime_api", None) == "omniphysics_deformable_body":
            omni_deformables[name] = deformable
            deformables.pop(name)
    return omni_deformables


def _restore_deformables(env, deformables):
    if not deformables:
        return
    scene_deformables = getattr(env.scene, "_deformable_objects", None)
    if isinstance(scene_deformables, dict):
        scene_deformables.update(deformables)


def _reset_omniphysics_deformables(deformables, env_ids):
    if not deformables:
        return
    from robofinals.utils.isaac_data_compat import as_torch
    import torch

    for deformable in deformables.values():
        default_state = deformable.data.default_nodal_state_w
        if isinstance(default_state, torch.Tensor):
            nodal_state = default_state[env_ids].clone()
        else:
            nodal_state = as_torch(default_state)[env_ids].clone()
        deformable.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)

