from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from robofinals.utils.pinocchio_ik.x7s_ik import X7SBimanualIK


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor


class WrappedX7SPinocchioIKAction(ActionTerm):
    cfg: "WrappedX7SPinocchioIKCfg"
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _ik_solver: X7SBimanualIK
    """The IK solver used to compute the joint positions."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    def __init__(self, cfg: "WrappedX7SPinocchioIKCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        else:
            self._offset = torch.zeros_like(self._raw_actions)

        if self.cfg.hand_name == "left":
            self._ik_solver = env.cfg.isaaclab_arena_env.embodiment._bi_ik.left
        elif self.cfg.hand_name == "right":
            self._ik_solver = env.cfg.isaaclab_arena_env.embodiment._bi_ik.right
        else:
            raise ValueError(f"Invalid hand name: {self.cfg.hand_name}")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 7  # (x, y, z, qx, qy, qz, qw)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "WrappedX7SPinocchioIKAction"
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # raw_action is the ee_pose based on the robot_base_link
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions + self._offset

    def apply_actions(self):
        target_pose = self._processed_actions.detach().cpu().numpy()
        target_pose[:, 3:] = target_pose[:, [6, 3, 4, 5]]  # solve_pose_to_joint need wxyz
        q_np, _ = self._ik_solver.solve_pose_to_joints(target_pose)
        arm_action = torch.tensor(q_np, device=self.device, dtype=torch.float32)
        arm_action = arm_action + self._offset
        self._asset.set_joint_position_target_index(target=arm_action, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._raw_actions[env_ids] = 0.0


@configclass
class WrappedX7SPinocchioIKCfg(ActionTermCfg):
    """Configuration for the wrapped Piper Pinocchio IK action term."""

    class_type: type[ActionTerm] = WrappedX7SPinocchioIKAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    hand_name: str = MISSING
    """Name of the hand. ["left", "right"]"""

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.
    """
