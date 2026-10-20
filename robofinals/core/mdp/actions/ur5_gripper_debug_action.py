# Copyright 2025 Lightwheel Team
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils.configclass import configclass


def _tensor_row(values, row: int = 0) -> list[float]:
    if hasattr(values, "detach"):
        tensor = values.detach()
    else:
        tensor = torch.as_tensor(values)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return [round(float(v), 4) for v in tensor[row].cpu().tolist()]


class UR5GripperDebugAction(BinaryJointPositionAction):
    """Binary gripper action with explicit UR5 gripper joint diagnostics."""

    cfg: "UR5GripperDebugActionCfg"

    def __init__(self, cfg: "UR5GripperDebugActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._debug_step = 0
        self._last_raw_action: float | None = None
        self._debug_joint_ids, self._debug_joint_names = self._asset.find_joints(self.cfg.debug_joint_names)
        if not self._debug_joint_ids:
            self._debug_joint_ids, self._debug_joint_names = self._joint_ids, self._joint_names
        print(
            "[UR5GripperDebugAction] resolved "
            f"joint_names={self._joint_names} joint_ids={self._joint_ids} "
            f"debug_joint_names={self._debug_joint_names} debug_joint_ids={self._debug_joint_ids} "
            f"open={_tensor_row(self._open_command)} close={_tensor_row(self._close_command)}",
            flush=True,
        )

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        raw_value = round(float(actions.detach().flatten()[0].cpu()), 4)
        if self._last_raw_action != raw_value:
            print(
                "[UR5GripperDebugAction] process "
                f"raw={raw_value:+.4f} processed={_tensor_row(self._processed_actions)}",
                flush=True,
            )
            self._last_raw_action = raw_value

    def apply_actions(self):
        super().apply_actions()
        self._debug_step += 1
        if self._debug_step <= int(self.cfg.debug_initial_steps) or self._debug_step % int(self.cfg.debug_every_n_steps) == 0:
            joint_pos_all = self._asset.data.joint_pos
            joint_target_all = self._asset.data.joint_pos_target
            if not hasattr(joint_pos_all, "detach"):
                import warp as wp

                joint_pos_all = wp.to_torch(joint_pos_all)
            if not hasattr(joint_target_all, "detach"):
                import warp as wp

                joint_target_all = wp.to_torch(joint_target_all)
            joint_ids = torch.as_tensor(self._debug_joint_ids, device=joint_pos_all.device, dtype=torch.long)
            joint_pos = joint_pos_all.index_select(1, joint_ids)
            joint_target = joint_target_all.index_select(1, joint_ids)
            print(
                "[UR5GripperDebugAction] state "
                f"step={self._debug_step} "
                f"pos={_tensor_row(joint_pos)} "
                f"target={_tensor_row(joint_target)} "
                f"processed={_tensor_row(self._processed_actions)}",
                flush=True,
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._last_raw_action = None


@configclass
class UR5GripperDebugActionCfg(BinaryJointPositionActionCfg):
    class_type: type[UR5GripperDebugAction] = UR5GripperDebugAction
    debug_joint_names: list[str] = MISSING
    debug_initial_steps: int = 20
    debug_every_n_steps: int = 30
