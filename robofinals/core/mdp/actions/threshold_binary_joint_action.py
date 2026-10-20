from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils.configclass import configclass


class ThresholdBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary joint position action that closes above a configurable threshold."""

    cfg: "ThresholdBinaryJointPositionActionCfg"

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        binary_mask = actions > self.cfg.threshold
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class ThresholdBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    class_type: type[ThresholdBinaryJointPositionAction] = ThresholdBinaryJointPositionAction
    threshold: float = 0.5
