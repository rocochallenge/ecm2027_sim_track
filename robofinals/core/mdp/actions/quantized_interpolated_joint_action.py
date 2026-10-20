# Copyright 2026 Lightwheel Team
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils.configclass import configclass


class QuantizedInterpolatedJointPositionAction(BinaryJointPositionAction):
    """Interpolate between open and closed targets using discrete input levels."""

    cfg: "QuantizedInterpolatedJointPositionActionCfg"

    def __init__(self, cfg: "QuantizedInterpolatedJointPositionActionCfg", env):
        if cfg.num_levels < 2:
            raise ValueError("num_levels must be at least 2")
        super().__init__(cfg, env)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        close_fraction = (actions.to(dtype=self._open_command.dtype).clamp(-1.0, 1.0) + 1.0) * 0.5
        intervals = self.cfg.num_levels - 1
        close_fraction = torch.round(close_fraction * intervals) / intervals
        self._processed_actions = torch.lerp(
            self._open_command,
            self._close_command,
            close_fraction,
        )

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class QuantizedInterpolatedJointPositionActionCfg(BinaryJointPositionActionCfg):
    class_type: type[QuantizedInterpolatedJointPositionAction] = QuantizedInterpolatedJointPositionAction
    num_levels: int = 10
    """Number of discrete gripper positions, including fully open and fully closed."""
