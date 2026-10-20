from collections.abc import Sequence
from dataclasses import MISSING
from typing import Callable

import numpy as np
import time
import torch
from isaaclab.utils.configclass import configclass
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.assets.articulation import Articulation


class WujiRetargetingAction(ActionTerm):
    cfg: "WujiRetargetingActionCfg"
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: "WujiRetargetingActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=True)
        self._last_target: torch.Tensor | None = None
        self._last_apply_ts: float | None = None
        self._default_control_dt = 1.0 / 50.0
        self.reset()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions
        finger_pos = self.raw_actions.reshape([self.num_envs, -1, 4])

        self._fingers_actions[:, :] = finger_pos


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids_ = list(env_ids) if env_ids is not None else list(range(self.num_envs))

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        processed_actions = torch.zeros(len(env_ids_), self.num_fingers, 4, device=self.device)
        if env_ids is None:
            self._fingers_actions = processed_actions
            self._last_target = None
            self._last_apply_ts = None
        else:
            self._fingers_actions[env_ids, :] = processed_actions

    def _estimate_control_dt(self) -> float:
        now = time.time()
        if self._last_apply_ts is None:
            self._last_apply_ts = now
            return self._default_control_dt
        dt = max(now - self._last_apply_ts, 1.0e-3)
        self._last_apply_ts = now
        return dt

    def _finger_velocity_budget(self, control_dt: float) -> float | None:
        try:
            vel = self._asset.actuators["fingers"].velocity_limit
            vel_arr = vel.reshape(-1)
            vel_arr = vel_arr[torch.isfinite(vel_arr)]
            if vel_arr.numel() == 0:
                return None
            return float(vel_arr.mean().item()) * control_dt
        except Exception:
            return None

    def apply_actions(self):
        processed_actions = self._fingers_actions.cpu().numpy()
        if np.sum(abs(processed_actions)) > 0.01:
            control_dt = self._estimate_control_dt()
            joint_pos_rad = self.cfg.post_process_fn(processed_actions, len(self._joint_ids))
            joint_pos_rad = torch.tensor(joint_pos_rad, dtype=torch.float32, device=self.processed_actions.device)

            velocity_budget = self._finger_velocity_budget(control_dt)
            if self._last_target is not None and velocity_budget is not None and velocity_budget > 0.0:
                joint_pos_rad = torch.clamp(
                    joint_pos_rad,
                    min=self._last_target - velocity_budget,
                    max=self._last_target + velocity_budget,
                )

            self._asset.set_joint_position_target(joint_pos_rad, self._joint_ids)
            self._last_target = joint_pos_rad.clone()

    @property
    def num_fingers(self) -> int:
        return 5

    @property
    def finger_names(self) -> list[str]:
        return self.cfg.joint_names

    @property
    def action_dim(self) -> int:
        num = self.num_fingers
        return num * 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions


@configclass
class WujiRetargetingActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = WujiRetargetingAction
    # retargeting_index: list[int] = MISSING
    joint_names: list[str] = MISSING
    post_process_fn: Callable = lambda x, y: x
