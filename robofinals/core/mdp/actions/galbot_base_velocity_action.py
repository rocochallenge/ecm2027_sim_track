"""Galbot mobile-base velocity command action."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import MISSING
import os
from typing import Sequence

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from robofinals.utils.isaac_data_compat import as_torch, sim_quat_raw_to_xyzw_torch


class GalbotBaseVelocityAction(ActionTerm):
    """Accept body-frame ``[vx, vy, wz]`` and drive the free base and omni wheels."""

    cfg: "GalbotBaseVelocityActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "GalbotBaseVelocityActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._wheel_ids, _ = self._asset.find_joints(self.cfg.wheel_joint_names, preserve_order=True)
        if len(self._wheel_ids) != 4:
            raise ValueError(f"Expected four Galbot wheel joints, found {len(self._wheel_ids)}")
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._wheel_scale = float(self.cfg.wheel_scale)
        try:
            self._smoothing_alpha = float(
                os.environ.get("ROBOFINALS_GALBOT_WHEEL_SMOOTHING_ALPHA", str(self.cfg.smoothing_alpha))
            )
        except ValueError:
            self._smoothing_alpha = float(self.cfg.smoothing_alpha)
        self._smoothing_alpha = min(max(self._smoothing_alpha, 0.0), 1.0)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @contextmanager
    def adapt_full_action(self, action):
        """Accept native 19-D commands or legacy 20-D recorded wheel values.

        Legacy wheel values already include teleop smoothing. Decode them
        without applying smoothing a second time. Both formats use apply_actions.
        """
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2 or action.shape[-1] not in (19, 20):
            raise ValueError("Galbot-Mobile expects (num_envs, 19) or (num_envs, 20) actions")
        if action.shape[0] not in (1, self.num_envs):
            raise ValueError(f"Expected action batch 1 or {self.num_envs}, got {action.shape[0]}")
        if not torch.isfinite(action).all():
            raise ValueError("Galbot-Mobile actions must be finite")
        legacy = action.shape[-1] == 20
        if legacy:
            w1, w2, w3, w4 = action[:, 16:20].unbind(dim=-1)
            velocity = torch.stack((
                (w1 + w2 + w3 + w4) / 4,
                (-w1 + w2 + w3 - w4) / 4,
                (-w1 + w2 - w3 + w4) / 4,
            ), dim=-1)
            # Project arbitrary four-wheel inputs onto planar body motion.
            action = torch.cat((action[:, :16], velocity), dim=-1)
        previous = getattr(self, "_legacy_wheel_input", False)
        self._legacy_wheel_input = legacy
        try:
            yield action
        finally:
            self._legacy_wheel_input = previous

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions[:, :3]
        if getattr(self, "_legacy_wheel_input", False):
            self._processed_actions[:] = self._raw_actions
        else:
            self._processed_actions[:] = torch.lerp(
                self._processed_actions, self._raw_actions, self._smoothing_alpha
            )

    def apply_actions(self) -> None:
        vx, vy, wz = self._processed_actions.unbind(dim=-1)

        # Preserve the existing Galbot omni-wheel mapping used by teleop.
        wheel_vel = torch.stack(
            (vx - vy - wz, vx + vy + wz, vx + vy - wz, vx - vy + wz), dim=-1
        )
        self._asset.set_joint_velocity_target(
            wheel_vel * self._wheel_scale, joint_ids=self._wheel_ids
        )

        quat_xyzw = sim_quat_raw_to_xyzw_torch(as_torch(self._asset.data.root_quat_w))
        vel_body = torch.stack((vx, vy, torch.zeros_like(vx)), dim=-1)
        vel_world = quat_apply(quat_xyzw, vel_body)
        root_vel = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
        root_vel[:, :3] = vel_world
        root_vel[:, 5] = wz
        self._asset.write_root_velocity_to_sim(root_vel)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


@configclass
class GalbotBaseVelocityActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = GalbotBaseVelocityAction
    wheel_joint_names: list[str] = MISSING
    wheel_scale: float = 0.01
    smoothing_alpha: float = 0.1
