from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
import isaaclab.utils.math as math_utils


class UR5NewtonAction(ActionTerm):
    """UR5 task-space action for Newton large-scene robot worlds.

    This follows the Newton large-scene teleop path: solve task-space commands
    with the backend robot world's finite-difference IK, then write joint
    position targets to the Isaac Lab articulation facade.
    """

    cfg: "UR5NewtonActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "UR5NewtonActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for UR5 body name {self.cfg.body_name!r}; "
                f"found {len(body_ids)}: {body_names}."
            )
        self._body_idx = int(body_ids[0])
        self._body_name = body_names[0]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.as_tensor(self.cfg.scale, device=self.device, dtype=torch.float32)

        self._joint_q_indices_by_env = [self._joint_q_indices(env_index) for env_index in range(self.num_envs)]
        initial_targets = self._configured_initial_joint_targets()
        if initial_targets is not None:
            self._snap_joint_positions_and_targets(initial_targets)
            self._hold_joint_targets = initial_targets
        else:
            self._hold_joint_targets = self._current_joint_positions_torch()
        self._processed_actions[:] = self._hold_joint_targets
        self._ik_solvers = [self._make_ik_solver(env_index) for env_index in range(self.num_envs)]
        self._write_joint_targets(self._hold_joint_targets)
        self._sync_newton_visual_state()

    @property
    def action_dim(self) -> int:
        return 6 if self.cfg.use_relative_mode else 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions[:, : self.action_dim]
        command = self._raw_actions * self._scale
        if self.cfg.use_relative_mode and torch.max(torch.abs(command)).item() <= 1.0e-8:
            self._processed_actions[:] = self._hold_joint_targets
            return

        ee_pos, ee_quat = self._current_ee_pose_torch()
        if self.cfg.use_relative_mode:
            target_pos, target_quat = math_utils.apply_delta_pose(ee_pos, ee_quat, command)
        else:
            target_pos = command[:, :3]
            target_quat = command[:, 3:7]

        rows = []
        for env_index, ik in enumerate(self._ik_solvers):
            q_seed = self._joint_positions(env_index)
            target = ik.solve(
                target_pos[env_index].detach().cpu().numpy(),
                target_quat[env_index].detach().cpu().numpy(),
                q_seed,
            )
            rows.append(target[list(self._joint_q_indices_by_env[env_index])])
        self._processed_actions[:] = torch.as_tensor(np.asarray(rows), device=self.device, dtype=torch.float32)
        self._hold_joint_targets[:] = self._processed_actions

    def apply_actions(self) -> None:
        self._write_joint_targets(self._processed_actions)

    def _write_joint_targets(self, targets: torch.Tensor) -> None:
        self._asset.set_joint_position_target_index(
            target=targets,
            joint_ids=self._joint_ids,
        )


    def _configured_initial_joint_targets(self) -> torch.Tensor | None:
        values = getattr(self.cfg, "initial_joint_positions", None)
        if values is None:
            default_by_joint = {
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.5708,
                "elbow_joint": 1.5708,
                "wrist_1_joint": -1.5708,
                "wrist_2_joint": -1.5708,
                "wrist_3_joint": 0.0,
            }
            if not all(joint_name in default_by_joint for joint_name in self._joint_names):
                return None
            values = tuple(default_by_joint[joint_name] for joint_name in self._joint_names)
        if len(values) != self._num_joints:
            raise ValueError(
                f"UR5 Newton initial_joint_positions length {len(values)} does not match "
                f"controlled joint count {self._num_joints}."
            )
        row = torch.as_tensor(values, device=self.device, dtype=torch.float32).reshape(1, self._num_joints)
        return row.repeat(self.num_envs, 1)

    def _snap_joint_positions_and_targets(self, targets: torch.Tensor) -> None:
        if hasattr(self._asset, "write_joint_position_to_sim_index"):
            self._asset.write_joint_position_to_sim_index(
                position=targets,
                joint_ids=self._joint_ids,
            )
        self._write_joint_targets(targets)
        self._sync_newton_visual_state()

    def _sync_newton_visual_state(self) -> None:
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        configured_targets = self._configured_initial_joint_targets()
        if env_ids is None:
            self._raw_actions.zero_()
            self._hold_joint_targets[:] = configured_targets if configured_targets is not None else self._current_joint_positions_torch()
            self._processed_actions[:] = self._hold_joint_targets
            self._snap_joint_positions_and_targets(self._hold_joint_targets)
        else:
            self._raw_actions[env_ids] = 0.0
            env_list = [int(env_id) for env_id in env_ids]
            if configured_targets is not None:
                current_positions = configured_targets[env_list]
            else:
                current_positions = self._current_joint_positions_torch(env_list)
            self._hold_joint_targets[env_list] = current_positions
            self._processed_actions[env_list] = current_positions
            if hasattr(self._asset, "write_joint_position_to_sim_index"):
                self._asset.write_joint_position_to_sim_index(
                    position=current_positions,
                    joint_ids=self._joint_ids,
                    env_ids=env_list,
                )
            self._asset.set_joint_position_target_index(
                target=current_positions,
                joint_ids=self._joint_ids,
                env_ids=env_list,
            )
            self._sync_newton_visual_state()

    def _make_ik_solver(self, env_index: int):
        from app.teleop.ik import EndEffectorIK

        world = self._asset.robot_worlds[env_index]
        joint_lower = _as_numpy(getattr(world.model, "joint_limit_lower", None), world.model.joint_coord_count)
        joint_upper = _as_numpy(getattr(world.model, "joint_limit_upper", None), world.model.joint_coord_count)
        return EndEffectorIK(
            model=world.model,
            state=world.state,
            ee_body_index=self._body_idx,
            active_q_indices=self._joint_q_indices_by_env[env_index],
            device=world.device,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        )

    def _joint_q_indices(self, env_index: int) -> tuple[int, ...]:
        world = self._asset.robot_worlds[env_index]
        indices = []
        missing = []
        for name in self._joint_names:
            index = world.joint_q_by_name.get(name)
            if index is None:
                missing.append(name)
            else:
                indices.append(int(index))
        if missing:
            raise ValueError(f"UR5 Newton action joint(s) not found in robot world: {', '.join(missing)}")
        return tuple(indices)

    def _joint_positions(self, env_index: int) -> np.ndarray:
        return np.asarray(self._asset.robot_worlds[env_index].current_joint_positions(), dtype=np.float32).copy()

    def _current_joint_positions_torch(self, env_indices: Sequence[int] | None = None) -> torch.Tensor:
        rows = []
        if env_indices is None:
            env_indices = range(len(self._asset.robot_worlds))
        for env_index in env_indices:
            env_index = int(env_index)
            world = self._asset.robot_worlds[env_index]
            positions = np.asarray(world.current_joint_positions(), dtype=np.float32)
            rows.append(positions[list(self._joint_q_indices_by_env[env_index])])
        return torch.as_tensor(np.asarray(rows), device=self.device, dtype=torch.float32)

    def _current_ee_pose_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = []
        quats = []
        for world in self._asset.robot_worlds:
            body_q = np.asarray(world.state.body_q.numpy(), dtype=np.float32)
            positions.append(body_q[self._body_idx, :3])
            quats.append(body_q[self._body_idx, 3:7])
        return (
            torch.as_tensor(np.asarray(positions), device=self.device, dtype=torch.float32),
            torch.as_tensor(np.asarray(quats), device=self.device, dtype=torch.float32),
        )


def _as_numpy(values, fallback_count: int) -> np.ndarray:
    if values is None:
        return np.full(int(fallback_count), np.nan, dtype=np.float32)
    return np.asarray(values.numpy() if hasattr(values, "numpy") else values, dtype=np.float32)


@configclass
class UR5NewtonActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = UR5NewtonAction

    preserve_order: bool = True
    joint_names: list[str] = MISSING
    body_name: str = MISSING
    scale: float | tuple[float, ...] = 1.0
    use_relative_mode: bool = True
    initial_joint_positions: tuple[float, ...] | None = None
