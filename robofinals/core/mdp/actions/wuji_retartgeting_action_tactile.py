import os
import time
from collections.abc import Sequence
from dataclasses import MISSING
from typing import Callable, Optional

import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass


class WujiRetargetingTactileAction(ActionTerm):
    cfg: "WujiRetargetingTactileActionCfg"
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: "WujiRetargetingTactileActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=True)
        all_joint_names = list(self._asset.data.joint_names)
        self._non_hand_joint_ids = [
            idx for idx, joint_name in enumerate(all_joint_names) if "finger" not in joint_name
        ]
        first_joint_name = self._joint_names[0] if self._joint_names else ""
        if first_joint_name.startswith("left_"):
            self._hold_flag_attr = "_hold_current_left_hand_joints"
        elif first_joint_name.startswith("right_"):
            self._hold_flag_attr = "_hold_current_right_hand_joints"
        else:
            self._hold_flag_attr = None
        self._last_target: torch.Tensor | None = None
        self._last_apply_ts: float | None = None
        self._default_control_dt = 1.0 / 50.0
        self._invalid_source_warned = False
        self._debug_enabled = os.environ.get("WUJI_HAND_DEBUG", "0") == "1"
        self._debug_every_n = max(1, int(os.environ.get("WUJI_HAND_DEBUG_EVERY_N", "50")))
        self._debug_apply_count = 0
        self._last_calib_debug_ts = 0.0
        self._joint_order_logged = False
        self._actuator_debug_logged = False
        self.reset()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if (
            self._use_manus_relative_hand_mode()
            and self._hold_flag_attr is not None
            and getattr(self._env.cfg, self._hold_flag_attr, False)
        ):
            current_target = self._get_current_joint_targets()
            if current_target is not None:
                self._processed_actions = current_target.detach().clone()
                self._fingers_actions.zero_()
                return
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
            self._last_calib_debug_ts = 0.0
        else:
            self._fingers_actions[env_ids, :] = processed_actions

    def _estimate_control_dt(self) -> float:
        env_step_dt = float(getattr(self._env, "step_dt", 0.0) or 0.0)
        if env_step_dt > 0.0:
            return env_step_dt
        return self._estimate_wall_clock_control_dt()

    def _estimate_wall_clock_control_dt(self) -> float:
        now = time.time()
        if self._last_apply_ts is None:
            self._last_apply_ts = now
            return self._default_control_dt
        dt = max(now - self._last_apply_ts, 1.0e-3)
        self._last_apply_ts = now
        return dt

    def _finger_velocity_budget(self, control_dt: float) -> float | None:
        if os.environ.get("MANUS_DISABLE_FINGER_VELOCITY_BUDGET", "0") == "1":
            return None
        explicit_budget = os.environ.get("MANUS_FINGER_VELOCITY_BUDGET")
        if explicit_budget not in (None, ""):
            return float(explicit_budget)
        cfg_budget = getattr(self._env.cfg, "_manus_finger_velocity_budget", None)
        if cfg_budget not in (None, ""):
            return float(cfg_budget)
        try:
            actuator = self._asset.actuators["fingers"]
            vel = getattr(actuator, "velocity_limit", None)
            if vel is None:
                vel = getattr(actuator, "velocity_limit_sim", None)
            if vel is None:
                return None
            vel_arr = torch.as_tensor(vel, device=self.device, dtype=torch.float32).reshape(-1)
            vel_arr = vel_arr[torch.isfinite(vel_arr)]
            if vel_arr.numel() == 0:
                return None
            return float(vel_arr.mean().item()) * control_dt
        except Exception:
            return None

    def _legacy_finger_velocity_budget(self, control_dt: float) -> float | None:
        try:
            vel = self._asset.actuators["fingers"].velocity_limit
            vel_arr = vel.reshape(-1)
            vel_arr = vel_arr[torch.isfinite(vel_arr)]
            if vel_arr.numel() == 0:
                return None
            return float(vel_arr.mean().item()) * control_dt
        except Exception:
            return None

    def _use_manus_relative_hand_mode(self) -> bool:
        # Tactile teleop now reuses the same absolute Manus hand mapping as the
        # non-tactile ViveManusDevice path. Keep the legacy relative-hand code
        # dormant to avoid changing unrelated tactile/arm features.
        return False

    def _apply_legacy_actions(self) -> None:
        processed_actions = self._fingers_actions.cpu().numpy()
        if np.sum(abs(processed_actions)) > 0.01:
            control_dt = self._estimate_control_dt()
            joint_pos_rad = self.cfg.post_process_fn(processed_actions, len(self._joint_ids))
            joint_pos_rad = torch.tensor(joint_pos_rad, dtype=torch.float32, device=self.processed_actions.device)

            velocity_budget = self._legacy_finger_velocity_budget(control_dt)
            if self._last_target is not None and velocity_budget is not None and velocity_budget > 0.0:
                joint_pos_rad = torch.clamp(
                    joint_pos_rad,
                    min=self._last_target - velocity_budget,
                    max=self._last_target + velocity_budget,
                )

            self._asset.set_joint_position_target(joint_pos_rad, self._joint_ids)
            self._last_target = joint_pos_rad.clone()

    def _get_joint_limit_tensors(self) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        try:
            joint_limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
            return joint_limits[..., 0], joint_limits[..., 1]
        except Exception:
            return None, None

    def _get_current_joint_targets(self) -> torch.Tensor | None:
        try:
            return self._asset.data.joint_pos[:, self._joint_ids]
        except Exception:
            return None

    def _get_current_joint_commands(self) -> torch.Tensor | None:
        try:
            return self._asset.data.joint_pos_target[:, self._joint_ids]
        except Exception:
            return None

    def _get_current_joint_positions(self) -> torch.Tensor | None:
        try:
            return self._asset.data.joint_pos[:, self._joint_ids]
        except Exception:
            return None

    def _maybe_log_left_actuator_debug(self) -> None:
        if self._actuator_debug_logged:
            return
        if self._hold_flag_attr != "_hold_current_left_hand_joints":
            return
        try:
            actuator = self._asset.actuators["fingers"]
        except Exception as exc:
            print(f"[WuJi left-hand] actuator_debug_failed=missing_fingers_actuator error={exc}")
            self._actuator_debug_logged = True
            return

        actuator_joint_names = list(getattr(actuator, "joint_names", []))
        actuator_joint_indices = getattr(actuator, "joint_indices", [])
        actuator_name_to_idx = {name: i for i, name in enumerate(actuator_joint_names)}
        missing = [name for name in self._joint_names if name not in actuator_name_to_idx]
        shared = [name for name in self._joint_names if name in actuator_name_to_idx]
        controlled_joint_ids = torch.as_tensor(self._joint_ids, dtype=torch.long).reshape(-1)

        def _fmt_values(value) -> str:
            if value is None:
                return "None"
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                return np.array2string(arr, precision=3, suppress_small=True, max_line_width=100000)
            except Exception:
                return str(value)

        def _take_joint_param(attr_name: str) -> str:
            value = getattr(actuator, attr_name, None)
            if value is None:
                return "None"
            try:
                tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
                picked = [float(tensor[actuator_name_to_idx[joint_name]].item()) for joint_name in shared]
                return np.array2string(np.asarray(picked, dtype=np.float32), precision=3, suppress_small=True)
            except Exception:
                return str(value)

        def _take_asset_buffer(attr_name: str) -> str:
            try:
                tensor = getattr(self._asset.data, attr_name)
                picked = tensor[0, self._joint_ids].detach().cpu().numpy().reshape(-1)
                return _fmt_values(picked)
            except Exception as exc:
                return f"unavailable ({exc})"

        def _take_physx_runtime(getter_name: str) -> str:
            try:
                tensor = getattr(self._asset.root_physx_view, getter_name)()
                picked = tensor[0, controlled_joint_ids].detach().cpu().numpy().reshape(-1)
                return _fmt_values(picked)
            except Exception as exc:
                return f"unavailable ({exc})"

        fixed_tendon_names = sorted(set(shared).intersection(set(getattr(self._asset, "fixed_tendon_names", []))))
        spatial_tendon_names = sorted(set(shared).intersection(set(getattr(self._asset, "spatial_tendon_names", []))))

        joint_lower, joint_upper = self._get_joint_limit_tensors()
        joint_limit_str = "None"
        if joint_lower is not None and joint_upper is not None:
            try:
                lower_deg = np.degrees(joint_lower[0].detach().cpu().numpy().reshape(-1))
                upper_deg = np.degrees(joint_upper[0].detach().cpu().numpy().reshape(-1))
                joint_limit_str = (
                    "lower="
                    + np.array2string(lower_deg, precision=1, suppress_small=True)
                    + " upper="
                    + np.array2string(upper_deg, precision=1, suppress_small=True)
                )
            except Exception:
                joint_limit_str = "unavailable"

        print(
            "[WuJi left-hand][actuator]\n"
            f"  controlled_joint_names={self._joint_names}\n"
            f"  actuator_joint_count={len(actuator_joint_names)} joint_indices={actuator_joint_indices}\n"
            f"  shared_joint_count={len(shared)} missing_joint_names={missing}\n"
            f"  joint_limits_deg={joint_limit_str}\n"
            f"  stiffness={_take_joint_param('stiffness')}\n"
            f"  damping={_take_joint_param('damping')}\n"
            f"  velocity_limit={_take_joint_param('velocity_limit')}\n"
            f"  effort_limit={_take_joint_param('effort_limit')}\n"
            f"  velocity_limit_sim={_take_joint_param('velocity_limit_sim')}\n"
            f"  effort_limit_sim={_take_joint_param('effort_limit_sim')}\n"
            f"  asset_buffer_joint_stiffness={_take_asset_buffer('joint_stiffness')}\n"
            f"  asset_buffer_joint_damping={_take_asset_buffer('joint_damping')}\n"
            f"  asset_buffer_joint_vel_limits={_take_asset_buffer('joint_vel_limits')}\n"
            f"  asset_buffer_joint_effort_limits={_take_asset_buffer('joint_effort_limits')}\n"
            f"  physx_runtime_stiffness={_take_physx_runtime('get_dof_stiffnesses')}\n"
            f"  physx_runtime_damping={_take_physx_runtime('get_dof_dampings')}\n"
            f"  physx_runtime_vel_limits={_take_physx_runtime('get_dof_max_velocities')}\n"
            f"  physx_runtime_effort_limits={_take_physx_runtime('get_dof_max_forces')}\n"
            f"  fixed_tendon_controlled_joints={fixed_tendon_names}\n"
            f"  spatial_tendon_controlled_joints={spatial_tendon_names}"
        )
        self._actuator_debug_logged = True

    def apply_actions(self):
        if not self._use_manus_relative_hand_mode():
            self._apply_legacy_actions()
            return

        if getattr(self._env.cfg, "_hold_current_non_hand_joints", False):
            non_hand_baseline = getattr(self._env.cfg, "_non_hand_joint_target_baseline", None)
            if non_hand_baseline is not None:
                self._asset.set_joint_position_target(
                    non_hand_baseline[:, self._non_hand_joint_ids],
                    self._non_hand_joint_ids,
                )

        if getattr(self._env.cfg, "_manus_fixed_hand_target_debug", False):
            baseline_attr = None
            if self._hold_flag_attr == "_hold_current_left_hand_joints":
                baseline_attr = "_manus_hand_target_baseline_left"
            elif self._hold_flag_attr == "_hold_current_right_hand_joints":
                baseline_attr = "_manus_hand_target_baseline_right"
            if baseline_attr is not None:
                fixed_target = getattr(self._env.cfg, baseline_attr, None)
                if fixed_target is not None:
                    fixed_target = torch.as_tensor(
                        fixed_target, dtype=torch.float32, device=self.processed_actions.device
                    )
                    if fixed_target.ndim == 1:
                        fixed_target = fixed_target.unsqueeze(0)
                    self._asset.set_joint_position_target(fixed_target, self._joint_ids)
                    self._last_target = fixed_target.detach().clone()
                    return

        if self._hold_flag_attr is not None and getattr(self._env.cfg, self._hold_flag_attr, False):
            current_target = self._get_current_joint_targets()
            if current_target is not None:
                self._asset.set_joint_position_target(current_target, self._joint_ids)
                self._processed_actions = current_target.detach().clone()
                self._last_target = current_target.detach().clone()
            return

        processed_actions = self._fingers_actions.cpu().numpy()
        source_is_all_zero = bool(np.max(np.abs(processed_actions)) < 1.0e-6)
        side = "left" if self._hold_flag_attr == "_hold_current_left_hand_joints" else "right"
        left_only_debug = bool(
            getattr(self._env.cfg, "_manus_left_only_debug", False)
            or os.environ.get("WUJI_LEFT_ONLY_DEBUG", "0") == "1"
        )
        if source_is_all_zero:
            fallback_target = self._last_target
            if fallback_target is None:
                fallback_target = self._get_current_joint_targets()
            if fallback_target is not None:
                if not self._invalid_source_warned:
                    if (not left_only_debug) or side == "left":
                        print(
                            f"[WuJi teleop] Ignoring invalid {side} hand Manus data: hand source packet is all zeros. "
                            "Reusing previous stable target."
                        )
                    self._invalid_source_warned = True
                self._asset.set_joint_position_target(fallback_target, self._joint_ids)
                self._processed_actions = fallback_target.detach().clone()
                self._last_target = fallback_target.detach().clone()
                return
        else:
            self._invalid_source_warned = False

        joint_pos_rad = None
        unclipped_joint_pos_rad = None
        mapped_joint_pos_np = None
        mapped_delta_np = None

        calib_origin_attr = None
        legacy_joint_angle_origin_attr = None
        calib_baseline_attr = None
        if self._hold_flag_attr == "_hold_current_left_hand_joints":
            calib_origin_attr = "_manus_hand_source_origin_left"
            legacy_joint_angle_origin_attr = "_manus_joint_angle_origin_left"
            calib_baseline_attr = "_manus_hand_target_baseline_left"
        elif self._hold_flag_attr == "_hold_current_right_hand_joints":
            calib_origin_attr = "_manus_hand_source_origin_right"
            legacy_joint_angle_origin_attr = "_manus_joint_angle_origin_right"
            calib_baseline_attr = "_manus_hand_target_baseline_right"

        calib_origin = getattr(self._env.cfg, calib_origin_attr, None) if calib_origin_attr is not None else None
        if (
            calib_origin is None
            and legacy_joint_angle_origin_attr is not None
            and getattr(self._env.cfg, "hand_retarget_mode", None) != "position"
        ):
            calib_origin = getattr(self._env.cfg, legacy_joint_angle_origin_attr, None)
        calib_baseline = getattr(self._env.cfg, calib_baseline_attr, None) if calib_baseline_attr is not None else None

        should_apply_hand_target = np.sum(abs(processed_actions)) > 0.01 or calib_baseline is not None
        if should_apply_hand_target:
            control_dt = self._estimate_control_dt()
            mapped_joint_pos_np = self.cfg.post_process_fn(processed_actions, len(self._joint_ids))
            joint_pos_np = mapped_joint_pos_np

            if calib_origin is not None and calib_baseline is not None:
                if torch.is_tensor(calib_origin):
                    origin_np = calib_origin.detach().cpu().numpy()
                else:
                    origin_np = np.asarray(calib_origin, dtype=np.float32)
                origin_np = origin_np.reshape(1, self.num_fingers, 4)
                delta_post_process_fn = getattr(self.cfg, "delta_post_process_fn", None)
                if delta_post_process_fn is not None:
                    joint_pos_np = delta_post_process_fn(processed_actions - origin_np, len(self._joint_ids))
                else:
                    mapped_origin = self.cfg.post_process_fn(origin_np, len(self._joint_ids))
                    joint_pos_np = joint_pos_np - mapped_origin
                delta_scale = float(getattr(self._env.cfg, "_manus_hand_delta_scale", 1.0))
                delta_limit = getattr(self._env.cfg, "_manus_hand_delta_limit", None)
                joint_pos_np = joint_pos_np * delta_scale
                if delta_limit is not None:
                    joint_pos_np = np.clip(joint_pos_np, -float(delta_limit), float(delta_limit))
                mapped_delta_np = np.asarray(joint_pos_np, dtype=np.float32).reshape(-1)
                joint_pos_rad = torch.as_tensor(
                    calib_baseline, dtype=torch.float32, device=self.processed_actions.device
                )
                if joint_pos_rad.ndim == 1:
                    joint_pos_rad = joint_pos_rad.unsqueeze(0)
                joint_pos_rad = joint_pos_rad + torch.tensor(
                    joint_pos_np, dtype=torch.float32, device=self.processed_actions.device
                )
            else:
                joint_pos_rad = torch.tensor(
                    joint_pos_np, dtype=torch.float32, device=self.processed_actions.device
                )
            unclipped_joint_pos_rad = joint_pos_rad.clone()

            joint_lower, joint_upper = self._get_joint_limit_tensors()
            if joint_lower is not None and joint_upper is not None:
                joint_pos_rad = torch.clamp(joint_pos_rad, min=joint_lower, max=joint_upper)

            velocity_budget = self._finger_velocity_budget(control_dt)
            if velocity_budget is not None and velocity_budget > 0.0:
                reference_target = self._last_target
                if reference_target is None and calib_baseline is not None:
                    reference_target = torch.as_tensor(
                        calib_baseline, dtype=torch.float32, device=self.processed_actions.device
                    )
                    if reference_target.ndim == 1:
                        reference_target = reference_target.unsqueeze(0)
                if reference_target is None:
                    reference_target = self._get_current_joint_targets()
                if reference_target is not None:
                    joint_pos_rad = torch.clamp(
                        joint_pos_rad,
                        min=reference_target - velocity_budget,
                        max=reference_target + velocity_budget,
                    )

            if joint_lower is not None and joint_upper is not None:
                joint_pos_rad = torch.clamp(joint_pos_rad, min=joint_lower, max=joint_upper)

        if joint_pos_rad is not None:
            self._asset.set_joint_position_target(joint_pos_rad, self._joint_ids)
            self._processed_actions = joint_pos_rad.detach().clone()
            self._last_target = joint_pos_rad.clone()
            if calib_origin is not None and calib_baseline is not None:
                now = time.time()
                target_delta = joint_pos_rad - torch.as_tensor(
                    calib_baseline, dtype=torch.float32, device=self.processed_actions.device
                )
                max_target_delta = float(target_delta.detach().abs().max().item()) if target_delta.numel() else 0.0
                should_log_left_focus = (
                    left_only_debug
                    and side == "left"
                    and (now - self._last_calib_debug_ts)
                    >= float(getattr(self._env.cfg, "_manus_left_debug_interval_s", 1.0))
                )
                should_log_default = (
                    (not left_only_debug)
                    and max_target_delta >= 0.05
                    and (now - self._last_calib_debug_ts) >= 1.0
                )
                if should_log_left_focus or should_log_default:
                    self._last_calib_debug_ts = now
                    if should_log_left_focus:
                        self._maybe_log_left_actuator_debug()
                    raw_manus_rad = processed_actions.reshape(-1)
                    target_joint_rad = joint_pos_rad.detach().cpu().numpy().reshape(-1)
                    target_delta_rad = target_delta.detach().cpu().numpy().reshape(-1)
                    mapped_joint_rad = None
                    if mapped_joint_pos_np is not None:
                        mapped_joint_rad = np.asarray(mapped_joint_pos_np, dtype=np.float32).reshape(-1)
                    command_joint_rad = None
                    current_command = self._get_current_joint_commands()
                    if current_command is not None:
                        command_joint_rad = current_command.detach().cpu().numpy().reshape(-1)
                    actual_joint_rad = None
                    tracking_err_max = None
                    actual_joint_pos = self._get_current_joint_positions()
                    if actual_joint_pos is not None:
                        actual_joint_rad = actual_joint_pos.detach().cpu().numpy().reshape(-1)
                        tracking_err = (joint_pos_rad - actual_joint_pos).detach().abs()
                        tracking_err_max = float(tracking_err.max().item()) if tracking_err.numel() else 0.0
                    applied_effort = None
                    computed_effort = None
                    try:
                        actuator = self._asset.actuators["fingers"]
                        actuator_joint_names = list(getattr(actuator, "joint_names", []))
                        actuator_name_to_idx = {name: i for i, name in enumerate(actuator_joint_names)}
                        pick_indices = [
                            actuator_name_to_idx[name]
                            for name in self._joint_names
                            if name in actuator_name_to_idx
                        ]
                        if pick_indices:
                            applied_effort = actuator.applied_effort[0, pick_indices].detach().cpu().numpy().reshape(-1)
                            computed_effort = actuator.computed_effort[0, pick_indices].detach().cpu().numpy().reshape(-1)
                    except Exception:
                        applied_effort = None
                        computed_effort = None
                    clip_max = 0.0
                    if unclipped_joint_pos_rad is not None:
                        clip_max = float((joint_pos_rad - unclipped_joint_pos_rad).detach().abs().max().item())
                    if should_log_left_focus:
                        if not self._joint_order_logged:
                            self._joint_order_logged = True
                            print(f"[WuJi left-hand] joint_order={self._joint_names}")

                        def _fmt_deg(arr: np.ndarray | None) -> str:
                            if arr is None:
                                return "None"
                            return np.array2string(
                                np.degrees(np.asarray(arr, dtype=np.float32).reshape(-1)),
                                precision=1,
                                suppress_small=True,
                                max_line_width=100000,
                            )

                        tracking_err_max_deg_str = (
                            f"{np.degrees(tracking_err_max):.1f}" if tracking_err_max is not None else "None"
                        )
                        print(
                            "[WuJi left-hand]\n"
                            f"  manus_raw_deg={_fmt_deg(raw_manus_rad)}\n"
                            f"  mapped_wuji_deg={_fmt_deg(mapped_joint_rad)}\n"
                            f"  mapped_delta_deg={_fmt_deg(mapped_delta_np)}\n"
                            f"  sim_target_deg={_fmt_deg(target_joint_rad)}\n"
                            f"  sim_command_deg={_fmt_deg(command_joint_rad)}\n"
                            f"  sim_joint_deg={_fmt_deg(actual_joint_rad)}\n"
                            f"  sim_error_deg={_fmt_deg(target_joint_rad - actual_joint_rad if actual_joint_rad is not None else None)}\n"
                            f"  actuator_computed_effort={np.array2string(np.asarray(computed_effort, dtype=np.float32), precision=2, suppress_small=True, max_line_width=100000) if computed_effort is not None else 'None'}\n"
                            f"  actuator_applied_effort={np.array2string(np.asarray(applied_effort, dtype=np.float32), precision=2, suppress_small=True, max_line_width=100000) if applied_effort is not None else 'None'}\n"
                            f"  clip_max_deg={np.degrees(clip_max):.1f} tracking_err_max_deg={tracking_err_max_deg_str}"
                        )
                    else:
                        source_head = raw_manus_rad[:8]
                        delta_head = target_delta_rad[:8]
                        target_head = target_joint_rad[:8]
                        command_head = command_joint_rad[:8] if command_joint_rad is not None else None
                        actual_head = actual_joint_rad[:8] if actual_joint_rad is not None else None
                        tracking_err_str = f"{tracking_err_max:.3f}" if tracking_err_max is not None else "None"
                        print(
                            f"[WuJi hand apply][{side}] "
                            f"source_head={np.array2string(source_head, precision=3)} "
                            f"delta_head={np.array2string(delta_head, precision=3)} "
                            f"target_head={np.array2string(target_head, precision=3)} "
                            f"command_head={np.array2string(command_head, precision=3) if command_head is not None else None} "
                            f"actual_head={np.array2string(actual_head, precision=3) if actual_head is not None else None} "
                            f"max_target_delta={max_target_delta:.3f} "
                            f"clip_max={clip_max:.3f} "
                            f"tracking_err_max={tracking_err_str} "
                            f"vel_budget={velocity_budget if 'velocity_budget' in locals() else None}"
                        )
            if self._debug_enabled:
                self._debug_apply_count += 1
                if self._debug_apply_count == 1 or (self._debug_apply_count % self._debug_every_n) == 0:
                    source_head = processed_actions.reshape(-1)[:8]
                    target_head = joint_pos_rad.detach().cpu().numpy().reshape(-1)[:8]
                    print(
                        f"[wuji-hand-debug][{side}] "
                        f"source_head={np.array2string(source_head, precision=4)} "
                        f"target_head={np.array2string(target_head, precision=4)} "
                        f"hold={getattr(self._env.cfg, self._hold_flag_attr, False) if self._hold_flag_attr else None} "
                        f"delta_scale={getattr(self._env.cfg, '_manus_hand_delta_scale', None)} "
                        f"delta_limit={getattr(self._env.cfg, '_manus_hand_delta_limit', None)}"
                    )

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
class WujiRetargetingTactileActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = WujiRetargetingTactileAction
    # retargeting_index: list[int] = MISSING
    joint_names: list[str] = MISSING
    post_process_fn: Callable = lambda x, y: x
    delta_post_process_fn: Optional[Callable] = None
