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

import math

import torch

from robofinals.core.checks.base_checker import BaseChecker


class TrajectorySmoothnessChecker(BaseChecker):
    """
    Trajectory smoothness checker measuring motion jerk in both Cartesian and joint space.

    Jerk is the third derivative of position/joint angle with respect to time.
    High jerk indicates sudden stops, vibrations, or poor trajectory quality.
    A warning is triggered when the Cartesian jerk magnitude exceeds jerk_threshold.

    Per-step metrics:
        avg_cartesian_jerk    Average Cartesian jerk magnitude over the episode (m/s³).
        rms_cartesian_jerk    RMS Cartesian jerk (more sensitive to outlier spikes) (m/s³).
        max_cartesian_jerk    Peak Cartesian jerk seen so far (m/s³).
        current_cartesian_jerk  Jerk magnitude at the current step (m/s³).
        jerk_ratio            avg_cartesian_jerk / jerk_threshold. Values > 1.0 indicate violation.
        avg_joint_jerk        Average joint-space jerk magnitude (rad/s³).
        rms_joint_jerk        RMS joint-space jerk (rad/s³).
        max_joint_jerk        Peak joint-space jerk seen so far (rad/s³).
        current_joint_jerk    Joint-space jerk at the current step (rad/s³).

    Optional baseline comparison (set baseline_jerk to enable):
        jerk_vs_baseline_pct  Percentage deviation of avg_cartesian_jerk from the baseline.
                              Negative means smoother than baseline, positive means rougher.
                              Formula: (avg - baseline) / baseline * 100.

    Optional per-frame recording (set record_frames=True to enable):
        Stores (replay_step, cart_jerk, joint_jerk) for every valid frame of env[0].
        Accessible via get_frame_log(). Spike analysis (frames exceeding spike_multiplier×avg)
        is computed post-hoc in get_frame_log().
        This is intended for offline analysis only; keep disabled during training.

    Adapted from RoboEval metric_rollout.py for Isaac Lab's tensor-parallel architecture.

    Args:
        jerk_threshold:    Cartesian jerk magnitude (m/s³) above which a warning is issued.
                           Default 100.0. Also used as the denominator for jerk_ratio.
        ee_link_name:      End-effector body link name. For X7S right arm: "right_hand_link".
        baseline_jerk:     Optional reference avg_cartesian_jerk from an expert/reference episode.
                           When set, jerk_vs_baseline_pct is included in the metrics output.
        record_frames:     If True, record per-frame jerk values for env[0] for offline analysis.
                           Default False. Enable for replay/debug; disable for training.
        spike_multiplier:  Frames with jerk > spike_multiplier × avg are considered spikes in
                           the frame log analysis. Default 5.0.
        warmup_steps:      Number of steps to skip after reset before accumulating jerk.
                           Must be >= 3 (the minimum required to fill the 3-point finite-
                           difference stencil). Default 7.
                           - Steps 0-2: stencil not yet valid (hard minimum).
                           - Steps 3+:  prev_accel buffer may still contain near-zero values
                             from reset, producing artificially large jerk spikes.
                           Increase this value if you observe initialization artifacts in the
                           frame log (e.g. joint_jerk > 10× the episode average in early steps).

    Warm-up window:
        Accumulation begins at step warmup_steps onward (step_counts >= warmup_steps).
        Confirmed for X7S: steps 3-4 show joint_jerk ~300k-567k rad/s³ due to cold buffer;
        warmup_steps=7 safely clears all initialization artifacts.
    """

    type = "trajectory_smoothness"

    def __init__(
        self,
        warning_on_screen=False,
        jerk_threshold=100.0,
        ee_link_name="right_hand_link",
        baseline_jerk=None,
        record_frames=False,
        spike_multiplier=5.0,
        warmup_steps=7,
    ):
        super().__init__(warning_on_screen)
        self.jerk_threshold = jerk_threshold
        self.ee_link_name = ee_link_name
        self.baseline_jerk = baseline_jerk
        self.record_frames = record_frames
        self.spike_multiplier = spike_multiplier
        # Clamp to minimum 3: the 3-point finite-difference stencil requires 3 prior frames.
        self.warmup_steps = max(3, int(warmup_steps))
        self._is_initialized = False
        self._ee_idx = None
        # Per-frame log for env[0]: list of {"step": int, "cart_jerk": float, "joint_jerk": float}
        self._frame_log = [] if record_frames else None

    def _init_buffers(self, num_envs, num_joints, device):
        # Cartesian space buffers (end-effector position)
        self._prev_cart_pos = torch.zeros((num_envs, 3), device=device)
        self._prev_cart_vel = torch.zeros((num_envs, 3), device=device)
        self._prev_cart_accel = torch.zeros((num_envs, 3), device=device)
        self._cart_jerk_sum = torch.zeros(num_envs, device=device)
        self._cart_jerk_sq_sum = torch.zeros(num_envs, device=device)   # for RMS
        self._max_cart_jerk = torch.zeros(num_envs, device=device)

        # Joint space buffers (all robot joints)
        self._prev_joint_pos = torch.zeros((num_envs, num_joints), device=device)
        self._prev_joint_vel = torch.zeros((num_envs, num_joints), device=device)
        self._prev_joint_accel = torch.zeros((num_envs, num_joints), device=device)
        self._joint_jerk_sum = torch.zeros(num_envs, device=device)
        self._joint_jerk_sq_sum = torch.zeros(num_envs, device=device)  # for RMS
        self._max_joint_jerk = torch.zeros(num_envs, device=device)

        self._step_counts = torch.zeros(num_envs, device=device)
        self._is_initialized = True

    def reset(self, env_ids=None):
        if not self._is_initialized:
            return
        if env_ids is None:
            self._prev_cart_pos.zero_()
            self._prev_cart_vel.zero_()
            self._prev_cart_accel.zero_()
            self._cart_jerk_sum.zero_()
            self._cart_jerk_sq_sum.zero_()
            self._max_cart_jerk.zero_()
            self._prev_joint_pos.zero_()
            self._prev_joint_vel.zero_()
            self._prev_joint_accel.zero_()
            self._joint_jerk_sum.zero_()
            self._joint_jerk_sq_sum.zero_()
            self._max_joint_jerk.zero_()
            self._step_counts.zero_()
            if self._frame_log is not None:
                self._frame_log.clear()
        else:
            self._prev_cart_pos[env_ids] = 0
            self._prev_cart_vel[env_ids] = 0
            self._prev_cart_accel[env_ids] = 0
            self._cart_jerk_sum[env_ids] = 0
            self._cart_jerk_sq_sum[env_ids] = 0
            self._max_cart_jerk[env_ids] = 0
            self._prev_joint_pos[env_ids] = 0
            self._prev_joint_vel[env_ids] = 0
            self._prev_joint_accel[env_ids] = 0
            self._joint_jerk_sum[env_ids] = 0
            self._joint_jerk_sq_sum[env_ids] = 0
            self._max_joint_jerk[env_ids] = 0
            self._step_counts[env_ids] = 0
            # Clear frame log only when env[0] is among the reset environments
            if self._frame_log is not None and (0 in env_ids.tolist()):
                self._frame_log.clear()

    def _get_ee_idx(self, robot):
        if self._ee_idx is None:
            ids, _ = robot.find_bodies(self.ee_link_name)
            if len(ids) == 0:
                raise ValueError(
                    f"[TrajectorySmoothnessChecker] Cannot find body '{self.ee_link_name}'. "
                    f"Available: {robot.data.body_names}"
                )
            self._ee_idx = ids[0]
        return self._ee_idx

    def _check(self, env):
        robot = env.scene["robot"]
        num_envs = env.num_envs
        device = robot.device
        dt = env.step_dt

        # Guard against degenerate dt (should not happen in normal simulation,
        # but protects against NaN/Inf cascading through three dt divisions).
        if dt <= 0:
            return {"success": True, "warning_text": None, "metrics": {}}

        cur_joint_pos = robot.data.joint_pos  # (num_envs, num_joints)
        num_joints = cur_joint_pos.shape[1]

        if not self._is_initialized:
            self._init_buffers(num_envs, num_joints, device)

        ee_idx = self._get_ee_idx(robot)
        cur_cart_pos = robot.data.body_state_w[:, ee_idx, 0:3]  # (num_envs, 3)

        # --- Cartesian jerk ---
        cur_cart_vel = (cur_cart_pos - self._prev_cart_pos) / dt
        cur_cart_accel = (cur_cart_vel - self._prev_cart_vel) / dt
        cart_jerk_vec = (cur_cart_accel - self._prev_cart_accel) / dt
        cur_cart_jerk = torch.norm(cart_jerk_vec, dim=-1)  # (num_envs,)

        # --- Joint space jerk ---
        cur_joint_vel = (cur_joint_pos - self._prev_joint_pos) / dt
        cur_joint_accel = (cur_joint_vel - self._prev_joint_vel) / dt
        joint_jerk_vec = (cur_joint_accel - self._prev_joint_accel) / dt
        cur_joint_jerk = torch.norm(joint_jerk_vec, dim=-1)  # (num_envs,)

        # Skip the first warmup_steps steps to avoid initialization impulses.
        # The minimum is 3 (stencil fill), but extra steps flush the near-zero
        # prev_accel buffer that causes artificially large jerk spikes after reset.
        valid_mask = self._step_counts >= self.warmup_steps

        self._cart_jerk_sum[valid_mask] += cur_cart_jerk[valid_mask]
        self._cart_jerk_sq_sum[valid_mask] += cur_cart_jerk[valid_mask] ** 2
        self._max_cart_jerk = torch.where(
            valid_mask & (cur_cart_jerk > self._max_cart_jerk), cur_cart_jerk, self._max_cart_jerk
        )
        self._joint_jerk_sum[valid_mask] += cur_joint_jerk[valid_mask]
        self._joint_jerk_sq_sum[valid_mask] += cur_joint_jerk[valid_mask] ** 2
        self._max_joint_jerk = torch.where(
            valid_mask & (cur_joint_jerk > self._max_joint_jerk), cur_joint_jerk, self._max_joint_jerk
        )

        # Record per-frame jerk for env[0] when frame logging is enabled.
        # This runs every valid step (step_counts > 2) and is intended for offline analysis.
        if self._frame_log is not None and valid_mask[0].item():
            self._frame_log.append({
                "step": int(self._step_counts[0].item()),
                "cart_jerk": round(cur_cart_jerk[0].item(), 4),
                "joint_jerk": round(cur_joint_jerk[0].item(), 4),
            })

        self._step_counts += 1
        self._prev_cart_pos = cur_cart_pos.clone()
        self._prev_cart_vel = cur_cart_vel.clone()
        self._prev_cart_accel = cur_cart_accel.clone()
        self._prev_joint_pos = cur_joint_pos.clone()
        self._prev_joint_vel = cur_joint_vel.clone()
        self._prev_joint_accel = cur_joint_accel.clone()

        is_violated = (cur_cart_jerk > self.jerk_threshold) & valid_mask

        # Jerk accumulation starts when step_counts > 4 (i.e. from the 6th call onward).
        # After that call step_counts is incremented, so the number of accumulated
        # samples equals max(0, step_counts - 5).  We floor at 1.0 to avoid division
        # by zero during the warm-up phase.
        valid_steps = max(1.0, (self._step_counts[0] - self.warmup_steps).item())
        avg_cart_jerk = (self._cart_jerk_sum[0] / valid_steps).item()
        # Protect math.sqrt from tiny negative values caused by floating-point rounding.
        rms_cart_jerk = math.sqrt(max(0.0, (self._cart_jerk_sq_sum[0] / valid_steps).item()))
        avg_joint_jerk = (self._joint_jerk_sum[0] / valid_steps).item()
        rms_joint_jerk = math.sqrt(max(0.0, (self._joint_jerk_sq_sum[0] / valid_steps).item()))

        # jerk_ratio: avg / threshold. <1 smooth, >1 exceeds threshold.
        jerk_ratio = round(avg_cart_jerk / max(self.jerk_threshold, 1e-6), 4)

        metrics = {
            "avg_cartesian_jerk": round(avg_cart_jerk, 3),
            "rms_cartesian_jerk": round(rms_cart_jerk, 3),
            "max_cartesian_jerk": round(self._max_cart_jerk[0].item(), 3),
            "current_cartesian_jerk": round(cur_cart_jerk[0].item(), 3),
            "jerk_ratio": jerk_ratio,
            "avg_joint_jerk": round(avg_joint_jerk, 3),
            "rms_joint_jerk": round(rms_joint_jerk, 3),
            "max_joint_jerk": round(self._max_joint_jerk[0].item(), 3),
            "current_joint_jerk": round(cur_joint_jerk[0].item(), 3),
        }

        # Optional baseline comparison
        if self.baseline_jerk is not None and self.baseline_jerk > 0:
            deviation_pct = (avg_cart_jerk - self.baseline_jerk) / self.baseline_jerk * 100.0
            metrics["jerk_vs_baseline_pct"] = round(deviation_pct, 1)

        if is_violated[0].item():
            warning_text = (
                f"trajectory_smoothness [HIGH] | cur={cur_cart_jerk[0].item():.1f} avg={avg_cart_jerk:.1f}"
                f" max={self._max_cart_jerk[0].item():.1f} m/s3 ratio={jerk_ratio:.2f}"
            )
        else:
            warning_text = (
                f"trajectory_smoothness | cur={cur_cart_jerk[0].item():.1f} avg={avg_cart_jerk:.1f}"
                f" max={self._max_cart_jerk[0].item():.1f} m/s3 ratio={jerk_ratio:.2f}"
            )

        return {
            "success": not is_violated[0].item(),
            "warning_text": warning_text,
            "metrics": metrics,
        }

    def get_frame_log(self):
        """
        Return per-frame jerk analysis for env[0]. Only available when record_frames=True.

        Returns a dict with:
            frames        List of {"step", "cart_jerk", "joint_jerk"} for each valid frame.
            cart_spikes   Subset of frames where cart_jerk > spike_multiplier × avg_cart_jerk.
            joint_spikes  Subset of frames where joint_jerk > spike_multiplier × avg_joint_jerk.
            summary       High-level stats (total_frames, n_cart_spikes, n_joint_spikes,
                          max_cart_step, max_joint_step).

        Returns None if record_frames=False or no frames have been recorded yet.
        """
        if self._frame_log is None or len(self._frame_log) == 0:
            return None

        frames = self._frame_log
        n = len(frames)

        cart_values = [f["cart_jerk"] for f in frames]
        joint_values = [f["joint_jerk"] for f in frames]

        avg_cart = sum(cart_values) / n
        avg_joint = sum(joint_values) / n

        cart_spike_thresh = avg_cart * self.spike_multiplier
        joint_spike_thresh = avg_joint * self.spike_multiplier

        cart_spikes = [f for f in frames if f["cart_jerk"] > cart_spike_thresh]
        joint_spikes = [f for f in frames if f["joint_jerk"] > joint_spike_thresh]

        # Find the step index of the global maximum for each metric
        max_cart_frame = max(frames, key=lambda f: f["cart_jerk"])
        max_joint_frame = max(frames, key=lambda f: f["joint_jerk"])

        return {
            "frames": frames,
            "cart_spikes": cart_spikes,
            "joint_spikes": joint_spikes,
            "summary": {
                "total_frames": n,
                "avg_cart_jerk": round(avg_cart, 3),
                "avg_joint_jerk": round(avg_joint, 3),
                "spike_multiplier": self.spike_multiplier,
                "cart_spike_threshold": round(cart_spike_thresh, 3),
                "joint_spike_threshold": round(joint_spike_thresh, 3),
                "n_cart_spikes": len(cart_spikes),
                "n_joint_spikes": len(joint_spikes),
                "max_cart_step": max_cart_frame["step"],
                "max_cart_value": max_cart_frame["cart_jerk"],
                "max_joint_step": max_joint_frame["step"],
                "max_joint_value": max_joint_frame["joint_jerk"],
            },
        }
