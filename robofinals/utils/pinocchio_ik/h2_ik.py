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

"""Pinocchio IK for Unitree H2 arms (x7s/piper-style reduced-model DLS)."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from robofinals.utils.pinocchio_loader import ensure_robotics_pinocchio_loaded

ensure_robotics_pinocchio_loaded()

import pinocchio as pin  # noqa: E402


H2_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
H2_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _default_urdf_path() -> str:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    robofinals_dir = os.path.dirname(os.path.dirname(cur_dir))
    return os.path.join(robofinals_dir, "utils", "h2_description", "H2_Plus.urdf")


class H2PinocchioIK:
    """Single-arm Pinocchio IK for H2 using a reduced model + damped least squares."""

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        joints_to_keep: Optional[list[str]] = None,
        ee_joint_name: Optional[str] = None,
        dls_lambda: float = 1e-2,
        max_iters: int = 80,
        tol: float = 1e-3,
        step_gain: float = 1.0,
    ):
        if urdf_path is None:
            urdf_path = _default_urdf_path()
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"H2 URDF not found: {urdf_path}")

        if joints_to_keep is None:
            joints_to_keep = list(H2_LEFT_ARM_JOINTS)
        if ee_joint_name is None:
            ee_joint_name = joints_to_keep[-1]

        full_model = pin.buildModelFromUrdf(urdf_path)
        keep = set(joints_to_keep)
        joints_to_lock_ids: list[int] = []
        for jid in range(1, full_model.njoints):
            jname = full_model.names[jid]
            if jname not in keep:
                joints_to_lock_ids.append(int(jid))

        ref_q = pin.neutral(full_model)
        self._model = pin.buildReducedModel(full_model, joints_to_lock_ids, ref_q)
        jee = self._model.getJointId(ee_joint_name)
        if jee <= 0:
            raise ValueError(f"Required joint '{ee_joint_name}' not found in reduced H2 model.")
        self._model.addFrame(
            pin.Frame(
                "ee",
                jee,
                pin.SE3(pin.Quaternion(1.0, 0.0, 0.0, 0.0), np.zeros(3)),
                pin.FrameType.OP_FRAME,
            )
        )
        self._frame_id = self._model.getFrameId("ee")
        self._data = self._model.createData()

        self._lambda = float(dls_lambda)
        self._max_iters = int(max_iters)
        self._tol = float(tol)
        self._gain = float(step_gain)
        self._q_lower = np.array(self._model.lowerPositionLimit, dtype=np.float64)
        self._q_upper = np.array(self._model.upperPositionLimit, dtype=np.float64)
        self._last_q: Optional[np.ndarray] = None
        # Teleop: keep moving with best-effort IK. Freezing on mild IK error causes
        # stop-and-jump stuttering when VR targets move quickly.
        self._err_gate_thresh = 1.0
        self._max_delta_per_step = 0.5
        self.joint_names = [self._model.names[jid] for jid in range(1, self._model.njoints)]

    @staticmethod
    def _homogeneous_from_pose(pos_wxyz: np.ndarray) -> np.ndarray:
        vec = np.asarray(pos_wxyz, dtype=np.float64).reshape(7)
        t = np.eye(4, dtype=np.float64)
        t[0:3, 3] = vec[0:3]
        w, x, y, z = vec[3:7]
        t[0:3, 0:3] = pin.Quaternion(w, x, y, z).toRotationMatrix()
        return t

    def _ensure_batch(self, batch_size: int):
        if self._last_q is None or self._last_q.shape[0] != batch_size:
            self._last_q = np.zeros((batch_size, self._model.nq), dtype=np.float64)

    def _ik_single(self, target_T: np.ndarray, warm_q: Optional[np.ndarray] = None) -> tuple[np.ndarray, bool, float]:
        q = np.zeros(self._model.nq, dtype=np.float64) if warm_q is None else np.asarray(warm_q, dtype=np.float64).copy()
        q = np.clip(q, self._q_lower, self._q_upper)
        target_se3 = pin.SE3(target_T[0:3, 0:3], target_T[0:3, 3])
        err_norm = float("inf")
        converged = False
        for _ in range(self._max_iters):
            pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)
            cur = self._data.oMf[self._frame_id]
            err6 = pin.log6(cur.inverse() * target_se3).vector
            err_norm = float(np.linalg.norm(err6))
            if err_norm < self._tol:
                converged = True
                break
            J = pin.computeFrameJacobian(
                self._model,
                self._data,
                q,
                self._frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            jjt = J @ J.T + (self._lambda ** 2) * np.eye(6)
            dq = self._gain * (J.T @ np.linalg.solve(jjt, err6))
            q = np.clip(q + dq, self._q_lower, self._q_upper)
        return q, converged, err_norm

    def solve_pose_to_joints(
        self,
        targets_pos_wxyz: np.ndarray,
        warm_start: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(targets_pos_wxyz, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
        batch = int(targets.shape[0])
        self._ensure_batch(batch)

        q_out = np.zeros((batch, self._model.nq), dtype=np.float64)
        success = np.zeros((batch,), dtype=bool)
        for i in range(batch):
            warm_q = None if warm_start is None else np.asarray(warm_start[i], dtype=np.float64)
            if warm_q is None and self._last_q is not None:
                warm_q = self._last_q[i]
            sol_q, converged, err_norm = self._ik_single(self._homogeneous_from_pose(targets[i]), warm_q)
            prev_q = None if self._last_q is None else self._last_q[i]
            # Prefer the latest DLS iterate. Only fall back to previous q on catastrophic error.
            if prev_q is not None and err_norm > self._err_gate_thresh:
                chosen_q = prev_q
            else:
                chosen_q = sol_q
            if prev_q is not None:
                delta = np.clip(chosen_q - prev_q, -self._max_delta_per_step, self._max_delta_per_step)
                chosen_q = prev_q + delta
            q_out[i] = chosen_q
            success[i] = bool(converged) and (err_norm <= self._err_gate_thresh)

        self._last_q = q_out.copy()
        return q_out, success

    def reset(self):
        self._last_q = None


class H2BimanualIK:
    """Two-arm Pinocchio IK wrapper for H2."""

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        dls_lambda: float = 1e-2,
        max_iters: int = 80,
        tol: float = 1e-3,
    ):
        self.left = H2PinocchioIK(
            urdf_path=urdf_path,
            joints_to_keep=list(H2_LEFT_ARM_JOINTS),
            ee_joint_name=H2_LEFT_ARM_JOINTS[-1],
            dls_lambda=dls_lambda,
            max_iters=max_iters,
            tol=tol,
        )
        self.right = H2PinocchioIK(
            urdf_path=urdf_path,
            joints_to_keep=list(H2_RIGHT_ARM_JOINTS),
            ee_joint_name=H2_RIGHT_ARM_JOINTS[-1],
            dls_lambda=dls_lambda,
            max_iters=max_iters,
            tol=tol,
        )

    def solve_pose_to_joints(
        self,
        left_targets_pos_wxyz: np.ndarray,
        right_targets_pos_wxyz: np.ndarray,
        left_warm_start: Optional[np.ndarray] = None,
        right_warm_start: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        l_q, l_succ = self.left.solve_pose_to_joints(left_targets_pos_wxyz, warm_start=left_warm_start)
        r_q, r_succ = self.right.solve_pose_to_joints(right_targets_pos_wxyz, warm_start=right_warm_start)
        return l_q, l_succ, r_q, r_succ

    def reset(self):
        self.left.reset()
        self.right.reset()
