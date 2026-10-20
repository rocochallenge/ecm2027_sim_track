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

"""LeRobot deferred viewport frame buffer: viewport CPU capture per step, flush without GPU replay."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Sequence

import numpy as np
import torch

from robofinals.utils.profile_utils import RollingPerfLogger


@dataclass
class BufferedFrame:
    joint_pos: np.ndarray
    action: np.ndarray
    cameras: dict[str, np.ndarray]


@dataclass
class EpisodeFrameBuffer:
    """In-memory CPU buffer for one deferred-capture teleop episode."""

    frames: list[BufferedFrame] = field(default_factory=list)
    _viewport_capture: Any | None = field(default=None, init=False, repr=False)
    _pending_joint_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _pending_action: np.ndarray | None = field(default=None, init=False, repr=False)
    _debug_append_count: int = field(default=0, init=False, repr=False)

    _capture_env: Any | None = field(default=None, init=False, repr=False)
    _perf: RollingPerfLogger = field(
        default_factory=lambda: RollingPerfLogger("viewport_capture"), init=False, repr=False
    )

    def clear(self) -> None:
        self.frames.clear()
        if self._viewport_capture is not None:
            self._viewport_capture.discard_pending()
        if self._capture_env is not None:
            setattr(self._capture_env, "_robofinals_viewport_capture_pending", False)
        self._capture_env = None
        self._pending_joint_pos = None
        self._pending_action = None
        self._debug_append_count = 0

    def _ensure_viewport_capture(self) -> Any:
        if self._viewport_capture is None:
            from robofinals.utils.viewport_capture import AsyncPolicyViewportCapture

            self._viewport_capture = AsyncPolicyViewportCapture()
        return self._viewport_capture

    def _append_completed_capture(self, cameras: dict[str, np.ndarray] | None) -> bool:
        if cameras is None:
            return False
        if self._pending_joint_pos is None or self._pending_action is None:
            raise RuntimeError("completed viewport capture has no matching robot state")
        self.frames.append(
            BufferedFrame(
                joint_pos=self._pending_joint_pos,
                action=self._pending_action,
                cameras=cameras,
            )
        )
        self._pending_joint_pos = None
        self._pending_action = None
        return True

    def append_step(
        self,
        env: Any,
        actions: torch.Tensor,
        *,
        viewports: dict[str, Any] | None,
        camera_policy_keys: Sequence[str],
        joint_pos_key: str = "joint_pos",
        skip_sim_render: bool = False,
    ) -> bool:
        """Collect a prior readback and submit this sample if the slot is free."""
        perf_enabled = self._perf.enabled
        perf_started = time.perf_counter() if perf_enabled else 0.0
        robot = getattr(getattr(env, "scene", None), "articulations", {}).get("robot")
        if robot is None:
            raise RuntimeError("robot articulation not found for deferred recording")
        joint_pos = robot.data.joint_pos[0].detach().cpu().float().numpy().reshape(-1)
        if isinstance(actions, torch.Tensor):
            action_row = actions[0] if actions.ndim > 1 else actions
            action = action_row.detach().cpu().float().numpy().reshape(-1)
        else:
            action = np.asarray(actions, dtype=np.float32).reshape(-1)

        state_done = time.perf_counter() if perf_enabled else 0.0
        capture = self._ensure_viewport_capture()
        completed = capture.take_completed()
        poll_done = time.perf_counter() if perf_enabled else 0.0
        if completed is not None:
            self._append_completed_capture(completed)
            setattr(env, "_robofinals_viewport_capture_pending", False)
        if capture.pending:
            if perf_enabled:
                self._perf.record(
                    total=poll_done - perf_started,
                    state_d2h=state_done - perf_started,
                    capture_poll=poll_done - state_done,
                    capture_submit=0.0,
                    pending_count=1.0,
                )
            return False

        self._pending_joint_pos = joint_pos
        self._pending_action = action
        try:
            debug = self._debug_append_count < 3
            if debug:
                print(f"[RecorderDebug] capture submit begin pending={capture.pending} skip_sim_render={skip_sim_render}", flush=True)
            capture.submit(viewports, camera_policy_keys)
            self._capture_env = env
            setattr(env, "_robofinals_viewport_capture_pending", True)
            if debug:
                print(f"[RecorderDebug] capture submit returned pending={capture.pending}; next Newton render will drive viewport callback", flush=True)
            self._debug_append_count += 1
            if perf_enabled:
                submit_done = time.perf_counter()
                self._perf.record(
                    total=submit_done - perf_started,
                    state_d2h=state_done - perf_started,
                    capture_poll=poll_done - state_done,
                    capture_submit=submit_done - poll_done,
                    pending_count=0.0,
                )
        except Exception:
            self._pending_joint_pos = None
            self._pending_action = None
            setattr(env, "_robofinals_viewport_capture_pending", False)
            self._capture_env = None
            raise
        return True

    def flush_pending_capture(self, env: Any | None = None) -> bool:
        """Finish one in-flight capture without entering a Kit update loop."""
        capture = self._viewport_capture
        if capture is None or not capture.pending:
            return False

        completed = capture.take_completed()
        if completed is None and env is not None:
            sim_render = getattr(getattr(env, "sim", None), "render", None)
            if callable(sim_render):
                sim_render()
                completed = capture.take_completed()
        if completed is None:
            capture.discard_pending()
            if self._capture_env is not None:
                setattr(self._capture_env, "_robofinals_viewport_capture_pending", False)
            self._capture_env = None
            self._pending_joint_pos = None
            self._pending_action = None
            print("[RecorderDebug] dropping unfinished viewport capture during shutdown", flush=True)
            return False

        appended = self._append_completed_capture(completed)
        if self._capture_env is not None:
            setattr(self._capture_env, "_robofinals_viewport_capture_pending", False)
        self._capture_env = None
        return appended

    def flush_cpu_frames_to_lerobot(
        self,
        lerobot_recorder: Any,
        *,
        camera_policy_keys: Sequence[str],
        task_description: str,
        joint_pos_key: str = "joint_pos",
    ) -> int:
        """Write buffered CPU frames to LeRobot (no sim replay, no GPU camera read)."""
        from robofinals.utils.lerobot_isaac_export import build_frame_from_policy_obs

        if not self.frames:
            return 0

        written = 0
        for buffered in self.frames:
            policy_obs: dict[str, Any] = {joint_pos_key: buffered.joint_pos}
            policy_obs.update(buffered.cameras)
            frame = build_frame_from_policy_obs(
                policy_obs,
                buffered.action,
                camera_policy_keys,
                task_description,
                joint_pos_key=joint_pos_key,
                camera_hwc_shapes=getattr(lerobot_recorder, "_camera_hwc_shapes", None),
            )
            if lerobot_recorder._dataset is None:
                lerobot_recorder._ensure_dataset(frame)
            assert lerobot_recorder._dataset is not None
            lerobot_recorder._dataset.add_frame(frame)
            written += 1
        self.clear()
        return written
