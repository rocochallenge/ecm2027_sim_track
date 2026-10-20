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

"""LeRobot v3 helpers for Isaac teleop and HDF5 batch convert.

Teleop writes v3 on disk at ``<dirname(dataset_file)>/<hdf5_stem>/`` by default, or at
``lerobot_dataset_root`` when set. Batch convert still expects
``dataset_with_video.hdf5`` per clip (from ``replay_main_local.py --record``), not teleop's raw HDF5.

Teleop-specific behavior: ``save_episode(parallel_encoding=False)`` (Kit breaks multiprocessing
ffmpeg); ``close()`` calls ``finalize()`` plus ``atexit`` (streaming temp dirs); PNG staging under
``images/`` is removed after each flush; ``attach_lerobot_recorder_export_hook`` ties LeRobot flush
to ``RecorderManager.export_episodes`` (success path, not only the ``T`` key).

Deferred viewport capture: ``defer_camera_capture`` captures Kit viewport CPU buffers each step (no RTX sensor
readback), then flushes LeRobot from the in-memory buffer without ``observation_manager.compute()``.
"""

from __future__ import annotations

import atexit
import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from robofinals.utils.profile_utils import RollingPerfLogger

from robofinals.core.backends import make_backend_runtime, resolve_backend_kind

try:
    from termcolor import colored as _colored
except Exception:  # pragma: no cover

    def _colored(msg: str, _color: str) -> str:
        return msg


def _log_ok(msg: str) -> None:
    print(_colored(msg, "green"), flush=True)

# Defaults aligned with ``convert_hdf5_to_lerobot_dataset.py``.
# Teleop in Kit uses synchronous encoding by default; background encoder workers
# can block Kit callbacks. The explicit argument remains overridable for offline use.
LEROBOT_STREAMING_KWARGS = {
    "streaming_encoding": False,
    "encoder_queue_maxsize": 0,
    "vcodec": "h264",
}

# Teleop ``from_args`` defaults (keep ``teleop_base.yml`` minimal; override via merged YAML when needed).
DEFAULT_LEROBOT_TELEOP_CAMERA_POLICY_KEYS: tuple[str, ...] = (
    "first_person_camera_rgb",
    "left_hand_camera_rgb",
    "right_hand_camera_rgb",
)
DEFAULT_LEROBOT_FRAME_START = 5
DEFAULT_LEROBOT_FPS = 30
DEFAULT_LEROBOT_RECORD_STRIDE = None
DEFAULT_LEROBOT_JOINT_POS_POLICY_KEY = "joint_pos"
DEFAULT_LEROBOT_STREAMING_ENCODING = False
DEFAULT_LEROBOT_ROBOT_TYPE_FALLBACK = "X7S-Abs"
DEFAULT_LEROBOT_TASK_DESCRIPTION_FALLBACK = "teleop"


def _teleop_camera_policy_keys_from_args(args: Any) -> list[str]:
    """Use YAML list when set and non-empty; otherwise ``DEFAULT_LEROBOT_TELEOP_CAMERA_POLICY_KEYS``."""
    cams = getattr(args, "lerobot_camera_policy_keys", None)
    if cams is None:
        return list(DEFAULT_LEROBOT_TELEOP_CAMERA_POLICY_KEYS)
    if isinstance(cams, str):
        return [cams] if cams.strip() else list(DEFAULT_LEROBOT_TELEOP_CAMERA_POLICY_KEYS)
    try:
        seq = list(cams)
    except TypeError:
        return list(DEFAULT_LEROBOT_TELEOP_CAMERA_POLICY_KEYS)
    if len(seq) == 0:
        return list(DEFAULT_LEROBOT_TELEOP_CAMERA_POLICY_KEYS)
    return [str(x) for x in seq]


def infer_policy_camera_hwc_shapes(env: Any, camera_policy_keys: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    """Map policy obs camera keys to (H, W, 3) from embodiment ``observation_cameras``."""
    shapes: dict[str, tuple[int, int, int]] = {}
    embodiment = getattr(getattr(env.cfg, "isaaclab_arena_env", None), "embodiment", None)
    obs_cams = getattr(embodiment, "observation_cameras", None) or {}
    for policy_key in camera_policy_keys:
        logical = policy_key[: -len("_rgb")] if policy_key.endswith("_rgb") else policy_key
        cam_info = obs_cams.get(logical)
        if isinstance(cam_info, dict) and "camera_cfg" in cam_info:
            cfg = cam_info["camera_cfg"]
            h, w = int(cfg.height), int(cfg.width)
            shapes[policy_key] = (h, w, 3)
        else:
            from robofinals.utils.viewport_capture import canonical_camera_hwc_shape

            shapes[policy_key] = canonical_camera_hwc_shape(policy_key)
    return shapes


def sync_viewport_capture_sizes(camera_hwc_shapes: dict[str, tuple[int, int, int]]) -> None:
    """Align Kit viewport capture/resize targets with embodiment camera resolutions."""
    from robofinals.utils import viewport_capture as vc

    for key, (h, w, _c) in camera_hwc_shapes.items():
        vc._POLICY_CAPTURE_TEXTURE_SIZE[key] = (int(w), int(h))


def should_defer_teleop_camera_sensors(args: Any) -> bool:
    """Skip RTX TiledCamera on wrist-camera USD prims; use Kit viewports instead.

    Newton always defers in interactive camera sessions so ``record`` cannot change
    physics/render timing. PhysX defers during recording when docked hand-camera
    viewports are shown, because Replicator and Kit share the same USD camera prims.
    """
    if not getattr(args, "enable_cameras", False):
        return False
    if getattr(args, "headless", False):
        return False
    if getattr(args, "record_lerobot", False) and getattr(args, "enable_multiple_viewports", False):
        return True
    runtime = make_backend_runtime(resolve_backend_kind(getattr(args, "physics_backend", None)))
    return runtime.recording.should_defer_camera_capture(args)


def _lerobot_record_stride_from_args(args: Any) -> int:
    """Subsample LeRobot frames during teleop to match ``lerobot_fps`` without blocking VR."""
    override = getattr(args, "lerobot_record_stride", DEFAULT_LEROBOT_RECORD_STRIDE)
    if override is not None:
        try:
            return max(1, int(override))
        except (TypeError, ValueError):
            pass
    step_hz = int(getattr(args, "step_hz", 50) or 50)
    fps = int(getattr(args, "lerobot_fps", DEFAULT_LEROBOT_FPS) or DEFAULT_LEROBOT_FPS)
    if fps <= 0:
        fps = DEFAULT_LEROBOT_FPS
    return max(1, int(round(step_hz / fps)))


VIDEO_INFO = {
    "video.fps": 30.0,
    "video.codec": "h264",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "has_audio": False,
}

# Env attribute read by ``monkey_patch.patch_recorder_manager_ep_meta`` after HDF5 export.
LEROBOT_EXPORT_HOOK_ATTR = "_lw_lerobot_on_recorder_export"


def attach_lerobot_recorder_export_hook(env: Any, recorder: Any | None) -> None:
    """Register ``recorder.on_recorder_flushed`` on ``env`` for the patched ``export_episodes``."""
    if env is None:
        return
    cb = getattr(recorder, "on_recorder_flushed", None) if recorder is not None else None
    setattr(env, LEROBOT_EXPORT_HOOK_ATTR, cb)


def build_lerobot_features(
    camera_basenames: Sequence[str],
    state_dim: int,
    action_dim: int,
    camera_shapes: dict[str, tuple[int, ...]] | Sequence[int],
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    shared_shape = None if isinstance(camera_shapes, dict) else tuple(int(x) for x in camera_shapes)
    for name in camera_basenames:
        shape = tuple(int(x) for x in camera_shapes[name]) if isinstance(camera_shapes, dict) else shared_shape
        if shape is None:
            raise KeyError(f"camera_shapes missing entry for {name!r}")
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
            "video_info": VIDEO_INFO,
        }
    return features


def _to_numpy_f32_vec(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy().reshape(-1).astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _action_vec_from_row(actions_row: torch.Tensor | np.ndarray) -> np.ndarray:
    """Normalize a per-env action row to a 1D float32 vector.

    Newton deferred capture passes a numpy 1D vector; indexing ``row[0]`` would
    keep only the first scalar. Torch rows may be 1D (already one env) or 2D.
    """
    if isinstance(actions_row, torch.Tensor):
        row = actions_row[0] if actions_row.ndim > 1 else actions_row
    else:
        arr = np.asarray(actions_row, dtype=np.float32)
        row = arr[0] if arr.ndim > 1 else arr
    return _to_numpy_f32_vec(row)


def _find_tensor_by_key(obj: Any, key: str) -> Any:
    """Depth-first search for ``key`` inside nested dict observations."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_tensor_by_key(v, key)
            if found is not None:
                return found
    return None


def flatten_obs_for_lerobot(
    env: Any,
    obs: dict[str, Any],
    camera_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Merge IsaacLab step observations into one flat dict (like HDF5 ``obs/*`` layout."""
    out: dict[str, Any] = {}
    pol = obs.get("policy")
    if isinstance(pol, dict):
        out.update(pol)
        cam_sub = out.pop("camera_obs", None)
        if isinstance(cam_sub, dict):
            out.update(cam_sub)
    gen = obs.get("embodiment_general_obs")
    if isinstance(gen, dict):
        for k, v in gen.items():
            out.setdefault(k, v)
    if "joint_pos" not in out and env is not None and getattr(env, "scene", None) is not None:
        robot = env.scene.articulations.get("robot")
        if robot is not None and hasattr(robot, "data"):
            out["joint_pos"] = robot.data.joint_pos

    camera_keys = list(camera_keys) if camera_keys else []
    for cam in camera_keys:
        if cam in out and out[cam] is not None:
            continue
        found = _find_tensor_by_key(obs, cam)
        if found is not None:
            out[cam] = found

    if camera_keys and env is not None and hasattr(env, "observation_manager"):
        missing = [c for c in camera_keys if c not in out or out.get(c) is None]
        if missing:
            try:
                extra = env.observation_manager.compute()
            except Exception:
                extra = None
            if isinstance(extra, dict):
                for cam in missing:
                    found = _find_tensor_by_key(extra, cam)
                    if found is not None:
                        out[cam] = found
    return out


def _image_to_numpy_hwc_u8(
    img: torch.Tensor | np.ndarray,
    policy_key: str | None = None,
    camera_hwc_shapes: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if policy_key is not None:
        from robofinals.utils.viewport_capture import canonical_camera_hwc_shape, _resize_rgb_hwc

        if camera_hwc_shapes and policy_key in camera_hwc_shapes:
            target_shape = camera_hwc_shapes[policy_key]
        else:
            target_shape = canonical_camera_hwc_shape(policy_key)
        if tuple(arr.shape) != target_shape:
            arr = _resize_rgb_hwc(arr, target_shape[:2])
    return np.ascontiguousarray(arr)


def build_frame_from_policy_obs(
    policy_obs: dict[str, Any],
    actions_row: torch.Tensor | np.ndarray,
    camera_policy_keys: Sequence[str],
    task_description: str,
    joint_pos_key: str = "joint_pos",
    camera_hwc_shapes: dict[str, tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    if joint_pos_key not in policy_obs:
        raise KeyError(
            f"policy obs missing {joint_pos_key!r}; keys: {list(policy_obs.keys())[:40]}"
        )
    jp = policy_obs[joint_pos_key]
    if isinstance(jp, torch.Tensor) and jp.ndim == 2:
        jp = jp[0]
    state = _to_numpy_f32_vec(jp)
    act = _action_vec_from_row(actions_row)
    frame: dict[str, Any] = {
        "observation.state": state,
        "action": act,
        "task": task_description,
    }
    for cam in camera_policy_keys:
        if cam not in policy_obs:
            raise KeyError(f"policy obs missing camera {cam!r}")
        frame[f"observation.images.{cam}"] = _image_to_numpy_hwc_u8(
            policy_obs[cam], cam, camera_hwc_shapes
        )
    return frame


def _lerobot_root_is_resumable(root: Path) -> bool:
    """Return whether ``root`` has all local metadata required by ``resume()``.

    LeRobot falls back to downloading missing metadata from the Hub.  That is
    inappropriate on Kit main thread and can make teleop appear frozen, so a
    partially written dataset must never enter the resume path.
    """
    info_path = root / "meta" / "info.json"
    try:
        with info_path.open("r", encoding="utf-8") as stream:
            info = json.load(stream)
        total_episodes = int(info.get("total_episodes", 0) or 0)
        total_frames = int(info.get("total_frames", 0) or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False

    if total_episodes <= 0 or total_frames <= 0:
        return False

    return (
        (root / "meta" / "tasks.parquet").is_file()
        and (root / "meta" / "stats.json").is_file()
        and any((root / "meta" / "episodes").rglob("*.parquet"))
        and any((root / "data").rglob("*.parquet"))
    )


def _archive_incomplete_lerobot_root(root: Path) -> Path | None:
    """Move a partial dataset aside so a clean local writer can start."""
    if not root.exists():
        return None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    archive = root.with_name(f"{root.name}.incomplete-{timestamp}")
    suffix = 1
    while archive.exists():
        archive = root.with_name(f"{root.name}.incomplete-{timestamp}-{suffix}")
        suffix += 1

    root.rename(archive)
    print(
        _colored(f"[LeRobot] archived incomplete dataset {root} -> {archive}", "yellow"),
        flush=True,
    )
    return archive


class TeleopLeRobotRecorder:
    """One LeRobot v3 dataset; save episodes only after successful Isaac recorder exports."""

    def __init__(
        self,
        *,
        dataset_root: str,
        repo_id: str,
        robot_type: str,
        task_description: str,
        camera_policy_keys: list[str],
        frame_start: int = 5,
        fps: int = 30,
        joint_pos_key: str = "joint_pos",
        streaming_encoding: bool = True,
        defer_camera_capture: bool = False,
        record_stride: int = 1,
    ):
        self.dataset_root = str(Path(dataset_root).expanduser().resolve())
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.task_description = task_description
        self.camera_policy_keys = list(camera_policy_keys)
        self.frame_start = int(frame_start)
        self.fps = int(fps)
        self.joint_pos_key = joint_pos_key
        self.streaming_encoding = bool(streaming_encoding)
        self.defer_camera_capture = bool(defer_camera_capture)
        self.record_stride = max(1, int(record_stride))
        self._viewports: dict[str, Any] = {}
        self._camera_hwc_shapes: dict[str, tuple[int, int, int]] | None = None
        self._episode_buffer = None
        if self.defer_camera_capture:
            from robofinals.utils.lerobot_common.episode_frame_buffer import EpisodeFrameBuffer
            self._episode_buffer = EpisodeFrameBuffer()
        self._dataset = None
        self._frame_in_episode = 0
        self._frames_since_flush = 0
        self._enabled = True
        self._closed = False
        self._capture_perf = RollingPerfLogger("lerobot_capture")
        atexit.register(self._atexit_finalize_writer)

    @classmethod
    def from_args(cls, args: Any, output_dir: str, output_stem: str) -> TeleopLeRobotRecorder | None:
        if not getattr(args, "record_lerobot", False):
            return None
        ext_root = getattr(args, "lerobot_dataset_root", None)
        if ext_root and str(ext_root).strip():
            root = str(Path(ext_root).expanduser().resolve())
        else:
            sub = output_stem if output_stem else "lerobot_dataset"
            root = str(Path(output_dir).expanduser().resolve() / str(sub))
        rid = getattr(args, "lerobot_repo_id", None) or Path(root).name
        robot = getattr(args, "lerobot_robot_type", None) or getattr(
            args, "robot", DEFAULT_LEROBOT_ROBOT_TYPE_FALLBACK
        )
        task_desc = getattr(args, "lerobot_task_description", None) or getattr(
            args, "task", DEFAULT_LEROBOT_TASK_DESCRIPTION_FALLBACK
        )
        cams = _teleop_camera_policy_keys_from_args(args)
        frame_start = int(getattr(args, "lerobot_frame_start", DEFAULT_LEROBOT_FRAME_START))
        fps = int(getattr(args, "lerobot_fps", DEFAULT_LEROBOT_FPS))
        jp_key = getattr(args, "lerobot_joint_pos_policy_key", DEFAULT_LEROBOT_JOINT_POS_POLICY_KEY)
        use_stream = bool(
            getattr(args, "lerobot_streaming_encoding", DEFAULT_LEROBOT_STREAMING_ENCODING)
        )
        defer_cameras = should_defer_teleop_camera_sensors(args)
        record_stride = _lerobot_record_stride_from_args(args)
        return cls(
            dataset_root=root,
            repo_id=str(rid),
            robot_type=str(robot),
            task_description=str(task_desc),
            camera_policy_keys=list(cams),
            frame_start=frame_start,
            fps=fps,
            joint_pos_key=str(jp_key),
            streaming_encoding=use_stream,
            defer_camera_capture=defer_cameras,
            record_stride=record_stride,
        )

    def _ensure_dataset(self, frame0: dict[str, Any]) -> None:
        if self._dataset is not None:
            return
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        state_dim = int(np.asarray(frame0["observation.state"]).reshape(-1).shape[0])
        action_dim = int(np.asarray(frame0["action"]).reshape(-1).shape[0])
        if self._camera_hwc_shapes:
            camera_shapes = dict(self._camera_hwc_shapes)
        else:
            from robofinals.utils.viewport_capture import canonical_camera_shapes

            camera_shapes = canonical_camera_shapes(self.camera_policy_keys)
        for cam in self.camera_policy_keys:
            key = f"observation.images.{cam}"
            if key in frame0:
                camera_shapes[cam] = tuple(int(x) for x in np.asarray(frame0[key]).shape)
        features = build_lerobot_features(
            self.camera_policy_keys, state_dim, action_dim, camera_shapes
        )
        Path(self.dataset_root).parent.mkdir(parents=True, exist_ok=True)
        create_kw = {
            **LEROBOT_STREAMING_KWARGS,
            "streaming_encoding": self.streaming_encoding,
        }
        root_resolved = str(Path(self.dataset_root).expanduser().resolve())
        root_path = Path(root_resolved)
        if _lerobot_root_is_resumable(root_path):
            self._dataset = self._resume_existing_lerobot_dataset(
                LeRobotDataset, root_resolved, create_kw
            )
            return
        if root_path.exists():
            _archive_incomplete_lerobot_root(root_path)
        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=root_resolved,
            fps=self.fps,
            robot_type=self.robot_type,
            features=features,
            **create_kw,
        )
        _log_ok(
            f"[LeRobot] dataset create root={root_resolved!r} repo_id={self.repo_id!r} "
            f"streaming_encoding={self.streaming_encoding} "
            f"state_dim={state_dim} action_dim={action_dim} cams={self.camera_policy_keys} "
            f"camera_shapes={camera_shapes} record_stride={self.record_stride}"
        )

    def _resume_existing_lerobot_dataset(self, LeRobotDataset: Any, root: str, create_kw: dict) -> Any:
        """Open existing on-disk dataset for append (offline; never pulls from Hub)."""
        try:
            ds = LeRobotDataset.resume(
                repo_id=self.repo_id,
                root=root,
                force_cache_sync=False,
                vcodec=str(create_kw.get("vcodec", "h264")),
                streaming_encoding=self.streaming_encoding,
                encoder_queue_maxsize=int(create_kw.get("encoder_queue_maxsize", 0)),
                batch_encoding_size=int(create_kw.get("batch_encoding_size", 1)),
            )
        except Exception as e:
            raise RuntimeError(
                f"[LeRobot] cannot resume existing dataset at {root!r}: {e!r}. "
                "Delete the directory or fix its layout."
            ) from e
        _log_ok(
            f"[LeRobot] dataset resume (append) root={root!r} repo_id={self.repo_id!r} "
            f"streaming_encoding={self.streaming_encoding}"
        )
        return ds

    def bind_env_camera_shapes(self, env: Any) -> None:
        """Set LeRobot/video feature resolutions from embodiment camera cfg (Galbot, X7S, ...)."""
        shapes = infer_policy_camera_hwc_shapes(env, self.camera_policy_keys)
        self._camera_hwc_shapes = shapes
        sync_viewport_capture_sizes(shapes)

    def attach_viewports(self, viewports: dict[str, Any] | None) -> None:
        """Bind Kit docked viewports for Newton viewport capture (``left_hand`` / ``right_hand``)."""
        self._viewports = dict(viewports) if viewports else {}

    def will_capture_at_index(self, idx: int) -> bool:
        """True when ``on_recording_step`` would capture at episode frame *idx*."""
        if not self._enabled:
            return False
        if idx < self.frame_start:
            return False
        return (idx - self.frame_start) % self.record_stride == 0

    def on_recording_started(self) -> None:
        self._frame_in_episode = 0
        self._frames_since_flush = 0
        if self._episode_buffer is not None:
            self._episode_buffer.clear()
        if self.record_stride > 1:
            print(
                _colored(
                    f"[LeRobot] teleop record_stride={self.record_stride} "
                    f"(capture ~every {self.record_stride} sim steps, target_fps={self.fps})",
                    "cyan",
                )
            )

    def on_recording_step(
        self,
        env: Any,
        obs: dict[str, Any],
        actions: torch.Tensor,
        *,
        skip_sim_render: bool = False,
    ) -> None:
        """One env.step while recording: skip first ``frame_start`` steps like offline convert."""
        if not self._enabled:
            return
        idx = self._frame_in_episode
        self._frame_in_episode += 1
        if idx < self.frame_start:
            return
        record_idx = idx - self.frame_start
        if record_idx % self.record_stride != 0:
            return
        if self.defer_camera_capture:
            if self._episode_buffer is None:
                raise RuntimeError("deferred camera capture requires an episode frame buffer")
            accepted = self._episode_buffer.append_step(
                env,
                actions,
                viewports=self._viewports,
                camera_policy_keys=self.camera_policy_keys,
                joint_pos_key=self.joint_pos_key,
                skip_sim_render=skip_sim_render,
            )
            if accepted:
                self._frames_since_flush += 1
            return
        capture_started = time.perf_counter() if self._capture_perf.enabled else 0.0
        policy_obs = flatten_obs_for_lerobot(env, obs, camera_keys=self.camera_policy_keys)
        flatten_done = time.perf_counter() if self._capture_perf.enabled else 0.0
        frame = build_frame_from_policy_obs(
            policy_obs,
            actions[0],
            self.camera_policy_keys,
            self.task_description,
            joint_pos_key=self.joint_pos_key,
            camera_hwc_shapes=self._camera_hwc_shapes,
        )
        frame_done = time.perf_counter() if self._capture_perf.enabled else 0.0
        if self._dataset is None:
            self._ensure_dataset(frame)
        dataset_done = time.perf_counter() if self._capture_perf.enabled else 0.0
        assert self._dataset is not None
        self._dataset.add_frame(frame)
        add_done = time.perf_counter() if self._capture_perf.enabled else 0.0
        self._frames_since_flush += 1
        if self._capture_perf.enabled:
            self._capture_perf.record(
                total=add_done - capture_started,
                flatten_obs=flatten_done - capture_started,
                build_frame=frame_done - flatten_done,
                ensure_dataset=dataset_done - frame_done,
                add_frame=add_done - dataset_done,
            )

    def _flush_deferred_cpu_buffer(self, env: Any | None = None) -> int:
        if self._episode_buffer is None:
            return 0
        self._episode_buffer.flush_pending_capture(env)
        if not self._episode_buffer.frames:
            return 0
        written = self._episode_buffer.flush_cpu_frames_to_lerobot(
            self,
            camera_policy_keys=self.camera_policy_keys,
            task_description=self.task_description,
            joint_pos_key=self.joint_pos_key,
        )
        return written

    def _save_episode_teleop(self) -> None:
        assert self._dataset is not None
        self._dataset.save_episode(parallel_encoding=False)
        self._remove_staging_images_tree()

    def _remove_staging_images_tree(self) -> None:
        """Remove non-streaming PNG staging (``images/``) after ffmpeg has built ``videos/``."""
        img_root = Path(self.dataset_root) / "images"
        if not img_root.is_dir():
            return
        try:
            shutil.rmtree(img_root)
        except OSError as e:
            print(_colored(f"[LeRobot] could not remove staging {img_root}: {e}", "yellow"))

    def on_recorder_flushed(self, env: Any | None = None) -> None:
        if self.defer_camera_capture:
            written = self._flush_deferred_cpu_buffer(env)
            if written > 0:
                self._save_episode_teleop()
                _log_ok(f"[LeRobot] save_episode ({written} frames, Newton viewport capture)")
            self._frames_since_flush = 0
            self._frame_in_episode = 0
            return
        if self._dataset is None or self._frames_since_flush == 0:
            self._frame_in_episode = 0
            return
        self._save_episode_teleop()
        _log_ok(f"[LeRobot] save_episode ({self._frames_since_flush} frames)")
        self._frames_since_flush = 0
        self._frame_in_episode = 0

    def on_stop_recording(self, env: Any | None = None) -> None:
        """Stop recording and discard an attempt that was not committed by the success hook."""
        self.on_recording_reset()

    def on_recording_reset(self) -> None:
        """Discard the current teleop attempt without saving an episode."""
        if self._episode_buffer is not None:
            self._episode_buffer.clear()
        if self._dataset is not None:
            self._dataset.clear_episode_buffer(delete_images=True)
        self._frames_since_flush = 0
        self._frame_in_episode = 0

    def _finalize_dataset_writer(self) -> None:
        """Discard uncommitted frames, finalize saved episodes, and release the writer."""
        self.on_recording_reset()
        ds = self._dataset
        self._dataset = None
        if ds is None:
            return
        try:
            ds.finalize()
            _log_ok("[LeRobot] dataset finalize()")
            self._remove_staging_images_tree()
        except Exception as e:
            print(_colored(f"[LeRobot] finalize failed: {e}", "yellow"))

    def finalize_dataset(self) -> None:
        """Finalize the on-disk dataset for upload while keeping the recorder usable."""
        if self._closed:
            return
        self._finalize_dataset_writer()
        self._frame_in_episode = 0
        self._frames_since_flush = 0
        if self._episode_buffer is not None:
            self._episode_buffer.clear()

    def _atexit_finalize_writer(self) -> None:
        """``atexit`` hook: ``finalize()`` is not reliably triggered via ``__del__`` in Kit."""
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Discard pending frames, finalize saved episodes, and drop staging images."""
        if self._closed:
            return
        self._closed = True
        self._enabled = False
        self._finalize_dataset_writer()
