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

"""Replay entry for action/state modes: `scene_backend` / `task_backend` come from HDF5 `env_args` (with CLI override)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import h5py
import torch
from isaaclab.app import AppLauncher
from robofinals.utils.teleop_utils import get_state_by_frame
from robofinals.utils.env import ExecuteMode
from robofinals.utils.env import parse_env_cfg

CLOUD_INPUT_METADATA_PATH = Path("/input/cloud-input-param-metadata.json")
WORKSPACE_SCENE_ROOT = Path("/workspace")


def _load_replay_episode(handler, episode_name: str, device: str, replay_mode: str):
    """Load only trajectory inputs; recorded RGB is regenerated, never consumed."""
    from isaaclab.utils.datasets import EpisodeData
    from isaaclab.utils.datasets.hdf5_dataset_file_handler import convert_pose_quat_wxyz_to_xyzw

    group = handler._hdf5_data_group[episode_name]
    legacy = handler.is_legacy_quaternion_format()

    def read(node):
        if isinstance(node, h5py.Group):
            return {name: read(child) for name, child in node.items()}
        array = node[()]
        if legacy and node.name.rsplit("/", 1)[-1] == "root_pose" and array.shape[-1] == 7:
            array = convert_pose_quat_wxyz_to_xyzw(array)
        return torch.as_tensor(array, device=device)

    required = ("initial_state", "states" if replay_mode == "state" else "actions")
    episode = EpisodeData()
    episode.data = {name: read(group[name]) for name in required}
    for name in ("seed", "success"):
        if name in group.attrs:
            setattr(episode, name, group.attrs[name])
    episode.env_id = handler.get_env_name()
    print(f"[replay_local] loaded trajectory only: {', '.join(required)} (old RGB skipped)", flush=True)
    return episode


class _CameraFrameStore:
    """Append camera frames to disk during replay instead of retaining them in RAM."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file = h5py.File(self.path, "w")
        self.datasets: dict[str, h5py.Dataset] = {}

    def append(self, name: str, image: torch.Tensor) -> None:
        frame = image.detach().cpu().numpy()
        dataset = self.datasets.get(name)
        if dataset is None:
            dataset = self.file.create_dataset(
                name,
                shape=(0, *frame.shape),
                maxshape=(None, *frame.shape),
                dtype=frame.dtype,
                chunks=(1, *frame.shape),
                compression="lzf",
            )
            self.datasets[name] = dataset
        index = dataset.shape[0]
        dataset.resize((index + 1, *dataset.shape[1:]))
        dataset[index] = frame

    def names(self):
        return tuple(self.datasets)

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay local task episodes (action/state only).")
    parser.add_argument("--dataset_file", type=str, required=True, help="Input dataset .hdf5 file path.")
    parser.add_argument("--replay_mode", type=str, default="action", choices=["action", "state"], help="Replay mode.")
    parser.add_argument("--layout", type=str, default=None, help="Override scene layout path/name.")
    parser.add_argument(
        "--enable_full_local_scene", action="store_true", default=False,
        help="Load the complete local scene, matching full-scene teleoperation.",
    )
    parser.add_argument(
        "--scene_backend",
        type=str,
        default=None,
        help="Override scene backend (default: read from HDF5 env_args, else robocasa).",
    )
    parser.add_argument(
        "--task_backend",
        type=str,
        default=None,
        help="Override task backend (default: read from HDF5 env_args, else robocasa).",
    )
    parser.add_argument("--first_person_view", action="store_true", default=False, help="Use first-person view.")
    parser.add_argument("--record", action="store_true", default=False, help="Record replay rollout to dataset.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _resolve_layout_from_cloud_input_metadata(metadata_path: Path = CLOUD_INPUT_METADATA_PATH) -> str | None:
    """Resolve local USD layout path from cloud metadata."""
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        params = payload.get("params", {})
        scene_name = params.get("scene_name")
        layout = params.get("layout")

        scene_name = str(scene_name).strip() if scene_name else ""
        layout = str(layout).strip() if layout else ""

        # Prefer `params.layout` first so cloud metadata can force the exact target path.
        if layout and layout.endswith(".usd") and os.path.isabs(layout):
            return layout
        if scene_name and scene_name.endswith(".usd"):
            return scene_name if os.path.isabs(scene_name) else str(WORKSPACE_SCENE_ROOT / scene_name)

        if scene_name and layout:
            layout_file = layout if layout.endswith(".usd") else f"{layout}.usd"
            return str(WORKSPACE_SCENE_ROOT / scene_name / layout_file)

        if layout:
            return layout if layout.endswith(".usd") else f"{layout}.usd"
    except Exception as e:
        print(f"Failed to parse cloud input metadata ({metadata_path}): {e}")
    return None


def _to_env_action(action, env):
    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action, dtype=torch.float32, device=env.device)
    else:
        action = action.to(device=env.device, dtype=torch.float32)
    if action.ndim == 1:
        action = action.unsqueeze(0)
    elif action.ndim > 2:
        action = action.reshape(action.shape[0], -1)
    if action.shape[0] != 1:
        raise ValueError(f"Invalid action batch shape {tuple(action.shape)} for num_envs=1.")
    return action


def _record_video_frame(env, video_recorder, frame_store=None):
    if video_recorder is None:
        return
    obs = env.observation_manager.compute()
    policy_obs = obs.get("policy", {}) if isinstance(obs, dict) else {}
    # Some embodiment configs expose the same camera tag more than once.
    # Keep the first occurrence so each HDF5 camera stream and video panel is
    # written exactly once per replay transition.
    camera_names = list(dict.fromkeys(
        getattr(env.cfg.isaaclab_arena_env.embodiment, "active_observation_camera_names", [])
    ))

    images, used_names = [], []
    for camera_name in camera_names:
        image = policy_obs.get(camera_name)
        if isinstance(image, torch.Tensor) and image.ndim in (3, 4):
            image = image[0] if image.ndim == 4 else image
            images.append(image)
            used_names.append(camera_name)
            if frame_store is not None:
                if hasattr(frame_store, "append"):
                    frame_store.append(camera_name, image)
                else:
                    frame_store.setdefault(camera_name, []).append(image.detach().clone().cpu())

    if not images:
        return

    if len(images) == 1:
        combined, combined_name = images[0], used_names[0]
    else:
        max_height = max(img.shape[0] for img in images)
        padded = [img if img.shape[0] == max_height else torch.cat([
            img,
            torch.full((max_height - img.shape[0], img.shape[1], img.shape[2]), 255 if img.dtype == torch.uint8 else 1.0, dtype=img.dtype, device=img.device)
        ], dim=0) for img in images]
        combined = torch.cat(padded, dim=1)
        combined_name = "__".join(used_names)

    if video_recorder.video_writer is None:
        video_recorder.start_recording(combined_name, (int(combined.shape[0]), int(combined.shape[1])))
    video_recorder.add_frame(combined)


def _infer_scene_backend(env_args: dict, cli_override: str | None, scene_name: str) -> str:
    """HDF5 often stores a cached floorplan `usd_path` even for Robocasa; do not treat that as local backend."""
    if cli_override:
        return cli_override
    if "scene_backend" in env_args:
        return str(env_args["scene_backend"])
    sn = str(scene_name).strip()
    # Match parse_env_cfg/get_scene_type: only *semantic* scene string ending in .usd uses Local-Scene-Usd.
    if sn.endswith(".usd"):
        return "local"
    return "robocasa"


def _infer_task_backend(env_args: dict, cli_override: str | None, scene_backend: str) -> str:
    if cli_override:
        return cli_override
    if "task_backend" in env_args:
        return str(env_args["task_backend"])
    # CoolKettle 等任务只注册为 Robocasa-Task-*；勿在 robocasa 场景上推断为 local。
    if scene_backend == "local":
        return "local"
    return "robocasa"


def _resolve_scene_name_from_env_args(env_args: dict, layout_override: str | None) -> str:
    """Match replay_demos / teleop_main: scene string from HDF5 env_args or explicit --layout."""
    if layout_override:
        return layout_override
    if "scene_type" in env_args and "layout_id" in env_args and "style_id" in env_args:
        return f"{env_args['scene_type']}-{env_args['layout_id']}-{env_args['style_id']}"
    if "layout_id" in env_args and "style_id" in env_args:
        scene_type = (
            "libero"
            if env_args.get("usd_path") and "libero" in str(env_args["usd_path"])
            else "robocasakitchen"
        )
        return f"{scene_type}-{env_args['layout_id']}-{env_args['style_id']}"
    if env_args.get("usd_path"):
        return str(env_args["usd_path"])
    cloud = _resolve_layout_from_cloud_input_metadata()
    if cloud:
        print(
            "Using layout from cloud metadata: "
            f"{CLOUD_INPUT_METADATA_PATH} -> {cloud}"
        )
        return cloud
    raise KeyError(
        "Missing scene metadata in env_args for replay. "
        "Pass --layout, or record scene_type/layout_id/style_id or usd_path in env_args, "
        f"or provide {CLOUD_INPUT_METADATA_PATH}."
    )


def _build_local_env(
    args,
    app_launcher,
    env_args: dict,
    task_name: str,
    robot_name: str,
    scene_name: str,
    episode_name: str,
    *,
    scene_backend: str,
    task_backend: str,
):

    execute_mode = ExecuteMode.REPLAY_ACTION if args.replay_mode == "action" else ExecuteMode.REPLAY_STATE
    add_camera_to_observation = bool(app_launcher._enable_cameras and args.record)

    env_cfg = parse_env_cfg(
        enable_full_local_scene=args.enable_full_local_scene,
        scene_backend=scene_backend,
        task_backend=task_backend,
        task_name=task_name,
        robot_name=robot_name,
        scene_name=scene_name,
        robot_scale=1.0,
        device=args.device,
        num_envs=1,
        use_fabric=True,
        replay_cfgs={
            "hdf5_path": args.dataset_file,
            "ep_meta": env_args,
            "ep_names": [episode_name],
            "add_camera_to_observation": add_camera_to_observation,
        },
        first_person_view=args.first_person_view,
        enable_cameras=app_launcher._enable_cameras,
        execute_mode=execute_mode,
        usd_simplify=env_args.get("usd_simplify", False),
        seed=env_args.get("seed", None),
        sources=env_args.get("sources", env_args.get("source", None)),
        object_projects=env_args.get("object_projects", None),
        headless_mode=args.headless,
    )

    env_name = f"ReplayMainLocal-{task_name}-{robot_name}-v0"
    env_cfg.env_name = env_name
    env_cfg.terminations.time_out = None

    from robofinals.core.runtime.isaac import lifecycle_env_from_config

    return lifecycle_env_from_config(
        env_cfg,
        env_name=env_name,
        application="replay",
        task_name=task_name,
        robot_name=robot_name,
        scene_name=scene_name,
        execute_mode=execute_mode.name,
    )


def _replay_action_episode(env, episode_data, simulation_app, video_recorder=None, camera_frames: dict[str, list] | None = None):
    initial_state = episode_data.get_initial_state()
    if initial_state is None:
        return None
    actions = []
    while simulation_app.is_running() and (action := episode_data.get_next_action()) is not None:
        actions.append(_to_env_action(action, env))

    from robofinals.applications.replay import ReplayApplication, ReplayProfile, Trajectory

    task = getattr(getattr(env.cfg, "isaaclab_arena_env", None), "task", None)
    split_frame_idx: int | None = None

    class Observer:
        def on_session_start(self, _context):
            return None

        def on_transition(self, context, _transition):
            nonlocal split_frame_idx
            _record_video_frame(env, video_recorder, frame_store=camera_frames)
            if split_frame_idx is None and task is not None and hasattr(task, "check_part023_done"):
                try:
                    mask = task.check_part001_done(env)
                    if isinstance(mask, torch.Tensor) and mask.numel() > 0 and bool(mask.reshape(-1)[0].item()):
                        split_frame_idx = int(context.frame_index)
                except Exception:
                    pass

        def on_session_close(self, _context, _error):
            return None

    ReplayApplication(env.lifecycle, profile=ReplayProfile.ACTION, observers=(Observer(),)).replay(
        "replay",
        Trajectory(schema_version=1, initial_state=initial_state, actions=tuple(actions)),
    )
    return split_frame_idx


def _replay_state_episode(env, args, episode_data, simulation_app, is_relative: bool, video_recorder=None, camera_frames: dict[str, list] | None = None):
    initial_state = episode_data.get_initial_state()
    if initial_state is None:
        return
    states = []
    frame_index = 0
    while simulation_app.is_running() and (state := get_state_by_frame(episode_data, frame_index)) is not None:
        states.append(state)
        frame_index += 1

    from robofinals.applications.replay import ReplayApplication, ReplayProfile, Trajectory

    class Observer:
        initial_reset = True

        def on_session_start(self, _context):
            return None

        def on_transition(self, _context, _transition):
            if self.initial_reset:
                self.initial_reset = False
                return
            # State playback does not advance simulation time. Render even in
            # headless mode and invalidate camera caches for the new pose.
            env.sim.forward()
            env.sim.render()
            for sensor in env.scene.sensors.values():
                sensor.update(env.step_dt, force_recompute=True)
            _record_video_frame(env, video_recorder, frame_store=camera_frames)

        def on_session_close(self, _context, _error):
            return None

    ReplayApplication(env.lifecycle, profile=ReplayProfile.STATE, observers=(Observer(),)).replay(
        "replay",
        Trajectory(
            schema_version=1,
            initial_state=initial_state,
            states=tuple(states),
            metadata={"is_relative": is_relative},
        ),
    )


def main():
    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    parser = _build_parser()
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset_file)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    dataset_handler = HDF5DatasetFileHandler()
    open_mode = "r+" if args.replay_mode == "action" else "r"
    dataset_handler.open(dataset_path, mode=open_mode)
    if dataset_handler.get_num_episodes() == 0:
        print("No episodes found in dataset.")
        return

    episode_names = sorted(dataset_handler.get_episode_names(), key=lambda v: int(v.split("_")[-1]))
    episode_name = episode_names[-1]

    env_args = json.loads(dataset_handler._hdf5_data_group.attrs["env_args"])
    if "LW_API_ENDPOINT" in env_args:
        os.environ["LW_API_ENDPOINT"] = env_args["LW_API_ENDPOINT"]

    task_name, robot_name = env_args["task_name"], env_args["robot_name"]
    scene_name = _resolve_scene_name_from_env_args(env_args, args.layout)
    scene_backend = _infer_scene_backend(env_args, args.scene_backend, scene_name)
    task_backend = _infer_task_backend(env_args, args.task_backend, scene_backend)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Discover modules only after SimulationApp has started.  Several task
    # modules import pxr/Isaac Sim bindings; importing them before AppLauncher
    # causes USD schema wrappers to be initialized against the wrong runtime.
    from robofinals.utils.env import discover_and_import_robofinals_modules

    discover_and_import_robofinals_modules()

    from robofinals.core.compat import CompatibilityLayer

    CompatibilityLayer(backend=getattr(args, "physics_backend", None)).enable()

    from robofinals.utils.place_utils.env_utils import reset_physx
    env = _build_local_env(
        args,
        app_launcher,
        env_args,
        task_name,
        robot_name,
        scene_name,
        episode_name,
        scene_backend=scene_backend,
        task_backend=task_backend,
    )
    reset_physx(env)
    env.reset(seed=env.cfg.seed)

    if args.enable_full_local_scene and robot_name == "Galbot-Mobile" and task_name == "ChamberServiceTask":
        from pxr import UsdGeom

        # Galbot already spawns a visible ground at z=0. Hide only the
        # coplanar authored floor mesh, leaving both support colliders intact.
        floor = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Scene/FlatGrid/Environment/Geometry")
        if floor.IsValid():
            UsdGeom.Imageable(floor).MakeInvisible()
            print("[replay_local] hidden duplicate scene floor mesh; collision retained", flush=True)

    video_recorder = None
    if args.record and app_launcher._enable_cameras:
        from robofinals.utils.video_recorder import VideoRecorder
        video_save_dir = os.path.dirname(args.dataset_file)
        video_recorder = VideoRecorder(save_dir=video_save_dir, fps=30, task=task_name, robot=robot_name, layout=scene_name)
        print(f"[replay_local] video save dir: {video_recorder.save_dir}")

    episode_data = _load_replay_episode(dataset_handler, episode_name, env.device, args.replay_mode)
    if args.replay_mode == "action":
        camera_frames = {}
        split_frame_idx = _replay_action_episode(env, episode_data, simulation_app, video_recorder, camera_frames)
        try:
            ep_group = dataset_handler._hdf5_data_group[episode_name]
            ep_group.attrs["part001_split_frame_idx"] = -1 if split_frame_idx is None else int(split_frame_idx)
            ep_group.attrs["part001_split_found"] = split_frame_idx is not None
            ep_group.attrs["part001_split_condition"] = (
                "part001_in_bin & part001_released & part001_stable & part001_contact_ok"
            )
            dataset_handler._hdf5_data_group.file.flush()
            print(f"[replay_local] part001_split_frame_idx={ep_group.attrs['part001_split_frame_idx']}")
        except Exception as e:
            print(f"[replay_local] failed to write split_frame_idx to hdf5: {e}")
    else:
        camera_frames = _CameraFrameStore(
            Path(dataset_path).with_suffix(".camera_frames.h5")
        )
        is_relative = str(robot_name).lower().endswith("rel")
        _replay_state_episode(env, args, episode_data, simulation_app, is_relative, video_recorder, camera_frames)

        if args.record:
            camera_frames.flush()
            expected_frames = int(dataset_handler._hdf5_data_group[episode_name].attrs["num_samples"])
            if not camera_frames.names():
                raise RuntimeError("No camera frames captured; output dataset was not replaced.")
            for name in camera_frames.names():
                actual = camera_frames.file[name].shape[0]
                if actual != expected_frames:
                    raise RuntimeError(f"{name}: captured {actual} frames, expected {expected_frames}; output not replaced.")
            dataset_handler.close()
            new_dataset_path = os.path.join(os.path.dirname(dataset_path), "dataset_with_video.hdf5")

            with h5py.File(dataset_path, "r") as f_in, h5py.File(new_dataset_path, "w") as f_out:
                # Retain format_version so current XYZW poses are not later
                # interpreted as legacy WXYZ poses.
                for attr_name, value in f_in.attrs.items():
                    f_out.attrs[attr_name] = value
                f_in.copy("data", f_out, "data")
                for attr_name in f_in["data"].attrs.keys():
                    f_out["data"].attrs[attr_name] = f_in["data"].attrs[attr_name]

                # Persist the layout actually used for rendering.  Otherwise
                # the regenerated camera frames and the dataset metadata would
                # describe different USD scenes.
                output_env_args = json.loads(f_out["data"].attrs["env_args"])
                output_env_args["scene_name"] = scene_name
                output_env_args["usd_path"] = scene_name
                f_out["data"].attrs["env_args"] = json.dumps(output_env_args)

                ep_group = f_out["data"][episode_name]
                obs_group = ep_group.require_group("obs")
                for cam_name in camera_frames.names():
                    if cam_name in obs_group:
                        del obs_group[cam_name]
                    source_frames = camera_frames.file[cam_name]
                    target = obs_group.create_dataset(
                        cam_name,
                        shape=source_frames.shape,
                        dtype=source_frames.dtype,
                        chunks=(1, *source_frames.shape[1:]),
                        compression="lzf",
                    )
                    for start in range(0, source_frames.shape[0], 64):
                        target[start : start + 64] = source_frames[start : start + 64]

            print(f"Dataset with video saved to: {new_dataset_path}")

        camera_frames.close()
        camera_frames.path.unlink(missing_ok=True)

    if video_recorder is not None:
        video_recorder.stop_recording()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
