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

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import tqdm

from isaaclab.app import AppLauncher

from robofinals.utils.isaac_data_compat import as_torch
from robofinals.utils.isaaclab_utils import get_robot_joint_target_from_scene, update_sensors
from robofinals.utils.log_utils import get_default_logger, get_logger
from robofinals.utils.video_recorder import VideoProcessor, calculate_camera_layout, compute_replay_video_shape

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument("--width", type=int, default=1920, help="Width of the rendered image.")
parser.add_argument("--height", type=int, default=1080, help="Height of the rendered image.")
parser.add_argument("--without_image", action="store_true", default=False, help="without image")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument("--robot_scale", type=float, default=1.0, help="robot scale")
parser.add_argument("--first_person_view", action="store_true", default=False, help="first person view")
parser.add_argument("--replay_mode", type=str, default="action", help="replay mode(action or joint_target)")
parser.add_argument("--layout", type=str, default=None, help="layout name")
parser.add_argument("--replay_all_clips", action="store_true", help="replay all clips, otherwise only replay the last clips")
parser.add_argument("--record", action="store_true", default=False, help="record the replayed actions")
parser.add_argument("--demo", type=int, default=-1, help="demo num in hdf5.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Isaac Lab Fabric.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True

if args_cli.replay_mode not in ("action", "joint_target"):
    raise ValueError(f"Invalid replay mode: {args_cli.replay_mode}, can only be 'action' or 'joint_target'")

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

if not args_cli.without_image:
    import cv2


def _resolve_replay_scene(layout, env_args, project_root=None):
    project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[4]

    def existing_path(value):
        if not value:
            return None
        path = Path(os.path.expandvars(os.path.expanduser(str(value))))
        candidates = [path]
        if not path.is_absolute():
            candidates.append(project_root / path)
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        return None

    if layout:
        return existing_path(layout) or os.path.expanduser(layout)

    recorded_scene = env_args.get("usd_path")
    scene_name = env_args.get("scene_name")
    resolved_recorded = existing_path(recorded_scene)
    if resolved_recorded:
        return resolved_recorded

    resolved_scene_name = existing_path(scene_name)
    return resolved_scene_name or recorded_scene or scene_name


# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
from robofinals.core.compat import CompatibilityLayer

CompatibilityLayer(backend=getattr(args_cli, "physics_backend", None)).enable()

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import torch

if not args_cli.headless:
    from robofinals.core.devices.keyboard.se3_keyboard import Se3Keyboard
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from robofinals.utils.place_utils.env_utils import set_seed

is_paused = False

obj_state_logger = get_logger("obj_state_logger")


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> (bool, str):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index].squeeze(0)
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name].squeeze(0)
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


def get_active_check_points(episode_data):
    """Get the active check points from the episode data."""
    active_check_points = []
    check_points = episode_data.data.get("checkpoints", {})
    frame_index_triggered = check_points.get("frame_index_triggered")
    frame_index = check_points.get("frame_index")

    if frame_index_triggered is None or frame_index is None:
        return active_check_points

    for i in range(len(frame_index_triggered)):
        if frame_index_triggered[i].item() > 0:
            active_check_points.append(int(frame_index[i].item()))
    return active_check_points


class ActionReplayObserver:
    """Collect action/joint-target replay artifacts from SessionRunner transitions."""

    def __init__(self, env, episode_data, *, mode, args, video_processor=None, validate_states=False):
        self.env = env
        self.episode_data = episode_data
        self.mode = mode
        self.args = args
        self.video_processor = video_processor
        self.validate_states = validate_states
        self.ee_poses = []
        self.joint_pos = []
        self.joint_targets = []
        self.gt_joint_targets = []
        self.object_states = {}
        self.object_forces = {}
        self.state_validation = []
        self.success = False
        self.active_check_points = get_active_check_points(episode_data)
        self.checkpoint_path = None

    def on_session_start(self, _context):
        return None

    def on_transition(self, context, transition):
        from robofinals.core.session import StepCommand

        if not isinstance(transition.command, StepCommand):
            return
        result = transition.result
        obs = getattr(result, "observation", result)
        if isinstance(obs, tuple):
            obs = obs[0]

        if self.checkpoint_path and context.frame_index in self.active_check_points:
            from robofinals.utils.teleop_utils import load_checkpoint, save_checkpoint

            save_checkpoint(self.env, self.checkpoint_path)
            load_checkpoint(self.env, self.checkpoint_path)

        general = obs["embodiment_general_obs"]
        self.ee_poses.append(general["eef_pos"].detach().cpu().numpy())
        self.joint_pos.append(general["joint_pos"].detach().cpu().numpy())
        if self.mode == "joint_target":
            self.joint_targets.append(
                get_robot_joint_target_from_scene(self.env.scene)["joint_pos_target"].detach().cpu().numpy()
            )

        if self.video_processor is not None:
            camera_names = self.env.cfg.isaaclab_arena_env.embodiment.active_observation_camera_names
            self.video_processor.add_frame(obs["policy"], camera_names)

        state_from_dataset = self.episode_data.get_next_state()
        if self.validate_states and state_from_dataset is not None:
            matched, details = compare_states(state_from_dataset, self.env.scene.get_state(is_relative=True), 0)
            self.state_validation.append((context.frame_index, matched, details))

        self._collect_object_artifacts()
        terminated = getattr(result, "terminated", False)
        if bool(terminated.any().item()) if hasattr(terminated, "any") else bool(terminated):
            self.success = True

    def on_session_close(self, _context, _error):
        return None

    def _collect_object_artifacts(self):
        current = self.env.scene.get_state(is_relative=True).get("rigid_object", {})
        state_data = self.episode_data._data.get("states", {}).get("rigid_object", {})
        state_index = max(0, int(getattr(self.episode_data, "next_state_index", 0)) - 1)
        for obj_name, obj_state in current.items():
            data = self.object_states.setdefault(
                obj_name,
                {"obj_root_pose": [], "obj_root_vel": [], "ref_root_pose": [], "ref_root_vel": []},
            )
            data["obj_root_pose"].append(obj_state["root_pose"].detach().cpu().numpy())
            data["obj_root_vel"].append(obj_state["root_velocity"].detach().cpu().numpy())
            reference = state_data.get(obj_name, {})
            if state_index < len(reference.get("root_pose", ())):
                data["ref_root_pose"].append(reference["root_pose"][state_index].unsqueeze(0).cpu().numpy())
                data["ref_root_vel"].append(reference["root_velocity"][state_index].unsqueeze(0).cpu().numpy())

            sensor_name = f"{obj_name}_contact"
            if sensor_name in self.env.scene.sensors:
                force = self.env.backend_runtime.contacts.read_contact(sensor_name).net_forces_w[:, 0, :]
                self.object_forces.setdefault(obj_name, []).append(force.detach().cpu().numpy())

    def write_artifacts(
        self,
        *,
        episode_name,
        pose_file,
        object_file,
        force_file,
        gt_ee_poses=None,
        reference_targets=None,
    ):
        ee_poses = np.concatenate(self.ee_poses, axis=0) if self.ee_poses else np.empty((0, 0, 3))
        joint_pos = np.concatenate(self.joint_pos, axis=0) if self.joint_pos else np.empty((0, 0))
        pose_file.create_dataset(f"data/{episode_name}/ee_poses", data=ee_poses)
        pose_file.create_dataset(f"data/{episode_name}/joint_pos", data=joint_pos)
        if self.joint_targets:
            pose_file.create_dataset(
                f"data/{episode_name}/joint_target",
                data=np.concatenate(self.joint_targets, axis=0),
            )
        if reference_targets:
            reference_values = [value.detach().cpu().numpy() for value in reference_targets]
            reference = (
                np.stack(reference_values, axis=0)
                if reference_values[0].ndim == 1
                else np.concatenate(reference_values, axis=0)
            )
            pose_file.create_dataset(f"data/{episode_name}/gt_joint_target", data=reference)
            if self.joint_targets:
                replayed = np.concatenate(self.joint_targets, axis=0)
                count = min(len(replayed), len(reference))
                width = min(replayed.shape[-1], reference.shape[-1])
                pose_file.create_dataset(
                    f"data/{episode_name}/joint_target_divergence",
                    data=replayed[:count, :width] - reference[:count, :width],
                )
        if gt_ee_poses is not None and len(ee_poses) > 1:
            count = min(len(ee_poses) - 1, len(gt_ee_poses))
            replayed_pose = ee_poses[:count, :3]
            reference_pose = np.asarray(gt_ee_poses[:count])
            if reference_pose.ndim == 3:
                reference_pose = reference_pose[:, 0, :3]
            elif reference_pose.ndim == 2:
                reference_pose = reference_pose[:, :3]
            divergence = np.linalg.norm(replayed_pose - reference_pose, axis=-1)
            pose_file.create_dataset(f"data/{episode_name}/pose_divergence", data=divergence)
        for obj_name, data in self.object_states.items():
            if not data["ref_root_pose"]:
                continue
            group = f"data/{episode_name}/{obj_name}"
            obj_pose = np.concatenate(data["obj_root_pose"], axis=0)
            ref_pose = np.concatenate(data["ref_root_pose"], axis=0)
            length = min(len(obj_pose), len(ref_pose))
            object_file.create_dataset(f"{group}/obj_root_pose", data=obj_pose[:length])
            object_file.create_dataset(f"{group}/ref_root_pose", data=ref_pose[:length])
            object_file.create_dataset(f"{group}/obj_root_pose_divergence", data=obj_pose[:length] - ref_pose[:length])
            obj_vel = np.concatenate(data["obj_root_vel"], axis=0)
            ref_vel = np.concatenate(data["ref_root_vel"], axis=0)
            velocity_length = min(len(obj_vel), len(ref_vel))
            object_file.create_dataset(f"{group}/obj_root_vel", data=obj_vel[:velocity_length])
            object_file.create_dataset(f"{group}/ref_root_vel", data=ref_vel[:velocity_length])
            object_file.create_dataset(
                f"{group}/obj_root_vel_divergence",
                data=obj_vel[:velocity_length] - ref_vel[:velocity_length],
            )
        for obj_name, forces in self.object_forces.items():
            if forces:
                force_file.create_dataset(
                    f"data/{episode_name}/{obj_name}/obj_force",
                    data=np.concatenate(forces, axis=0),
                )


def main():
    """Replay action or joint-target demonstrations through ReplayService."""
    from isaaclab.envs import ManagerBasedRLEnv

    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    episode_count = dataset_file_handler.get_num_episodes()
    if episode_count == 0:
        print("No episodes found in the dataset.")
        return

    episode_indices = args_cli.select_episodes or (
        list(range(episode_count)) if args_cli.replay_all_clips else [episode_count - 1]
    )
    episode_names = sorted(
        dataset_file_handler.get_episode_names(),
        key=lambda name: int(name.split("_")[-1]),
    )
    selected_names = [episode_names[index] for index in episode_indices]

    env_args = json.loads(dataset_file_handler._hdf5_data_group.attrs["env_args"])
    usd_simplify = env_args.get("usd_simplify", False)
    requested_scene = _resolve_replay_scene(args_cli.layout, env_args)
    if not args_cli.layout and requested_scene != env_args.get("usd_path"):
        print(f"[Replay] Using locally available scene: {requested_scene}")
    is_local_scene = str(env_args.get("scene_backend", "")).lower() == "local" or (
        isinstance(requested_scene, str) and requested_scene.endswith(".usd")
    )

    # Keep the legacy direct-Gym replay path for IsaacLab task IDs. Local USD
    # tasks are rebuilt from their HDF5 metadata below, even though teleop gives
    # them a ``Robocasa-...`` environment name for compatibility.
    if not is_local_scene and "-" in env_args.get("env_name", "") and not env_args.get("env_name", "").startswith("Robocasa"):
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        task_name = args_cli.task.strip()
        robot_name = "runtime"
        scene_name = "runtime"
        execute_mode = "REPLAY_ACTION" if args_cli.replay_mode == "action" else "REPLAY_JOINT_TARGETS"
        env_name = args_cli.task
    else:
        from robofinals.utils.env import ExecuteMode, parse_env_cfg

        scene_backend = env_args.get("scene_backend", "robocasa")
        task_backend = env_args.get("task_backend", "robocasa")
        task_name = (env_args.get("task_name") if args_cli.task is None else args_cli.task) or ""
        task_name = task_name.strip()
        robot_name = env_args["robot_name"]
        if robot_name == "double_piper_abs":
            robot_name = "DoublePiper-Abs"
        elif robot_name == "double_piper_rel":
            robot_name = "DoublePiper-Rel"
        if requested_scene:
            scene_name = requested_scene
        else:
            scene_type = env_args.get(
                "scene_type", "libero" if "libero" in env_args.get("usd_path", "") else "robocasakitchen"
            )
            scene_name = f"{scene_type}-{env_args['layout_id']}-{env_args['style_id']}"
        execute_mode = ExecuteMode.REPLAY_ACTION if args_cli.replay_mode == "action" else ExecuteMode.REPLAY_JOINT_TARGETS
        env_cfg = parse_env_cfg(
            scene_backend=scene_backend,
            task_backend=task_backend,
            task_name=task_name,
            robot_name=robot_name,
            scene_name=scene_name,
            robot_scale=args_cli.robot_scale,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            physics_backend=env_args.get("physics_backend", getattr(args_cli, "physics_backend", None)),
            use_fabric=not args_cli.disable_fabric,
            replay_cfgs={
                "hdf5_path": args_cli.dataset_file,
                "ep_meta": env_args,
                "render_resolution": (args_cli.width, args_cli.height),
                "ep_names": selected_names,
                "add_camera_to_observation": True,
            },
            first_person_view=args_cli.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=execute_mode,
            usd_simplify=usd_simplify,
            seed=env_args.get("seed"),
            sources=env_args.get("sources"),
            object_projects=env_args.get("object_projects"),
            headless_mode=args_cli.headless,
            enable_full_local_scene=env_args.get("enable_full_local_scene", is_local_scene),
        )
        env_name = f"Robocasa-{task_name}-{robot_name}-v0"
        gym.register(
            id=env_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )

    if args_cli.record:
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args_cli.dataset_file)
        env_cfg.recorders.dataset_filename = (
            os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
            + f"_{args_cli.replay_mode}_replay_record"
        )
    else:
        env_cfg.recorders = {}
    env_cfg.terminations.time_out = None

    from robofinals.core.runtime.isaac import lifecycle_env_from_config
    from robofinals.applications.replay import ReplayApplication, ReplayProfile, Trajectory

    env = lifecycle_env_from_config(
        env_cfg,
        env_name=env_name,
        application="replay",
        task_name=task_name,
        robot_name=robot_name,
        scene_name=scene_name,
        execute_mode=execute_mode.name if hasattr(execute_mode, "name") else execute_mode,
    )
    set_seed(env_cfg.seed, env)

    if not args_cli.headless and env.common_step_counter <= 1:
        for _ in range(env.cfg.warmup_steps):
            update_sensors(env, env.physics_dt)

    import carb
    import h5py

    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
    env.reset(seed=env.cfg.seed)

    active_camera_names = env.cfg.isaaclab_arena_env.embodiment.active_observation_camera_names
    num_cameras = len(active_camera_names)
    num_rows, cameras_per_row_list, max_cameras_in_row = calculate_camera_layout(num_cameras, 4)
    print(f"Video grid layout -> rows: {num_rows}, per_row: {cameras_per_row_list} total cameras: {num_cameras}")
    video_height, video_width = compute_replay_video_shape(
        active_camera_names,
        env.cfg.isaaclab_arena_env.embodiment.observation_cameras,
        args_cli,
        4,
    )
    replayed_episode_count = 0

    def batch_action(action):
        if args_cli.num_envs == 1:
            return action
        if not isinstance(action, torch.Tensor):
            return action
        target_shape = tuple(env.action_space.shape)
        if tuple(action.shape) == target_shape:
            return action
        batched = torch.zeros(target_shape, dtype=action.dtype, device=action.device)
        if action.ndim == len(target_shape) - 1:
            batched[0] = action
        else:
            raise ValueError(f"Cannot batch action shape {tuple(action.shape)} into {target_shape}")
        return batched

    for episode_index in tqdm.tqdm(episode_indices, desc=f"Replaying actions {args_cli.replay_mode}"):
        episode_name = episode_names[episode_index]
        episode_data = dataset_file_handler.load_episode(episode_name, env.device)
        initial_state = episode_data.get_initial_state()
        if initial_state is None:
            get_default_logger().warning(f"Episode {episode_name} has no initial state; skipping")
            continue

        values = []
        if args_cli.replay_mode == "action":
            while (value := episode_data.get_next_action()) is not None:
                values.append(batch_action(value))
            profile = ReplayProfile.ACTION
        else:
            while (value := episode_data.get_next_joint_target()) is not None:
                values.append(batch_action(value["joint_pos_target"]))
            profile = ReplayProfile.JOINT_TARGET

        save_dir = (
            Path(args_cli.dataset_file).parent
            if not args_cli.replay_all_clips
            else Path(args_cli.dataset_file).parent / "replay_results" / episode_name
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        replay_mp4_path = save_dir / f"isaac_replay_action_{args_cli.replay_mode}.mp4"
        video_processor = (
            VideoProcessor(replay_mp4_path, video_height, video_width, args_cli)
            if app_launcher._enable_cameras
            else None
        )
        observer = ActionReplayObserver(
            env,
            episode_data,
            mode=args_cli.replay_mode,
            args=args_cli,
            video_processor=video_processor,
            validate_states=bool(args_cli.validate_states and args_cli.num_envs == 1),
        )
        observer.checkpoint_path = str(
            save_dir / f"isaac_replay_action_{args_cli.replay_mode}_checkpoint.pt"
        )

        with (
            contextlib.suppress(KeyboardInterrupt),
            h5py.File(save_dir / f"isaac_replay_action_{args_cli.replay_mode}_pose_divergence.hdf5", "w") as pose_file,
            h5py.File(save_dir / f"isaac_replay_action_{args_cli.replay_mode}_obj_pose_divergence.hdf5", "w") as object_file,
            h5py.File(save_dir / f"isaac_replay_action_{args_cli.replay_mode}_obj_force.hdf5", "w") as force_file,
        ):
            pose_file.create_group("data")
            object_file.create_group("data")
            force_file.create_group("data")
            trajectory = Trajectory(
                schema_version=1,
                initial_state=initial_state,
                actions=tuple(values),
                metadata={
                    "env_ids": torch.tensor([0], device=env.device),
                    "is_relative": False,
                },
            )
            report = ReplayApplication(
                env.lifecycle,
                profile=profile,
                observers=(observer,),
            ).replay(episode_name, trajectory)
            if not report.completed:
                raise RuntimeError(f"Replay failed for {episode_name}: {'; '.join(report.errors)}")

            gt_ee_poses = None
            gt_path = save_dir / "replay_state_ee_poses.hdf5"
            if gt_path.exists():
                with h5py.File(gt_path, "r") as gt_file:
                    if f"data/{episode_name}/ee_poses" in gt_file:
                        gt_ee_poses = gt_file[f"data/{episode_name}/ee_poses"][:]
            observer.write_artifacts(
                episode_name=episode_name,
                pose_file=pose_file,
                object_file=object_file,
                force_file=force_file,
                gt_ee_poses=gt_ee_poses,
                reference_targets=values if args_cli.replay_mode == "joint_target" else None,
            )
            with open(save_dir / f"isaac_replay_action_{args_cli.replay_mode}.json", "w") as metrics_file:
                json.dump(
                    {
                        episode_name: {
                            "success": observer.success,
                            "frames": report.frames,
                            "state_validation": [
                                {"frame": frame, "matched": matched, "details": details}
                                for frame, matched, details in observer.state_validation
                            ],
                        }
                    },
                    metrics_file,
                    indent=2,
                )

        if video_processor is not None:
            video_processor.shutdown()
        if args_cli.record:
            env.recorder_manager.export_episodes()
        replayed_episode_count += 1

    metrics_data = env_cfg.isaaclab_arena_env.task.get_checker_results()
    if metrics_data:
        metrics_path = Path(args_cli.dataset_file).parent / "metrics.json"
        with open(metrics_path, "w") as metrics_file:
            json.dump(metrics_data, metrics_file, indent=2)
        print(f"Metrics saved to: {metrics_path}")

    print(f"Finished replaying {replayed_episode_count} episode(s).", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
