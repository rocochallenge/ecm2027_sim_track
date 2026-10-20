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

"""TeleopSession — core teleoperation session class.

Isaac/Kit startup and task-specific collaborators live here as methods of
:class:`TeleopSession`.  The simulation loop is owned by
``TeleopApplication`` and ``SessionRunner``; ``teleop_main.py`` only wires the
public entry point.
"""

import json
import os
import time
from collections.abc import Callable
from types import SimpleNamespace

import torch
from termcolor import colored

from robofinals.utils.log_utils import get_default_logger
from robofinals.core.backends import make_backend_runtime, resolve_backend_kind
from robofinals.utils.profile_utils import trace_profile
from robofinals.utils.render_utils import optimize_rendering
from robofinals.utils.viewport_capture import WRIST_VIEWPORT_SIZE
from robofinals.core.session import ApplicationCommand, ResetCommand
from robofinals.applications.teleop.commands import CommandMailbox
from robofinals.applications.teleop.spec import TeleopSpec

_DISABLED_BACKEND_DEBUG = SimpleNamespace(
    reset_physx=False,
    no_immediate_step=False,
    vr_auto_start=False,
    skip_ui_setup=False,
    verbose_step=False,
)


class TeleopSession:
    """Encapsulates a complete teleoperation session.

    ``TeleopApplication.from_bootstrap`` constructs this bootstrap, calls
    :meth:`setup`, and then owns execution through ``SessionRunner``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, args, app_launcher, simulation_app, *, spec: TeleopSpec | None = None):
        self.args = args
        self.spec = spec or TeleopSpec.from_args(args)
        self.app_launcher = app_launcher
        self.simulation_app = simulation_app

        # Derived paths (computed before Isaac Sim imports)
        self.output_dir = os.path.dirname(args.dataset_file)
        self.output_file_name = os.path.splitext(os.path.basename(args.dataset_file))[0]
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Setup default checkpoint path if not provided
        if getattr(args, "checkpoint_path", None) in (None, ""):
            args.checkpoint_path = os.path.join(
                self.output_dir, f"{self.output_file_name}.ckpt.pt"
            )

        # These are set during setup() / run()
        self.env = None
        self.teleop_interface = None
        self.lerobot_recorder = None
        self.video_recorder = None
        self.anim_recorder = None
        self.mcap_recorder = None
        self.mcap_controller = None
        self._teleop_viewports: dict | None = None
        self.sliding_window_manager = None
        self.garment_manager = None
        self._teleoperation_callbacks: dict[str, Callable[[], None]] = {}
        self.command_mailbox = CommandMailbox()

        # Runner/application state
        self.teleoperation_active: bool = False

        # Upload dialog state
        self.upload_dialog_state: dict = {"shown": False, "result": None, "window": None}
        self.awaiting_upload_choice: bool = False

    # ------------------------------------------------------------------
    # Callback methods (replace closures / nonlocal pattern)
    # ------------------------------------------------------------------

    def reset_env(self):
        # Fixed local USD scenes own their movable objects inside the composed
        # stage rather than ``task.object_cfgs``. A normal env.reset() restores
        # registered Isaac Lab assets but does not reload those USD rigid bodies.
        # Rebuild the fixed scene so X truly restores its authored layout.
        arena_env = getattr(getattr(self.env, "cfg", None), "isaaclab_arena_env", None)
        orchestrator = getattr(arena_env, "orchestrator", None)
        task = getattr(orchestrator, "task", None)
        object_cfgs = getattr(task, "object_cfgs", None)
        scene = getattr(orchestrator, "scene", None)
        if object_cfgs == {} and getattr(scene, "scene_type", None) == "local":
            print("reset fixed local scene from USD", flush=True)
            self.command_mailbox.publish(ApplicationCommand("remake_fixed"))
            return

        print("reset env", flush=True)
        self.command_mailbox.publish(ResetCommand(seed=getattr(self.args, "seed", None)))

    def reset_env_keep_placement(self):
        print("reset env and keep placement", flush=True)
        self.command_mailbox.publish(ApplicationCommand("reset_keep_placement"))

    def remake_env_with_random_layout(self):
        print("remake env with random layout", flush=True)
        self.command_mailbox.publish(ApplicationCommand("remake_random"))

    def remake_env_with_fixed_layout(self):
        print("remake env with fixed layout", flush=True)
        self.command_mailbox.publish(ApplicationCommand("remake_fixed"))

    def start_teleoperation(self):
        print("[RecorderDebug] start_teleoperation callback received", flush=True)
        # Keep the device-side latch and the runner gate in sync. VR normally
        # sets ``started`` before invoking this callback; setting it here also
        # makes auto-start and other command sources follow the same path.
        interface = self.teleop_interface
        if interface is not None and hasattr(interface, "started"):
            interface.started = True
        self.command_mailbox.publish(ApplicationCommand("start_teleoperation"))

    def stop_teleoperation(self):
        self.command_mailbox.publish(ApplicationCommand("stop_teleoperation"))

    def flush_recorder_manager(self):
        self.command_mailbox.publish(ApplicationCommand("flush_recorder"))

    def rollback_to_checkpoint(self):
        self.command_mailbox.publish(ApplicationCommand("rollback_checkpoint"))

    def call_save_checkpoint(self):
        self.command_mailbox.publish(ApplicationCommand("save_checkpoint"))

    def _should_defer_camera_obs(self) -> bool:
        """Keep RTX TiledCamera off shared wrist-camera prims when required for stable rendering."""
        from robofinals.utils.lerobot_isaac_export import should_defer_teleop_camera_sensors

        return should_defer_teleop_camera_sensors(self.args)

    def _replay_cfgs_add_camera_to_observation(self, replay_cfgs: dict | None) -> dict | None:
        if not self.app_launcher._enable_cameras:
            return replay_cfgs
        if not (getattr(self.args, "record", False) or getattr(self.args, "record_lerobot", False)):
            return replay_cfgs
        if self._should_defer_camera_obs():
            return replay_cfgs
        if replay_cfgs is None:
            return {"add_camera_to_observation": True}
        return {**replay_cfgs, "add_camera_to_observation": True}

    def _replay_cfgs_with_deferred_cameras(self, replay_cfgs: dict | None) -> dict | None:
        if not self._should_defer_camera_obs():
            return replay_cfgs
        if replay_cfgs is None:
            return {"defer_camera_sensors": True}
        return {**replay_cfgs, "defer_camera_sensors": True}

    def setup(self):
        """Create env, attach recorder, create teleop interface. Returns self."""
        args = self.args

        # --- Isaac-Sim-dependent imports (must be after AppLauncher) ---
        import gymnasium as gym
        import numpy as np

        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.datasets import HDF5DatasetFileHandler
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.manager_based.manipulation.lift import mdp  # noqa: F401
        from isaaclab.devices.teleop_device_factory import create_teleop_device
        from multiprocessing import shared_memory
        import carb
        from omni.log import get_log
        import omni.log

        channels = [
            "omni.physx.tensors.plugin",
            "omni.usd",
            "omni.physicsschema.plugin",
            "omni.physx.plugin",
            "omni.graph.core.plugin",
            "rtx.neuraylib.plugin",
            "rtx.scenedb.plugin",
            "omni.hydra",
            "omni.isaac.dynamic_control",
            "omni.replicator.core.scripts.extension",
            "omni.usd-abi.plugin",
            "omni.kit.menu.utils.app_menu",
        ]
        for channel in channels:
            get_log().set_channel_enabled(channel, False, omni.log.SettingBehavior.OVERRIDE)

        print(f"teleop_main working directory: {os.getcwd()}")
        print(f"teleop_main dataset_file: {args.dataset_file}")

        from robofinals.utils.lerobot_isaac_export import (
            TeleopLeRobotRecorder,
            attach_lerobot_recorder_export_hook,
        )

        self.lerobot_recorder = TeleopLeRobotRecorder.from_args(
            args, self.output_dir, self.output_file_name
        )
        if getattr(args, "record_lerobot", False) and not getattr(args, "enable_cameras", False):
            print(
                colored(
                    "[LeRobot] record_lerobot is on but enable_cameras is false; "
                    "LeRobot export needs RGB cameras — enable_cameras: true recommended.",
                    "yellow",
                )
            )

        print("[teleop] building environment config...", flush=True)
        self.env = self._create_env_config()
        print("[teleop] environment ready", flush=True)
        self._setup_garment_manager()
        attach_lerobot_recorder_export_hook(self.env, self.lerobot_recorder)
        if self.lerobot_recorder is not None:
            self.lerobot_recorder.bind_env_camera_shapes(self.env)

        recording = self.env.backend_runtime.recording
        if getattr(args, "anim_recording_enabled", False) and recording.supports_animation_recording():
            from robofinals.applications.teleop.anim_recording import TeleopAnimationRecorder

            self.anim_recorder = TeleopAnimationRecorder(args, get_default_logger)
            self.anim_recorder.prepare(self.env)
            delivery_dir = getattr(args, "anim_recording_output_dir", None)
            recording.attach_animation_on_close(self.env, delivery_dir=delivery_dir)

        if args.enable_rendering_optimization:
            optimize_rendering(self.env, args)

        get_default_logger().info(f"env_cfg: {self.env.cfg}")

        self._create_teleop_interface()

        from robofinals.utils.ui_utils import hide_ui_windows
        if not args.headless:
            hide_ui_windows(self.simulation_app)

        print("Starting teleoperation - will check for task signals during execution")

        return self

    def initialize_for_runner(self) -> None:
        """Perform startup reset work before SessionRunner samples the device."""
        env = self.env
        self._maybe_reset_physx(env)
        env.reset(seed=env.cfg.seed)
        self._initialize_garment_after_env_reset(env)
        self._maybe_newton_immediate_step(env, tag="runner_startup")
        self._validate_sensor_runtime(env, require_ready=False)
        if self.teleop_interface is not None:
            self.teleop_interface.reset()
            # ``reset()`` clears the VR device's own start latch. Re-apply the
            # Newton diagnostic auto-start after reset so it exercises the same
            # action path as a physical B press.
            if self._newton_debug().vr_auto_start and hasattr(self.teleop_interface, "started"):
                self.teleop_interface.started = True
                self.teleoperation_active = True
        if getattr(self.args, "teleop_auto_start", False):
            self.start_teleoperation()
        if self._should_setup_teleop_ui():
            viewports, _overlay = self._setup_env_config_with_args()
            self._teleop_viewports = viewports
            self._attach_lerobot_viewports(viewports)
            print("[teleop] runner viewport setup complete", flush=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Env config
    # ------------------------------------------------------------------

    def _create_env_config(self, object_cfgs=None, cache_usd_version=None, scene_name=None, initial_state=None):
        """Create environment configuration based on task type."""
        args = self.args

        import omni.usd
        omni.usd.get_context().new_stage()

        from robofinals.core.runtime.environment_factory import EnvironmentFactory
        from robofinals.utils.env import ExecuteMode
        import gymnasium as gym
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.datasets import HDF5DatasetFileHandler
        from robofinals.utils.seed_utils import set_seed

        with trace_profile("parse_env_cfg"):
            # Load replay dataset
            if hasattr(args, "input_dataset_file") and os.path.exists(args.input_dataset_file):
                dataset_file_handler = HDF5DatasetFileHandler()
                dataset_file_handler.open(args.input_dataset_file)
                env_args = json.loads(dataset_file_handler._hdf5_data_group.attrs["env_args"])
                scene_backend = env_args["scene_backend"] if "scene_backend" in env_args else "robocasa"
                task_backend = env_args["task_backend"] if "task_backend" in env_args else "robocasa"
                task_name = env_args["task_name"].strip()
                robot_name = env_args["robot_name"]
                args.robot = robot_name
                scene_name = f"{env_args['scene_type']}-{env_args['layout_id']}-{env_args['style_id']}"
                usd_simplify = env_args["usd_simplify"] if "usd_simplify" in env_args else False
                if getattr(args, "launcher_managed_upload", False):
                    replay_seed = getattr(args, "seed", None)
                else:
                    replay_seed = env_args.get("seed")
                episode_count = dataset_file_handler.get_num_episodes()
                episode_indices_to_replay = list(range(episode_count))
                episode_names = list(dataset_file_handler.get_episode_names())
                episode_names.sort(key=lambda x: int(x.split("_")[-1]))
                episode_names_to_replay = [episode_names[idx] for idx in episode_indices_to_replay]
                dataset_file_handler.close()
                _replay_from_hdf5 = {
                    "hdf5_path": args.input_dataset_file,
                    "ep_meta": env_args,
                    "ep_names": episode_names_to_replay,
                }
                if self.app_launcher._enable_cameras and (
                    getattr(args, "record", False) or getattr(args, "record_lerobot", False)
                ):
                    _replay_from_hdf5 = self._replay_cfgs_add_camera_to_observation(_replay_from_hdf5)
                _replay_from_hdf5 = self._replay_cfgs_with_deferred_cameras(_replay_from_hdf5)
                env_cfg = EnvironmentFactory().compile_config_from_kwargs(dict(
                    scene_backend=scene_backend,
                    task_backend=task_backend,
                    task_name=task_name,
                    robot_name=robot_name,
                    scene_name=scene_name,
                    robot_scale=args.robot_scale,
                    gripper_closed_gap=getattr(args, "gripper_closed_gap", 0.0),
                    device=args.device,
                    num_envs=args.num_envs,
                    physics_backend=args.physics_backend,
                    newton_deformable_solver=getattr(args, "newton_deformable_solver", None),
                    use_fabric=not args.disable_fabric,
                    replay_cfgs=_replay_from_hdf5,
                    first_person_view=args.first_person_view,
                    enable_cameras=self.app_launcher._enable_cameras,
                    execute_mode=ExecuteMode.REPLAY_TELEOP,
                    usd_simplify=usd_simplify,
                    seed=replay_seed,
                    sources=env_args["sources"] if "sources" in env_args else None,
                    object_projects=env_args["object_projects"] if "object_projects" in env_args else None,
                    headless_mode=args.headless,
                ))
            else:
                ep_meta = {}
                if hasattr(args, "init_robot_base_pos") and args.init_robot_base_pos is not None:
                    try:
                        pos = [float(v) for v in args.init_robot_base_pos]
                        if len(pos) == 3:
                            ep_meta["init_robot_base_pos"] = pos
                    except Exception:
                        pass
                if hasattr(args, "init_robot_base_ori") and args.init_robot_base_ori is not None:
                    try:
                        ori = [float(v) for v in args.init_robot_base_ori]
                        if len(ori) == 3:
                            ep_meta["init_robot_base_ori"] = ori
                    except Exception:
                        pass
                if object_cfgs is not None:
                    ep_meta["object_cfgs"] = object_cfgs
                if cache_usd_version is not None:
                    ep_meta["cache_usd_version"] = cache_usd_version
                replay_cfgs = None
                if len(ep_meta) > 0:
                    replay_cfgs = {"ep_meta": ep_meta}
                if self.app_launcher._enable_cameras and (
                    getattr(args, "record", False) or getattr(args, "record_lerobot", False)
                ):
                    replay_cfgs = self._replay_cfgs_add_camera_to_observation(replay_cfgs)
                replay_cfgs = self._replay_cfgs_with_deferred_cameras(replay_cfgs)
                object_init_offset = [0.0, 0.0]
                if hasattr(args, "object_init_offset") and args.object_init_offset is not None:
                    try:
                        object_init_offset = [float(v) for v in args.object_init_offset]
                    except Exception:
                        pass
                if scene_name is None:
                    scene_name = args.layout
                kwargs = {"debug_assets": args.debug_assets}
                if args.debug_assets == "object":
                    execute_mode = ExecuteMode.TEST_OBJECT
                    kwargs["test_object_paths"] = args.test_object_paths
                elif args.debug_assets == "fixture":
                    execute_mode = ExecuteMode.TEST_FIXTURE
                    kwargs["test_fixture_path"] = args.test_fixture_path
                    kwargs["test_fixture_type"] = args.test_fixture_type
                else:
                    execute_mode = ExecuteMode.TELEOP
                if hasattr(args, "robot_to_fixture_dist") and args.robot_to_fixture_dist is not None:
                    kwargs["robot_to_fixture_dist"] = args.robot_to_fixture_dist
                if hasattr(args, "init_robot_base_pos") and args.init_robot_base_pos is not None:
                    kwargs["init_robot_base_pos"] = [float(v) for v in args.init_robot_base_pos]
                if hasattr(args, "init_robot_base_ori") and args.init_robot_base_ori is not None:
                    kwargs["init_robot_base_ori"] = [float(v) for v in args.init_robot_base_ori]
                env_cfg = EnvironmentFactory().compile_config_from_kwargs(dict(
                    scene_backend=args.scene_backend,
                    task_backend=args.task_backend,
                    task_name=args.task,
                    robot_name=args.robot,
                    scene_name=scene_name,
                    robot_scale=args.robot_scale,
                    gripper_closed_gap=getattr(args, "gripper_closed_gap", 0.0),
                    device=args.device,
                    num_envs=args.num_envs,
                    physics_backend=args.physics_backend,
                    newton_deformable_solver=getattr(args, "newton_deformable_solver", None),
                    use_fabric=not args.disable_fabric,
                    replay_cfgs=replay_cfgs,
                    first_person_view=args.first_person_view,
                    enable_cameras=self.app_launcher._enable_cameras,
                    execute_mode=execute_mode,
                    usd_simplify=args.usd_simplify,
                    object_init_offset=object_init_offset,
                    max_scene_retry=args.max_scene_retry,
                    max_object_placement_retry=args.max_object_placement_retry,
                    seed=args.seed,
                    sources=args.sources,
                    object_projects=args.object_projects,
                    headless_mode=args.headless,
                    initial_state=initial_state,
                    teleop_device=args.teleop_device,
                    resample_objects_placement_on_reset=args.resample_objects_placement_on_reset,
                    resample_robot_placement_on_reset=args.resample_robot_placement_on_reset,
                    **kwargs,
                ))
            env_name = f"Robocasa-{args.task}-{args.robot}-v0"
            env_cfg.env_name = env_name
            env_cfg.terminations.time_out = None
            if args.record:
                env_cfg.recorders.dataset_export_dir_path = self.output_dir
                env_cfg.recorders.dataset_filename = self.output_file_name


        make_backend_runtime(
            resolve_backend_kind(getattr(self.args, "physics_backend", None)),
        ).simulation.configure_before_env_make(self, env_cfg)

        physics_type = type(getattr(env_cfg.sim, "physics", None)).__name__
        print(
            "[teleop] environment construction starting "
            f"(backend={getattr(self.args, 'physics_backend', None)}, "
            f"sim.physics={physics_type}, "
            f"seed={env_cfg.seed}, "
            f"num_envs={env_cfg.scene.num_envs}, "
            f"replicate_physics={env_cfg.scene.replicate_physics})...",
            flush=True,
        )
        from robofinals.core.runtime.environment_factory import EnvironmentFactory
        from robofinals.core.runtime.run_plan import ConfigResolver

        plan = ConfigResolver.resolve(
            {
                "application": "teleop",
                "execute_mode": execute_mode.name,
                "scene_name": scene_name or args.layout,
                "robot_name": args.robot,
                "task_name": args.task,
                "physics_backend": getattr(args, "physics_backend", None),
                "device": args.device,
                "num_envs": getattr(env_cfg.scene, "num_envs", args.num_envs),
                "headless_mode": args.headless,
                "env_name": env_name,
            }
        )
        lifecycle = EnvironmentFactory().create_from_config(plan, env_cfg, env_name=env_name)
        new_env = lifecycle.env
        print("[teleop] environment construction complete", flush=True)
        self._configure_camera_read_deferral(new_env)
        self._validate_sensor_runtime(new_env)
        set_seed(env_cfg.seed, new_env)
        return new_env

    def _configure_camera_read_deferral(self, env) -> None:
        """Skip RTX camera readback when it would destabilize the interactive render loop."""
        if not self._should_defer_camera_obs():
            return
        env._robofinals_defer_camera_read = True
        print(
            "[teleop] Deferred RTX camera GPU readback; Kit/VR viewport rendering remains enabled",
            flush=True,
        )

    def _setup_garment_manager(self):
        """Create and add the optional LeHome garment manager for the current stage."""
        from robofinals.applications.teleop.garment_runtime import GarmentManager

        self.garment_manager = GarmentManager.from_args(self.args)
        if self.garment_manager is not None:
            self.garment_manager.add_to_stage()
        task = getattr(getattr(self.env, "cfg", None), "isaaclab_arena_env", None)
        task = getattr(task, "task", None)
        if task is not None:
            task.garment_manager = self.garment_manager
        self._register_garment_scene_handle(self.env)

    def _initialize_garment_after_env_reset(self, env):
        """Initialize optional LeHome garment after an env reset/remake."""
        task = getattr(getattr(env, "cfg", None), "isaaclab_arena_env", None)
        task = getattr(task, "task", None)
        if task is not None:
            task.garment_manager = self.garment_manager
        if self.garment_manager is not None:
            self.garment_manager.initialize_after_env_reset(env)
        self._register_garment_scene_handle(env)

    def _recreate_garment_for_current_stage(self):
        """Recreate optional LeHome garment handles after a stage remake."""
        if self.garment_manager is None:
            self._setup_garment_manager()
            return
        self.garment_manager.mark_stale()
        self.garment_manager.add_to_stage()
        task = getattr(getattr(self.env, "cfg", None), "isaaclab_arena_env", None)
        task = getattr(task, "task", None)
        if task is not None:
            task.garment_manager = self.garment_manager
        self._register_garment_scene_handle(self.env)

    def _register_garment_scene_handle(self, env):
        """Expose the active garment through env.scene for task-side access."""
        if env is None:
            return
        scene = getattr(env, "scene", None)
        extras = getattr(scene, "extras", None)
        if extras is None:
            return
        garment = getattr(self.garment_manager, "garment", None) if self.garment_manager is not None else None
        if garment is None:
            extras.pop("garment", None)
            return
        extras["garment"] = garment

    def _uses_newton_physics(self) -> bool:
        return resolve_backend_kind(getattr(self.args, "physics_backend", None)).value == "newton"

    def _newton_debug(self):
        if not self._uses_newton_physics():
            return _DISABLED_BACKEND_DEBUG
        from robofinals.core.backends.newton.debug import get_newton_teleop_debug

        return get_newton_teleop_debug()

    def _maybe_reset_physx(self, env) -> None:
        """Skip extra reset_physx on Newton (benchmark path); opt-in via TELEOP_NEWTON_DEBUG."""
        if self._uses_newton_physics() and not self._newton_debug().reset_physx:
            return
        from robofinals.utils.place_utils.env_utils import reset_physx

        reset_physx(env)

    def _device_required_sensor_names(self, env) -> tuple[str, ...]:
        embodiment = getattr(getattr(getattr(env, "cfg", None), "isaaclab_arena_env", None), "embodiment", None)
        names = getattr(embodiment, "device_required_sensor_names", ()) if embodiment is not None else ()
        return tuple(names or ())

    def _validate_sensor_runtime(self, env, *, require_ready: bool = False) -> None:
        env.backend_runtime.contacts.validate_scene_sensors()
        names = self._device_required_sensor_names(env)
        if require_ready and names:
            env.backend_runtime.contacts.assert_required_sensors_ready(names)

    def _maybe_newton_immediate_step(self, env, *, tag: str) -> None:
        """Run one zero-action env.step() after reset to mirror benchmark_x7s_newton timing."""
        if not self._uses_newton_physics() or self._newton_debug().no_immediate_step:
            return
        actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
        with torch.inference_mode():
            self._env_step(env, actions, tag=f"immediate_step:{tag}")

    def _env_step(self, env, actions, *, tag: str, frame_idx: int | None = None):
        debug = self._newton_debug()
        verbose = debug.verbose_step and (
            frame_idx is None
            or frame_idx <= 3
            or (frame_idx is not None and frame_idx % 100 == 0)
        )
        if verbose:
            print(f"[newton_debug] env.step BEGIN tag={tag}", flush=True)
        result = env.step(actions)
        if verbose:
            print(f"[newton_debug] env.step END tag={tag}", flush=True)

        # Poll remote MCAP control if available
        if self.mcap_controller is not None and self.mcap_controller.available:
            timestamp_ns = int(time.time() * 1e9)
            self.mcap_controller.poll(timestamp_ns)

        return result

    def _uses_newton_visualizer(self) -> bool:
        """Return whether this Newton session explicitly requested its standalone viewer."""
        visualizers = getattr(self.args, "visualizer", None) or ()
        if isinstance(visualizers, str):
            visualizers = visualizers.split(",")
        return (
            self._uses_newton_physics()
            and any(
                str(visualizer).strip().lower() == "newton"
                for visualizer in visualizers
            )
        )

    @staticmethod
    def _newton_visualizers(env) -> list:
        visualizers = getattr(getattr(env, "sim", None), "visualizers", ())
        return [
            visualizer
            for visualizer in visualizers
            if getattr(getattr(visualizer, "cfg", None), "visualizer_type", None) == "newton"
        ]

    def _newton_first_person_camera_pose(self):
        """Compute a Newton-viewer pose without mutating Kit's camera state."""
        arena_cfg = getattr(getattr(self.env, "cfg", None), "isaaclab_arena_env", None)
        embodiment = getattr(arena_cfg, "embodiment", None)
        viewport_cfg = getattr(embodiment, "viewport_cfg", None)
        if not viewport_cfg:
            return None

        offset = viewport_cfg.get("offset")
        lookat = viewport_cfg.get("lookat")
        if offset is None or lookat is None:
            return None

        from robofinals.utils.place_utils.env_utils import compute_camera_follow_pose

        position, target = compute_camera_follow_pose(self.env, offset, lookat)
        return tuple(float(value) for value in position), tuple(float(value) for value in target)

    def _register_newton_viewer_callbacks(self, callbacks: dict[str, Callable[[], None]]) -> None:
        """Route teleop commands through Newton without changing Kit callbacks or cameras."""
        if not self._uses_newton_visualizer():
            return
        for visualizer in self._newton_visualizers(self.env):
            if getattr(self.args, "first_person_view", False):
                visualizer.register_camera_follow_callback(
                    lambda _state: self._newton_first_person_camera_pose(),
                    locked=True,
                )
            for key, callback in callbacks.items():
                visualizer.register_key_callback(key, callback)

    def _remake_env(self, scene_name=None):
        """Recreate the environment with a fresh configuration."""
        from robofinals.utils.place_utils.env_utils import reset_obj_cache
        from robofinals.utils.lerobot_isaac_export import attach_lerobot_recorder_export_hook

        reset_obj_cache()
        self._close_env_for_remake()
        new_env = self._create_env_config(scene_name=scene_name)
        get_default_logger().info(f"env_cfg: {new_env.cfg}")

        self._maybe_reset_physx(new_env)
        new_env.reset(seed=new_env.cfg.seed)
        self.env = new_env
        self._recreate_garment_for_current_stage()
        self._initialize_garment_after_env_reset(new_env)
        self._maybe_newton_immediate_step(new_env, tag="remake")
        self._validate_sensor_runtime(new_env, require_ready=True)
        self._register_newton_viewer_callbacks(self._teleoperation_callbacks)

        if self.args.enable_rendering_optimization:
            optimize_rendering(new_env, self.args)
        if self.teleop_interface is not None:
            if hasattr(self.teleop_interface, "refresh_env"):
                self.teleop_interface.refresh_env(new_env)
            self.teleop_interface.reset()

        self.env = new_env
        attach_lerobot_recorder_export_hook(self.env, self.lerobot_recorder)
        if self.lerobot_recorder is not None:
            self.lerobot_recorder.bind_env_camera_shapes(new_env)
            ep_meta = new_env.cfg.isaaclab_arena_env.orchestrator.get_ep_meta()
            self.lerobot_recorder.task_description = ep_meta.get("lang", self.args.task)
        if self.anim_recorder is not None:
            self.anim_recorder.prepare(self.env)

    def _close_env_for_remake(self):
        """Close the current env and release recorder file handles before recreating it."""
        if self.env is None:
            return

        recorder_manager = getattr(self.env, "recorder_manager", None)
        if recorder_manager is not None:
            for attr_name in ("_dataset_file_handler", "_failed_episode_dataset_file_handler"):
                dataset_file_handler = getattr(recorder_manager, attr_name, None)
                if dataset_file_handler is None:
                    continue
                try:
                    dataset_file_handler.close()
                except Exception as exc:
                    print(colored(
                        f"Warning: failed to close {attr_name} before env remake: {exc}",
                        "yellow",
                    ))

        self.env.close()

    def _create_teleop_interface(self):
        """Create a registered device plugin; keep legacy construction available for rollback."""
        args = self.args
        allow_headless_newton_vr = (
            self._uses_newton_visualizer()
            and str(getattr(args, "teleop_device", "")).lower().startswith("vr")
        )
        if args.headless and not allow_headless_newton_vr:
            self.teleop_interface = None
            self.teleoperation_active = True
            return

        from robofinals.applications.teleop import DeviceContext, build_default_device_registry
        from robofinals.utils.teleop_utils import quick_rewind

        callbacks: dict[str, Callable[[], None]] = {
            "X": self.reset_env,
            "R": self.reset_env_keep_placement,
            "Y": self.remake_env_with_random_layout,
            "U": self.remake_env_with_fixed_layout,
            "T": self.flush_recorder_manager,
            "M": self.call_save_checkpoint,
            "N": self.rollback_to_checkpoint,
            "B": self.start_teleoperation,
            "P": lambda: quick_rewind(self.env, 10),
        }
        registry = build_default_device_registry()
        embodiment = getattr(getattr(self.env.cfg, "isaaclab_arena_env", None), "embodiment", None)
        device_names = getattr(getattr(embodiment, "teleop_devices", None), "devices", ())
        registry.register_embodiment_devices(device_names)
        binding = registry.create(args.teleop_device, DeviceContext(self.env, args, callbacks))
        interface = binding.interface

        self.teleoperation_active = binding.active

        self.teleop_interface = interface
        self._register_newton_viewer_callbacks(callbacks)
        self._teleoperation_callbacks = callbacks

    def _needs_lerobot_viewport_capture(self) -> bool:
        recorder = self.lerobot_recorder
        return (
            recorder is not None
            and getattr(recorder, "defer_camera_capture", False)
            and not getattr(self.args, "headless", False)
        )

    def _should_setup_teleop_ui(self) -> bool:
        if getattr(self.args, "headless", False):
            return False
        if self._needs_lerobot_viewport_capture():
            return True
        return not (self._uses_newton_physics() and self._newton_debug().skip_ui_setup)

    def _attach_lerobot_viewports(self, viewports) -> None:
        if self.lerobot_recorder is not None:
            self.lerobot_recorder.attach_viewports(viewports)

    def _setup_env_config_with_args(self, viewports=None):
        """Configure viewports and overlay window. Returns (viewports, overlay_window)."""
        args = self.args
        env = self.env

        from robofinals.utils.ui_utils import (
            set_viewport_to_first_person,
            setup_cameras,
            setup_task_description_ui,
        )

        if not args.headless and args.enable_cameras:
            if args.enable_multiple_viewports:
                viewports = setup_cameras(env, viewports)
            if args.first_person_view:
                # Multiple docked camera viewports and the main first-person
                # viewport are independent features and may be enabled together.
                set_viewport_to_first_person(env)
            viewports = viewports or {}
            for key, v_p in viewports.items():
                res = v_p.viewport_api.get_texture_resolution()
                sca = v_p.viewport_api.get_texture_resolution_scale()
                v_p.viewport_api.set_texture_resolution(WRIST_VIEWPORT_SIZE)
                env.sim.render()
                res_new = v_p.viewport_api.get_texture_resolution()
                print(f"Viewport {key} resolution: {res}, scale: {sca}, new resolution: {res_new}")
        overlay_window = None
        if not args.headless:
            overlay_window = setup_task_description_ui(env, viewports)
        return viewports, overlay_window
