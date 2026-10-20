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

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import random
from functools import partial

import os
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import gymnasium as gym

from isaaclab.app import AppLauncher

AppLauncher._SIM_APP_CFG_TYPES.setdefault("limit_cpu_threads", [int])

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--remote_protocol", type=str, default="ipc", help="Remote protocol, can be ipc or restful")
parser.add_argument("--ipc_host", type=str, default="127.0.0.1", help="IPC host")
parser.add_argument("--ipc_port", type=int, default=50000, help="IPC port")
parser.add_argument("--ipc_authkey", type=str, default="lightwheel", help="IPC authkey")
parser.add_argument("--restful_host", type=str, default="0.0.0.0", help="Restful host")
parser.add_argument("--restful_port", type=int, default=8000, help="Restful port")
parser.add_argument("--enable_rtsp", action="store_true", default=False, help="Publish per-env RTSP streams")
parser.add_argument("--rtsp_camera_name", type=str, default="first_person_camera", help="Scene sensor name to stream")
parser.add_argument("--rtsp_base_port", type=int, default=18554, help="Base RTSP port; env_i uses this port")
parser.add_argument("--rtsp_width", type=int, default=1280, help="Stream width in pixels")
parser.add_argument("--rtsp_height", type=int, default=640, help="Stream height in pixels")
parser.add_argument("--rtsp_mount_prefix", type=str, default="/env", help="RTSP mount path prefix, appended with _{i}")
parser.add_argument("--rtsp_advertise_host", type=str, default=None, help="Hostname/IP advertised in stream URLs")
parser.add_argument("--rtsp_max_streams", type=int, default=None, help="Cap number of streams (default: all envs)")
parser.add_argument("--rtsp_bitrate", type=str, default="4M", help="FFmpeg video bitrate, e.g. 4M, 8M, 10M")
parser.add_argument(
    "--anim_recording_output_dir",
    type=str,
    default=None,
    help=(
        "Destination for the self-contained animation delivery folder after OVD bake. "
        "Default: <IsaacLab>/anim_recordings/<timestamp>_delivery"
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if getattr(args_cli, "livestream", -1) not in (1, 2):
    args_cli.headless = False

# Long-running env_server bakes on exit; avoid Isaac Lab wall-clock auto-stop/shutdown.
if getattr(args_cli, "anim_recording_enabled", False):
    if args_cli.headless:
        raise ValueError("Animation recording requires a non-headless Kit UI (--viz kit).")
    args_cli.anim_recording_stop_time = 999999999.0

if args_cli.remote_protocol == "restful":
    from robofinals.distributed.restful import RestfulEnvWrapper
    RemoteEnvWrapper = partial(RestfulEnvWrapper, address=(args_cli.restful_host, args_cli.restful_port))
elif args_cli.remote_protocol == "ipc":   # ipc
    from robofinals.distributed.ipc import IpcDistributedEnvWrapper
    RemoteEnvWrapper = partial(IpcDistributedEnvWrapper, address=(args_cli.ipc_host, args_cli.ipc_port), authkey=args_cli.ipc_authkey.encode())

app_launcher_args = vars(args_cli)
app_launcher_args["limit_cpu_threads"] = 1

if args_cli.enable_rtsp:
    app_launcher_args["enable_cameras"] = True

# WebRTC livestream: SimulationApp defaults to window_width=1440, window_height=900 while
# the WebRTC client negotiates 1920x1080, causing "Cannot stream video frame" warnings and
# a black client. Pin render/window size and pass --no-window. Do NOT force
# isaaclab.python.headless.rendering.kit here: that experience omits omni.kit.widget.toolbar,
# but AppLauncher._hide_stop_button() still imports it whenever livestream >= 1.
if getattr(args_cli, "livestream", -1) in (1, 2):
    _stream_w, _stream_h = 1280, 720
    app_launcher_args["width"] = _stream_w
    app_launcher_args["height"] = _stream_h
    app_launcher_args["window_width"] = _stream_w
    app_launcher_args["window_height"] = _stream_h
    _kit_args = (app_launcher_args.get("kit_args") or "").strip()
    if "--no-window" not in _kit_args:
        _kit_args = f"{_kit_args} --no-window".strip()
    # Allow client window resize without dropping frames (optional fallback).
    if "allowDynamicResize" not in _kit_args:
        _kit_args = (
            f"{_kit_args} "
            "--/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true"
        ).strip()
    app_launcher_args["kit_args"] = _kit_args
    print(
        f"[env_server] livestream enabled: render/window locked to {_stream_w}x{_stream_h}, "
        "kit_args includes '--no-window' and allowDynamicResize."
    )

app_launcher = None
simulation_app = None
_attached_env = None


def _shutdown_env_server(env_server):
    """Bake OVD animation (if enabled) before tearing down the sim."""
    global _attached_env
    env = getattr(env_server, "_env", None) or _attached_env
    if getattr(args_cli, "anim_recording_enabled", False):
        runtime = getattr(env, "backend_runtime", None)
        if runtime is not None:
            delivery_dir = getattr(args_cli, "anim_recording_output_dir", None)
            runtime.recording.finish_animation(env, delivery_dir=delivery_dir)
    if args_cli.enable_rtsp:
        from robofinals.utils.rtsp_streaming import teardown_rtsp_streams

        teardown_rtsp_streams()
    env_server.close()
    global simulation_app
    if simulation_app is not None:
        simulation_app.close()
        simulation_app = None


def make_env_cfg(cfg):
    from isaaclab_tasks.utils import parse_env_cfg

    if "-" in cfg.task:
        env_cfg = parse_env_cfg(
            cfg.task, device=cfg.device, num_envs=cfg.num_envs, use_fabric=not cfg.disable_fabric
        )
        task_name = cfg.task
    else:
        from robofinals.core.runtime.environment_factory import EnvironmentFactory
        from robofinals.utils.env import ExecuteMode

        requested_enable_cameras = bool(getattr(cfg, "enable_cameras", False))
        if requested_enable_cameras and not app_launcher._enable_cameras:
            print(
                "[env_server] WARNING: cfg.enable_cameras=True but server AppLauncher was started "
                "without --enable_cameras. Camera observations will be unavailable."
            )

        env_cfg = EnvironmentFactory().compile_config_from_kwargs(dict(
            scene_backend=cfg.scene_backend,
            task_backend=cfg.task_backend,
            task_name=cfg.task,
            robot_name=cfg.robot,
            scene_name=cfg.layout,
            rl_name=cfg.rl,
            robot_scale=cfg.robot_scale,
            device=cfg.device,
            num_envs=cfg.num_envs,
            use_fabric=not cfg.disable_fabric,
            first_person_view=cfg.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=ExecuteMode.EVAL,
            headless_mode=args_cli.headless,
            usd_simplify=cfg.usd_simplify,
            seed=cfg.seed,
            sources=cfg.sources,
            object_projects=cfg.object_projects,
            replay_cfgs=cfg.replay_cfgs,
            physics_backend=getattr(cfg, "physics_backend", None),
            enable_full_local_scene=cfg.enable_full_local_scene,
            init_robot_base_pos=getattr(cfg, "init_robot_base_pos", None),
            init_robot_base_ori=getattr(cfg, "init_robot_base_ori", None),
            rtsp_camera_name=args_cli.rtsp_camera_name if args_cli.enable_rtsp else None,
            rtsp_width=args_cli.rtsp_width if args_cli.enable_rtsp else None,
            rtsp_height=args_cli.rtsp_height if args_cli.enable_rtsp else None,
        ))
        task_name = f"Robocasa-{cfg.task}-{cfg.robot}-v0"

        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True
        )

    env_cfg.observations.policy.concatenate_terms = cfg.concatenate_terms
    # modify configuration
    env_cfg.terminations.time_out = None
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = cfg.num_envs if cfg.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = cfg.device if cfg.device is not None else env_cfg.sim.device
    # multi-gpu training config
    if cfg.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # randomly sample a seed if seed = -1
    if cfg.seed == -1:
        cfg.seed = random.randint(0, 10000)
    return task_name, env_cfg


def make_env(cfg, launcher_args, args_override: dict = None):
    global app_launcher
    global simulation_app
    global _attached_env
    if app_launcher is None:
        args_override = args_override or {}
        app_launcher_args_ = {**launcher_args, **args_override}
        app_launcher = AppLauncher(app_launcher_args_)
        simulation_app = app_launcher.app
        from robofinals.core.compat import CompatibilityLayer

        CompatibilityLayer(backend=app_launcher_args_.get("physics_backend")).enable()
    from robofinals.utils.place_utils.env_utils import warmup_rendering
    task_name, env_cfg = make_env_cfg(cfg)
    from robofinals.core.runtime.isaac import lifecycle_env_from_config

    env = lifecycle_env_from_config(
        env_cfg,
        env_name=task_name,
        application="eval",
        task_name=getattr(cfg, "task", task_name),
        robot_name=getattr(cfg, "robot", "runtime"),
        scene_name=getattr(cfg, "layout", "runtime"),
        execute_mode="EVAL",
    )
    from robofinals.applications.eval import RolloutApplication

    application = RolloutApplication(env.lifecycle)
    warmup_rendering(application)
    if args_cli.enable_rtsp:
        from robofinals.utils.rtsp_streaming import setup_rtsp_streams
        setup_rtsp_streams(
            application,
            camera_name=args_cli.rtsp_camera_name,
            base_port=args_cli.rtsp_base_port,
            width=args_cli.rtsp_width,
            height=args_cli.rtsp_height,
            mount_prefix=args_cli.rtsp_mount_prefix,
            advertise_host=args_cli.rtsp_advertise_host,
            max_streams=args_cli.rtsp_max_streams,
            bitrate=args_cli.rtsp_bitrate,
        )
    _attached_env = application
    if getattr(args_cli, "anim_recording_enabled", False):
        runtime = getattr(application, "backend_runtime", None)
        if runtime is not None and runtime.recording.supports_animation_recording():
            delivery_dir = getattr(args_cli, "anim_recording_output_dir", None)
            runtime.recording.attach_animation_on_close(application, delivery_dir=delivery_dir)
    return application


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    """Rest everything follows."""
    env_server = RemoteEnvWrapper(env_initializer=partial(make_env, launcher_args=app_launcher_args))
    try:
        env_server.serve()
    finally:
        _shutdown_env_server(env_server)


if __name__ == "__main__":
    main()
