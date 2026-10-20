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
import logging
import multiprocessing
import sys
import os

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"
# ---------------------------------------------------------------------------
# Cap host-side BLAS / OpenMP thread pools BEFORE importing numpy / torch /
# pinocchio.
#
# CPU-side Pinocchio + IPOPT for IK; together with isaaclab 3.x's
# Newton/Warp backend and PyTorch this oversubscribes physical cores and
# IPOPT stalls inside OpenMP spinlocks. Pinning to 1 thread eliminates the
# contention and brings X7S IK back to single-digit ms. G1 (GPU IK) is
# unaffected either way.
#
# IMPORTANT: these env vars are only honored when set before the first
# import of numpy / mkl / openblas / openmp, so we set them here as the
# very first action of the script. ``setdefault`` keeps any explicit
# override from launch.json / shell environment.
# ---------------------------------------------------------------------------


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _prepend_nvidia_cuda_libs_to_ld_library_path() -> None:
    """Expose conda-shipped CUDA 13 NVRTC libs to the dynamic linker.

    PyTorch 2.x+cu130 JIT-compiles CUDA kernels via nvrtc and expects
    ``libnvrtc-builtins.so.13.0`` on ``LD_LIBRARY_PATH``. Kit/AppLauncher
    does not add the conda ``nvidia/cu13/lib`` directory automatically.
    """
    import site

    for site_pkg in site.getsitepackages():
        for rel in ("nvidia/cu13/lib", "nvidia/cu12/lib"):
            lib_dir = os.path.join(site_pkg, rel)
            if not os.path.isdir(lib_dir):
                continue
            current = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir in current.split(":"):
                return
            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}" if current else lib_dir
            return


_prepend_nvidia_cuda_libs_to_ld_library_path()


_NOISY_INFO_LOG_PREFIXES = (
    "ogn_registration",
    "isaacsim.asset.transformer.rules.utils",
    "i2rt.robots.utils",
)


def _quiet_isaac_ogn_registration_logs() -> None:
    """Suppress verbose third-party INFO logs emitted on stderr by Kit startup."""
    if getattr(logging.Logger, "_robofinals_quiet_info_patch", False):
        return

    original_handle = logging.Logger.handle

    def handle_quiet_info(self, record):
        if record.levelno <= logging.INFO and record.name.startswith(_NOISY_INFO_LOG_PREFIXES):
            return
        return original_handle(self, record)

    logging.Logger.handle = handle_quiet_info
    logging.Logger._robofinals_quiet_info_patch = True
    for logger_name in _NOISY_INFO_LOG_PREFIXES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


_quiet_isaac_ogn_registration_logs()

import scipy.spatial.transform
from robofinals.utils.config_loader import config_loader, merge_task_yaml_with_cli
from robofinals.core.backends import resolve_backend_kind
from robofinals.utils.teleop_utils import download_file

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

# Force before first pxr import; Kit may overwrite after startup — re-applied in main().
from isaaclab.app import AppLauncher

AppLauncher._SIM_APP_CFG_TYPES.setdefault("limit_cpu_threads", [int])

_BASE_EXCLUDED_EXTENSIONS = (
    "omni.usd.config",
    "isaacsim.ros2.bridge",
    "isaacsim.ros2.core",
    "isaacsim.ros2.examples",
    "isaacsim.ros2.nodes",
    "isaacsim.ros2.ui",
)
_BASE_LOCAL_KIT_ARGS = (
    "--/log/level=warning",
    "--/log/pythonLevel=warning",
    "--/log/channels/ogn_registration/level=warning",
    "--/telemetry/enableAnonymousData=false",
    "--/telemetry/enableAnonymousAppName=false",
    # Isaac-Sim Fabric caches viewport "Show All Colliders" (value 2) in user.config.json.
    "--/persistent/physics/visualizationDisplayColliders=0",
)
_DEFAULT_KIT_EXPERIENCE = "isaacsim.exp.full.fabric.kit"


def _configure_internal_ros2_runtime() -> None:
    """Make Isaac Sim bundled Jazzy ROS2 libraries discoverable when present."""
    ros_lib = os.path.join(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "isaacsim",
        "exts",
        "isaacsim.ros2.core",
        "jazzy",
        "lib",
    )
    if not os.path.isdir(ros_lib):
        return
    os.environ.setdefault("ROS_DISTRO", "jazzy")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = current.split(os.pathsep) if current else []
    if ros_lib not in entries:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([ros_lib, *entries])


def _build_local_kit_args(physics_backend: str | None) -> tuple[str, ...]:
    """Build Kit overrides without changing the Warp runtime used by PhysX."""
    excluded_extensions = list(_BASE_EXCLUDED_EXTENSIONS)
    if resolve_backend_kind(physics_backend).value == "newton":
        # Newton must use the environment's warp-lang package instead of Kit's
        # older omni.warp.core extension, whose API is incompatible with Newton.
        excluded_extensions.append("omni.warp.core")
    excluded = ",".join(f'"{extension}"' for extension in excluded_extensions)
    return (f"--/app/extensions/excluded=[{excluded}]", *_BASE_LOCAL_KIT_ARGS)


# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--task_config", type=str, default=None, help="task config")
parser.add_argument("--checkpoint_path", type=str, default=None, help="path to save/load teleop checkpoint (env state)")
parser.add_argument("--auto_load_checkpoint", action="store_true", default=False, help="auto-load checkpoint on start if available")
parser.add_argument("--batch_name", type=str, default="default-batch", help="batch name for data collection")
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Override parallel environment count; omit to use value from task yaml.",
)
parser.add_argument(
    "--record_animation",
    action="store_true",
    default=False,
    help="[Deprecated] Use --anim_recording_enabled. Teleop animation recording uses Isaac Lab's OVD recorder.",
)
parser.add_argument(
    "--anim_recording_output_dir",
    type=str,
    default=None,
    help=(
        "Destination directory for the self-contained delivery folder produced after OVD bake. "
        "If omitted, the delivery is written next to the raw OVD run at "
        "``<IsaacLab>/anim_recordings/<ts>_delivery``. If provided, the delivery is written to "
        "this directory (created/overwritten). The raw OVD run itself always stays in "
        "``<IsaacLab>/anim_recordings/<ts>`` for debugging."
    ),
)
parser.add_argument(
    "--physics_backend",
    type=str,
    default="physx",
    choices=["newton", "physx"],
    help="Simulation physics backend (PhysX vs Newton). Omit to use task YAML or robofinals_PHYSICS_BACKEND.",
)
parser.add_argument(
    "--disable_physics_picking",
    action="store_true",
    default=False,
    help=(
        "Disable PhysX <-> USD/Fabric writeback overrides that enable Kit's Shift+LeftClick "
        "force-push tool. By default the overrides are ON for interactive (non-headless) PhysX "
        "teleop so picking works like in IsaacSim. Turn this on to recover IsaacLab's default "
        "fast-path (no USD writeback) for pure data-throughput runs."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
anim_cli_override_names = (
    "record_animation",
    "anim_recording_enabled",
    "anim_recording_start_time",
    "anim_recording_stop_time",
    "anim_recording_output_dir",
)
anim_cli_overrides = {}
for override_name in anim_cli_override_names:
    flag = f"--{override_name}"
    if hasattr(args_cli, override_name) and any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:]):
        anim_cli_overrides[override_name] = getattr(args_cli, override_name)
yaml_args = config_loader.load(args_cli.task_config)
merge_task_yaml_with_cli(args_cli, yaml_args)
for override_name, override_value in anim_cli_overrides.items():
    setattr(args_cli, override_name, override_value)
_configure_internal_ros2_runtime()

if getattr(args_cli, "record_animation", False):
    args_cli.anim_recording_enabled = True

# Long-running teleop bakes on exit; avoid Isaac Lab wall-clock auto-stop/shutdown.
if getattr(args_cli, "anim_recording_enabled", False):
    if args_cli.headless:
        raise ValueError("Animation recording requires a non-headless Kit UI (--viz kit).")
    args_cli.anim_recording_stop_time = 999999999.0

if getattr(args_cli, "anim_recording_enabled", False) and getattr(args_cli, "disable_fabric", False):
    print("[AnimRecording] OVD recording enabled; forcing disable_fabric=False so GPU teleop remains visible.")
    args_cli.disable_fabric = False

if getattr(args_cli, "disable_fabric", False) and "cuda" in str(getattr(args_cli, "device", "")).lower():
    raise ValueError(
        "`disable_fabric=true` with `device=cuda` is not a valid teleop setup in Isaac Lab: physics may run, but "
        "the renderer will not receive transform updates. Use Fabric-enabled teleop (`disable_fabric=false`) and "
        "launch with `--anim_recording_enabled` to record via the OVD recorder instead."
    )
app_launcher_args = vars(args_cli).copy()

# Isaac Lab interprets this AppLauncher flag as a request for the PhysX OVD
# extension. Newton records public runtime state directly, so keep the feature
# enabled on args_cli while preventing an unrelated PhysX recorder from loading.
if (
    str(getattr(args_cli, "physics_backend", "physx")).lower() == "newton"
    and getattr(args_cli, "anim_recording_enabled", False)
):
    app_launcher_args["anim_recording_enabled"] = False

if "kit" in (args_cli.visualizer or []) and not str(app_launcher_args.get("experience") or ""):
    app_launcher_args["experience"] = _DEFAULT_KIT_EXPERIENCE
local_kit_args = _build_local_kit_args(getattr(args_cli, "physics_backend", None))
_existing_kit_args = (app_launcher_args.get("kit_args") or "").strip()
app_launcher_args["kit_args"] = " ".join((*local_kit_args, _existing_kit_args)).strip()

app_launcher_args["profiler_backend"] = ["tracy"]
app_launcher_args["limit_cpu_threads"] = 1

# ---------------------------------------------------------------------------
# Kit / render profile
#
# benchmark_x7s_newton.py (known-good Newton + Kit path) only sets:
#   --/physics/fabricUseGPUInterop=false
# and does NOT run optimize_rendering() (no DLSS/async RTX carb toggles).
#
# PhysX teleop keeps the picking writeback overrides below.
# ---------------------------------------------------------------------------


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    import time
    start_time = time.time()
    print("starting isaacsim")
    app_launcher = AppLauncher(app_launcher_args)
    simulation_app = app_launcher.app
    from robofinals.core.compat import CompatibilityLayer

    CompatibilityLayer(backend=getattr(args_cli, "physics_backend", None)).enable()
    if getattr(args_cli, "enable_cameras", False):
        from isaaclab.app.settings_manager import get_settings_manager
        get_settings_manager().set_bool("/isaaclab/cameras_enabled", True)
    print(f"isaacsim started in {time.time() - start_time:.2f}s")

    from robofinals.applications.teleop import TeleopApplication

    session = None
    try:
        session = TeleopApplication.from_bootstrap(args_cli, app_launcher, simulation_app)
        session.setup().run()
    except BaseException as exc:
        import traceback

        print(f"[teleop] startup/session failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        try:
            if session is not None and getattr(args_cli, "anim_recording_enabled", False):
                env = getattr(session, "env", None)
                runtime = getattr(env, "backend_runtime", None)
                if runtime is not None:
                    delivery_dir = getattr(args_cli, "anim_recording_output_dir", None)
                    runtime.recording.finish_animation(env, delivery_dir=delivery_dir)
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
