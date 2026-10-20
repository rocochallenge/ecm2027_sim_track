# Chamber Service: Teleoperation and Evaluation

This repository contains the Chamber Service task, its local USD scene and textures, the Galbot description, and source launchers for teleoperation, demonstration replay, and policy evaluation. The launchers run this checkout directly; they do not start or require a RoboFinals Docker image.

## Runtime Prerequisites

- Linux x86_64, Python 3.12, an NVIDIA RTX GPU, and a driver supporting the selected CUDA 13.0 PyTorch wheels and Isaac Sim 6.0.1. Run `nvidia-smi` before installing. The existing development machine uses driver 580.173.02; this is an observed configuration, not a universal minimum-driver claim.
- The Python environment must provide Isaac Lab PhysX and the Isaac Lab Arena APIs used by this task. The small Arena Python subset needed by the local-scene task is included under `vendor/isaaclab_arena` with its Apache-2.0 license.
- Git LFS, to materialize the USD textures and robot assets.
- X11 access for the Kit teleoperation and evaluation server. VR teleoperation also needs the headset/browser to reach port `8012` on the host.

Isaac Sim and Isaac Lab are external simulator dependencies, installed from NVIDIA's package index and the public Isaac Lab source repository by the steps below, not from a RoboFinals Docker image or private GitLab submodule. You must accept NVIDIA's applicable licenses yourself. The launch scripts use the active `python` by default; set `PYTHON_BIN` to its interpreter path if needed.

## Install

### 1. System packages and a new Python environment

Install Conda or Miniforge separately if `conda` is not already available. On Ubuntu, install the host tools and graphical runtime libraries:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config \
  libgl1 libegl1 libglx0 libglib2.0-0 libsm6 libice6 libxrender1 \
  libxext6 libxrandr2 libxi6 libxcursor1 libxinerama1 libxkbcommon0 \
  libvulkan1 vulkan-tools ffmpeg
nvidia-smi
conda create -n roco-chamber python=3.12 -y
conda activate roco-chamber
git lfs install
git clone https://github.com/LightwheelAI/RoCo-SG27-x-Lightwheel.git
cd RoCo-SG27-x-Lightwheel
git lfs pull
```

The NVIDIA driver is a host prerequisite; the installer does not install or change it. A separately installed CUDA toolkit is not required for the prebuilt PyTorch wheels. Allow ample disk space for Isaac Sim, its extension cache, downloaded wheels, and the Isaac Lab checkout. The Python environment, pip cache, and driver must belong to the same Linux host used to launch the simulator.

### 2. Install the simulator and source dependencies

```bash
bash scripts/install_env.sh
```

The installer requires an activated virtual/Conda environment and Python 3.12. It installs PyTorch 2.11.0 / torchvision 0.26.0 / torchaudio 2.11.0 from the CUDA 13.0 index, Isaac Sim 6.0.1 with its extension cache, the selected PhysX/Kit Isaac Lab packages, and this project. It fetches public Isaac Lab tag `v3.0.0-beta2` at commit `28a37cecdd433c22d9eabd6a5954add9f13a8951` into the ignored `.deps/IsaacLab` directory. Do not delete that directory after installation: Isaac Lab is installed in editable mode.

The public tag reports Isaac Lab package version 6.1.11 and provides the split `isaaclab_physx` API used here. It is not the internal fork used for the earlier local runtime checks. The installer deliberately avoids `isaaclab[all]`, `isaaclab.sh --install all`, private submodules, Newton extras, training stacks, and unrelated robot integrations. In particular, the old installer's PyTorch 2.7 / CUDA 12.8 commands must not be mixed with this Isaac Sim 6.0.1 installation, whose core package requires PyTorch 2.11.0.

Core versions are constrained in `requirements/runtime-constraints.txt`; this is a compatibility constraints file, not a complete transitive lockfile. The application dependencies remain declared in `pyproject.toml`. The bundled Arena subset is selected by the launch scripts; do not install a different Arena package over it to repair imports. The installer runs `pip check` and the preflight checker, and stops on failure instead of declaring the environment ready.

The upstream tag's `isaaclab/setup.py` pins `coverage==7.6.1` and `pillow==12.1.1`, whereas Isaac Sim 6.0.1 requires `coverage==7.4.4` and `pillow==12.2.0`. These are otherwise unsatisfiable dependencies. The installer applies the checked-in `patches/isaaclab-6.1.11-isaacsim-6.0.1.patch` to align those two metadata pins with Isaac Sim; it does not modify simulator/task logic or bypass dependency checks with `--no-deps`. The Isaac Lab clone skips optional LFS downloads; this task's scene/robot/texture files are instead materialized by this repository's `git lfs pull`. Isaac Sim itself pulls some Newton dependencies transitively even though this project installs no Newton extras and uses PhysX.

### 3. Check installation and start the real runtime

```bash
python scripts/check_environment.py
python -m pip check
./scripts/run_teleop.sh --help
./scripts/run_eval_env.sh --help
./scripts/run_replay_eval.sh --help
./scripts/run_teleop.sh
```

The preflight checks Python/package versions, CUDA availability, LFS asset materialization, public Isaac Lab provenance, and local module locations without launching Kit. Passing it or `--help` is not a simulation test. Teleoperation requires a working desktop/X11 session (`echo "$DISPLAY"`), accessible GPU, and a VR headset connection to port 8012. WebXR also requires a browser-trusted secure context or an appropriately configured secure tunnel; merely opening the port does not establish a VR session. Use the replay and policy commands below for their separate runtime checks.

**Validation boundary (2026-10-20):** dependency resolution of the combined public-source simulator/Lab/project requirements passed in an empty Python 3.12 virtual environment after applying the two metadata fixes (264 packages; dry run, not a full installation). Shell/Python syntax and the teleop/eval/replay `--help` checks passed. A simulator-server startup using public Lab source and the existing machine's binary packages reached the IPC listener in one run; the subsequent Chamber attachment test timed out during NVIDIA Kit extension-registry synchronization, before task construction. This is not a validated clean-machine install or Chamber rollout. The existing development environment also differs from the constrained versions and is not a substitute for that test. Real VR interaction, replay completion, and policy rollout remain unverified. A trained policy/checkpoint is not included.

If Kit appears to hang at `Syncing extension registry`, inspect its latest log and check access to NVIDIA's extension service (`ovextensionsprod.blob.core.windows.net`); pip installation success does not prove that the first-launch network access is available. The current evaluation server opens Kit even when `--headless` is passed without livestream configuration, so the documented server command requires an X11 desktop; genuine display-free evaluation is not claimed here.

## Teleoperation

From a graphical session with the VR controller reachable:

```bash
./scripts/run_teleop.sh
```

The default config selects `Galbot-Mobile`, `vr-controller`, `ChamberServiceTask`, and `robofinals/data/assets/Scene/scene04_1.usd`. The HDF5 recording is written to `recordings/galbot.hdf5`.

```bash
PYTHON_BIN=/path/to/isaac/python ./scripts/run_teleop.sh
```

## Demonstration Replay

Run these commands from this repository's root in the installed simulator environment. No Docker image is required.

### Headless replay and success checks

Replay a recorded action dataset without a GUI:

```bash
./scripts/run_replay_eval.sh /absolute/path/to/recording.hdf5 action
```

This wrapper explicitly selects `--headless --without_image --device=cpu --record`. It is intended for replay and task success checks, not interactive viewing. Use `joint_target` instead of `action` for joint-target recordings. Do not use this wrapper for GUI replay: appending `--viz kit` does not remove its explicit `--headless` flag.

### GUI replay with three cameras

From a graphical session with a working `DISPLAY`, enable the Kit visualizer explicitly:

```bash
export PYTHONPATH="$PWD:$PWD/vendor/isaaclab_arena${PYTHONPATH:+:$PYTHONPATH}"
python -m robofinals.applications.replay.entrypoints.replay_action_demo \
  --dataset_file="$HOME/Downloads/ChamberServiceTask_1792484589627111/galbot_success.hdf5" \
  --replay_mode=action \
  --num_envs=1 \
  --enable_cameras \
  --viz kit
```

Replace the example dataset path with your own. Use `$HOME` or an absolute path rather than a quoted literal `~`. For joint-target recordings, change `--replay_mode=action` to `--replay_mode=joint_target`. Add `--width=640 --height=480` if lower camera/video resolution is sufficient; the defaults are 1920 by 1080 per camera.

`--enable_cameras` enables sensor rendering, not a GUI. The supported Isaac Lab launcher can default to headless when no visualizer is selected; use `--viz kit` and do not pass `--headless` for local GUI replay. Galbot replay selects three observation streams: the head camera and the two wrist cameras. The progress bar counts episodes, not frames, so a single episode can remain at `0/1` while its frames are replaying.

### Recordings from another machine or Docker

An HDF5 file may contain an absolute `usd_path` from the recording machine, such as `/workspace/robofinals/.../scene04_1_enabled.usd`. The replay entry point uses an existing recorded USD when available; otherwise it resolves the recorded `scene_name` relative to the working directory or this repository's root. For Chamber recordings with `scene_name=robofinals/data/assets/Scene/scene04_1.usd`, this selects the local source scene and lets the scene loader generate its enabled variant, rather than selecting a generated sibling and creating `_enabled_enabled.usd`. The original HDF5 metadata is not rewritten.

If the recorded source path is also unavailable, explicitly select the matching local scene:

```bash
# Append to the GUI replay command above, or to the headless wrapper command:
--layout="$PWD/robofinals/data/assets/Scene/scene04_1.usd"
```

`--layout` takes precedence over the recorded scene path. The local scene and assets must match the recording; resolving a path does not establish asset-version equivalence. Replay does not require a teleoperation device or controller connection.

### Outputs and troubleshooting

By default only the last episode is replayed. Use `--replay_all_clips` for all episodes or `--select_episodes 0` to select an episode by index. Replay JSON, divergence HDF5 files, and camera video when cameras are enabled are written beside the input HDF5, not under `output/`. With `--replay_all_clips`, per-episode artifacts go under `replay_results/<episode>/` beside the dataset; task `metrics.json`, when available, remains beside the dataset. `--record` additionally exports a replay-recording HDF5 beside the input. Ensure that directory is writable.

The replay entry point includes the `--disable_fabric` option (Fabric remains enabled by default), avoids duplicate camera registration, and no longer calls the teleoperation-only `teleop_interface.reset()` on GUI startup. A `rerender_on_reset` deprecation warning or a missing `omni.kit.context_menu` integration warning alone does not indicate replay failure. `RTX streaming completed` confirms RTX initialization, not that a desktop window was opened.

## Policy Evaluation

Policy evaluation separates the simulator server from a policy client. Start the server in one terminal:

```bash
./scripts/run_eval_env.sh
```

In another terminal, copy `configs/evaluation/chamber_policy_template.yml`, update the scene path and checkpoint, and make the policy module available in the active Python environment. `policy_name` uses the form `python_module:PolicyClass`; the class implements the interface in `policy/base.py`. Set the environment in the YAML to `ChamberServiceTask`, `Galbot-Mobile`, and local backends. For example:

```yaml
policy_name: my_chamber_policy:MyPolicy
checkpoint: /absolute/path/to/checkpoint
test_num: 1
time_out_limit: 100
instruction: Service the chamber
env_cfg:
  task: ChamberServiceTask
  scene_backend: local
  task_backend: local
  robot: Galbot-Mobile
  layout: /absolute/path/to/this/repository/robofinals/data/assets/Scene/scene04_1.usd
  enable_full_local_scene: true
  physics_backend: physx
  device: cpu
  num_envs: 1
  rl: null
  sources: []
  object_projects: []
```

```bash
./scripts/run_policy_eval.sh /absolute/path/to/policy.yml
```

Results are written to `eval_result/eval_results.json`. The policy implementation and model checkpoint are user-supplied; this repository provides the environment, IPC server/client, task success checks, and policy interface, not a trained model.

## Scope

The Chamber Service task uses bundled local assets and does not acquire its scene or task objects from a remote registry. The source subset retains the shared runtime modules needed by the launchers and evaluation path; RoboStack and unused JoyLo, Manus, Vive, and OpenXR device packages are excluded. The policy implementation and checkpoint remain user-supplied. A compatible Isaac Sim/Isaac Lab installation is still required because those simulator binaries and licensed components are not source code in this repository.
