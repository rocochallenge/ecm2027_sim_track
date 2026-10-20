# Chamber Service: Teleoperation and Evaluation

This repository contains the Chamber Service task, its local USD scene and textures, the Galbot description, and source launchers for teleoperation, demonstration replay, and policy evaluation. The launchers run this checkout directly; they do not start or require a RoboFinals Docker image.

## Runtime Prerequisites

- Linux x86_64, an NVIDIA GPU and driver supported by Isaac Sim, and a separately installed Isaac Sim 6.0.1 plus a compatible Isaac Lab PhysX Python environment (validated with Isaac Lab 6.1.11).
- The Python environment must provide Isaac Lab PhysX and the Isaac Lab Arena APIs used by this task. The small Arena Python subset needed by the local-scene task is included under `vendor/isaaclab_arena` with its Apache-2.0 license.
- Git LFS, to materialize the USD textures and robot assets.
- X11 access for the Kit teleoperation and evaluation server. VR teleoperation also needs the headset/browser to reach port `8012` on the host.

Isaac Sim and Isaac Lab are external NVIDIA simulator prerequisites, not bundled in this repository or supplied by a Docker image. Install them separately using NVIDIA's Isaac Sim and Isaac Lab installation instructions, then activate that Python environment before running these commands. The checked-in Python code and task assets are loaded from this repository. The scripts use the active `python` by default; set `PYTHON_BIN` to its interpreter path if needed.

## Install

```bash
git lfs install
git lfs pull
python -m pip install -e .
```

The editable install installs this repository's Python dependencies and registers its task, robot, and scene modules. It does not install Isaac Sim, Isaac Lab, PyTorch, or CUDA. Use the simulator-compatible Python environment for this step; do not install these application dependencies into a separate system Python and then launch with Isaac Sim's Python.

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

Replay a recorded action dataset and run the task's success checks:

```bash
./scripts/run_replay_eval.sh /absolute/path/to/recording.hdf5 action
```

Use `joint_target` for joint-target recordings. Replay outputs are written under `output/`.

## Policy Evaluation

Policy evaluation separates the simulator server from a policy client. Start the server in one terminal:

```bash
./scripts/run_eval_env.sh --headless
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
