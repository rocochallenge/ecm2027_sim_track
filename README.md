# Singapore Embodied AI Challenge for Manufacturing

**Official competition website:** [Singapore Embodied AI Challenge for Manufacturing](https://nrp.gov.sg/SGEAIchallenge/)

**ECM — technical repository and participant guide**

**Repository status:** The supplier Chamber Service source, local simulation assets, Galbot robot description, teleoperation, replay, and local policy evaluation tools are integrated. See [Code usage](#code-usage) for installation and launch commands. The official 12-point evaluator, intervention accounting, Docker submission service, trained baselines, and demonstration datasets remain pending. The current implementation differs from the official task specification as detailed below.

## Background

The Singapore Embodied AI Challenge for Manufacturing advances embodied AI for industrial robotic manipulation, with a focus on assembly and semiconductor cleanroom automation. The challenge combines an online simulation benchmark with a subsequent physical robot evaluation to study how policies perform on realistic, sequential manufacturing tasks.

This repository provides the simulation task, robot-control integration, and local evaluation tools for policy development. Baseline policies and official submission validation tools are still to be released.

## Task

The task is a chamber maintenance and component replacement sequence. A robot must open the chamber, extract an existing large component, pack it into a box, wipe the chamber interior, install a replacement component, and close the chamber.

The six subtasks are attempted in order. Each subtask has two action-specific scoring milestones: an intermediate achievement and completion of the required operation. Each milestone is worth 1 point, for a maximum of 2 points per subtask and **12 points per episode**. The milestones are described below; their exact detection criteria and numeric thresholds must be published with the evaluator.

| Subtask | Required operation | Preliminary completion condition | Available points |
| - | - | - | - |
| 1 | Lift off the chamber lid | The lid is lifted by at least a specified distance. | 1 for securely grasping the lid; 1 for lifting it clear of the chamber to the required distance. |
| 2 | Extract the large component | The original component is removed and moved a specified distance away from its initial position. | 1 for securely grasping the original component; 1 for extracting it and moving it the required distance from its initial position. |
| 3 | Pack the component into a box | The component is released into the box, contacts the box, and its center is below the box's highest point. | 1 for moving the held component above the box opening; 1 for releasing and successfully placing it inside the box. |
| 4 | Wipe the chamber interior | Contact is registered at all required sampling points: eight on the bottom, excluding the central support region, and three on each of the four side walls, for 20 points in total. | 1 for picking up the wiping tool; 1 for wiping all 20 required sampling points with the tool. |
| 5 | Unbox and place the replacement | The replacement component is within a specified distance of the original component's initial position. | 1 for picking up and lifting the replacement component out of its box; 1 for placing it at the required installation position. |
| 6 | Close and lock the chamber | The chamber lid is within a specified distance of its initial position; a locking-state check remains to be defined. | 1 for picking up the removed lid for reinstallation; 1 for replacing it in the required closed position and satisfying the finalized locking criterion. |

**Task details to publish:** robot embodiment and joint limits; chamber and component geometry; object masses and friction; box and wiping-tool specifications; initial conditions and randomization ranges; available sensors; intermediate and completion milestone criteria for each subtask; and annotated subtask illustrations.

The preliminary subtask 6 completion test checks lid position. A separate locking-state check, and any orientation or insertion checks needed for subtask 5, remain to be defined before the rules are finalized. Intermediate and completion checks must distinguish the two milestones in each subtask. For subtask 3, define the target region above the box opening and a successful release and placement; picking the component again is not a scoring requirement.

## Current implementation and official rules

The task table above and the scoring requirements below describe the **official specification to finalize**. The imported implementation in [`robofinals_tasks/chamber_service_task.py`](robofinals_tasks/chamber_service_task.py) currently provides a development success checker with six sequential, latched stage flags and a `process_score` from **0 to 6**. Each stage requires the preceding stage; it does not implement the two milestones per subtask or distinguish robot achievements from manual interventions. Do not interpret its progress score as the official score out of 12.

| Operation | Current development check | Difference from the official specification |
| - | - | - |
| Open lid | Normalized articulated lid joint position `>= 0.4`. | Joint rotation rather than a detached-lid grasp and lift-distance check. |
| Extract original part | Part is outside the chamber's XY proximity region (`distance >= 0.4 m`). | No separate grasp or 3D displacement milestone. |
| Pack part | Part's XY distance to the empty box is `< 0.15 m`. | No above-box milestone, release, height, or box-contact check. |
| Wipe chamber | Three separate sponge/chamber contacts, each continuously lasting `>= 0.5 s`; contact must break before another qualifies. | No tool-pick milestone or coverage check for the 20 required sampling points. |
| Install replacement | Replacement's XY distance to the chamber is `< 0.4 m`. | No unboxing milestone, orientation, or insertion check. |
| Close lid | Normalized articulated lid joint position `<= 0.005`. | No lid-pick milestone or independent locking-state check. |

The proximity checks use object/fixture positions resolved by [`object_utils.py`](robofinals/utils/object_utils.py), not a validated box-volume or insertion test. The task sets `episode_length_s = 300`, but the policy server explicitly disables the environment timeout termination; policy implementations must enforce their rollout limit. `time_out_limit` in the policy YAML is passed to the policy and is not enforced by the launcher itself.

## Policy evaluation and scoring

### Subtask scoring and manual intervention

The official evaluator must record two milestones for each of the six subtasks:

- **Intermediate achievement: 1 point** when the robot achieves the first milestone specified for that subtask in the table, such as grasping the lid, moving the held component above the box, or unboxing the replacement.
- **Subtask completion: 1 point** when the robot achieves the second milestone specified for that subtask, such as lifting the lid clear, releasing the component into the box, or installing the replacement.

Each milestone can earn credit only once per episode. Repeating an action or satisfying the same milestone again does not earn additional points. The episode score is the sum of all credited milestones, from **0 to 12**.

Manual manipulation of objects is permitted under the official intervention procedure to skip a challenging part and allow the robot to continue with the remaining work. A failed or skipped milestone does not prevent later milestones or subtasks from earning credit. Manual manipulation itself earns no points: only milestones achieved by the robot are credited. If an intervention supplies a pick or other prerequisite, the robot may still earn the point for a subsequent operation it completes, provided the evaluator can distinguish that operation from the manually completed portion.

For example, if the component is manually moved above the box and the robot then releases it and successfully places it inside, subtask 3 earns 1 point for placement and 0 points for the manually supplied positioning milestone. Subtasks 4–6 remain eligible for credit. Achieving both milestones in all six subtasks autonomously earns 12 points.

The intervention procedure must define who may manipulate objects, when intervention is allowed, permitted object movements, how execution pauses and resumes, how policy state is handled, and how affected milestones are marked as skipped or still eligible. Interventions must be logged, and manually achieved conditions must not be mistaken for robot achievements.

The evaluator should report the episode score; intermediate and completion milestone outcomes and times for every subtask; elapsed episode time; intervention records; and termination reason. Milestone outcomes should distinguish robot-achieved, manually supplied or skipped, failed, and not attempted states. Across episodes, report mean score out of 12, per-milestone robot success rates, autonomous full-task success rate, and intervention frequency. Official aggregation and ranking rules are **TBD**.

### Policy evaluation

Official evaluation is planned to use containerized policies so teams can manage their own software dependencies while keeping evaluation consistent and reproducible. The current local development path uses a Python policy client and simulator IPC server; it does not implement the Docker submission protocol described here.

For the official submission workflow, teams will submit a **Docker image** containing their policy, dependencies, and model weights. The planned standardized service interface is:

- `health()`: report readiness.
- `reset(task_info)`: initialize a new episode and reset policy state.
- `predict(observation)`: return an action or an allowed action sequence.
- `close()`: release resources and shut down.

The official evaluator manages the robot or simulator, sends observations, validates and executes actions, and computes scores. Teams may choose their internal software stack, provided it meets the published hardware compatibility requirements, resource limits, and interface specification. Policy services do not directly control scoring or manual intervention.

All submissions must run **without external network access** and pass the official validation checks before evaluation. Observation formats, action definitions, timing constraints, episode reset rules, and failure handling are standardized across teams. Image digests, evaluation configurations, and execution logs are recorded for reproducibility.

### Evaluation specification to finalize

Simulation platform collaborators should publish the following alongside the evaluator:

- Exact intermediate and completion milestone criteria for every subtask, including valid grasps, lifting and extraction, positioning above the box, release and placement, tool pickup and wiping, unboxing and installation, and lid replacement and locking. Define how objects already held between subtasks are handled without requiring an unnecessary repeated pick.
- Numeric thresholds, units, reference frames, and exact comparison operators for all distance tests.
- Definitions of component center, box height, valid box contact, and valid wiping contact; identify which object or tool must make contact.
- Coordinates and tolerances of the 20 wiping points, whether contacts accumulate over time, and any required contact duration.
- Whether milestone success is latched or must remain satisfied; handling of dropped objects, collisions, invalid actions, resets, and timeouts.
- The manual intervention procedure, milestone eligibility after intervention, intervention logging, and whether intervention time counts toward episode limits.
- Episode time or step limit, number of evaluation episodes, seeds, scene variations, and permitted retries.
- Official score aggregation, tie-break rules, and the relationship between local validation and final evaluation.
- Permitted policy observations and control modes, including whether privileged simulator state is excluded.
- Policy service transport and message schemas for `health()`, `reset(task_info)`, `predict(observation)`, and `close()`, including readiness, error responses, and shutdown behavior.
- Hardware compatibility requirements, container runtime requirements, CPU/GPU and memory limits, and image submission and validation procedures.
- Standard observation and action definitions, allowed action sequences, timing and synchronization constraints, episode reset rules, intervention-resume behavior, and failure handling.

Keep the published rules, evaluator configuration, and implementation consistent and versioned together.

## Code usage

### Repository structure

| Path | Contents |
| - | - |
| `pyproject.toml` | Python package metadata, dependencies, and module registrations. |
| `robofinals/` | Simulator runtime, robot control, devices, teleoperation, replay, IPC client/server, and utilities. |
| `robofinals_tasks/` | Chamber Service task and shared task implementation. |
| `robofinals/modules/robots/galbot/` | Galbot configuration, action definitions, cameras, and embodiment registration. |
| `robofinals/data/assets/` | Local chamber scene, textures, Galbot USD description, and sponge assets. |
| `policy/base.py` | Abstract interface for user-supplied local policies. |
| `configs/` | Teleoperation defaults and policy evaluation template. |
| `scripts/` | Environment installer, preflight checker, and source launchers. |
| `requirements/` and `patches/` | Runtime version constraints and Isaac Lab dependency compatibility patch. |
| `robofinals/tests/` | Replay scene-resolution and camera-registration regression tests. |
| `vendor/isaaclab_arena/` | Required Arena source subset and Apache-2.0 license. |

The Python distribution is `ecm2027-sim-track`. Existing `robofinals`, `robofinals_tasks`, `policy`, and `isaaclab_arena` import names are retained for compatibility. Run commands below from the `ecm2027_sim_track` repository root on the supported Linux host. The launchers use this checkout directly and require no RoboFinals Docker image.

### Runtime Prerequisites

- Linux x86_64, Python 3.12, an NVIDIA RTX GPU, and a driver supporting the selected CUDA 13.0 PyTorch wheels and Isaac Sim 6.0.1. Run `nvidia-smi` before installing. The supplier reports driver 580.173.02 on its development machine; this is not a universal minimum-driver claim or a configuration verified by this integration.
- The Python environment must provide Isaac Lab PhysX and the Isaac Lab Arena APIs used by this task. The small Arena Python subset needed by the local-scene task is included under `vendor/isaaclab_arena` with its Apache-2.0 license.
- Git LFS, to materialize the USD textures and robot assets.
- X11 access for the Kit teleoperation and evaluation server. VR teleoperation also needs the headset/browser to reach port `8012` on the host.

Isaac Sim and Isaac Lab are external simulator dependencies, installed from NVIDIA's package index and the public Isaac Lab source repository by the steps below, not from a RoboFinals Docker image or private GitLab submodule. You must accept NVIDIA's applicable licenses yourself. The launch scripts use the active `python` by default; set `PYTHON_BIN` to its interpreter path if needed.

### Install

#### 1. System packages and a new Python environment

Install Conda or Miniforge separately if `conda` is not already available. On Ubuntu, install the host tools and graphical runtime libraries:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config \
  libgl1 libegl1 libglx0 libglib2.0-0 libsm6 libice6 libxrender1 \
  libxext6 libxrandr2 libxi6 libxcursor1 libxinerama1 libxkbcommon0 \
  libvulkan1 vulkan-tools ffmpeg
nvidia-smi
conda create -n ecm2027-sim-track python=3.12 -y
conda activate ecm2027-sim-track
git lfs install
git clone https://github.com/rocochallenge/ecm2027_sim_track.git
cd ecm2027_sim_track
git lfs pull
```

The NVIDIA driver is a host prerequisite; the installer does not install or change it. A separately installed CUDA toolkit is not required for the prebuilt PyTorch wheels. Allow ample disk space for Isaac Sim, its extension cache, downloaded wheels, and the Isaac Lab checkout. The Python environment, pip cache, and driver must belong to the same Linux host used to launch the simulator.

#### 2. Install the simulator and source dependencies

```bash
bash scripts/install_env.sh
```

The installer requires an activated virtual/Conda environment and Python 3.12. It installs PyTorch 2.11.0 / torchvision 0.26.0 / torchaudio 2.11.0 from the CUDA 13.0 index, Isaac Sim 6.0.1 with its extension cache, the selected PhysX/Kit Isaac Lab packages, and this project. It fetches public Isaac Lab tag `v3.0.0-beta2` at commit `28a37cecdd433c22d9eabd6a5954add9f13a8951` into the ignored `.deps/IsaacLab` directory. Do not delete that directory after installation: Isaac Lab is installed in editable mode.

The supplier specifies Isaac Lab package version 6.1.11 at this public tag, providing the split `isaaclab_physx` API used here. The installer deliberately avoids `isaaclab[all]`, `isaaclab.sh --install all`, private submodules, Newton extras, training stacks, and unrelated robot integrations. Keep the pinned PyTorch 2.11.0 / CUDA 13.0 environment consistent with the selected Isaac Sim release.

Core versions are constrained in `requirements/runtime-constraints.txt`; this is a compatibility constraints file, not a complete transitive lockfile. The application dependencies remain declared in `pyproject.toml`. The bundled Arena subset is selected by the launch scripts; do not install a different Arena package over it to repair imports. The installer runs `pip check` and the preflight checker, and stops on failure instead of declaring the environment ready.

The upstream tag's `isaaclab/setup.py` pins `coverage==7.6.1` and `pillow==12.1.1`, whereas Isaac Sim 6.0.1 requires `coverage==7.4.4` and `pillow==12.2.0`. These are otherwise unsatisfiable dependencies. The installer applies the checked-in `patches/isaaclab-6.1.11-isaacsim-6.0.1.patch` to align those two metadata pins with Isaac Sim; it does not modify simulator/task logic or bypass dependency checks with `--no-deps`. The Isaac Lab clone skips optional LFS downloads; this task's scene/robot/texture files are instead materialized by this repository's `git lfs pull`. Isaac Sim itself pulls some Newton dependencies transitively even though this project installs no Newton extras and uses PhysX.

#### 3. Check installation and start the real runtime

```bash
export PYTHONPATH="$PWD:$PWD/vendor/isaaclab_arena${PYTHONPATH:+:$PYTHONPATH}"
python scripts/check_environment.py
python -m pip check
bash scripts/run_teleop.sh --help
bash scripts/run_eval_env.sh --help
python -m robofinals.applications.replay.entrypoints.replay_action_demo --help
bash scripts/run_teleop.sh
```

The preflight checks Python/package versions, CUDA availability, LFS asset materialization, public Isaac Lab provenance, and local module locations without launching Kit. Passing it or `--help` is not a simulation test. Teleoperation requires a working desktop/X11 session (`echo "$DISPLAY"`), accessible GPU, and a VR headset connection to port 8012. WebXR also requires a browser-trusted secure context or an appropriately configured secure tunnel; merely opening the port does not establish a VR session. Use the replay and policy commands below for their separate runtime checks.

**Local integration validation (2026-10-21):** all 409 imported Python files passed AST parsing with Python 3.12 grammar on the available Windows Python 3.14 interpreter; the seven supplied replay regression tests passed. All six Shell scripts and 12 README Bash blocks passed syntax checks; the installer's `--help` ran successfully. Asset materialization and file-copy integrity were checked locally. The environment preflight failed because this host has Windows/Python 3.14 rather than the required Linux/Python 3.12 environment, missing simulator/PyTorch packages, and no accessible CUDA runtime. Dependency installation, Kit startup, USD dependency resolution, VR interaction, headless replay, policy rollouts, and simulator-dependent CLI `--help` have not been validated here. These checks do not establish Python 3.12 or simulator compatibility. A trained policy/checkpoint and demonstration HDF5 are not included.

If Kit appears to hang at `Syncing extension registry`, inspect its latest log and check access to NVIDIA's extension service (`ovextensionsprod.blob.core.windows.net`); pip installation success does not prove that the first-launch network access is available. The current evaluation server opens Kit even when `--headless` is passed without livestream configuration, so the documented server command requires an X11 desktop; genuine display-free evaluation is not claimed here.

### Teleoperation

From a graphical session with the VR controller reachable:

```bash
bash scripts/run_teleop.sh
```

The default config selects `Galbot-Mobile`, `vr-controller`, `ChamberServiceTask`, and `robofinals/data/assets/Scene/scene04_1.usd`. The HDF5 recording is written to `recordings/galbot.hdf5`.

```bash
PYTHON_BIN=/path/to/isaac/python bash scripts/run_teleop.sh
```

### Demonstration Replay

Run these commands from this repository's root in the installed simulator environment. No Docker image is required.

#### Headless replay and success checks

Replay a recorded action dataset without a GUI:

```bash
bash scripts/run_replay_eval.sh /absolute/path/to/recording.hdf5 action
```

This wrapper explicitly selects `--headless --without_image --device=cpu --record`. It is intended for replay and task success checks, not interactive viewing. Use `joint_target` instead of `action` for joint-target recordings. Do not use this wrapper for GUI replay: appending `--viz kit` does not remove its explicit `--headless` flag.

#### GUI replay with three cameras

From a graphical session with a working `DISPLAY`, enable the Kit visualizer explicitly:

```bash
export PYTHONPATH="$PWD:$PWD/vendor/isaaclab_arena${PYTHONPATH:+:$PYTHONPATH}"
python -m robofinals.applications.replay.entrypoints.replay_action_demo \
  --dataset_file="$PWD/recordings/galbot.hdf5" \
  --replay_mode=action \
  --num_envs=1 \
  --enable_cameras \
  --viz kit
```

Replace the example dataset path with your own. Use `$HOME` or an absolute path rather than a quoted literal `~`. For joint-target recordings, change `--replay_mode=action` to `--replay_mode=joint_target`. Add `--width=640 --height=480` if lower camera/video resolution is sufficient; the defaults are 1920 by 1080 per camera.

`--enable_cameras` enables sensor rendering, not a GUI. The supported Isaac Lab launcher can default to headless when no visualizer is selected; use `--viz kit` and do not pass `--headless` for local GUI replay. Galbot replay selects three observation streams: the head camera and the two wrist cameras. The progress bar counts episodes, not frames, so a single episode can remain at `0/1` while its frames are replaying.

#### Recordings from another machine or Docker

An HDF5 file may contain an absolute `usd_path` from the recording machine, such as `/workspace/robofinals/.../scene04_1_enabled.usd`. The replay entry point uses an existing recorded USD when available; otherwise it resolves the recorded `scene_name` relative to the working directory or this repository's root. For Chamber recordings with `scene_name=robofinals/data/assets/Scene/scene04_1.usd`, this selects the local source scene and lets the scene loader generate its enabled variant, rather than selecting a generated sibling and creating `_enabled_enabled.usd`. The original HDF5 metadata is not rewritten.

If the recorded source path is also unavailable, explicitly select the matching local scene:

```bash
# Append to the GUI replay command above, or to the headless wrapper command:
--layout="$PWD/robofinals/data/assets/Scene/scene04_1.usd"
```

`--layout` takes precedence over the recorded scene path. The local scene and assets must match the recording; resolving a path does not establish asset-version equivalence. Replay does not require a teleoperation device or controller connection.

#### Outputs and troubleshooting

By default only the last episode is replayed. Use `--replay_all_clips` for all episodes or `--select_episodes 0` to select an episode by index. Replay JSON, divergence HDF5 files, and camera video when cameras are enabled are written beside the input HDF5, not under `output/`. With `--replay_all_clips`, per-episode artifacts go under `replay_results/<episode>/` beside the dataset; task `metrics.json`, when available, remains beside the dataset. `--record` additionally exports a replay-recording HDF5 beside the input. Ensure that directory is writable.

The replay entry point includes the `--disable_fabric` option (Fabric remains enabled by default), avoids duplicate camera registration, and no longer calls the teleoperation-only `teleop_interface.reset()` on GUI startup. A `rerender_on_reset` deprecation warning or a missing `omni.kit.context_menu` integration warning alone does not indicate replay failure. `RTX streaming completed` confirms RTX initialization, not that a desktop window was opened.

### Policy Evaluation

Policy evaluation separates the simulator server from a policy client. Start the server in one terminal:

```bash
bash scripts/run_eval_env.sh
```

In another terminal, activate the same installed environment and create a local policy config:

```bash
cp configs/evaluation/chamber_policy_template.yml configs/evaluation/chamber_policy.local.yml
```

Edit `configs/evaluation/chamber_policy.local.yml` to set your policy import and checkpoint, and make the policy module available in the active Python environment. The template's scene path is relative to this checkout; the launcher changes to the repository root before reading it. `policy_name` uses the form `python_module:PolicyClass`; the class implements the interface in `policy/base.py`. Set the environment in the YAML to `ChamberServiceTask`, `Galbot-Mobile`, and local backends. For example:

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
  layout: robofinals/data/assets/Scene/scene04_1.usd
  enable_full_local_scene: true
  physics_backend: physx
  device: cpu
  num_envs: 1
  rl: null
  sources: []
  object_projects: []
```

```bash
bash scripts/run_policy_eval.sh configs/evaluation/chamber_policy.local.yml
```

Results are written to `eval_result/eval_results.json`. The policy implementation and model checkpoint are user-supplied; this repository provides the environment, IPC server/client, task success checks, and policy interface, not a trained model.

### Local policy and robot interface

The current policy class accepts `usr_args` in its constructor and implements `get_model(usr_args)`, `get_action()`, `reset_model()`, and `eval(task_env, observation, usr_args, video_writer)` as defined in [`policy/base.py`](policy/base.py). `eval` owns the rollout loop and returns a success result per environment. The launcher resets the environment and model between batches and calls `format_eval_results`; its default JSON contains `test_count`, `success_count`, `success_rate`, and per-episode success booleans. It does not produce official 12-point milestone or intervention reports.

The IPC client connects to `127.0.0.1:50000`; the server constructs the scene after the client attaches its environment configuration. `task_env.reset()` returns observations and reset extras, while `task_env.step(action)` returns observations, reward, termination, truncation, and extras. See [`policy/base.py`](policy/base.py) and [`policy.py`](robofinals/applications/eval/entrypoints/policy.py) for the executable integration contract. `BasePolicy.step_environment` moves NumPy actions to CUDA; with the supplied CPU physics config, a custom policy should instead pass a tensor on the configured environment device or implement its own stepping logic.

`Galbot-Mobile` configures left/right differential-IK arm commands, binary gripper commands, and wheel joint velocities. Actions and observation ordering are defined by [`galbot.py`](robofinals/modules/robots/galbot/galbot.py) and [`assets_cfg.py`](robofinals/modules/robots/galbot/assets_cfg.py); obtain the action dimension from server metadata rather than assuming a vector size. Camera streams are the head and both wrists; policy observations also include joint state and previous actions. The default CPU setting selects physics placement and does not remove the NVIDIA RTX rendering requirement. A complete participant observation/action schema, units, calibration, resource limits, and container service specification remain to be published.

### Resources, baselines, and troubleshooting

The local task uses `robofinals/data/assets/Scene/scene04_1.usd`, `galbot_one_golf_description`, and `roco_chamber_service/Sponge027` under `robofinals/data/assets/`. Run `git lfs pull` after cloning; `python scripts/check_environment.py` checks whether assets are still LFS pointers. Generated `*_enabled.usd` scenes, recordings, dependency checkouts, and evaluation output are ignored. Keep the bundled Arena source in `PYTHONPATH`, as done by the launchers. Download and cache external simulator components during preparation; local assets alone do not establish offline Kit startup.

No demonstration dataset, trained ACT/π0/π0.5 checkpoint, baseline Docker image, or official image validator is included. Supply a compatible recording for replay and your own policy module/checkpoint for policy evaluation. Minimum VRAM, RAM, storage, and a validated driver compatibility matrix remain to be confirmed with a clean supported-host rollout. Native Windows operation is not supported by the installer.

This is a Chamber/Galbot runtime subset. The generic scripts under `vendor/isaaclab_arena/isaaclab_arena/scripts/` refer to Arena example environments that are not bundled. The shared `robofinals/utils/lerobot_utils.py` also refers to SO100/SO101 robot asset configuration absent from this delivery; installing the optional LeRobot dependencies does not supply those robot modules. Those tools are outside the documented Chamber launch paths and remain unsupported in this subset.

For lightweight regression checks that do not require the simulator:

```bash
python -m unittest discover -s robofinals/tests -p 'test_*.py' -v
```

The root [LICENSE](LICENSE) remains in place. Preserve the imported Lightwheel copyright/Apache-2.0 file headers and the [Arena license](vendor/isaaclab_arena/LICENSE.md); separate scene, robot, texture, dataset, and checkpoint redistribution terms still need confirmation before participant publication.

## Data release plan

**Recommended hosting:** a dedicated Hugging Face dataset repository for demonstrations and a model repository for baseline checkpoints. The current local scene assets are included with Git LFS; future larger asset releases should use external hosting with versioned manifests and download utilities. Participants must package required model weights and runtime dependencies into their Docker images before evaluation; hosting links are for preparation and distribution.

| Resource | Planned contents | Release trigger | Storage link |
| - | - | - | - |
| Simulation assets | Chamber scene, Galbot description, textures, and sponge are bundled; asset licensing and a versioned checksum manifest remain pending. | Included in this source integration. | [`robofinals/data/assets/`](robofinals/data/assets/) (Git LFS) |
| Starter demonstrations | Small sample dataset and loader examples for validating the pipeline. | With the quick-start release. | TBD — Hugging Face dataset repository |
| Training demonstrations | Versioned trajectories with observations, actions, episode metadata, intermediate and completion milestone annotations, and intervention records where applicable. | After schema and quality checks. | TBD — Hugging Face dataset repository |
| Public validation resources | Reproducible validation configurations and any designated public data. | With the evaluator release. | TBD |
| Baseline checkpoints | Weights, model cards, dependencies, and reference results. | With each baseline release. | TBD — Hugging Face model repository |

Release dates, dataset sizes, modalities, formats, and access conditions are **TBD**. Publish dataset cards covering provenance, licenses, schema, splits, normalization, calibration, limitations, and known issues. Provide download instructions, revision identifiers, checksums, and sample-loading code.

**Proposed split policy:** keep training and public validation resources separate; keep final evaluation seeds, scenarios, and any held-out data private until an approved release. Confirm the split policy before publishing data.

## Additional repository requirements

- **Licensing:** include a code license and separate licenses or usage terms for assets, datasets, and model weights.
- **Versioning:** maintain a changelog and compatibility matrix for code, simulator, assets, datasets, checkpoints, policy service interface, and evaluation rules.
- **Validation:** include checks that examples run and that the evaluator handles milestone scoring, manual interventions, threshold boundaries, and termination consistently. Verify the 12-point maximum, no duplicate credit, no credit for manually supplied milestones, and continued scoring after skipped milestones. Validate policy containers against the official service, hardware, resource, and network requirements.
- **Participant support:** provide troubleshooting, known limitations, issue templates, and a technical contribution guide.
- **Benchmark reference:** add task documentation and a citation entry when the benchmark reference is available.

## Simulation platform collaborators' preparation checklist

- [x] Import the supplier simulator integration, task assets, robot interface, and runtime dependency constraints (full runtime validation pending).
- [x] Replace the proposed paths with the actual repository structure.
- [ ] Add tested installation, robot-control, policy container, rollout, and evaluation examples.
- [ ] Implement the six-subtask, 12-point evaluator with the action-specific intermediate and completion milestones listed in the task table.
- [ ] Finalize intermediate milestone criteria, completion thresholds, manual intervention procedures, aggregation, and termination rules.
- [ ] Publish API and sensor documentation, including observation and action schemas.
- [ ] Publish the Docker policy service specification, hardware compatibility requirements, resource limits, and official validation checks.
- [ ] Verify evaluation without external network access and record image digests, evaluation configurations, and execution logs.
- [ ] Release a working baseline image and reference evaluation results.
- [ ] Create the data and checkpoint repositories; replace storage-link placeholders.
- [ ] Publish dataset loaders, manifests, checksums, and licensing information.
- [ ] Document the participant policy and image submission interface.
- [ ] Verify the complete quick start from a clean environment and remove resolved TBD entries.
