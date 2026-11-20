# Chamber Service: Teleoperation and Evaluation

This repository provides the Chamber Service simulation, evaluation service, teleoperation and replay tools. Participants provide a policy Docker image; the evaluator runs the simulator and policy and records the evaluation results.

## Participant policy interface

The policy connects as a gRPC client to `ecm27.policy.v1.Policy`. The evaluator runs the gRPC server and supplies the endpoint to the policy container. The complete message definitions are in [`policy_v1.proto`](robofinals/distributed/policy_v1.proto); the field semantics are specified below. The service sends observations and accepts actions; scores and evaluator records are not part of the policy interface.

### Build and deliver a policy image

Build a Linux `amd64` policy image whose default entrypoint starts the client without additional command-line arguments. The evaluator sets the environment variable `ECM27_EVAL_ENDPOINT` to the server's `host:port` address. The client connects to this address; it does not need to listen on an incoming port. The evaluator keeps the policy process running while evaluation is in progress and stops it afterwards. Include the model weights and runtime dependencies needed by the entrypoint in the image, and provide the image's CPU, GPU and GPU-memory requirements with the image.

The policy must handle the evaluator-selected number of environments as a batch, returning one action per environment on each call and resetting any per-environment policy state when that environment ends. The [example client](examples/policy_docker/example_policy.py) includes observation decoding and per-environment resets. Its actions only demonstrate the interface; replace `ExamplePolicy` with your model.

```bash
# From the repository root. Edit the example Dockerfile and policy for your model.
docker build -f examples/policy_docker/Dockerfile -t ecm27-policy:latest .

# Deliver either this image archive or a registry image reference.
docker save -o ecm27-policy.tar ecm27-policy:latest
```

The example's [`Dockerfile.dockerignore`](examples/policy_docker/Dockerfile.dockerignore) restricts the build context; add your model files to its allowlist when adapting it. The evaluator provides the network and endpoint, for example by starting the image with `--network <evaluator-network> -e ECM27_EVAL_ENDPOINT=simulator:50000`. The simulator must be reachable at that address. A simulator running outside the policy container binds to an address reachable on the selected network using `--ipc_host`; `127.0.0.1` inside a policy container refers to that container itself.

The example uses Python 3.12, `grpcio`, `protobuf` and NumPy, and imports the client and generated message definitions from this repository. Participants may use another language or implementation that follows the same protobuf definition. Isaac Sim and Isaac Lab are required by the evaluator, not by the policy client.

### Requests and results

Only two calls are defined. The field meanings, types and shapes are fixed below; no metadata is exchanged.

| RPC | Request | Response |
|---|---|---|
| `GetObservation` | Empty | Current observation and `terminated`/`truncated` flags, one entry per environment. |
| `Step` | `ActionBatch` containing one 19- or 20-dimensional float32 action per environment. All actions in one batch must have the same dimension. | Next observation and per-environment flags; when an episode ends, the last observation before reset is also returned. |

The number of environments `N` is selected by the evaluator. The ordered `observations` list has one `Observation` per environment, so the policy can use its length as `N`; the current 64 MiB gRPC message limit permits 1–27 environments. This is a protocol limit and the actual hardware capacity can be lower. Each `Observation` has nine fixed byte fields listed below. Float arrays are little-endian float32 and images are uint8 RGB in `[height,width,3]` order; no names, types, shapes or episode identifiers are carried in the data.

The policy sends an action for every environment and waits for `Step` to finish before sending the next batch. One accepted call advances all environments by one control step. On normal episode completion, `observations[i]` is the new episode's first observation and `final_observations[i]` is the finished episode's last observation when `terminated[i]` or `truncated[i]` is true. If neither is true, that entry in `final_observations` must be ignored; the list is empty when no episode ended. Reset each affected environment's policy state after its end flag.

The evaluator enforces a default **300-minute wall-clock limit per environment episode**. When an episode reaches this limit, that environment is reset without restarting the simulator. Until the policy reads the reset result with `GetObservation`, `Step` is rejected with `FAILED_PRECONDITION` without applying the supplied actions. The returned `truncated` flag identifies the timed-out environment. The wall-clock watchdog runs between simulator calls; a simulator call stuck internally cannot be interrupted by this in-process timer.

Invalid action dimensions, non-finite values, invalid quaternions, and values outside the numerical protection limits below return `INVALID_ARGUMENT` before simulation advances. They are not separate scoring penalties. The provided Python client retries a safe read at most three times and retries `Step` only on the explicit server-busy rejection that occurs before execution. A lost or timed-out `Step` reply may follow an already executed action; do not blindly retry such a request.

### Observation fields

| Field | Shape | Meaning |
|---|---|---|
| `first_person_camera_rgb` | `[480,640,3]` uint8 | Head RGB image; sensor update period 1/30 s. |
| `left_hand_camera_rgb`, `right_hand_camera_rgb` | each `[224,224,3]` uint8 | Left and right wrist RGB images; sensor update period 0.05 s. |
| `actions` | `[19]` float32 | Last accepted action after quaternion normalization and legacy-to-19 conversion, before native base-velocity smoothing; zero after reset. |
| `joint_pos` | `[77]` float32 | Joint positions in radians relative to the default positions, in the order below. |
| `joint_vel` | `[77]` float32 | Joint velocities in rad/s relative to the default velocities, in the same order. |
| `eef_pos` | `[2,3]` float32 | Left then right end-effector position in world-axis coordinates, relative to the environment origin, in metres. |
| `eef_quat` | `[2,4]` float32 | Left then right end-effector world orientation, quaternion `[x,y,z,w]`. |
| `gripper_pos` | `[2]` float32 | Absolute positions of the left and right driven gripper joints, in radians. |

Each row above describes **one environment**. All 77 listed joints are revolute; default velocities are zero. The 77 joint names, in their exact observation order, are:

```python
joint_names = [
    "leg_joint1", "wheel1_joint", "wheel2_joint", "wheel3_joint", "wheel4_joint", "leg_joint2",
] + [f"wheel_{wheel}_passive_{joint}_joint"
     for wheel in (1, 2, 3, 4) for joint in range(10)] + [
    'leg_joint3',
    'leg_joint4',
    'leg_joint5',
    'head_joint1',
    'left_arm_joint1',
    'right_arm_joint1',
    'head_joint2',
    'left_arm_joint2',
    'right_arm_joint2',
    'left_arm_joint3',
    'right_arm_joint3',
    'left_arm_joint4',
    'right_arm_joint4',
    'left_arm_joint5',
    'right_arm_joint5',
    'left_arm_joint6',
    'right_arm_joint6',
    'left_arm_joint7',
    'right_arm_joint7',
    'left_gripper_joint',
    'left_gripper_l_inner_knuckle_joint',
    'left_gripper_l_knuckle_joint',
    'left_gripper_r_inner_knuckle_joint',
    'right_gripper_joint',
    'right_gripper_l_inner_knuckle_joint',
    'right_gripper_l_knuckle_joint',
    'right_gripper_r_inner_knuckle_joint',
    'left_gripper_l_finger_joint',
    'left_gripper_r_finger_joint',
    'right_gripper_l_finger_joint',
    'right_gripper_r_finger_joint',
]
assert len(joint_names) == 77
```

Default joint positions are zero except for the following radians (indices are zero-based):

| Index | Joint | Default position |
|---:|---|---:|
| 0 | `leg_joint1` | 0.93 |
| 5 | `leg_joint2` | 2.5 |
| 46 | `leg_joint3` | 1.56 |
| 50 | `left_arm_joint1` | 1.134464 |
| 51 | `right_arm_joint1` | -1.134464 |
| 52 | `head_joint2` | 0.35 |
| 53 | `left_arm_joint2` | -1.308997 |
| 54 | `right_arm_joint2` | 1.308997 |
| 55 | `left_arm_joint3` | -0.523599 |
| 56 | `right_arm_joint3` | 0.523599 |
| 57 | `left_arm_joint4` | -1.570796 |
| 58 | `right_arm_joint4` | 1.570796 |

### Actions

Prefer the 19-dimensional format. Both arms use absolute end-effector target poses in the robot root frame, `[x,y,z,qx,qy,qz,qw]`; positions are metres and quaternions use `[x,y,z,w]`. The service normalizes each arm's quaternion before applying the action and rejects quaternion norms below `1e-6` as invalid. The action layout is:

| Index | Command |
|---|---|
| `0:7` | Left arm target pose. |
| `7:14` | Right arm target pose. |
| `14`, `15` | Left and right gripper: a negative value closes; zero or positive opens. |
| `16:19` | Body-frame base velocity `[vx,vy,wz]`, in m/s, m/s and rad/s. |

The 20-dimensional legacy format keeps the first 16 values and replaces the base command with recorded `[w1,w2,w3,w4]`. The simulator converts it by `vx=(w1+w2+w3+w4)/4`, `vy=(-w1+w2+w3-w4)/4`, and `wz=(-w1+w2-w3+w4)/4`. These legacy values are not arbitrary physical wheel speeds. Native 19-dimensional base velocity uses exponential smoothing with alpha 0.1 per control step, initialized to zero on reset; legacy 20-dimensional base inputs bypass that smoothing. Do not truncate a 20-dimensional model output to 19 dimensions.

To protect simulator numerical stability, each arm target position component must be within ±10 m; each decoded base velocity component must be within ±10 m/s or rad/s. For the legacy format, each raw wheel value must also be within ±30. These are input-validity checks, not safety-related scoring items; a rejected action does not advance the environment or incur a score deduction.

The service's control step is 0.02 seconds of simulation time. Task success sets `terminated`; a 900-second simulation-time limit sets `truncated`. Participant policy execution time is wall-clock time and does not advance simulation until the next `Step` request.

## Evaluator: install and run the simulator

The evaluation runtime requires Linux x86_64, Python 3.12, an NVIDIA GPU with a compatible driver, Git LFS, FFmpeg and X11. On Ubuntu, install the host packages and prepare a Python environment:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config \
  libgl1 libegl1 libglx0 libglib2.0-0 libsm6 libice6 libxrender1 \
  libxext6 libxrandr2 libxi6 libxcursor1 libxinerama1 libxkbcommon0 \
  libvulkan1 vulkan-tools ffmpeg xvfb
git lfs install
git clone https://github.com/LightwheelAI/RoCo-SG27-x-Lightwheel.git
cd RoCo-SG27-x-Lightwheel
git lfs pull
python3.12 -m venv .venv
source .venv/bin/activate
bash scripts/install_env.sh
python -m pip install -r requirements/policy-grpc.txt
python scripts/check_environment.py
```

The installer provides Isaac Sim 6.0.1, the Isaac Lab dependencies and this project. The launcher uses the active Python interpreter; set `PYTHON_BIN` to select another interpreter. On a machine without a desktop X11 session, run the evaluation launcher under `xvfb-run -a`.

Edit [`configs/evaluation/chamber_server.example.json`](configs/evaluation/chamber_server.example.json) for the local scene path, device and number of environments, then start the service:

```bash
./scripts/run_eval_env.sh --remote_protocol=grpc \
  --eval_env_config configs/evaluation/chamber_server.example.json \
  --ipc_host=0.0.0.0 --ipc_port=50000 \
  --eval_record_dir ./eval_result/episodes
```

Set `--ipc_host` to an address reachable by the policy and pass that address to its container as `ECM27_EVAL_ENDPOINT`. The evaluator owns the simulator configuration, episode resets, scores and recordings.

## Teleoperation and replay

Run teleoperation from an X11 session with a reachable VR controller:

```bash
./scripts/run_teleop.sh
```

The default recording is `recordings/galbot.hdf5`. Replay a recording and check task success with:

```bash
./scripts/run_replay_eval.sh /absolute/path/to/recording.hdf5 action
```

Use `joint_target` in place of `action` for joint-target recordings. Replay results are written beside the recording. Run each launcher with `--help` for its options.
