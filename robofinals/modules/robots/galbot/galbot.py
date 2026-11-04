"""Fixed-base dual-arm teleoperation embodiment for Galbot One Golf."""

from dataclasses import MISSING
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply
from isaaclab_arena.utils.pose import Pose

import robofinals.core.mdp as mdp
import robofinals.core.mdp as mdp_isaac_lab
from robofinals.core.mdp import ee_frame_pos, ee_frame_quat, gripper_pos
from robofinals.modules.robots.robot_arena_base import EmbodimentBasePolicyObservationCfg, LwEmbodimentBase
from robofinals.utils.env import ExecuteMode
from robofinals.utils.isaac_data_compat import as_torch, sim_quat_raw_to_xyzw_torch

from .assets_cfg import (
    GALBOT_CFG,
    GALBOT_LEFT_ARM_JOINTS,
    GALBOT_LEFT_GRIPPER_JOINTS,
    GALBOT_MOBILE_CFG,
    GALBOT_RIGHT_ARM_JOINTS,
    GALBOT_RIGHT_GRIPPER_JOINTS,
    GALBOT_WHEEL_JOINTS,
    OFFSET_CONFIG,
)

# Import USD/PhysX for collision filtering
try:
    from pxr import UsdPhysics, Sdf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

# TCP offset from arm_link7. FrameTransformer uses (w, x, y, z); DiffIK OffsetCfg uses (x, y, z, w).
_GALBOT_TCP_OFFSET_POS = (0.0, 0.0, 0.0)
_GALBOT_TCP_OFFSET_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)
_GALBOT_TCP_OFFSET_ROT_XYZW = (0.0, 0.0, 0.0, 1.0)

# xyzw: 180deg about X. (x,y,z)->(x,-y,-z).
_GALBOT_X180_XYZW = (1.0, 0.0, 0.0, 0.0)


def _quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1⊗q2 for xyzw quaternions (apply q2 first, then q1)."""
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def setup_galbot_gripper_collision_filtering(env_prim_path: str):
    """Setup collision filtering for Galbot gripper fingers.

    Disables collisions between gripper finger collision meshes to allow closure,
    while keeping collisions with external objects enabled for grasping.

    Args:
        env_prim_path: Path to the environment prim (e.g., "/World/envs/env_0")
    """
    if not USD_AVAILABLE:
        print("[WARN] USD not available, skipping gripper collision filtering")
        return

    stage = sim_utils.get_current_stage()
    robot_path = f"{env_prim_path}/Robot"

    # Define gripper finger links that should not collide with each other
    gripper_links = [
        f"{robot_path}/left_gripper_l_finger_link",
        f"{robot_path}/left_gripper_r_finger_link",
        f"{robot_path}/right_gripper_l_finger_link",
        f"{robot_path}/right_gripper_r_finger_link",
    ]

    # Apply filtered pairs to each gripper link's collision meshes
    for link_path in gripper_links:
        collisions_path = f"{link_path}/collisions"
        collisions_prim = stage.GetPrimAtPath(collisions_path)

        if not collisions_prim.IsValid():
            continue

        # Apply FilteredPairsAPI to the collisions group
        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(collisions_prim)
        filtered_pairs_rel = filtered_pairs.CreateFilteredPairsRel()

        # Add all other gripper links as filtered pairs
        for other_link_path in gripper_links:
            if other_link_path != link_path:
                other_collisions_path = f"{other_link_path}/collisions"
                target_sdf_path = Sdf.Path(other_collisions_path)
                if target_sdf_path not in filtered_pairs_rel.GetTargets():
                    filtered_pairs_rel.AddTarget(target_sdf_path)


@configclass
class ActionsCfg:
    left_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    right_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    left_hand_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_hand_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class MobileActionsCfg(ActionsCfg):
    wheels_action: mdp.JointVelocityActionCfg = MISSING


@configclass
class GalbotCameraCfg:
    first_person_camera: TiledCameraCfg = None
    left_hand_camera: TiledCameraCfg = None
    right_hand_camera: TiledCameraCfg = None


@configclass
class GalbotPolicyObservationsCfg(EmbodimentBasePolicyObservationCfg):
    actions = ObsTerm(func=mdp_isaac_lab.last_action)
    joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
    eef_pos = ObsTerm(func=ee_frame_pos)
    eef_quat = ObsTerm(func=ee_frame_quat)
    gripper_pos = ObsTerm(func=gripper_pos)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class GalbotSceneCfg:
    robot: ArticulationCfg = GALBOT_CFG
    ground: AssetBaseCfg | None = None
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_arm_link7",
                name="tool_left_hand",
                offset=OffsetCfg(pos=_GALBOT_TCP_OFFSET_POS, rot=_GALBOT_TCP_OFFSET_ROT_WXYZ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_arm_link7",
                name="tool_right_hand",
                offset=OffsetCfg(pos=_GALBOT_TCP_OFFSET_POS, rot=_GALBOT_TCP_OFFSET_ROT_WXYZ),
            ),
        ],
    )


class GalbotEnvCfg(LwEmbodimentBase):
    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None, mobile: bool = False):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.name = "Galbot-Mobile" if mobile else "Galbot-Abs"
        self.mobile = mobile
        self.robot_scale = robot_scale
        self.robot_base_link = "base_link"
        self.robot_base_offset = {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0]}
        self.robot_to_fixture_dist = 0.50
        self.offset_config = OFFSET_CONFIG
        self.camera_config = GalbotCameraCfg()
        self.observation_cameras = {
            "first_person_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/head_link2/head_end_effector_mount_link/camera_front_head_left_rgb",
                    # IsaacLab and golf_sensors.json both use ROS quaternions as (x, y, z, w).
                    offset=TiledCameraCfg.OffsetCfg(
                        pos=(0.082781626, -0.053015078, 0.031427263),
                        rot=(-0.094725615, 0.702703943, 0.094480690, 0.698790116),
                        convention="ros",
                    ),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                        intrinsic_matrix=[
                            198.96716907989332, 0.0, 343.12156681644376,
                            0.0, 198.96716907989332, 226.75427303925488,
                            0.0, 0.0, 1.0,
                        ],
                        width=640,
                        height=480,
                        clipping_range=(0.1, 15.0),
                    ),
                    width=640,
                    height=480,
                    update_period=1.0 / 30.0,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS, ExecuteMode.EVAL],
            },
            "left_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_arm_link7/left_arm_end_effector_mount_link/camera_left_wrist",
                    offset=TiledCameraCfg.OffsetCfg(
                        pos=(0.07089459385344977, 0.011084636915734618, 0.0475356786953811),
                        rot=(-0.5921350168801781, 0.5860286987303605, -0.38397145780165504, 0.3981361647004496),
                        convention="ros",
                    ),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                        intrinsic_matrix=[202.0, 0.0, 200.0, 0.0, 202.0, 112.0, 0.0, 0.0, 1.0],
                        width=400,
                        height=224,
                        clipping_range=(0.03, 10.0),
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS, ExecuteMode.EVAL],
            },
            "right_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_arm_link7/right_arm_end_effector_mount_link/camera_right_wrist",
                    offset=TiledCameraCfg.OffsetCfg(
                        pos=(0.07089459385344977, 0.011084636915734618, 0.0475356786953811),
                        rot=(0.5921350168801781, -0.5860286987303605, 0.38397145780165504, -0.3981361647004496),
                        convention="ros",
                    ),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                        intrinsic_matrix=[202.0, 0.0, 200.0, 0.0, 202.0, 112.0, 0.0, 0.0, 1.0],
                        width=400,
                        height=224,
                        clipping_range=(0.03, 10.0),
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS, ExecuteMode.EVAL],
            },
        }
        self.action_config = MobileActionsCfg() if mobile else ActionsCfg()
        self.policy_observation_config = GalbotPolicyObservationsCfg()
        self.scene_config = GalbotSceneCfg()
        if mobile:
            self.scene_config.robot = GALBOT_MOBILE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            self.scene_config.ground = AssetBaseCfg(
                prim_path="/World/galbot_ground",
                spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0), color=(0.18, 0.18, 0.18)),
            )
        self.scene_config.robot.spawn.scale = (robot_scale, robot_scale, robot_scale)

        self.action_config.left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=GALBOT_LEFT_ARM_JOINTS,
            body_name="left_arm_link7",
            controller=mdp.DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            scale=1.0,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=_GALBOT_TCP_OFFSET_POS,
                rot=_GALBOT_TCP_OFFSET_ROT_XYZW,
            ),
        )
        self.action_config.right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=GALBOT_RIGHT_ARM_JOINTS,
            body_name="right_arm_link7",
            controller=mdp.DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            scale=1.0,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=_GALBOT_TCP_OFFSET_POS,
                rot=_GALBOT_TCP_OFFSET_ROT_XYZW,
            ),
        )
        # Official Isaac Lab parallel-gripper pattern: binary open/close positions;
        # soft grasp comes from ImplicitActuatorCfg.effort_limit_sim on the driven joint.
        self.action_config.left_hand_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=GALBOT_LEFT_GRIPPER_JOINTS,
            open_command_expr={"left_gripper_joint": 0.0},
            close_command_expr={"left_gripper_joint": 1.703},
        )
        self.action_config.right_hand_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=GALBOT_RIGHT_GRIPPER_JOINTS,
            open_command_expr={"right_gripper_joint": 0.0},
            close_command_expr={"right_gripper_joint": 1.703},
        )
        if mobile:
            self.action_config.wheels_action = mdp.JointVelocityActionCfg(
                asset_name="robot",
                joint_names=GALBOT_WHEEL_JOINTS,
                scale=0.01,
                preserve_order=True,
            )

    def camera_viewport_prim_candidates(self) -> dict[str, tuple[str, ...]]:
        """USD camera prim names for Kit docked viewports (see ``ui_utils.setup_cameras``)."""
        return {
            "first_person": ("first_person_camera", "camera_front_head_left_rgb"),
            "left": ("left_hand_camera", "camera_left_wrist"),
            "right": ("right_hand_camera", "camera_right_wrist"),
        }

    def _debug_gripper_state(self, device, left_target: float, right_target: float) -> None:
        """Log Galbot gripper state when explicit diagnostics are enabled."""
        enabled = os.environ.get("ROBOFINALS_GALBOT_GRIPPER_DEBUG", "").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return

        count = getattr(self, "_galbot_gripper_debug_count", 0) + 1
        self._galbot_gripper_debug_count = count
        try:
            interval = max(1, int(os.environ.get("ROBOFINALS_GALBOT_GRIPPER_DEBUG_INTERVAL", "1")))
        except ValueError:
            interval = 1
        if count % interval != 0:
            return

        robot = device.env.scene.articulations.get("robot")
        if robot is None:
            print("[GalbotGripperDebug] robot articulation is unavailable", flush=True)
            return

        names = list(robot.data.joint_names)
        try:
            joint_ids = [names.index("left_gripper_joint"), names.index("right_gripper_joint")]
        except ValueError:
            print(f"[GalbotGripperDebug] gripper joints not found; joints={names}", flush=True)
            return

        pos = robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()
        vel = robot.data.joint_vel[0, joint_ids].detach().cpu().tolist()
        physics_dt = float(getattr(device.env, "physics_dt", getattr(device.env.cfg.sim, "dt", 0.0)))
        decimation = int(getattr(device.env.cfg, "decimation", 1))
        step_delta = [value * physics_dt for value in vel]
        targets = [float(left_target), float(right_target)]
        errors = [target - current for target, current in zip(targets, pos)]
        print(
            "[GalbotGripperDebug] "
            f"control_step={count} dt={physics_dt:.6f} decimation={decimation} "
            f"pos={pos} vel={vel} step_delta={step_delta} "
            f"target={targets} target_error={errors}",
            flush=True,
        )

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        num_envs = device.env.num_envs
        arm_actions = []
        for abs_arm in (action["left_arm_abs"], action["right_arm_abs"]):
            arm_action = abs_arm.clone()
            quat_xyzw = arm_action[3:7]
            if not torch.isfinite(quat_xyzw).all() or torch.linalg.norm(quat_xyzw) < 1e-6:
                quat_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0], device=arm_action.device, dtype=arm_action.dtype)
            quat_xyzw = quat_xyzw / torch.linalg.norm(quat_xyzw)
            arm_action[3:7] = quat_xyzw
            arm_actions.append(arm_action.repeat(num_envs, 1))
        # Isaac Lab BinaryJointPositionAction: action < 0 → close, >= 0 → open.
        # VR trigger is pressed → +1 (see VRDevice "Trigger (hold) = close gripper"),
        # so invert: pressed → -1 close, released → +1 open.
        left_cmd = -1.0 if action["left_gripper"] > 0 else 1.0
        right_cmd = -1.0 if action["right_gripper"] > 0 else 1.0
        left_gripper = torch.full((num_envs, 1), left_cmd, device=device.env.device)
        right_gripper = torch.full((num_envs, 1), right_cmd, device=device.env.device)
        self._debug_gripper_state(device, left_target=left_cmd, right_target=right_cmd)
        result = torch.cat([*arm_actions, left_gripper, right_gripper], dim=1)

        if self.mobile:
            # Mobile base: right stick = planar translate, left stick X = yaw.
            rbase = torch.as_tensor(
                action.get("rbase", (0.0, 0.0)), device=device.env.device, dtype=torch.float32
            ).reshape(-1)
            if rbase.numel() < 2:
                rbase = torch.nn.functional.pad(rbase, (0, 2 - rbase.numel()))
            lbase = torch.as_tensor(
                action.get("lbase", (0.0, 0.0)), device=device.env.device, dtype=torch.float32
            ).reshape(-1)
            if lbase.numel() < 2:
                lbase = torch.nn.functional.pad(lbase, (0, 2 - lbase.numel()))

            # OpenTeleVision remaps sticks as:
            #   thumbstick_x = -raw[1], thumbstick_y = -raw[0]
            # with raw ≈ [right, forward]. So physical forward/back is rbase[0],
            # physical left/right is rbase[1]. Body +X = forward, +Y = left.
            forward = rbase[0]
            lateral = rbase[1] * 0.7
            # Left stick: same remap, physical left/right is lbase[1].
            yaw = lbase[1]

            # Smooth the commanded planar velocity before applying it to both
            # wheel targets and the free root. The teleop action is sampled at
            # control rate, while PhysX advances several smaller sub-steps.
            try:
                smoothing_alpha = float(
                    os.environ.get("ROBOFINALS_GALBOT_WHEEL_SMOOTHING_ALPHA", "0.1")
                )
            except ValueError:
                smoothing_alpha = 0.1
            smoothing_alpha = min(max(smoothing_alpha, 0.0), 1.0)
            requested_command = torch.stack([forward, lateral, yaw]).to(
                device=device.env.device, dtype=torch.float32
            )
            previous_command = getattr(self, "_galbot_mobile_command", None)
            if (
                previous_command is None
                or previous_command.shape != requested_command.shape
                or previous_command.device != requested_command.device
            ):
                previous_command = torch.zeros_like(requested_command)
            smoothed_command = torch.lerp(previous_command, requested_command, smoothing_alpha)
            self._galbot_mobile_command = smoothed_command
            forward, lateral, yaw = smoothed_command.unbind()

            # Omnidirectional wheel velocities stay in body frame.
            wheel = torch.stack([
                forward - lateral - yaw,  # wheel1
                forward + lateral + yaw,  # wheel2
                forward + lateral - yaw,  # wheel3
                forward - lateral + yaw,  # wheel4
            ]).repeat(num_envs, 1)

            # Root velocity is world-frame: rotate body cmd by current root quat
            # (Isaac Lab 3+/Newton stores root_quat_w as xyzw).
            robot = device.env.scene.articulations.get("robot")
            self._write_galbot_base_velocity(robot, forward, lateral, yaw, num_envs, device.env.device)
            if (
                robot is not None
                and os.environ.get("ROBOFINALS_GALBOT_GRIPPER_DEBUG", "").strip().lower()
                in {"1", "true", "yes", "on"}
            ):
                count = getattr(self, "_galbot_gripper_debug_count", 0)
                try:
                    interval = max(1, int(os.environ.get("ROBOFINALS_GALBOT_GRIPPER_DEBUG_INTERVAL", "1")))
                except ValueError:
                    interval = 1
                if count % interval == 0:
                    print(
                        "[GalbotBaseDebug] "
                        f"control_step={count} command_vel="
                        f"[{float(forward):.6f}, {float(lateral):.6f}, {float(yaw):.6f}]",
                        flush=True,
                    )

            result = torch.cat([result, wheel], dim=1)

        return result

    @staticmethod
    def _write_galbot_base_velocity(robot, forward, lateral, yaw_rate, num_envs: int, device) -> None:
        """Write body-frame base command as world-frame root velocity."""
        if robot is None or not hasattr(robot, "write_root_velocity_to_sim"):
            return
        quat_xyzw = sim_quat_raw_to_xyzw_torch(as_torch(robot.data.root_quat_w))
        f = torch.as_tensor(forward, device=device, dtype=torch.float32).expand(num_envs)
        l = torch.as_tensor(lateral, device=device, dtype=torch.float32).expand(num_envs)
        wz = torch.as_tensor(yaw_rate, device=device, dtype=torch.float32).expand(num_envs)
        zeros = torch.zeros_like(f)
        vel_b = torch.stack([f, l, zeros], dim=-1)
        vel_w = quat_apply(quat_xyzw, vel_b)
        root_vel = torch.zeros((num_envs, 6), device=device, dtype=torch.float32)
        root_vel[:, 0:3] = vel_w
        root_vel[:, 5] = wz
        robot.write_root_velocity_to_sim(root_vel)

    def get_default_action(self, device: torch.device, num_envs: int) -> torch.Tensor:
        if self.mobile:
            return torch.zeros(num_envs, 20, device=device)  # 16 (arms+grippers) + 4 (wheels)
        else:
            return torch.zeros(num_envs, 16, device=device)

    def apply_pre_physics_step(self, env):
        """Re-apply the teleop base command before every physics sub-step."""
        if not self.mobile:
            return
        command = getattr(self, "_galbot_mobile_command", None)
        if command is None:
            return
        robot = env.scene.articulations.get("robot")
        self._write_galbot_base_velocity(
            robot, command[0], command[1], command[2], env.num_envs, env.device
        )

    def reset_robot_cfg_state(self):
        """Clear teleoperation state that must not leak across environment resets."""
        self._galbot_mobile_command = None
        self._galbot_gripper_command = None
        self._galbot_gripper_debug_count = 0

    def get_action_cfg(self):
        """Return action configuration for the environment."""
        return self.action_config

    def modify_env_cfg(self, env_cfg):
        env_cfg = super().modify_env_cfg(env_cfg)

        # The task reset restores joint state but, by default, leaves actuator
        # targets untouched. Galbot legs have no teleop action term, so reset
        # targets with the configured stand pose instead of pulling to zero.
        env_cfg.events.reset_all.params = {"reset_joint_targets": True}
        env_cfg.scene.global_light = AssetBaseCfg(
            prim_path="/World/global_light",
            spawn=sim_utils.DomeLightCfg(
                color=(0.75, 0.75, 0.75), intensity=950.0, visible_in_primary_ray=True
            ),
        )
        env_cfg.scene.preview_light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/preview_light",
            spawn=sim_utils.SphereLightCfg(
                radius=0.25, color=(1.0, 1.0, 1.0), intensity=2500.0
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.2, -1.2, 2.4)),
        )
        env_cfg.sim.dt = 1 / 200
        env_cfg.decimation = 4
        env_cfg.viewer.eye = (2.8, -2.8, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.8)
        return env_cfg

    def post_init(self, env):
        """Called after environment initialization to setup collision filtering."""
        if hasattr(super(), 'post_init'):
            super().post_init(env)

        # Setup gripper collision filtering for all environments
        for env_idx in range(env.num_envs):
            env_prim_path = f"/World/envs/env_{env_idx}"
            setup_galbot_gripper_collision_filtering(env_prim_path)


class GalbotAbsEnvCfg(GalbotEnvCfg):
    pass


class GalbotMobileEnvCfg(GalbotEnvCfg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mobile=True, **kwargs)
