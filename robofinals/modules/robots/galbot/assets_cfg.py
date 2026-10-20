"""Isaac Lab articulation configurations for the Galbot One Golf."""

import copy
from robofinals.data import robofinals_DATA_PATH
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


GALBOT_USD_PATH = (
    robofinals_DATA_PATH
    / "assets"
    / "galbot_one_golf_description"
    / "usd"
    / "galbot_one_golf.usda"
)

GALBOT_LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
GALBOT_RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
GALBOT_LEG_JOINTS = [f"leg_joint.*"]
GALBOT_HEAD_JOINTS = ["head_joint1", "head_joint2"]
GALBOT_LEFT_GRIPPER_JOINTS = ["left_gripper_joint"]
GALBOT_RIGHT_GRIPPER_JOINTS = ["right_gripper_joint"]
# Only the four actuated wheel joints. ``wheel_1_passive_*`` etc. are
# PhysX passive roller joints and must not become teleop action dimensions.
GALBOT_WHEEL_JOINTS = ["wheel1_joint", "wheel2_joint", "wheel3_joint", "wheel4_joint"]

# These are conservative startup gains. Tune them against a real Galbot launch
# rather than copying gains from a kinematically different embodiment.
GALBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(GALBOT_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # Higher than 3 to clear gripper/plate clip faster; 5 can fling thin parts.
            max_depenetration_velocity=4.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # Raise from 24/4: more position iters reduce finger/plate penetration
            # under grasp contact; velocity iters tighten friction hold.
            solver_position_iteration_count=48,
            solver_velocity_iteration_count=8,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Isaac joint targets at spawn. USD authored body Xforms are the q=0 pose;
            # these non-zero values move the robot to the L-shaped teleop stand pose.
            # Keep leg_joint1 below the imported USD's 0.937-rad hard limit.
            "leg_joint1": 0.93,
            "leg_joint2": 2.50,
            "leg_joint3": 1.56,
            "leg_joint4": 0.0,
            "leg_joint5": 0.0,
            "head_joint1": 0.0,
            "head_joint2": 0.35,  # head down
            "left_arm_joint1": 1.134464,
            "left_arm_joint2": -1.308997,
            "left_arm_joint3": -0.52359878,
            "left_arm_joint4": -1.57079633,
            "left_arm_joint5": 0.0,
            "left_arm_joint6": 0.0,
            "left_arm_joint7": 0.0,
            "right_arm_joint1": -1.134464,
            "right_arm_joint2": 1.308997,
            "right_arm_joint3": 0.52359878,
            "right_arm_joint4": 1.57079633,
            "right_arm_joint5": 0.0,
            "right_arm_joint6": 0.0,
            "right_arm_joint7": 0.0,
            "left_gripper_joint": 0.0,
            "right_gripper_joint": 0.0,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=GALBOT_LEG_JOINTS,
            effort_limit_sim=169.0,
            velocity_limit_sim=1.0,
            stiffness=200.0,
            damping=80.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=GALBOT_HEAD_JOINTS,
            effort_limit_sim=4.0,
            velocity_limit_sim=1.5,
            stiffness=80.0,
            damping=10.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=GALBOT_LEFT_ARM_JOINTS + GALBOT_RIGHT_ARM_JOINTS,
            # Hold against grasp wrenches (wrist twist); damping first, then stiffness.
            effort_limit_sim=1200.0,
            velocity_limit_sim=1.5,
            stiffness=1400.0,
            damping=320.0,
        ),
        # Driven revolute + USD mimic fingers, tuned like NVIDIA Robotiq 2F-85 guide:
        # https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/guides/gripper_tuning.html
        # Assumptions (teleop chamber-scale): m=1.5 kg, mu=0.4, finger lever r=0.05 m,
        # 1 drive + 5 mimic → tau_j0 = 6*(mg/mu/2)*r ≈ 6 Nm.
        # stiffness = tau_j0 / 5deg; damping for dr=1 with armature 5e-3; vmax ≈ 1.703/0.28.
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=GALBOT_LEFT_GRIPPER_JOINTS + GALBOT_RIGHT_GRIPPER_JOINTS,
            effort_limit_sim=6.0,
            velocity_limit_sim=6.1,
            stiffness=69.0,
            damping=1.17,
            armature=0.005,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Mobile variant: the same USD and upper-body setup, with a free root and
# velocity-controlled omni wheels. The fixed-base GALBOT_CFG remains the
# default used by Galbot-Abs.
GALBOT_MOBILE_CFG = copy.deepcopy(GALBOT_CFG)
GALBOT_MOBILE_CFG.spawn.articulation_props.fix_root_link = False
GALBOT_MOBILE_CFG.spawn.rigid_props.disable_gravity = False
GALBOT_MOBILE_CFG.spawn.rigid_props.linear_damping = 0.5
GALBOT_MOBILE_CFG.spawn.rigid_props.angular_damping = 8.0
GALBOT_MOBILE_CFG.init_state.pos = (0.0, 0.0, 0.03)
# Keep the mobile legs at the authored USD lock strength. The legs are not
# exposed as teleoperation or policy action joints, so their default targets
# should hold the stand pose against arm-contact reaction torques.
GALBOT_MOBILE_CFG.actuators["legs"].stiffness = 100000.0
GALBOT_MOBILE_CFG.actuators["legs"].damping = 10000.0
GALBOT_MOBILE_CFG.actuators["legs"].effort_limit_sim = 1000.0
GALBOT_MOBILE_CFG.actuators["wheels"] = ImplicitActuatorCfg(
    joint_names_expr=GALBOT_WHEEL_JOINTS,
    effort_limit_sim=40.0,
    velocity_limit_sim=1.0,
    stiffness=50000.0,
    damping=1000.0,
)

OFFSET_CONFIG = {
    # From USD Physics joint FK at GALBOT_CFG.init_state.joint_pos.
    # Note: authored USD body Xforms are the q=0 pose, NOT this init pose.
    "left_offset": np.array([0.349326, 0.299017, 0.988982]),
    "right_offset": np.array([0.349327, -0.299017, 0.988982]),
    # link7 orientation @ init pose relative to base_link (translation in *_offset).
    "left2arm_transform": np.array([
        [-0.997015, 0.019980, -0.074566, 0.0],
        [0.000000, 0.965925, 0.258819, 0.0],
        [0.077197, 0.258046, -0.963043, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]),
    "right2arm_transform": np.array([
        [-0.997015, -0.019980, -0.074566, 0.0],
        [0.000000, 0.965925, -0.258819, 0.0],
        [0.077197, -0.258046, -0.963043, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]),
    "left2finger_transform": np.eye(4),
    "right2finger_transform": np.eye(4),
    "robot_arm_length": 0.7,
    "arm_input_scale": [1, 1, 1],
}
