"""Fixed observation schema for the Chamber Service policy protocol."""

JOINT_NAMES = ('leg_joint1', 'wheel1_joint', 'wheel2_joint', 'wheel3_joint', 'wheel4_joint', 'leg_joint2', 'wheel_1_passive_0_joint', 'wheel_1_passive_1_joint', 'wheel_1_passive_2_joint', 'wheel_1_passive_3_joint', 'wheel_1_passive_4_joint', 'wheel_1_passive_5_joint', 'wheel_1_passive_6_joint', 'wheel_1_passive_7_joint', 'wheel_1_passive_8_joint', 'wheel_1_passive_9_joint', 'wheel_2_passive_0_joint', 'wheel_2_passive_1_joint', 'wheel_2_passive_2_joint', 'wheel_2_passive_3_joint', 'wheel_2_passive_4_joint', 'wheel_2_passive_5_joint', 'wheel_2_passive_6_joint', 'wheel_2_passive_7_joint', 'wheel_2_passive_8_joint', 'wheel_2_passive_9_joint', 'wheel_3_passive_0_joint', 'wheel_3_passive_1_joint', 'wheel_3_passive_2_joint', 'wheel_3_passive_3_joint', 'wheel_3_passive_4_joint', 'wheel_3_passive_5_joint', 'wheel_3_passive_6_joint', 'wheel_3_passive_7_joint', 'wheel_3_passive_8_joint', 'wheel_3_passive_9_joint', 'wheel_4_passive_0_joint', 'wheel_4_passive_1_joint', 'wheel_4_passive_2_joint', 'wheel_4_passive_3_joint', 'wheel_4_passive_4_joint', 'wheel_4_passive_5_joint', 'wheel_4_passive_6_joint', 'wheel_4_passive_7_joint', 'wheel_4_passive_8_joint', 'wheel_4_passive_9_joint', 'leg_joint3', 'leg_joint4', 'leg_joint5', 'head_joint1', 'left_arm_joint1', 'right_arm_joint1', 'head_joint2', 'left_arm_joint2', 'right_arm_joint2', 'left_arm_joint3', 'right_arm_joint3', 'left_arm_joint4', 'right_arm_joint4', 'left_arm_joint5', 'right_arm_joint5', 'left_arm_joint6', 'right_arm_joint6', 'left_arm_joint7', 'right_arm_joint7', 'left_gripper_joint', 'left_gripper_l_inner_knuckle_joint', 'left_gripper_l_knuckle_joint', 'left_gripper_r_inner_knuckle_joint', 'right_gripper_joint', 'right_gripper_l_inner_knuckle_joint', 'right_gripper_l_knuckle_joint', 'right_gripper_r_inner_knuckle_joint', 'left_gripper_l_finger_joint', 'left_gripper_r_finger_joint', 'right_gripper_l_finger_joint', 'right_gripper_r_finger_joint')

JOINT_DEFAULT_POS = (0.9300000071525574, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.559999942779541, 0.0, 0.0, 0.0, 1.1344640254974365, -1.1344640254974365, 0.3499999940395355, -1.3089970350265503, 1.3089970350265503, -0.5235987901687622, 0.5235987901687622, -1.5707963705062866, 1.5707963705062866, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Each tuple is (name, scalar dtype, dimensions after the environment axis).
OBSERVATION_SCHEMA = (
    ("first_person_camera_rgb", "uint8", (480, 640, 3)),
    ("left_hand_camera_rgb", "uint8", (224, 224, 3)),
    ("right_hand_camera_rgb", "uint8", (224, 224, 3)),
    ("actions", "float32", (19,)),
    ("joint_pos", "float32", (77,)),
    ("joint_vel", "float32", (77,)),
    ("eef_pos", "float32", (2, 3)),
    ("eef_quat", "float32", (2, 4)),
    ("gripper_pos", "float32", (2,)),
)
