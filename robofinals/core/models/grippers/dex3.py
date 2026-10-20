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

import math
import numpy as np

import robofinals.core.mdp as mdp

from .base_gripper import BaseGripperCfg


class Dex3GripperCfg(BaseGripperCfg):
    def __init__(self, left_retageting_file_name: str, right_retageting_file_name: str):
        super().__init__(left_retageting_file_name, right_retageting_file_name)
        self.left_contact_body_name = "left_hand_thumb_2_link"
        self.right_contact_body_name = "right_hand_thumb_2_link"

    def left_hand_action_cfg(self):
        return {
            "tracking": mdp.DexRetargetingActionCfg(
                asset_name="robot",
                config_name=self.left_retageting_file_name,
                retargeting_index=[0, 2, 4, 1, 3, 5, 6],  # [4,5,6],# in0,in1,mi0,mi1, th0,th1,th2 ==> in0,mi0,th0,th1,mi1,in1,th2
                joint_names=["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                             "left_hand_index_0_joint", "left_hand_index_1_joint",
                             "left_hand_middle_0_joint", "left_hand_middle_1_joint"],
                # joint_names=["left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint"]
                post_process_fn=self.post_process_left
            ),
            "handle": mdp.JointPositionMapActionCfg(
                asset_name="robot",
                joint_names=['left_hand_thumb_1_joint', 'left_hand_thumb_2_joint', 'left_hand_index.*', 'left_hand_middle.*'],
                post_process_fn=self.process_hand,
            ),
            "rl": mdp.BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["left_hand_.*"],
                open_command_expr={'left_hand_index.*': 0.0, 'left_hand_middle.*': 0.0, 'left_hand_thumb.*': 0.0},
                close_command_expr={'left_hand_thumb_0_joint': 0.0, 'left_hand_thumb_1_joint': -np.pi / 6,
                                    'left_hand_thumb_2_joint': -np.pi / 6, 'left_hand_index_1_joint': np.pi / 3, 'left_hand_middle_1_joint': np.pi / 3,
                                    'left_hand_index_0_joint': np.pi / 6, 'left_hand_middle_0_joint': np.pi / 6},
            )
        }

    def right_hand_action_cfg(self):
        return {
            "tracking": mdp.DexRetargetingActionCfg(
                asset_name="robot",
                config_name=self.right_retageting_file_name,
                retargeting_index=[0, 2, 4, 1, 3, 5, 6],  # [4,5,6],#in0,in1,mi0,mi1, th0,th1,th2 ==> in0,mi0,th0,th1,mi1,in1,th2
                joint_names=["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                             "right_hand_index_0_joint", "right_hand_index_1_joint",
                             "right_hand_middle_0_joint", "right_hand_middle_1_joint"],
                # joint_names=["right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint"]
                post_process_fn=self.post_process_right
            ),
            "handle": mdp.JointPositionMapActionCfg(
                asset_name="robot",
                joint_names=['right_hand_thumb_1_joint', 'right_hand_thumb_2_joint', 'right_hand_index.*', 'right_hand_middle.*'],
                post_process_fn=self.process_hand,
            ),
            "rl": mdp.BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["right_hand_.*"],
                open_command_expr={'right_hand_index.*': 0.0, 'right_hand_middle.*': 0.0, 'right_hand_thumb.*': 0.0},
                close_command_expr={'right_hand_thumb_0_joint': 0.0, 'right_hand_thumb_1_joint': -np.pi / 6,
                                    'right_hand_thumb_2_joint': -np.pi / 6, 'right_hand_index_1_joint': np.pi / 3, 'right_hand_middle_1_joint': np.pi / 3,
                                    'right_hand_index_0_joint': np.pi / 6, 'right_hand_middle_0_joint': np.pi / 6},
            )
        }

    def post_process_left(self, retargeted_actions, num_joints):
        scale = 1.3
        actions = np.zeros_like(retargeted_actions, dtype=np.float32)
        actions[:, 0] = (retargeted_actions[:, 0]) * np.pi / 2 * scale  # in 0
        actions[:, 1] = (retargeted_actions[:, 1]) * np.pi / 2 * scale  # mi 0
        actions[:, 2] = 0  # th 0
        actions[:, 3] = retargeted_actions[:, 0] * scale  # in 1
        actions[:, 4] = retargeted_actions[:, 1] * scale  # mi 1
        actions[:, 5] = np.pi / 3 + retargeted_actions[:, 2] * scale  # th 1
        actions[:, 6] = np.pi / 3 + retargeted_actions[:, 2] * scale  # th 2
        return actions

    def post_process_right(self, retargeted_actions, num_joints):
        scale = 1.3
        actions = np.zeros_like(retargeted_actions, dtype=np.float32)
        actions[:, 0] = (retargeted_actions[:, 0]) * np.pi / 2 * scale  # in 0
        actions[:, 1] = (retargeted_actions[:, 1]) * np.pi / 2 * scale  # mi 0
        actions[:, 2] = 0  # th 0
        actions[:, 3] = retargeted_actions[:, 0] * scale  # in 1
        actions[:, 4] = retargeted_actions[:, 1] * scale  # mi 1
        actions[:, 5] = -np.pi / 3 - retargeted_actions[:, 2] * scale  # th 1
        actions[:, 6] = -np.pi / 3 - retargeted_actions[:, 2] * scale  # th 2
        return actions

    def process_hand(self, target_pos, joint_names):
        if target_pos.shape[-1] == 1:
            target_pos = target_pos.repeat(1, len(joint_names))
        if joint_names[0].startswith('left'):
            target_pos = (target_pos + 1) / 2
        else:
            target_pos = -(target_pos + 1) / 2
        target_pos[:, 0:4] *= -1.4  # index 0 middle 0 index 1 middlrbe 1
        target_pos[:, 4] *= np.pi / 6  # thumb 1
        target_pos[:, 5] *= 1.3  # thumb 2
        return target_pos


class ManusDex3GripperCfg(Dex3GripperCfg):
    """Dedicated Dex3 tracking mapping for Manus hybrid teleop.

    This mapping is intentionally hard-coded (no env switches) to keep behavior stable:
    - explicit invert for Manus open/close direction
    - per-finger gains
    - per-joint clipping to avoid saturating joint limits
    """

    # Per-finger gains for the 3 retarget channels: [index, middle, thumb]
    _GAIN = np.array([1.45, 1.35, 0.70], dtype=np.float32)
    _THUMB_BIAS = np.pi / 10

    # Joint-wise clipping in post-process output order:
    # [in0, mi0, th0, in1, mi1, th1, th2]
    _LEFT_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.10, -0.10], dtype=np.float32)
    _LEFT_HIGH = np.array([2.40, 2.30, 0.0, 2.10, 2.00, 1.20, 1.20], dtype=np.float32)
    _RIGHT_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.20, -1.20], dtype=np.float32)
    _RIGHT_HIGH = np.array([2.40, 2.30, 0.0, 2.10, 2.00, 0.10, 0.10], dtype=np.float32)

    def _manus_src(self, retargeted_actions):
        # Manus close/open direction is opposite to Dex3 default tracking, invert explicitly.
        return 1.0 - retargeted_actions

    def post_process_left(self, retargeted_actions, num_joints):
        src = self._manus_src(retargeted_actions)
        index = src[:, 0] * self._GAIN[0]
        middle = src[:, 1] * self._GAIN[1]
        thumb = src[:, 2] * self._GAIN[2]

        actions = np.zeros_like(retargeted_actions, dtype=np.float32)
        actions[:, 0] = index * (np.pi / 2)  # in 0
        actions[:, 1] = middle * (np.pi / 2)  # mi 0
        actions[:, 2] = 0.0  # th 0
        actions[:, 3] = index  # in 1
        actions[:, 4] = middle  # mi 1
        actions[:, 5] = self._THUMB_BIAS + thumb  # th 1
        actions[:, 6] = self._THUMB_BIAS + thumb  # th 2
        return np.clip(actions, self._LEFT_LOW, self._LEFT_HIGH)

    def post_process_right(self, retargeted_actions, num_joints):
        src = self._manus_src(retargeted_actions)
        index = src[:, 0] * self._GAIN[0]
        middle = src[:, 1] * self._GAIN[1]
        thumb = src[:, 2] * self._GAIN[2]

        actions = np.zeros_like(retargeted_actions, dtype=np.float32)
        actions[:, 0] = index * (np.pi / 2)  # in 0
        actions[:, 1] = middle * (np.pi / 2)  # mi 0
        actions[:, 2] = 0.0  # th 0
        actions[:, 3] = index  # in 1
        actions[:, 4] = middle  # mi 1
        actions[:, 5] = -self._THUMB_BIAS - thumb  # th 1
        actions[:, 6] = -self._THUMB_BIAS - thumb  # th 2
        return np.clip(actions, self._RIGHT_LOW, self._RIGHT_HIGH)


class ManusWujiGripperCfg(BaseGripperCfg):

    # Per-finger gains for the 3 retarget channels: [index, middle, thumb]
    _GAIN = np.array([1.45, 1.35, 0.70], dtype=np.float32)
    _THUMB_BIAS = np.pi / 10

    # Per-joint bias/gain calibration vector: [5 fingers * 4 joints] -> 20 scalars.
    # Example: _LEFT_GAIN_VEC[0] maps to left_finger1_joint1, _LEFT_GAIN_VEC[1] maps to left_finger1_joint2, and so on.
    _LEFT_GAIN_VEC = np.ones(20, dtype=np.float32)
    _LEFT_BIAS_VEC = np.zeros(20, dtype=np.float32)
    _RIGHT_GAIN_VEC = np.ones(20, dtype=np.float32)
    _RIGHT_BIAS_VEC = np.zeros(20, dtype=np.float32)

    _LEFT_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.10, -0.10], dtype=np.float32)
    _LEFT_HIGH = np.array([2.40, 2.30, 0.0, 2.10, 2.00, 1.20, 1.20], dtype=np.float32)

    _RIGHT_LOW = np.array([2.11, -9.03, -26.57, -27.67,
                           -9.23, -23.17, -27.01, -26.61,
                           -9.85, -23.00, -26.54, -26.91,
                           -9.17, -23.69, -27.4, -27.65,
                           -9.59, -24.08, -27.52, -26.96], dtype=np.float32)
    _RIGHT_HIGH = np.array([92.39, 53.35, 89.42, 88.53,
                            89.31, 17.5, 88.83, 90.26,
                            88.79, 17.17, 89.46, 89.97,
                            89.0, 18.11, 88.51, 89.09,
                            89.03, 16.79, 88.35, 90.01], dtype=np.float32)

    def __init__(self, left_retageting_file_name: str, right_retageting_file_name: str):
        super().__init__(left_retageting_file_name, right_retageting_file_name)
        self.left_contact_body_name = "wujihand_left_filtered/left_finger1_tip_link"
        self.right_contact_body_name = "wujihand_right_filtered/right_finger1_tip_link"

    def _manus_src(self, retargeted_actions):
        # Manus close/open direction is opposite to Dex3 default tracking, invert explicitly.
        return 1.0 - retargeted_actions

    @staticmethod
    def _position_passthrough(retargeted_actions, num_joints):
        """Flatten (1, 5, 4) -> (1, 20). Used when retargeting is done in preprocess."""
        return retargeted_actions.reshape(1, -1)

    @staticmethod
    def _linear_delta_passthrough(delta_actions, gain_vec, bias_vec):
        """Map calibrated Manus deltas to WuJi finger deltas without legacy absolute offsets."""
        arr = np.asarray(delta_actions, dtype=np.float32).reshape(-1, 5, 4)
        mapped = arr[:, :, [1, 0, 2, 3]].reshape(arr.shape[0], 20)
        gain = np.asarray(gain_vec, dtype=np.float32).reshape(1, 20)
        bias = np.asarray(bias_vec, dtype=np.float32).reshape(1, 20)
        return mapped * gain + bias

    def post_process_left_delta(self, delta_actions, num_joints):
        return self._linear_delta_passthrough(delta_actions, self._LEFT_GAIN_VEC, self._LEFT_BIAS_VEC)

    def post_process_right_delta(self, delta_actions, num_joints):
        return self._linear_delta_passthrough(delta_actions, self._RIGHT_GAIN_VEC, self._RIGHT_BIAS_VEC)

    def _wuji_tactile_action_cfg(
        self,
        *,
        post_process_fn,
        joint_names: list[str],
        delta_post_process_fn=None,
    ):
        from robofinals.core.mdp.actions.wuji_retartgeting_action_tactile import (
            WujiRetargetingTactileActionCfg,
        )

        kwargs = {
            "asset_name": "robot",
            "post_process_fn": post_process_fn,
            "joint_names": joint_names,
        }
        if delta_post_process_fn is not None:
            kwargs["delta_post_process_fn"] = delta_post_process_fn
        return WujiRetargetingTactileActionCfg(**kwargs)

    def left_hand_action_cfg(self):
        if bool(getattr(self, "use_tactile_action", False)):
            joint_names = ["left_finger1_joint1", "left_finger1_joint2", "left_finger1_joint3", "left_finger1_joint4",
                           "left_finger2_joint1", "left_finger2_joint2", "left_finger2_joint3", "left_finger2_joint4",
                           "left_finger3_joint1", "left_finger3_joint2", "left_finger3_joint3", "left_finger3_joint4",
                           "left_finger4_joint1", "left_finger4_joint2", "left_finger4_joint3", "left_finger4_joint4",
                           "left_finger5_joint1", "left_finger5_joint2", "left_finger5_joint3", "left_finger5_joint4"]
            return {
                "tracking": self._wuji_tactile_action_cfg(
                    post_process_fn=self.post_process_left,
                    delta_post_process_fn=self.post_process_left_delta,
                    joint_names=joint_names,
                ),
                "position": self._wuji_tactile_action_cfg(
                    post_process_fn=self._position_passthrough,
                    joint_names=joint_names,
                ),
            }

        return {
            "tracking": mdp.WujiRetargetingActionCfg(
                asset_name="robot",
                post_process_fn=self.post_process_left,
                joint_names=["left_finger1_joint1", "left_finger1_joint2", "left_finger1_joint3", "left_finger1_joint4",
                             "left_finger2_joint1", "left_finger2_joint2", "left_finger2_joint3", "left_finger2_joint4",
                             "left_finger3_joint1", "left_finger3_joint2", "left_finger3_joint3", "left_finger3_joint4",
                             "left_finger4_joint1", "left_finger4_joint2", "left_finger4_joint3", "left_finger4_joint4",
                             "left_finger5_joint1", "left_finger5_joint2", "left_finger5_joint3", "left_finger5_joint4"]
            ),
            "position": mdp.WujiRetargetingActionCfg(
                asset_name="robot",
                post_process_fn=self._position_passthrough,
                joint_names=["left_finger1_joint1", "left_finger1_joint2", "left_finger1_joint3", "left_finger1_joint4",
                             "left_finger2_joint1", "left_finger2_joint2", "left_finger2_joint3", "left_finger2_joint4",
                             "left_finger3_joint1", "left_finger3_joint2", "left_finger3_joint3", "left_finger3_joint4",
                             "left_finger4_joint1", "left_finger4_joint2", "left_finger4_joint3", "left_finger4_joint4",
                             "left_finger5_joint1", "left_finger5_joint2", "left_finger5_joint3", "left_finger5_joint4"]
            ),
        }

    def right_hand_action_cfg(self):
        if bool(getattr(self, "use_tactile_action", False)):
            joint_names = ["right_finger1_joint1", "right_finger1_joint2", "right_finger1_joint3", "right_finger1_joint4",
                           "right_finger2_joint1", "right_finger2_joint2", "right_finger2_joint3", "right_finger2_joint4",
                           "right_finger3_joint1", "right_finger3_joint2", "right_finger3_joint3", "right_finger3_joint4",
                           "right_finger4_joint1", "right_finger4_joint2", "right_finger4_joint3", "right_finger4_joint4",
                           "right_finger5_joint1", "right_finger5_joint2", "right_finger5_joint3", "right_finger5_joint4"]
            return {
                "tracking": self._wuji_tactile_action_cfg(
                    post_process_fn=self.post_process_right,
                    delta_post_process_fn=self.post_process_right_delta,
                    joint_names=joint_names,
                ),
                "position": self._wuji_tactile_action_cfg(
                    post_process_fn=self._position_passthrough,
                    joint_names=joint_names,
                ),
            }

        return {
            "tracking": mdp.WujiRetargetingActionCfg(
                asset_name="robot",
                post_process_fn=self.post_process_right,
                joint_names=["right_finger1_joint1", "right_finger1_joint2", "right_finger1_joint3", "right_finger1_joint4",
                             "right_finger2_joint1", "right_finger2_joint2", "right_finger2_joint3", "right_finger2_joint4",
                             "right_finger3_joint1", "right_finger3_joint2", "right_finger3_joint3", "right_finger3_joint4",
                             "right_finger4_joint1", "right_finger4_joint2", "right_finger4_joint3", "right_finger4_joint4",
                             "right_finger5_joint1", "right_finger5_joint2", "right_finger5_joint3", "right_finger5_joint4"]
            ),
            "position": mdp.WujiRetargetingActionCfg(
                asset_name="robot",
                post_process_fn=self._position_passthrough,
                joint_names=["right_finger1_joint1", "right_finger1_joint2", "right_finger1_joint3", "right_finger1_joint4",
                             "right_finger2_joint1", "right_finger2_joint2", "right_finger2_joint3", "right_finger2_joint4",
                             "right_finger3_joint1", "right_finger3_joint2", "right_finger3_joint3", "right_finger3_joint4",
                             "right_finger4_joint1", "right_finger4_joint2", "right_finger4_joint3", "right_finger4_joint4",
                             "right_finger5_joint1", "right_finger5_joint2", "right_finger5_joint3", "right_finger5_joint4"]
            ),
        }

    def post_process_left(self, retargeted_actions, num_joints):
        """
        retargeted_actions: (1, 5, 4)
        Flattening order:
        finger1_joint1..4, finger2_joint1..4, ..., finger5_joint1..4 -> 20 dims
        """
        arr = retargeted_actions  # shape: (1, 5, 4)
        result = np.zeros((1, 20), dtype=np.float32)

        # finger1
        result[:, 0] = math.radians(math.degrees(arr[0, 0, 1]) * -4.5 + 137)   # finger1_joint1
        result[:, 1] = math.radians(math.degrees(arr[0, 0, 0]) * -3 + 81)      # finger1_joint2 [30--10]-> [-9-51]
        result[:, 2] = arr[0, 0, 2]  # finger1_joint3
        result[:, 3] = arr[0, 0, 3]  # finger1_joint4
        # finger2
        result[:, 4] = arr[0, 1, 1]  # finger2_joint1
        result[:, 5] = arr[0, 1, 0]  # finger2_joint2
        result[:, 6] = arr[0, 1, 2]  # finger2_joint3
        result[:, 7] = arr[0, 1, 3]  # finger2_joint4
        # finger3
        result[:, 8] = arr[0, 2, 1]  # finger3_joint1
        result[:, 9] = arr[0, 2, 0]  # finger3_joint2
        result[:, 10] = arr[0, 2, 2]  # finger3_joint3
        result[:, 11] = arr[0, 2, 3]  # finger3_joint4
        # finger4
        result[:, 12] = arr[0, 3, 1]  # finger4_joint1
        result[:, 13] = arr[0, 3, 0]  # finger4_joint2
        result[:, 14] = arr[0, 3, 2]  # finger4_joint3
        result[:, 15] = arr[0, 3, 3]  # finger4_joint4
        # finger5
        result[:, 16] = arr[0, 4, 1]  # finger5_joint1
        result[:, 17] = arr[0, 4, 0]  # finger5_joint2
        result[:, 18] = arr[0, 4, 2]  # finger5_joint3
        result[:, 19] = arr[0, 4, 3]  # finger5_joint4

        # Per-joint calibration can be applied above as *gain + bias, for example:
        # result[:, 0] = arr[0, 0, 0] * g0 + b0
        print("left_result", result)
        return result

    def post_process_right(self, retargeted_actions, num_joints):
        """
        Same flattening order and mapping as the left hand.
        retargeted_actions: (1, 5, 4)
        """
        arr = retargeted_actions  # shape: (1, 5, 4)
        result = np.zeros((1, 20), dtype=np.float32)

        # finger1
        result[:, 0] = math.radians(math.degrees(arr[0, 0, 1]) * -4.5 + 137)  # finger1_joint1 [30--10] -> [2-92]
        result[:, 1] = math.radians(math.degrees(arr[0, 0, 0]) * -3 + 81)  # finger1_joint2 [30--10]-> [-9-51]
        result[:, 2] = arr[0, 0, 2]  # finger1_joint3
        result[:, 3] = arr[0, 0, 3]  # finger1_joint4
        # finger2
        result[:, 4] = arr[0, 1, 1]  # finger2_joint1
        result[:, 5] = arr[0, 1, 0]  # finger2_joint2
        result[:, 6] = arr[0, 1, 2]  # finger2_joint3
        result[:, 7] = arr[0, 1, 3]  # finger2_joint4
        # finger3
        result[:, 8] = arr[0, 2, 1]  # finger3_joint1
        result[:, 9] = arr[0, 2, 0]  # finger3_joint2
        result[:, 10] = arr[0, 2, 2]  # finger3_joint3
        result[:, 11] = arr[0, 2, 3]  # finger3_joint4
        # finger4
        result[:, 12] = arr[0, 3, 1]  # finger4_joint1
        result[:, 13] = arr[0, 3, 0]  # finger4_joint2
        result[:, 14] = arr[0, 3, 2]  # finger4_joint3
        result[:, 15] = arr[0, 3, 3]  # finger4_joint4
        # finger5
        result[:, 16] = arr[0, 4, 1]  # finger5_joint1
        result[:, 17] = arr[0, 4, 0]  # finger5_joint2
        result[:, 18] = arr[0, 4, 2]  # finger5_joint3
        result[:, 19] = arr[0, 4, 3]  # finger5_joint4
        print("right_result", result)
        return result

    def process_hand(self, target_pos, joint_names):
        if target_pos.shape[-1] == 1:
            target_pos = target_pos.repeat(1, len(joint_names))
        if joint_names[0].startswith('left'):
            target_pos = (target_pos + 1) / 2
        else:
            target_pos = -(target_pos + 1) / 2
        target_pos[:, 0:4] *= -1.4  # index 0 middle 0 index 1 middlrbe 1
        target_pos[:, 4] *= np.pi / 6  # thumb 1
        target_pos[:, 5] *= 1.3  # thumb 2
        return target_pos
