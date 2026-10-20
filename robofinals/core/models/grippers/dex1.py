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

import numpy as np

import robofinals.core.mdp as mdp

from .base_gripper import BaseGripperCfg


class Dex1GripperCfg(BaseGripperCfg):
    """Two-finger gripper mapping for Dex3-style controller teleop.

    Command convention:
    - -1.0: released / fully open
    - +1.0: pressed / closed
    """

    OPEN_POS = 0.0245
    CLOSE_POS = -0.02

    def __init__(self, left_retageting_file_name: str = "", right_retageting_file_name: str = ""):
        super().__init__(left_retageting_file_name, right_retageting_file_name)
        self.left_contact_body_name = "left_dex1_finger_link_1"
        self.right_contact_body_name = "right_dex1_finger_link_1"

    def left_hand_action_cfg(self):
        return {
            "handle": mdp.JointPositionMapActionCfg(
                asset_name="robot",
                joint_names=["left_dex1_finger_joint_1", "left_dex1_finger_joint_2"],
                scale=1.0,
                offset=0.0,
                use_default_offset=False,
                preserve_order=True,
                post_process_fn=self.process_hand,
            )
        }

    def right_hand_action_cfg(self):
        return {
            "handle": mdp.JointPositionMapActionCfg(
                asset_name="robot",
                joint_names=["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"],
                scale=1.0,
                offset=0.0,
                use_default_offset=False,
                preserve_order=True,
                post_process_fn=self.process_hand,
            )
        }

    def process_hand(self, target_pos, joint_names):
        if target_pos.shape[-1] == 1:
            target_pos = target_pos.repeat(1, len(joint_names))
        target_pos = (target_pos + 1.0) * 0.5
        if hasattr(target_pos, "clamp"):
            target_pos = target_pos.clamp(0.0, 1.0)
        else:
            target_pos = np.clip(target_pos, 0.0, 1.0)
        return self.OPEN_POS + target_pos * (self.CLOSE_POS - self.OPEN_POS)
