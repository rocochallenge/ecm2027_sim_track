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

from robofinals.core.checks.factory_obj_drop_checker import FactoryObjDropChecker
from robofinals.core.checks.gripper_blacklist_collision_checker import GripperBlacklistCollisionChecker
from robofinals.core.checks.path_efficiency_checker import PathEfficiencyChecker
from robofinals.core.checks.trajectory_smoothness_checker import TrajectorySmoothnessChecker
from robofinals.core.checks.action_state_inconsistency_checker import ActionStateInconsistencyChecker
from robofinals.core.checks.actuator_velocity_jump_checker import VelocityJumpChecker
from robofinals.core.checks.arm_joint_pos_checker import ArmJointAngleChecker
from robofinals.core.checks.clipping_checker import ClippingChecker
from robofinals.core.checks.gripper_collision_checker import GripperCollisionChecker
from robofinals.core.checks.kitchen_coffee_collision_checker import KitchenCoffeeCollisionChecker
from robofinals.core.checks.motion_checker import MotionChecker
from robofinals.core.checks.obj_drop_checker import ObjDropChecker
from robofinals.core.checks.start_object_move_checker import StartObjectMoveChecker

CHECKER_REGISTRY = {
    "motion": MotionChecker,
    "kitchen_coffee_collision": KitchenCoffeeCollisionChecker,
    "gripper_collision": GripperCollisionChecker,
    "gripper_blacklist_collision": GripperBlacklistCollisionChecker,
    "clipping": ClippingChecker,
    "velocity_jump": VelocityJumpChecker,
    # "start_object_move": StartObjectMoveChecker,
    # "obj_drop": ObjDropChecker,
    # "arm_joint_angle": ArmJointAngleChecker,
    # "action_state_inconsistency": ActionStateInconsistencyChecker,
    # New metrics for GTC demo (adapted from RoboEval)
    # "trajectory_smoothness": TrajectorySmoothnessChecker,
    # "path_efficiency": PathEfficiencyChecker,
    # "factory_obj_drop": FactoryObjDropChecker,
}


def get_checker(checker_type):
    if checker_type not in CHECKER_REGISTRY:
        raise ValueError(f"Checker type {checker_type} not found")
    return CHECKER_REGISTRY[checker_type]


def get_checkers_from_cfg(checkers_cfg):
    checkers = []
    for checker_type in checkers_cfg.keys():
        checker_cfg = checkers_cfg[checker_type]
        checker_cls = get_checker(checker_type)
        # Forward all config keys except warning_on_screen as kwargs to the checker constructor,
        # allowing TrajectorySmoothnessChecker / PathEfficiencyChecker to receive their own params.
        kwargs = {k: v for k, v in checker_cfg.items() if k != "warning_on_screen"}
        checkers.append(checker_cls(warning_on_screen=checker_cfg.get("warning_on_screen", False), **kwargs))
    return checkers


def form_checker_result(checkers_cfg):
    checkers_results = {}
    for check_type in checkers_cfg.keys():
        checkers_results[check_type] = {}
    return checkers_results
