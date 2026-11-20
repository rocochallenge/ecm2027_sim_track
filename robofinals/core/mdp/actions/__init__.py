from .dex_retartgeting_action import DexRetargetingAction, DexRetargetingActionCfg  # noqa: F401, F403
from .wuji_retartgeting_action import WujiRetargetingAction, WujiRetargetingActionCfg  # noqa: F401, F403
from .leg_position_action import LegPositionAction, LegPositionActionCfg
from .joint_position_map_action import JointPositionMapAction, JointPositionMapActionCfg
from .rel_joint_action import RelJointPositionAction, RelJointPositionActionCfg
from .g1_action import G1Action, G1ActionCfg
from .decoupled_wbc_action import G1DecoupledWBCAction, G1DecoupledWBCActionCfg
from .ur5_newton_action import UR5NewtonAction, UR5NewtonActionCfg
from .ur5_gripper_debug_action import UR5GripperDebugAction, UR5GripperDebugActionCfg
from .threshold_binary_joint_action import ThresholdBinaryJointPositionAction, ThresholdBinaryJointPositionActionCfg
from .quantized_interpolated_joint_action import (
    QuantizedInterpolatedJointPositionAction,
    QuantizedInterpolatedJointPositionActionCfg,
)
from .x7s_action import WrappedX7SPinocchioIKAction, WrappedX7SPinocchioIKCfg
from .galbot_base_velocity_action import GalbotBaseVelocityAction, GalbotBaseVelocityActionCfg

__all__ = [
    "DexRetargetingAction",
    "DexRetargetingActionCfg",
    "WujiRetargetingAction",
    "WujiRetargetingActionCfg",
    "LegPositionAction",
    "LegPositionActionCfg",
    "JointPositionMapAction",
    "JointPositionMapActionCfg",
    "RelJointPositionAction",
    "RelJointPositionActionCfg",
    "G1Action",
    "G1ActionCfg",
    "G1DecoupledWBCAction",
    "G1DecoupledWBCActionCfg",
    "UR5NewtonAction",
    "UR5NewtonActionCfg",
    "UR5GripperDebugAction",
    "UR5GripperDebugActionCfg",
    "ThresholdBinaryJointPositionAction",
    "ThresholdBinaryJointPositionActionCfg",
    "QuantizedInterpolatedJointPositionAction",
    "QuantizedInterpolatedJointPositionActionCfg",
    "PinkAction",
    "PinkActionCfg",
    "WrappedX7SPinocchioIKAction",
    "WrappedX7SPinocchioIKCfg",
    "GalbotBaseVelocityAction",
    "GalbotBaseVelocityActionCfg",
]


def __getattr__(name: str):
    if name in {"WujiRetargetingTactileAction", "WujiRetargetingTactileActionCfg"}:
        from .wuji_retartgeting_action_tactile import (
            WujiRetargetingTactileAction,
            WujiRetargetingTactileActionCfg,
        )

        return {
            "WujiRetargetingTactileAction": WujiRetargetingTactileAction,
            "WujiRetargetingTactileActionCfg": WujiRetargetingTactileActionCfg,
        }[name]
    if name in {"G1TactileAction", "G1TactileActionCfg"}:
        from .g1_action_tactile import G1TactileAction, G1TactileActionCfg

        return {
            "G1TactileAction": G1TactileAction,
            "G1TactileActionCfg": G1TactileActionCfg,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .pink_action import PinkAction, PinkActionCfg
