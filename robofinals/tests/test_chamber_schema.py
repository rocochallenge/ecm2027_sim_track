import unittest
from types import SimpleNamespace
import torch
from robofinals.distributed.chamber_schema import JOINT_NAMES, OBSERVATION_SCHEMA
from robofinals.distributed.grpc_policy import _observation
from robofinals.modules.robots.galbot.galbot import galbot_gripper_positions

class SchemaTests(unittest.TestCase):
    def test_fixed_fields_required(self):
        policy={name:torch.zeros((1,*shape),dtype=torch.uint8 if dtype=="uint8" else torch.float32)
                for name,dtype,shape in OBSERVATION_SCHEMA}
        self.assertEqual(len(_observation({"policy":policy})),1)
        with self.assertRaisesRegex(ValueError,"batch size"):
            _observation({"policy":policy}, expected_count=2)
        del policy["right_hand_camera_rgb"]
        with self.assertRaisesRegex(ValueError,"required observation missing"):
            _observation({"policy":policy})
        policy["right_hand_camera_rgb"]=torch.zeros((1,10,10,3),dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError,"invalid observation schema"):
            _observation({"policy":policy})

    def test_grippers_are_left_then_right_driven_joints(self):
        joints=torch.arange(len(JOINT_NAMES),dtype=torch.float32).reshape(1,-1)
        robot=SimpleNamespace(data=SimpleNamespace(joint_names=list(JOINT_NAMES),joint_pos=joints))
        env=SimpleNamespace(scene={"robot":robot})
        actual=galbot_gripper_positions(env)
        self.assertEqual(actual.tolist(),[[65.0,69.0]])

if __name__=="__main__": unittest.main()
