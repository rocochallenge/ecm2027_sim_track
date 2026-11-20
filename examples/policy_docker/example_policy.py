"""Minimal policy container entrypoint; replace ExamplePolicy with your own policy."""
import os

import grpc
import numpy as np

from robofinals.distributed import policy_v1_pb2 as pb
from robofinals.distributed.grpc_policy import PolicyClient


FIELD_SHAPES = {
    "first_person_camera_rgb": (np.uint8, (480, 640, 3)),
    "left_hand_camera_rgb": (np.uint8, (224, 224, 3)),
    "right_hand_camera_rgb": (np.uint8, (224, 224, 3)),
    "actions": (np.dtype("<f4"), (19,)),
    "joint_pos": (np.dtype("<f4"), (77,)),
    "joint_vel": (np.dtype("<f4"), (77,)),
    "eef_pos": (np.dtype("<f4"), (2, 3)),
    "eef_quat": (np.dtype("<f4"), (2, 4)),
    "gripper_pos": (np.dtype("<f4"), (2,)),
}


def decode(observation):
    return {name: np.frombuffer(getattr(observation, name), dtype=dtype).reshape(shape).copy()
            for name, (dtype, shape) in FIELD_SHAPES.items()}


class ExamplePolicy:
    def reset(self, env_index):
        """Reset state for one completed environment."""

    def act(self, observations):
        # Demonstration only: replace with model inference.
        # Nonzero quaternion components make each arm target a valid rotation.
        actions = np.zeros((len(observations), 19), dtype=np.float32)
        actions[:, 6] = 1.0
        actions[:, 13] = 1.0
        return actions


def main():
    endpoint = os.environ["ECM27_EVAL_ENDPOINT"]
    channel = grpc.insecure_channel(endpoint, options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)])
    grpc.channel_ready_future(channel).result(timeout=300 * 60)
    client = PolicyClient(channel)
    policy = ExamplePolicy()
    transition = client.get_observation(pb.Empty(), timeout=300 * 60)
    while True:
        for i, (terminated, truncated) in enumerate(zip(transition.terminated, transition.truncated)):
            if terminated or truncated:
                policy.reset(i)
        observations = [decode(obs) for obs in transition.observations]
        action_rows = policy.act(observations)
        request = pb.ActionBatch(actions=[pb.Action(values=row) for row in action_rows])
        try:
            transition = client.step(request, timeout=300 * 60)
        except grpc.RpcError as error:
            if error.code() != grpc.StatusCode.FAILED_PRECONDITION:
                raise
            transition = client.get_observation(pb.Empty(), timeout=300 * 60)


if __name__ == "__main__":
    main()
