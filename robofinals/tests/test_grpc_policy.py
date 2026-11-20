import socket
import threading
import unittest
from types import SimpleNamespace
import grpc
import numpy as np
import torch
from robofinals.distributed.grpc_policy import GrpcPolicyServer, PolicyClient
from robofinals.distributed import policy_v1_pb2 as pb
from robofinals.distributed.chamber_schema import JOINT_DEFAULT_POS, JOINT_NAMES, OBSERVATION_SCHEMA

class FakeEnv:
    num_envs = 2
    unwrapped = SimpleNamespace(step_dt=0.02, scene={"robot":SimpleNamespace(data=SimpleNamespace(joint_names=JOINT_NAMES,
        default_joint_pos=torch.tensor(JOINT_DEFAULT_POS).repeat(2,1), default_joint_vel=torch.zeros((2,77))))})
    action_space = SimpleNamespace(shape=(19,))
    def __init__(self): self.actions = []; self.corrupt = False
    def reset(self): return self.obs(), {"private": 123}
    def obs(self):
        policy = {name: torch.zeros((2,*shape), dtype=torch.uint8 if dtype == "uint8" else torch.float32)
                  for name,dtype,shape in OBSERVATION_SCHEMA}
        policy["secret_score"] = torch.ones((2, 1))
        if self.corrupt:
            policy.pop("left_hand_camera_rgb")
        return {"policy": policy, "secret_group": {"score": torch.ones(2)}}
    def step(self, action):
        self.actions.append(action)
        return self.obs(), torch.ones(2), torch.tensor([False, True]), torch.tensor([False, False]), {"score": 99, "final_obs": self.obs()}
    def close(self): pass

class GrpcPolicyTests(unittest.TestCase):
    def test_protobuf_contract_and_validation(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port=sock.getsockname()[1]
        env=FakeEnv()
        server=GrpcPolicyServer(lambda cfg: env, ("127.0.0.1",port))
        thread=threading.Thread(target=server.serve, daemon=True); thread.start()
        channel=grpc.insecure_channel(f"127.0.0.1:{port}", options=[("grpc.max_receive_message_length", 67108864)])
        try:
            grpc.channel_ready_future(channel).result(timeout=10)
            client=PolicyClient(channel)
            initial=client.get_observation(pb.Empty(), timeout=5)
            self.assertEqual(len(initial.observations),2)
            self.assertFalse(hasattr(client,"get_metadata"))
            self.assertEqual([m.name for m in pb.DESCRIPTOR.services_by_name["Policy"].methods],
                             ["GetObservation", "Step"])
            self.assertFalse(hasattr(initial,"episode_id"))
            self.assertEqual([f.name for f in pb.Observation.DESCRIPTOR.fields],
                             [x[0] for x in OBSERVATION_SCHEMA])
            for dim in (19,20):
                action=np.zeros((2,dim),dtype="<f4")
                action[:,6]=2.0; action[:,13]=3.0
                result=client.step(pb.ActionBatch(actions=[pb.Action(values=row) for row in action]), timeout=5)
                self.assertEqual(len(result.observations),2)
                self.assertEqual(len(result.observations[0].joint_pos),77*4)
                self.assertEqual(list(result.terminated), [False,True])
                self.assertEqual(len(result.final_observations),2)
            with self.assertRaises(grpc.RpcError) as err:
                client.step(pb.ActionBatch(actions=[pb.Action(values=[0.0])]), timeout=5)
            self.assertEqual(err.exception.code(),grpc.StatusCode.INVALID_ARGUMENT)
            data=np.full((2,19),np.nan,dtype="<f4").tobytes()
            with self.assertRaises(grpc.RpcError) as err:
                client.step(pb.ActionBatch(actions=[pb.Action(values=row) for row in np.frombuffer(data,dtype="<f4").reshape(2,19)]), timeout=5)
            self.assertEqual(err.exception.code(),grpc.StatusCode.INVALID_ARGUMENT)
            self.assertEqual(len(env.actions),2)
            self.assertTrue(torch.allclose(env.actions[0][:,6],torch.ones(2)))
            self.assertTrue(torch.allclose(env.actions[0][:,13],torch.ones(2)))
            zeros=np.zeros((2,19),dtype="<f4").tobytes()
            with self.assertRaises(grpc.RpcError) as err:
                client.step(pb.ActionBatch(actions=[pb.Action(values=row) for row in np.frombuffer(zeros,dtype="<f4").reshape(2,19)]),timeout=5)
            self.assertEqual(err.exception.code(),grpc.StatusCode.INVALID_ARGUMENT)
            self.assertEqual(len(env.actions),2)
            extreme=np.zeros((2,19),dtype="<f4"); extreme[:,6]=1; extreme[:,13]=1; extreme[0,0]=1e30
            with self.assertRaises(grpc.RpcError) as err:
                client.step(pb.ActionBatch(actions=[pb.Action(values=row) for row in extreme]),timeout=5)
            self.assertEqual(err.exception.code(),grpc.StatusCode.INVALID_ARGUMENT)
            self.assertEqual(len(env.actions),2)
            valid=np.zeros((2,19),dtype="<f4"); valid[:,6]=1; valid[:,13]=1
            client.step(pb.ActionBatch(actions=[pb.Action(values=row) for row in valid]),timeout=5)
            self.assertEqual(len(env.actions),3)
            env.corrupt=True
            valid=np.zeros((2,19),dtype="<f4"); valid[:,6]=1; valid[:,13]=1
            with self.assertRaises(grpc.RpcError) as err:
                client.step(pb.ActionBatch(actions=[pb.Action(values=row) for row in valid]),timeout=5)
            self.assertEqual(err.exception.code(),grpc.StatusCode.INTERNAL)
        finally:
            channel.close(); server.close(); thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

if __name__ == "__main__": unittest.main()
