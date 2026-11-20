import socket
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import grpc
import numpy as np
import torch
from robofinals.distributed.grpc_policy import GrpcPolicyServer, PolicyClient
from robofinals.distributed.chamber_schema import JOINT_DEFAULT_POS, JOINT_NAMES, OBSERVATION_SCHEMA
from robofinals.distributed import policy_v1_pb2 as pb

class Runtime:
    def __init__(self):
        self.num_envs=2; self.action_space=SimpleNamespace(shape=(19,)); self.resets=[]; self.steps=0
        self.unwrapped=SimpleNamespace(step_dt=0.02, device="cpu", extras={},
            scene={"robot":SimpleNamespace(data=SimpleNamespace(joint_names=JOINT_NAMES,
        default_joint_pos=torch.tensor(JOINT_DEFAULT_POS).repeat(2,1), default_joint_vel=torch.zeros((2,77))))},
            observation_manager=SimpleNamespace(compute=lambda **kwargs:self.obs()))
    def obs(self):
        return {"policy":{name:torch.zeros((2,*shape),dtype=torch.uint8 if dtype=="uint8" else torch.float32)
                          for name,dtype,shape in OBSERVATION_SCHEMA}}
    def reset(self, **kwargs):
        ids=kwargs.get("env_ids")
        if ids is not None: self.resets.append(ids.tolist())
        return self.obs(),{}
    def step(self, action):
        self.steps+=1
        return self.obs(),torch.zeros(2),torch.tensor([False,False]),torch.tensor([False,False]),{}
    def close(self): pass

class FakeError(grpc.RpcError):
    def __init__(self, code, details): self._code=code; self._details=details
    def code(self): return self._code
    def details(self): return self._details

class WallTimeoutTests(unittest.TestCase):
    def test_one_environment_times_out_and_requires_fresh_observation(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]
        env=Runtime()
        server=GrpcPolicyServer(lambda cfg:env,("127.0.0.1",port),wall_timeout_seconds=10)
        thread=threading.Thread(target=server.serve,daemon=True);thread.start()
        channel=grpc.insecure_channel(f"127.0.0.1:{port}",options=[("grpc.max_receive_message_length",67108864)])
        try:
            grpc.channel_ready_future(channel).result(timeout=10)
            client=PolicyClient(channel)
            server._episode_started[0]=time.monotonic()-11
            deadline=time.monotonic()+3
            while not server._pending_wall_timeout[0] and time.monotonic()<deadline: time.sleep(.01)
            self.assertEqual(server._pending_wall_timeout,[True,False])
            action=np.zeros((2,19),dtype=np.float32);action[:,6]=1;action[:,13]=1
            with self.assertRaises(grpc.RpcError) as err:
                client.step(pb.ActionBatch(actions=[pb.Action(values=row) for row in action]),timeout=5)
            self.assertEqual(err.exception.code(),grpc.StatusCode.FAILED_PRECONDITION)
            self.assertEqual(env.steps,0)
            obs=client.get_observation(pb.Empty(),timeout=5)
            self.assertEqual(list(obs.truncated),[True,False])
            self.assertEqual(list(obs.terminated),[False,False])
            self.assertEqual(len(obs.final_observations),2)
            self.assertEqual(env.resets,[[0]])
            client.step(pb.ActionBatch(actions=[pb.Action(values=row) for row in action]),timeout=5)
            self.assertEqual(env.steps,1)
        finally:
            channel.close();server.close();thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

    def test_retry_limit_never_replays_ambiguous_step(self):
        attempts=[]
        def call(request,timeout):
            attempts.append(1)
            raise FakeError(grpc.StatusCode.UNAVAILABLE,"connection lost")
        with patch("robofinals.distributed.grpc_policy.time.sleep"):
            with self.assertRaises(FakeError): PolicyClient._invoke(call,None,timeout=1,step=True)
            self.assertEqual(len(attempts),1)
            attempts.clear()
            with self.assertRaises(FakeError): PolicyClient._invoke(call,None,timeout=1,step=False)
            self.assertEqual(len(attempts),4)
            attempts.clear()
            def busy(request,timeout):
                attempts.append(1)
                raise FakeError(grpc.StatusCode.RESOURCE_EXHAUSTED,"another request is in progress")
            with self.assertRaises(FakeError): PolicyClient._invoke(busy,None,timeout=1,step=True)
            self.assertEqual(len(attempts),4)

if __name__=="__main__": unittest.main()
