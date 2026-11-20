"""Bounded two-process tests of policy cleanup over the real IPC transport.

The simulator is replaced by a tiny runtime; Isaac startup/physics are not tested.
"""
import multiprocessing
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from policy.base import BasePolicy


class Runtime:
    def __init__(self, events, outcome):
        self.events = events
        self.outcome = outcome
        self.action_space = SimpleNamespace(shape=(19,))
        self.cfg = SimpleNamespace(decimation=4)
        self.num_envs = 1

    def reset(self, seed=None):
        return {}, {}

    def step(self, action):
        assert torch.is_tensor(action)
        self.events.put(('action', tuple(action.shape)))
        if self.outcome == 'error':
            raise RuntimeError('simulated step failure')
        return {}, 0, torch.tensor([self.outcome == 'terminated']), torch.tensor([self.outcome == 'truncated']), {}

    def close(self):
        self.events.put(('closed', True))


def serve(ready, events, outcome):
    from robofinals.distributed.ipc import IpcDistributedEnvWrapper
    wrapper = IpcDistributedEnvWrapper(env_initializer=lambda cfg: Runtime(events, outcome), address=('127.0.0.1', 0))
    ready.send(wrapper._server.listener.address)
    ready.close()
    wrapper.serve()


class LoopPolicy:
    _video_writer = None

    def __init__(self, args):
        self.outcome = args['outcome']

    def reset_model(self):
        pass

    def eval(self, env, obs, args, video):
        if self.outcome == 'policy_error':
            raise RuntimeError('simulated policy failure')
        for _ in range(3):
            obs, done, _ = BasePolicy.step_environment(self, env, np.zeros((1, 20)), args)
            if bool(done.all()):
                return False  # Done must not be interpreted as task success.
        raise AssertionError('policy did not see episode end')

    def format_eval_results(self, results, args, count):
        return {'test_count': count}


def evaluate(address, directory, outcome):
    with patch.object(sys, 'argv', ['policy.py']):
        from robofinals.applications.eval.entrypoints import policy as entry
    from robofinals.distributed.proxy import RemoteEnv
    connect = RemoteEnv.make
    os.chdir(directory)
    with patch.object(RemoteEnv, 'make', side_effect=lambda **kwargs: connect(address)), \
            patch.object(entry, '_load_policy_class', return_value=LoopPolicy):
        entry.main({'policy_name': 'test', 'test_num': 1, 'outcome': outcome,
                    'env_cfg': {'num_envs': 1}})


class PolicyExitTests(unittest.TestCase):
    def test_done_masks_include_truncation_for_both_protocols(self):
        for tensor in (False, True):
            for arity in (3, 5):
                terminated, truncated = [False, True], [True, False]
                if tensor:
                    terminated, truncated = torch.tensor(terminated), torch.tensor(truncated)
                transition = ({}, terminated, truncated) if arity == 3 else ({}, 0, terminated, truncated, {})
                env = SimpleNamespace(step=lambda action: transition)
                _, done, _ = BasePolicy.step_environment(None, env, np.zeros((2, 19)), {})
                np.testing.assert_array_equal(done, [True, True])

    def test_two_process_exit_on_termination_truncation_and_exception(self):
        ctx = multiprocessing.get_context('spawn')
        for outcome in ('terminated', 'truncated', 'error', 'policy_error'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                reader, writer = ctx.Pipe(duplex=False)
                events = ctx.Queue()
                server = ctx.Process(target=serve, args=(writer, events, outcome))
                client = None
                try:
                    server.start()
                    self.assertTrue(reader.poll(15), 'IPC server failed to start')
                    address = reader.recv()
                    client = ctx.Process(target=evaluate, args=(address, directory, outcome))
                    client.start()
                    client.join(15)
                    self.assertFalse(client.is_alive(), 'policy process hung')
                    failed = outcome in ('error', 'policy_error')
                    self.assertEqual(client.exitcode, 1 if failed else 0)
                    server.join(15)
                    self.assertFalse(server.is_alive(), 'IPC server did not stop')
                    self.assertEqual(server.exitcode, 0)
                    if outcome != 'policy_error':
                        self.assertEqual(events.get(timeout=2), ('action', (1, 20)))
                    self.assertEqual(events.get(timeout=2), ('closed', True))
                    self.assertEqual(Path(directory, 'eval_result/eval_results.json').exists(), not failed)
                finally:
                    for process in (client, server):
                        if process is not None and process.is_alive():
                            process.terminate()
                            process.join(5)
                            if process.is_alive():
                                process.kill()
                                process.join(5)
                    reader.close()
                    writer.close()
                    events.close()


if __name__ == '__main__':
    unittest.main()
