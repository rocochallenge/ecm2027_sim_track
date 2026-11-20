import unittest
from types import SimpleNamespace
import torch
from robofinals_tasks.chamber_service_task import ChamberServiceTask

class MultiEnvScoreTests(unittest.TestCase):
    def test_snapshot_and_clear_only_selected_slot(self):
        task = ChamberServiceTask()
        env = SimpleNamespace(num_envs=2, device="cpu")
        task._ensure_state_buffers(env)
        task._step_dt = 0.02
        task._score_steps[:] = torch.tensor([9, 7])
        task._point_flags[0, :3] = True
        task._point_flags[1, :2] = True
        task._process_score[:] = torch.tensor([3, 2])
        observed = []
        task._episode_finished_callback = lambda i, metrics: observed.append((i, metrics.copy()))
        task._snapshot_finished_episode("reset", torch.tensor([1]))
        task._clear_scoring_state(torch.tensor([False, True]))
        self.assertEqual(task._process_score.tolist(), [3, 0])
        self.assertEqual(task._score_steps.tolist(), [9, 0])
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0], 1)
        self.assertEqual(observed[0][1]["process_score"], 2)
        self.assertEqual(observed[0][1]["episode_steps"], 7)
        self.assertEqual(task._finished_episodes[0]["env_id"], 1)

if __name__ == "__main__": unittest.main()
