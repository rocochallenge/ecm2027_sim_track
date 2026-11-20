import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch
from robofinals.applications.eval.episode_records import EpisodeRecords, CAMERAS

class RecordsTests(unittest.TestCase):
    def test_per_slot_completion_and_video(self):
        task = SimpleNamespace(POINT_NAMES=("p1",), _stage_names=("stage1",),
            _point_flags=torch.tensor([[True], [False]]), _stage_flags=torch.tensor([[True], [False]]),
            _score_steps=torch.tensor([3, 2]), _process_score=torch.tensor([1,0]))
        obs={"policy": {name: np.zeros((2, 32, 32, 3), dtype=np.uint8) for name in CAMERAS}}
        raw=SimpleNamespace(step_dt=0.02, termination_manager=SimpleNamespace(terminated=[False,False],
                            time_outs=[False,True]), extras={"final_obs":obs},
                            observation_manager=SimpleNamespace(compute=lambda: obs))
        env=SimpleNamespace(num_envs=2, unwrapped=raw, cfg=SimpleNamespace(isaaclab_arena_env=SimpleNamespace(task=task)))
        with tempfile.TemporaryDirectory() as d:
            records=EpisodeRecords(d, env, {"seed": 42})
            records.start(obs)
            records.before_step(np.zeros((2,19),dtype=np.float32))
            task._episode_finished_callback(1,{"points":{"p1":False}, "completed_stages":[],
                                              "episode_sim_time_s":0.04,"process_score":0})
            records.after_step(obs)
            root=records.root
            manifest=json.loads((root/"env_0001/episode_000000/manifest.json").read_text())
            score=json.loads((root/"env_0001/episode_000000/score.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(score["end_reason"], "truncated")
            self.assertTrue((root/"env_0000/episode_000000/manifest.json").exists())
            self.assertEqual(records.active[0]["episode_id"], 0)
            self.assertEqual(records.active[1]["episode_id"], 1)
            for name in CAMERAS:
                path=root/"env_0001/episode_000000"/(name+".mp4")
                cap=cv2.VideoCapture(str(path)); ok,_=cap.read(); cap.release()
                self.assertTrue(ok, str(path))
            records.close()
            interrupted=json.loads((root/"env_0000/episode_000000/manifest.json").read_text())
            self.assertEqual(interrupted["status"],"interrupted")

if __name__ == "__main__": unittest.main()
