import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from robofinals.distributed.eval_config import load_eval_config, locked_initializer
from robofinals.distributed.proxy import EnvService


class ConfigLockTests(unittest.TestCase):
    def test_config_and_attach_are_evaluator_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.json"
            path.write_text(json.dumps({"task": "ChamberServiceTask", "robot": "Galbot-Mobile",
                                        "layout": "/fixed/scene.usd", "seed": 9}), encoding="utf-8")
            cfg = load_eval_config(path)
        seen = []
        init = locked_initializer(lambda c, _: seen.append(c) or c, cfg, {})
        first = init({"task": "OtherTask", "seed": 999, "layout": "/attacker"})
        self.assertEqual((first.task, first.seed, first.layout), ("ChamberServiceTask", 9, "/fixed/scene.usd"))
        first.replay_cfgs["add_camera_to_observation"] = False
        self.assertTrue(init({}).replay_cfgs["add_camera_to_observation"])

    def test_invalid_config_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.json"
            for value in ([], {}, {"task": "x", "robot": "y", "layout": "z", "seed": -1},
                          {"task": "x", "robot": "y", "layout": "z", "num_envs": 0}):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_eval_config(path)

    def test_policy_reset_seed_is_ignored_in_locked_mode(self):
        seeds = []
        runtime = SimpleNamespace(_eval_config_locked=True, reset=lambda seed: (seeds.append(seed) or {}, {}))
        EnvService(runtime).reset(seed=12345)
        self.assertEqual(seeds, [None])
        runtime._eval_config_locked = False
        EnvService(runtime).reset(seed=12345)
        self.assertEqual(seeds[-1], 12345)

if __name__ == "__main__":
    unittest.main()
