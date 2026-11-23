import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robofinals_tasks.chamber_service_task import ChamberServiceTask
from robofinals_tasks.lightwheel_home_tasks.home_base_task import HomeBaseTask


class ChamberTableKinematicTests(unittest.TestCase):
    @patch.object(ChamberServiceTask, "KEEP_TABLE_STATIC", True)
    @patch.object(HomeBaseTask, "init_fixtures")
    def test_init_fixtures_sets_every_table_kinematic(self, super_init):
        attrs = []
        prims = {}
        for env_id in range(3):
            attr = Mock()
            attr.IsValid.return_value = True
            prim = Mock()
            prim.IsValid.return_value = True
            prim.GetAttribute.return_value = attr
            path = ChamberServiceTask.TABLE_PRIM_PATH.format(env_id=env_id)
            prims[path] = prim
            attrs.append(attr)

        stage = Mock()
        stage.GetPrimAtPath.side_effect = prims.__getitem__
        env = SimpleNamespace(num_envs=3, scene=SimpleNamespace(stage=stage))
        task = ChamberServiceTask()

        task.init_fixtures(env)

        super_init.assert_called_once_with(env, None)
        self.assertEqual(stage.GetPrimAtPath.call_count, env.num_envs)
        for attr in attrs:
            attr.Set.assert_called_once_with(True)

    @patch.object(ChamberServiceTask, "KEEP_TABLE_STATIC", True)
    @patch.object(HomeBaseTask, "modify_env_cfg")
    def test_modify_env_cfg_removes_table_from_generic_resets(self, super_modify):
        env_cfg = SimpleNamespace(
            scene=SimpleNamespace(Table291=object()),
            episode_length_s=0.0,
        )
        super_modify.return_value = env_cfg

        result = ChamberServiceTask().modify_env_cfg(env_cfg)

        self.assertIs(result, env_cfg)
        self.assertIsNone(env_cfg.scene.Table291)


if __name__ == "__main__":
    unittest.main()
