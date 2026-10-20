import ast
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE_PATH = Path(__file__).resolve().parents[1] / 'modules/robots/robot_arena_base.py'
CAMERA_NAMES = ('first_person_camera', 'left_hand_camera', 'right_hand_camera')


class ObservationTerm:
    pass


def load_camera_setup():
    tree = ast.parse(SOURCE_PATH.read_text())
    embodiment = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LwEmbodimentBase')
    setup = next(node for node in embodiment.body if isinstance(node, ast.FunctionDef) and node.name == '_setup_camera_config')
    camera_observations = make_dataclass('CameraObservations', [(name + '_rgb', ObservationTerm) for name in CAMERA_NAMES])
    namespace = {
        'ObsTerm': ObservationTerm,
        'make_camera_observation_cfg': lambda config: SimpleNamespace(camera_obs=camera_observations),
        'combine_configclass_instances': lambda *args: args[1],
    }
    exec(compile(ast.Module(body=[setup], type_ignores=[]), str(SOURCE_PATH), 'exec'), namespace)
    return type('CameraSetup', (), {'_setup_camera_config': namespace['_setup_camera_config'], 'active_observation_camera_names': []})


def make_robot(robot_type):
    robot = robot_type()
    robot.context = SimpleNamespace(execute_mode='replay')
    robot.observation_cameras = {name: {'execute_mode': ['replay'], 'camera_cfg': object()} for name in CAMERA_NAMES}
    robot.camera_config = SimpleNamespace()
    robot.enable_cameras = True
    robot.add_camera_to_observation = True
    robot.policy_observation_config = SimpleNamespace()
    return robot


class ReplayCameraConfigTests(unittest.TestCase):
    def test_repeated_setup_has_exactly_three_observation_streams(self):
        robot = make_robot(load_camera_setup())
        for attempt in range(3):
            robot._setup_camera_config('chamber')
            self.assertEqual(robot.active_observation_camera_names, [name + '_rgb' for name in CAMERA_NAMES])

    def test_robot_instances_do_not_share_camera_selection(self):
        robot_type = load_camera_setup()
        first = make_robot(robot_type)
        second = make_robot(robot_type)
        first._setup_camera_config('chamber')
        second._setup_camera_config('chamber')
        self.assertEqual(len(first.active_observation_camera_names), 3)
        self.assertEqual(len(second.active_observation_camera_names), 3)
        self.assertIsNot(first.active_observation_camera_names, second.active_observation_camera_names)
        self.assertEqual(robot_type.active_observation_camera_names, [])

    def test_disabled_camera_observations_clear_the_active_selection(self):
        robot = make_robot(load_camera_setup())
        robot._setup_camera_config('chamber')
        robot.add_camera_to_observation = False
        robot._setup_camera_config('chamber')
        self.assertEqual(robot.active_observation_camera_names, [])

    def test_local_task_registers_cameras_only_once(self):
        tree = ast.parse(SOURCE_PATH.read_text())
        local_branches = [node for node in ast.walk(tree) if isinstance(node, ast.If) and ast.unparse(node.test) == "orchestrator.task.context.task_backend == 'local'"]
        self.assertEqual(len(local_branches), 1)
        calls = [node for statement in local_branches[0].body for node in ast.walk(statement) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == '_setup_camera_config']
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
