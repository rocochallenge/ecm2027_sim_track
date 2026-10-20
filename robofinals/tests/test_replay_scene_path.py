import ast
import os
import tempfile
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "applications/replay/entrypoints/replay_action_demo.py"


def load_scene_resolver():
    tree = ast.parse(SOURCE_PATH.read_text())
    resolver = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_replay_scene"
    )
    namespace = {"Path": Path, "os": os, "__file__": str(SOURCE_PATH)}
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace["_resolve_replay_scene"]


class ReplayScenePathTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project_root = Path(self.temp_dir.name)
        self.scene_dir = self.project_root / "fixture_scenes/Scene"
        self.scene_dir.mkdir(parents=True)
        self.resolve_scene = load_scene_resolver()

    def metadata(self):
        return {
            "usd_path": "/workspace/robofinals/fixture_scenes/Scene/test_scene_enabled.usd",
            "scene_name": "fixture_scenes/Scene/test_scene.usd",
        }

    def test_stale_container_path_uses_source_scene_not_enabled_output(self):
        scene = self.scene_dir / "test_scene.usd"
        scene.touch()
        enabled_scene = self.scene_dir / "test_scene_enabled.usd"
        enabled_scene.touch()

        resolved = self.resolve_scene(None, self.metadata(), self.project_root)

        self.assertEqual(resolved, str(scene.resolve()))

    def test_existing_recorded_usd_is_preferred(self):
        recorded_scene = self.scene_dir / "recorded_scene.usd"
        recorded_scene.touch()
        metadata = self.metadata()
        metadata["usd_path"] = str(recorded_scene)

        resolved = self.resolve_scene(None, metadata, self.project_root)

        self.assertEqual(resolved, str(recorded_scene.resolve()))

    def test_explicit_layout_takes_precedence(self):
        scene = self.scene_dir / "override.usd"
        scene.touch()

        resolved = self.resolve_scene(str(scene), self.metadata(), self.project_root)

        self.assertEqual(resolved, str(scene.resolve()))


if __name__ == "__main__":
    unittest.main()
