# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaaclab_arena.assets.background import Background
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.scene.scene import Scene

from robofinals.core.settings import get_runtime_settings
from robofinals.core.models.scenes.scene_parser import get_fixture_cfgs, parse_fixtures
from robofinals.modules.scenes.kitchen.kitchen_arena import KitchenArena, task_preferred_fixture_names
from robofinals.utils.isaaclab_utils import NoDeepcopyMixin
from robofinals.utils.isaaclab_utils.assets.general_asset_arena import GeneralAssetArena
from robofinals.utils.usd_utils import OpenUsd as usd
from robofinals.utils.env import ExecuteMode


class LocalScene(Scene, NoDeepcopyMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.context = get_runtime_settings()
        self.num_envs = self.context.num_envs
        self.scene_name = self.context.scene_name
        self.scene_usd_path = self.context.scene_name
        self.scene_backend = self.context.scene_backend
        self.scene_type = "local"
        self.stage = None
        self.fixtures = {}
        self.fxtr_placements = {}
        self.is_replay_mode = self.context.execute_mode in (
            ExecuteMode.REPLAY_ACTION,
            ExecuteMode.REPLAY_JOINT_TARGETS,
            ExecuteMode.REPLAY_STATE,
        )
        assert self.scene_usd_path.endswith(".usd"), "Scene USD path must end with .usd"
        if self.context.usd_simplify:
            self.stage = usd.get_stage(self.scene_usd_path)

    def setup_env_config(self, orchestrator):
        # Parse fixtures from local USD scene to support fixture-dependent tasks.
        # Physics pruning (strip colliders/rigid bodies from unused fixtures) and
        # joint fix-ups run once in ``KitchenArena.__init__``; by the time we get
        # here ``self.scene_usd_path`` already points at the patched USD.
        preferred_fixture_names = task_preferred_fixture_names(orchestrator.task)
        self.robofinals_arena = KitchenArena(
            local_scene_path=self.scene_usd_path,
            enable_fixtures=orchestrator.task.enable_fixtures,
            movable_fixtures=orchestrator.task.movable_fixtures,
            layout_registry_names=orchestrator.task.layout_registry_names,
            preferred_fixture_names=preferred_fixture_names,
        )
        self.scene_usd_path = self.robofinals_arena.usd_path
        self.stage = self.robofinals_arena.stage
        self.fixtures = parse_fixtures(
            self.stage,
            self.context.num_envs,
            self.context.seed,
            self.context.device,
        )
        self.fxtr_placements = {}
        self.fixture_cfgs = get_fixture_cfgs(self)
        print(f"[LocalScene Debug] Parsed fixtures count: {len(self.fixtures)}")

        if self.context.enable_full_local_scene:
            background = GeneralAssetArena(
                name="Scene",
                usd_path=self.scene_usd_path,
                object_min_z=0.1,
                physics_backend=self.context.physics_backend,
            )
        else:
            background = Background(
                name="Scene",
                usd_path=self.scene_usd_path,
                object_min_z=0.1,
            )
        # flush self.assets
        self.assets = {}
        self.add_asset(background)

    def get_ep_meta(self):
        return {
            "floorplan_version": None,
            # A fixed local scene has no layout/style identifiers. Persist the
            # resolved USD path so a replay can rebuild the same scene.
            "scene_name": self.scene_name,
            "usd_path": self.scene_usd_path,
            "scene_backend": self.scene_backend,
            "task_backend": self.context.task_backend,
            "physics_backend": self.context.physics_backend,
            "enable_full_local_scene": self.context.enable_full_local_scene,
        }

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg):
        env_cfg.scene_backend = self.scene_backend
        env_cfg.sim.render.enable_translucency = True
        env_cfg.num_envs = self.num_envs

        # Floor for all rigid/articulation contacts (plates, table, sponge, …).
        # Per-actor counts below this are raised; matches Galbot grasp solver floor.
        from robofinals.core.backends import make_backend_runtime

        make_backend_runtime(self.context.physics_backend).simulation.set_scene_params(
            env_cfg.sim,
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
            min_position_iteration_count=48,
            min_velocity_iteration_count=8,
        )

        if self.context.enable_global_illumination:
            env_cfg.sim.render.enable_global_illumination = True

        return env_cfg
