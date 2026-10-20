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

import glob
import os
from copy import deepcopy
from dataclasses import MISSING
from typing import Any, ClassVar, Dict, List

import lazy_import
import numpy as np
import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_reference import ObjectReference
from robofinals.core.models.assets.LwObjectReference import LwObjectReference
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose

import robofinals.core.mdp as mdp
import robofinals.utils.fixture_utils as FixtureUtils
import robofinals.utils.math_utils.transform_utils.numpy_impl as Tn
import robofinals.utils.object_utils as OU
EnvUtils = lazy_import.lazy_module("robofinals.utils.place_utils.env_utils")

from robofinals.core.checks.checker_factory import get_checkers_from_cfg, form_checker_result
from robofinals.core.backends import make_backend_runtime
from robofinals.core.settings import get_runtime_settings
from robofinals.core.models.assets.LwObject import LwObject
from robofinals.core.models.fixtures import Fixture, FixtureType, fixture_is_type
from robofinals.core.models.scenes.scene_parser import register_fixture_from_obj
from robofinals.utils.csv_loader import csv_loader
from robofinals.utils.env import ExecuteMode
from robofinals.utils.isaaclab_utils import NoDeepcopyMixin
from robofinals.utils.log_utils import copy_dict_for_json
from robofinals.utils.place_utils.contact_queue import ContactQueue
from robofinals.utils.place_utils.usd_object import USDObject
from robofinals.utils.usd_utils import OpenUsd as usd


@configclass
class TaskBaseObservationCfg:
    """Observation specifications for the MDP."""
    pass


@configclass
class TaskBasePolicyObservationCfg(ObsGroup):
    """Observations for policy group."""

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class EventCfg:
    """Configuration for events."""

    # robot_physics_material: EventTerm = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 1.25),
    #         "dynamic_friction_range": (0.8, 1.25),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )

    init_task: EventTerm = MISSING

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # cabinet_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
    #         "static_friction_range": (1.0, 1.25),
    #         "dynamic_friction_range": (1.25, 1.5),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. Approach the handle
    # approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    # align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)

    # # 2. Grasp the handle
    # approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    # align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    # grasp_handle = RewTerm(
    #     func=mdp.grasp_handle,
    #     weight=0.5,
    #     params={
    #         "threshold": 0.03,
    #         "open_joint_pos": MISSING,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
    #     },
    # )

    # 3. Open the drawer
    # open_drawer_bonus = RewTerm(
    #     func=mdp.open_drawer_bonus,
    #     weight=7.5,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["corpus_to_drawer_0_0"])},
    # )
    # multi_stage_open_drawer = RewTerm(
    #     func=mdp.multi_stage_open_drawer,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["corpus_to_drawer_0_0"])},
    # )

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=True)
    success: DoneTerm = MISSING


class LwTaskBase(TaskBase, NoDeepcopyMixin):
    task_name: str = None
    task_type: str = "teleop"
    force_reset_env_enabled: bool = False
    resample_objects_placement_on_reset: bool = True
    resample_robot_placement_on_reset: bool = True
    init_robot_base_pos: list[float] | None = None
    init_robot_base_ori: list[float] | None = None

    BIYAM_CONTROLLER = "BiYam-Controller"
    X7S_ABS = "X7S-Abs"
    DEFAULT_OBJECT_INIT_OFFSET: ClassVar[list[float]] = [0.0, 0.0]

    init_robot_base_pos_by_embodiment: ClassVar[dict[str, list[float]] | None] = None
    init_robot_base_ori_by_embodiment: ClassVar[dict[str, list[float]] | None] = None
    object_init_offset_by_embodiment: ClassVar[dict[str, list[float]] | None] = None

    EXCLUDE_LAYOUTS: list = []
    OVEN_EXCLUDED_LAYOUTS: list = [1, 3, 5, 6, 8, 10, 11, 13, 14, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 33, 36, 38, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    DOUBLE_CAB_EXCLUDED_LAYOUTS: list = [32, 41, 59]
    DINING_COUNTER_EXCLUDED_LAYOUTS: list = [1, 3, 5, 6, 18, 20, 36, 39, 40, 43, 47, 50, 52]
    ISLAND_EXCLUDED_LAYOUTS: list = [1, 3, 5, 6, 8, 9, 10, 13, 18, 19, 22, 27, 30, 36, 40, 43, 46, 47, 49, 52, 53, 60]
    STOOL_EXCLUDED_LAYOUT: list = [1, 3, 5, 6, 18, 20, 36, 39, 40, 43, 47, 50, 52]
    SHELVES_INCLUDED_LAYOUT: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    DOUBLE_CAB_EXCLUDED_LAYOUTS: list = [32, 41, 59]
    FREEZER_EXCLUDED_LAYOUTS = [1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 25, 26, 27, 28, 30, 31, 33, 34, 35, 36, 37, 39, 40, 42, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60]
    FOUR_TOASTER_SLOT_EXCLUDE_STYLES = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 20, 21, 23, 24, 25, 27, 28, 31, 32, 35, 36, 39, 41, 43, 44, 46, 47, 48, 49, 51, 53, 54, 56, 58, 60]
    _start_success_check_count: int = 10
    enable_fixtures: list[str] = []
    movable_fixtures: list[str] = []
    layout_registry_names: list[str | FixtureType | int] | None = None
    pairwise_contacts: ClassVar[tuple[tuple[str, str], ...]] = ()
    success_stages: ClassVar[tuple[str, ...]] = ()

    def __init__(self):
        super().__init__()
        self.context = get_runtime_settings()
        self.backend_runtime = make_backend_runtime(self.context.physics_backend)
        self.task_backend = self.context.task_backend
        self.usd_simplify = self.context.usd_simplify
        self.exclude_layouts = self.EXCLUDE_LAYOUTS
        self.cache_usd_version = self.context.ep_meta.get("cache_usd_version", {})
        self.objects_version = self.cache_usd_version.get("objects_version", None)
        self.sources = self.context.sources
        self.object_projects = self.context.object_projects
        self.seed = self.context.seed
        self.rng = np.random.default_rng(self.seed)
        if self.context.resample_objects_placement_on_reset is not None:
            self.resample_objects_placement_on_reset = self.context.resample_objects_placement_on_reset
        if self.context.resample_robot_placement_on_reset is not None:
            self.resample_robot_placement_on_reset = self.context.resample_robot_placement_on_reset
        self.init_robot_base_ref = None
        self.events_cfg = EventCfg(
            init_task=EventTerm(func=self.init_fixtures, mode="startup")
        )
        self.termination_cfg = TerminationsCfg(
            success=DoneTerm(func=self.check_success_caller)
        )
        self.observation_cfg = TaskBaseObservationCfg()
        self.policy_observation_cfg = TaskBasePolicyObservationCfg()
        self.commands_cfg = None
        self.curriculum_cfg = None
        self.rewards_cfg = None
        self.assets = {}
        self._articulation_assets = {}
        self._deformable_assets = {}
        self.contact_sensors = {}
        self.init_checkers_cfg()
        self.checkers = get_checkers_from_cfg(self.checkers_cfg)
        self.checker_results = form_checker_result(self.checkers_cfg)
        self.fix_object_pose_cfg = None
        self.is_replay_mode = False
        self.contact_queues = [ContactQueue() for _ in range(self.context.num_envs)]

        if self.context.execute_mode == ExecuteMode.TEST_OBJECT:
            self.visible_obj_idx = 0
            self.test_object_paths = self.context.test_object_paths

        # Initialize retry counts
        self.scene_retry_count = 0
        self.object_retry_count = 0

        self._success_cache: torch.Tensor = torch.tensor([0], device=self.context.device, dtype=torch.int32).repeat(self.context.num_envs)
        self._success_flag: torch.Tensor = torch.tensor([False], device=self.context.device, dtype=torch.bool).repeat(self.context.num_envs)

    @classmethod
    def resolve_embodiment_robot_base_pose(
        cls,
        robot_name: str | None,
    ) -> tuple[list[float], list[float]] | None:
        """Return spawn pose when a task defines an embodiment-specific override."""
        if not robot_name:
            return None

        for task_cls in cls.__mro__:
            if task_cls is LwTaskBase:
                continue
            pos_map = task_cls.__dict__.get("init_robot_base_pos_by_embodiment")
            if not pos_map or robot_name not in pos_map:
                continue

            pos = pos_map[robot_name]
            ori_map = task_cls.__dict__.get("init_robot_base_ori_by_embodiment") or {}
            ori = ori_map.get(robot_name)
            if ori is None:
                ori = task_cls.__dict__.get("init_robot_base_ori") or [0.0, 0.0, 0.0]
            return [float(v) for v in pos], [float(v) for v in ori]
        return None

    @classmethod
    def resolve_embodiment_object_init_offset(cls, robot_name: str | None) -> list[float]:
        """Return XY placement offset for the active embodiment (default ``[0.0, 0.0]``)."""
        if robot_name:
            for task_cls in cls.__mro__:
                if task_cls is LwTaskBase:
                    continue
                offset_map = task_cls.__dict__.get("object_init_offset_by_embodiment")
                if offset_map and robot_name in offset_map:
                    offset = offset_map[robot_name]
                    return [float(offset[0]), float(offset[1])]
        return list(cls.DEFAULT_OBJECT_INIT_OFFSET)

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg.task_backend = self.task_backend
        if self.context.execute_mode == ExecuteMode.TELEOP:
            self._success_count = int(1 / env_cfg.sim.dt / 2)
        else:
            self._success_count = 1

        # general settings
        env_cfg.episode_length_s = 8.0
        env_cfg.viewer.eye = (3.0, -4.0, 2.0)
        env_cfg.viewer.lookat = (3.0, 1.0, 0.3)
        # env_cfg.viewer.origin_type = "asset_root"
        # env_cfg.viewer.asset_name = "robot"
        # simulation settings
        env_cfg.sim.dt = 1 / 100  # physics frequency: 100Hz
        env_cfg.sim.render_interval = 4  # render frequency: 25Hz
        env_cfg.decimation = 2  # action frequency: 50Hz
        self.backend_runtime.simulation.set_scene_params(
            env_cfg.sim,
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
        )
        self.backend_runtime.simulation.configure_task_env(env_cfg)
        return env_cfg

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_observation_cfg(self):
        return self.observation_cfg

    def get_policy_observation_cfg(self):
        return self.policy_observation_cfg

    def get_commands_cfg(self):
        return self.commands_cfg

    def get_curriculum_cfg(self):
        return self.curriculum_cfg

    def get_rewards_cfg(self):
        return self.rewards_cfg

    def _check_success(self, env):
        return torch.tensor([False], device=env.device).repeat(env.num_envs)

    def get_success_stage_flags(self) -> list[bool]:
        """Return one completion flag per label in ``success_stages``."""
        return [False] * len(self.success_stages)

    def check_declared_contact(
        self,
        env,
        contact_pair: tuple[str, str],
        threshold: float = 1.0e-8,
    ) -> torch.Tensor:
        if contact_pair not in self.pairwise_contacts:
            raise KeyError(f"Undeclared pairwise contact: {contact_pair!r}")
        return OU.check_declared_contact_sensor(env, contact_pair, threshold)

    def check_success_caller(self, env):
        arena_env = env.cfg.isaaclab_arena_env
        arena_env.orchestrator.update_state(env)

        for checker in arena_env.task.checkers:
            arena_env.task.checker_results[checker.type] = checker.check(env)

        # at the begining of the episode, dont check success for stabilization
        success_check_result = self._check_success(env)

        assert isinstance(success_check_result, torch.Tensor), f"_check_success must be a torch.Tensor, but got {type(success_check_result)}"
        assert len(success_check_result.shape) == 1 and success_check_result.shape[0] == env.num_envs, f"_check_success must be a torch.Tensor of shape ({env.num_envs},), but got {success_check_result.shape}"
        start_check_count = torch.tensor(self._start_success_check_count, device=env.episode_length_buf.device, dtype=env.episode_length_buf.dtype)
        success_check_result &= (env.episode_length_buf >= start_check_count)

        # success delay
        self._success_flag &= (self._success_cache < self._success_count)
        self._success_cache *= (self._success_cache < self._success_count)
        self._success_flag |= success_check_result
        self._success_cache += self._success_flag.int()
        return self._success_cache >= self._success_count

    def init_fixtures(self, env, env_ids=None):
        # set env instance in task
        self.env_instance = env
        # register placed fixture
        for name, articulation in self._articulation_assets.items():
            pos = self.object_placements[name][0]
            rot = self.object_placements[name][1]  # xyzw
            prim = usd.get_prim_by_name(env.scene.stage.GetPseudoRoot(), name, only_xform=True)
            register_fixture_from_obj(
                obj=articulation,
                prim=prim[0],
                fixtures_ref=self.fixture_refs,
                num_envs=env.num_envs,
                pos=pos,
                rot=rot,
            )
        for fixture_controller in self.fixture_refs.values():
            if isinstance(fixture_controller, Fixture):
                fixture_controller.setup_env(env)

    def init_checkers_cfg(self):
        # checkers
        if self.context.execute_mode in (ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS, ExecuteMode.REPLAY_STATE):
            print("INFO: Running in Replay Mode. Using replay-specific checker config.")
            self.checkers_cfg = {
                "motion": {
                    "warning_on_screen": False
                },
                "clipping": {
                    "warning_on_screen": False
                },
                "velocity_jump": {
                    "warning_on_screen": False
                }
            }
        else:
            print("INFO: Running in Teleop Mode. Using default checker config.")
            self.checkers_cfg = {
                # "motion": {
                #     "warning_on_screen": False
                # },
                # "clipping": {
                #     "warning_on_screen": False
                # },
                # "velocity_jump": {
                #     "warning_on_screen": False
                # },
                # "start_object_move": {
                #     "warning_on_screen": False
                # }
            }

    def get_checker_results(self):
        """
        Get all checker data for JSON export. This function integrates various types of checker results
        and can be extended to include additional results in the future.

        Returns:
            dict: Complete checker data combining all available checker results
        """
        checker_datas = {}

        for checker in self.checkers:
            checker_datas[checker.type] = checker.get_metrics(self.checker_results[checker.type])

        return checker_datas

    def get_metrics(self):
        return None

    def get_warning_text(self):
        warning_text = ""
        for checker in self.checkers:
            if self.checker_results[checker.type].get("warning_text"):
                warning_text += self.checker_results[checker.type].get("warning_text")
                warning_text += "\n"
        return warning_text

    def _get_obj_cfgs(self):
        """
        Returns a list of object configurations to use in the environment.
        The object configurations are usually environment-specific and should
        be implemented in the subclass.

        Returns:
            list: list of object configurations
        """

        return []

    def _create_objects(self):
        """
        Creates and places objects in the kitchen environment.
        Helper function called by _create_objects()
        """
        # add objects
        self.objects: Dict[str, USDObject] = {}
        if "object_cfgs" in self.context.ep_meta:
            self.object_cfgs: List[Dict[str, Any]] = self.context.ep_meta["object_cfgs"]
            for obj_num, obj_cfg in enumerate(self.object_cfgs):
                if "name" not in obj_cfg:
                    obj_cfg["name"] = "obj_{}".format(obj_num + 1)
                object_version = None
                if self.objects_version is not None:
                    for obj_version in self.objects_version:
                        if obj_cfg["name"] in obj_version:
                            object_version = obj_version[obj_cfg["name"]]
                            break
                obj_cfg["ep_meta_scale"] = obj_cfg["info"]["scale"]
                recorded_info_name = obj_cfg.get("info", {}).get("name")
                model, info = EnvUtils.create_obj(self, obj_cfg, version=object_version)
                obj_cfg["info"] = {**obj_cfg.get("info", {}), **info}
                if recorded_info_name:
                    obj_cfg["info"]["name"] = recorded_info_name
                self.objects[model.task_name] = model
        else:
            self.object_cfgs = self._get_obj_cfgs()
            self.object_cfgs = self.apply_object_init_offset(self.object_cfgs)
            all_obj_cfgs = []
            for obj_num, obj_cfg in enumerate(self.object_cfgs):
                obj_cfg["type"] = "object"
                if "name" not in obj_cfg:
                    obj_cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = EnvUtils.create_obj(self, obj_cfg)
                obj_cfg["info"] = {**obj_cfg.get("info", {}), **info}
                self.objects[model.task_name] = model

                # check and create merged object if needed
                if obj_cfg.get("merged_obj", None):
                    base_obj_dir = os.path.dirname(model.obj_path)
                    pattern = os.path.join(base_obj_dir, f"{model.name}_*/{model.name}_*.usd")
                    merged_obj_files = glob.glob(pattern)
                    if merged_obj_files:
                        for merged_obj_path in merged_obj_files:
                            merged_obj_name = os.path.basename(os.path.dirname(merged_obj_path))

                            merged_obj_cfg = deepcopy(obj_cfg)
                            merged_obj_cfg["name"] = f"{obj_cfg['name']}_{merged_obj_name.lower().split('_')[1]}"
                            merged_obj_cfg["obj_groups"] = merged_obj_path
                            merged_obj_cfg["type"] = "object"

                            merged_obj_placement = obj_cfg.get("auxiliary_obj_placement", None)
                            if merged_obj_placement is not None:
                                merged_obj_cfg["placement"] = merged_obj_placement
                                obj_cfg.pop("auxiliary_obj_placement", None)
                            else:
                                reset_regions = model.get_reset_regions()
                                # use reg_anchor site for merged part placement if exists
                                if "anchor" in reset_regions:
                                    anchor_region = reset_regions["anchor"]
                                    merged_obj_cfg["placement"] = dict(
                                        size=(0.01, 0.01),
                                        pos=anchor_region["offset"],
                                        ensure_object_boundary_in_range=False,
                                        ensure_valid_placement=False,
                                        sample_args=dict(
                                            reference=obj_cfg["name"],
                                            ref_fixture=obj_cfg["placement"]["fixture"],
                                        ),
                                    )
                                else:
                                    raise FileNotFoundError(f"Base object USD file does not contain the required anchor site.")

                            model, info = EnvUtils.create_obj(self, merged_obj_cfg)
                            merged_obj_cfg["info"] = info
                            all_obj_cfgs.append(merged_obj_cfg)
                            self.objects[model.task_name] = model
                    else:
                        raise FileNotFoundError(f"Merged object USD file not found.")

                # place object in a container and add container as an object to the scene
                try_to_place_in = obj_cfg["placement"].get("try_to_place_in", None)
                object_ref = obj_cfg["placement"].get("object", None)

                if try_to_place_in and (
                    "in_container" in obj_cfg["info"]["groups_containing_sampled_obj"]
                ):
                    container_cfg = {
                        "name": obj_cfg["name"] + "_container",
                        "obj_groups": obj_cfg["placement"].get("try_to_place_in"),
                        "placement": deepcopy(obj_cfg["placement"]),
                        "type": "object",
                    }

                    init_robot_here = obj_cfg.get("init_robot_here", False)
                    if init_robot_here is True:
                        obj_cfg["init_robot_here"] = False
                        container_cfg["init_robot_here"] = True

                    try_to_place_in_kwargs = obj_cfg["placement"].get(
                        "try_to_place_in_kwargs", None
                    )
                    if try_to_place_in_kwargs is not None:
                        for k, v in try_to_place_in_kwargs.items():
                            container_cfg[k] = v

                    container_kwargs = obj_cfg["placement"].get("container_kwargs", None)
                    if container_kwargs is not None:
                        for k, v in container_kwargs.items():
                            container_cfg[k] = v

                    # add in the new object to the model
                    all_obj_cfgs.append(container_cfg)
                    model, info = EnvUtils.create_obj(self, container_cfg)
                    container_cfg["info"] = {**container_cfg.get("info", {}), **info}
                    self.objects[model.task_name] = model

                    # modify object config to lie inside of container
                    reset_regions = model.get_reset_regions()
                    if "int" in reset_regions:
                        int_region = reset_regions["int"]
                    else:
                        int_region = reset_regions[model.bounded_region_name]
                    obj_cfg["placement"] = dict(
                        size=(int_region["size"][0] / 4, int_region["size"][1] / 4),
                        pos=int_region["offset"],
                        ensure_object_boundary_in_range=False,
                        sample_args=dict(
                            reference=container_cfg["name"],
                            ref_fixture=obj_cfg["placement"]["fixture"],
                        ),
                    )
                elif (
                    object_ref
                ):
                    parent = object_ref
                    if parent in self.objects:
                        container_name = parent
                        container_obj = self.objects[parent]
                        container_size = container_obj.size
                        smaller_dim = min(container_size[0], container_size[1])
                        sampling_size = (smaller_dim * 0.5, smaller_dim * 0.5)
                        obj_cfg["placement"] = {
                            "size": sampling_size,
                            "ensure_object_boundary_in_range": False,
                            "sample_args": {"reference": container_name},
                        }

                # append the config for this object
                all_obj_cfgs.append(obj_cfg)

            self.object_cfgs = all_obj_cfgs

        # update cache_usd_version
        objects_version = []
        for obj_cfg in self.object_cfgs:
            objects_version.append({obj_cfg["name"]: obj_cfg["info"]["obj_version"]})
        self.objects_version = objects_version

        for obj_cfg in self.object_cfgs:
            stage = usd.get_stage(obj_cfg["info"]["obj_path"])
            object_type = detect_object_type(stage=stage)
            # Preserve the existing task behavior for assets without an authored
            # physics object API. Before deformable support, every non-articulation
            # task object was treated as rigid.
            if object_type == ObjectType.BASE:
                object_type = ObjectType.RIGID
            # Placement uses object_scale; spawn uses spawn_scale (1.0 when scale is baked into USD).
            model = self.objects.get(obj_cfg["info"]["task_name"])
            obj_scale = model.spawn_scale.tolist()
            # Convert model.spawn_scale to tuple format
            if not isinstance(obj_scale, (tuple, list)):
                object_scale = (float(obj_scale), float(obj_scale), float(obj_scale))
            else:
                object_scale = obj_scale
            task_obj_name = obj_cfg["info"]["task_name"]
            if object_type == ObjectType.ARTICULATION:
                self._articulation_assets[task_obj_name] = obj_cfg
            elif object_type == ObjectType.DEFORMABLE:
                if not self._newton_deformables_enabled():
                    raise RuntimeError(
                        f"Task object '{task_obj_name}' is deformable, but "
                        "newton_deformable_solver is not configured by YAML/CLI or the task class. "
                        "Configure newton_deformable_solver to enable Newton soft-body simulation."
                    )
                self._track_deformable_asset(
                    task_obj_name,
                    source="spawned_object",
                    obj_cfg=obj_cfg,
                )
            self.add_asset(
                LwObject(
                    name=task_obj_name,
                    tags=["object"],
                    usd_path=obj_cfg["info"]["obj_path"],
                    prim_path=f"{{ENV_REGEX_NS}}/Scene/{task_obj_name}",
                    object_type=object_type,
                    scale=object_scale,
                    tetmesh_source=obj_cfg.get("tetmesh_source", "collision"),
                    backend_assets=self.backend_runtime.assets,
                )
            )
            # IsaacLab ContactSensorCfg is a rigid-body contact sensor. Deformables
            # are registered through scene.deformable_objects and must not enter it.
            if object_type != ObjectType.DEFORMABLE:
                # Object-level contact sensors remain part of the task contract;
                # backend-specific sensor implementations belong to the adapter.
                self.add_contact_sensor_cfg(
                    name=f"{task_obj_name}_contact",
                    cfg=ContactSensorCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/Scene/{task_obj_name}/{obj_cfg['info']['name']}",
                        update_period=0.0,
                        history_length=6,
                        debug_vis=False,
                        force_threshold=0.0,
                        filter_prim_paths_expr=[],
                    )
                )

    def get_obj_lang(self, obj_name="obj", get_preposition=False):
        """
        gets a formatted language string for the object (replaces underscores with spaces)
        """
        return OU.get_obj_lang(self, obj_name=obj_name, get_preposition=get_preposition)

    def _update_fxtr_obj_placement(self, object_placements):
        updated_obj_names = []
        for obj_name, obj_placement in object_placements.items():
            updated_placement = list(obj_placement)
            obj_cfg = [cfg for cfg in self.object_cfgs if cfg["name"] == obj_name][0]
            ref_fixture = None
            if "fixture" in obj_cfg["placement"]:
                ref_fixture = obj_cfg["placement"]["fixture"]
            if isinstance(ref_fixture, str):
                ref_fixture = self.get_fixture(ref_fixture)
            # TODO: add other sliding fxtr types
            if fixture_is_type(ref_fixture, FixtureType.DRAWER):
                ref_rot_mat = Tn.euler2mat(np.array([0, 0, ref_fixture.rot]))
                updated_placement[0] = np.array(updated_placement[0]) + ref_fixture._regions["int"]["per_env_offset"] @ ref_rot_mat.T
                updated_obj_names.append(obj_name)
            else:
                updated_placement[0] = np.array(updated_placement[0])[None, :].repeat(self.context.num_envs, axis=0)
            object_placements[obj_name] = updated_placement
        return object_placements, updated_obj_names

    def add_asset(self, asset: Asset):
        assert asset.name is not None, "Asset with the same name already exists"
        self.assets[asset.name] = asset

    def add_contact_sensor_cfg(self, name: str, cfg: ContactSensorCfg):
        assert name is not None, "Contact sensor with the same name already exists"
        self.contact_sensors[name] = cfg

    def get_scene_cfg(self):
        fields = []
        for asset in self.assets.values():
            for asset_cfg_name, asset_cfg in asset.get_object_cfg().items():
                fields.append((asset_cfg_name, type(asset_cfg), asset_cfg))
        for sensor_name, sensor_cfg in self.contact_sensors.items():
            fields.append((sensor_name, type(sensor_cfg), sensor_cfg))
        NewConfigClass = make_configclass("TaskCfg", fields)
        new_config_class = NewConfigClass()
        return new_config_class

    def apply_object_init_offset(self, cfgs):
        embodiment_offset = self.resolve_embodiment_object_init_offset(self.context.robot_name)
        context_offset = getattr(self.context, "object_init_offset", None) or self.DEFAULT_OBJECT_INIT_OFFSET
        combined = [
            embodiment_offset[0] + float(context_offset[0]),
            embodiment_offset[1] + float(context_offset[1]),
        ]
        if combined == self.DEFAULT_OBJECT_INIT_OFFSET:
            return cfgs
        return self._apply_xy_offset_to_cfgs(cfgs, combined)

    @staticmethod
    def _apply_xy_offset_to_cfgs(cfgs, offset: list[float]):
        for cfg in cfgs:
            if "placement" in cfg and "pos" in cfg["placement"]:
                pos = cfg["placement"]["pos"]
                cfg["placement"]["pos"] = (
                    (offset[0] + pos[0]) if isinstance(pos[0], (float, int)) else pos[0],
                    (offset[1] + pos[1]) if isinstance(pos[1], (float, int)) else pos[1],
                )
        return cfgs

    def _replay_ep_name(self) -> str:
        ep_names = self.context.replay_cfgs["ep_names"]
        current = self.context.replay_cfgs.get("current_ep_name")
        if current:
            return current
        return ep_names[-1]

    def _load_placement(self):
        objects_placement = {}
        import h5py
        ep_name = self._replay_ep_name()
        with h5py.File(self.context.replay_cfgs["hdf5_path"], 'r') as f:
            rigid_objects_path = f"data/{ep_name}/initial_state/rigid_object"
            articulation_objects_path = f"data/{ep_name}/initial_state/articulation"
            deformable_objects_path = f"data/{ep_name}/initial_state/deformable_object"

            object_paths = [rigid_objects_path, articulation_objects_path]
            if deformable_objects_path in f:
                object_paths.append(deformable_objects_path)

            for objects_path in object_paths:
                if objects_path not in f:
                    continue

                objects_group = f[objects_path]

                for obj_name in objects_group.keys():
                    if obj_name not in self.objects.keys():
                        continue
                    obj_group = f[f"{objects_path}/{obj_name}"]
                    if "root_pose" not in obj_group:
                        continue
                    root_pose = obj_group["root_pose"][0]
                    # Isaac Lab root_pose: [x, y, z, qx, qy, qz, qw] (xyzw)
                    objects_placement[obj_name] = (
                        tuple(root_pose[0:3].tolist()),
                        np.array(root_pose[3:7], dtype=np.float32),
                        self.objects[obj_name],
                    )
        return objects_placement

    def _load_robot_placement(self):
        """Load recorded robot root pose from HDF5 initial_state (relative to env origin)."""
        import h5py

        cached = getattr(self, "_cached_robot_placement", None)
        if cached is not None:
            return cached

        ep_name = self._replay_ep_name()
        with h5py.File(self.context.replay_cfgs["hdf5_path"], "r") as f:
            robot_path = f"data/{ep_name}/initial_state/articulation/robot"
            if robot_path not in f or "root_pose" not in f[robot_path]:
                self._cached_robot_placement = None
                return None
            root_pose = f[robot_path]["root_pose"][0]
            pos = tuple(root_pose[0:3].tolist())
            quat_xyzw = np.array(root_pose[3:7], dtype=np.float32)
            self._cached_robot_placement = (pos, quat_xyzw)
            print(
                f"[RobotPose] _load_robot_placement ep={ep_name} "
                f"hdf5_pos={pos} hdf5_quat_xyzw={quat_xyzw.tolist()}",
                flush=True,
            )
            return self._cached_robot_placement

    def _setup_kitchen_references(self, scene):
        """
        setup fixtures (and their references). this function is called within load_model function for kitchens
        """
        serialized_refs = self.context.ep_meta.get("fixture_refs", {})
        # unserialize refs
        self.fixture_refs = {
            k: self.get_fixture(v) for (k, v) in serialized_refs.items()
        }

    def get_fixture(self, id, ref=None, size=(0.2, 0.2), full_name_check=False, fix_id=None, full_depth_region=False) -> Fixture | None:
        """
        search fixture by id (name, object, or type)

        Args:
            id (str, Fixture, FixtureType): id of fixture to search for

            ref (str, Fixture, FixtureType): if specified, will search for fixture close to ref (within 0.10m)

            size (tuple): if sampling counter, minimum size (x,y) that the counter region must be

            full_depth_region (bool): if True, will only sample island counter regions that can be accessed

        Returns:
            Fixture: fixture object
        """
        # case 1: id refers to fixture object directly
        if isinstance(id, Fixture):
            return id
        # case 2: id refers to exact name of fixture
        elif id in self.fixtures.keys():
            return self.fixtures[id]

        if ref is None:
            # find all fixtures with names containing given name
            if isinstance(id, FixtureType) or isinstance(id, int):
                matches = [
                    name
                    for (name, fxtr) in self.fixtures.items()
                    if fixture_is_type(fxtr, id)
                ]
            else:
                if full_name_check:
                    matches = [name for name in self.fixtures.keys() if name == id]
                else:
                    matches = [name for name in self.fixtures.keys() if id in name]
            if id == FixtureType.COUNTER or id == FixtureType.COUNTER_NON_CORNER:
                matches = [
                    name
                    for name in matches
                    if FixtureUtils.is_fxtr_valid(self, self.fixtures[name], size)
                ]
            if (
                len(matches) > 1
                and any("island" in name for name in matches)
                and full_depth_region
            ):
                island_matches = [name for name in matches if "island" in name]
                if len(island_matches) >= 3:
                    depths = [self.fixtures[name].size[1] for name in island_matches]
                    sorted_indices = sorted(range(len(depths)), key=lambda i: depths[i])
                    min_depth = depths[sorted_indices[0]]
                    next_min_depth = (
                        depths[sorted_indices[1]] if len(depths) > 1 else min_depth
                    )
                    if min_depth < 0.8 * next_min_depth:
                        keep = [
                            i
                            for i in range(len(island_matches))
                            if i != sorted_indices[0]
                        ]
                        filtered_islands = [island_matches[i] for i in keep]
                        matches = [
                            name for name in matches if name not in island_matches
                        ] + filtered_islands

            if len(matches) == 0:
                return None
            # sample random key
            if fix_id is not None:
                key = matches[fix_id]
            else:
                key = self.rng.choice(matches)
            return self.fixtures[key]
        else:
            ref_fixture = self.get_fixture(ref)
            # assert isinstance(id, FixtureType)
            cand_fixtures: List[Fixture] = []
            for fxtr in self.fixtures.values():
                if not fixture_is_type(fxtr, id):
                    continue
                if fxtr is ref_fixture:
                    continue
                if id == FixtureType.COUNTER:
                    fxtr_is_valid = FixtureUtils.is_fxtr_valid(self, fxtr, size)
                    if not fxtr_is_valid:
                        continue
                cand_fixtures.append(fxtr)

            if len(cand_fixtures) == 0:
                raise ValueError(f"No fixture found for {id} with size {size}")

            # first, try to find fixture "containing" the reference fixture
            for fxtr in cand_fixtures:
                if OU.point_in_fixture(ref_fixture.pos, fxtr, only_2d=True):
                    return fxtr
            # if no fixture contains reference fixture, sample all close fixtures
            dists = [
                OU.fixture_pairwise_dist(ref_fixture, fxtr) for fxtr in cand_fixtures
            ]
            min_dist = np.min(dists)
            close_fixtures = [
                fxtr for (fxtr, d) in zip(cand_fixtures, dists) if d - min_dist < 0.10
            ]
            return self.rng.choice(close_fixtures)

    def register_fixture_ref(self, ref_name: str, fn_kwargs: dict = {}):
        """
        Registers a fixture reference for later use. Initializes the fixture
        if it has not been initialized yet.

        Args:
            ref_name (str): name of the reference

            fn_kwargs (dict): keyword arguments to pass to get_fixture

        Returns:
            Fixture: fixture object
        """
        existing = self.fixture_refs.get(ref_name)
        if existing is not None:
            return existing
        fixture = self.get_fixture(**fn_kwargs)
        if fixture is not None:
            self.fixture_refs[ref_name] = fixture
        return fixture

    def _resolve_fixture_object_type(self, fixtr, ref_prim):
        """Classify fixture refs from their authored physics structure."""
        articulation_root = (
            ref_prim
            if usd.is_articulation_root(ref_prim)
            else usd.find_articulation_root(ref_prim)
        )
        if articulation_root is not None:
            movable_joints = usd.get_all_joints_without_fixed(articulation_root)
            if movable_joints:
                return ObjectType.ARTICULATION, articulation_root

            # Newton exposes no bindable IsaacLab view for a zero-DOF
            # articulation: ArticulationView finds no articulation, RigidObject
            # cannot bind the fixed child body, and FrameView rejects the
            # ArticulationRootAPI prim. Keep it in the scene for physics/contact,
            # but do not create a task-level ObjectReference for it.
            if not self.backend_runtime.assets.bind_zero_dof_articulation():
                return None, articulation_root

            return ObjectType.BASE, ref_prim

        if (
            not self.backend_runtime.assets.bind_fixed_joint_rigid_body()
            and usd.get_prim_by_type(ref_prim, include_types=["PhysicsFixedJoint"])
        ):
            # Some static fixtures author only a rigid child and a FixedJoint,
            # without ArticulationRootAPI on the fixture root. Newton likewise
            # exposes no independent rigid-object view for that welded body.
            return None, ref_prim

        if usd.has_rigidbody_api(ref_prim):
            return ObjectType.RIGID, ref_prim
        if usd.has_deformable_body_api(ref_prim):
            return ObjectType.DEFORMABLE, ref_prim

        return ObjectType.BASE, ref_prim

    def _track_deformable_asset(self, name: str, **metadata) -> None:
        """Track deformables so Newton VBD solver composition runs before gym.make()."""
        self._deformable_assets[name] = {"name": name, **metadata}

    def _newton_deformables_enabled(self) -> bool:
        """Return True when YAML/CLI opted into the Newton deformable solver path."""
        return self.backend_runtime.assets.supports_deformable_solver(
            getattr(self.context, "newton_deformable_solver", None)
        )

    def _init_ref_fixtures(self):
        for fixtr in self.fixture_refs.values():
            if fixtr is None:
                continue
            # The rack is already part of the parent Scene asset. Newton collapses
            # its world-fixed articulation to a body that cannot bind to an
            # ArticulationView, RigidObject view, or FrameView.
            if isinstance(fixtr, Fixture) and fixture_is_type(fixtr, FixtureType.PALLET_TURNOVER_RACK):
                continue

            # Only classify the fixture prim itself as an articulation / rigid body.
            # Recursive checks can incorrectly promote a container prim when one of
            # its descendants has physics APIs, causing IsaacLab to look for an
            # articulation or rigid body at the wrong path.
            ref_prim = getattr(fixtr, "reference_prim", None)
            if ref_prim is None:
                ref_prim = getattr(fixtr, "prim", None)
            if ref_prim is None:
                continue
            object_type, ref_prim = self._resolve_fixture_object_type(fixtr, ref_prim)
            if object_type is None:
                continue
            if object_type == ObjectType.DEFORMABLE and not self._newton_deformables_enabled():
                continue

            # Preserve the actual prim path in the source USD instead of assuming
            # the fixture name is a direct child of /Scene. ObjectReference expects
            # the path relative to the source stage defaultPrim after /Scene.
            prim_path = ref_prim.GetPath().pathString
            default_prim = ref_prim.GetStage().GetDefaultPrim()
            if default_prim:
                default_prim_path = default_prim.GetPath().pathString
                if prim_path == default_prim_path:
                    prim_path = ""
                elif prim_path.startswith(default_prim_path + "/"):
                    prim_path = prim_path.removeprefix(default_prim_path)

            is_deformable = object_type == ObjectType.DEFORMABLE
            ref_cls = LwObjectReference if is_deformable else ObjectReference
            reference_kwargs = {"backend_assets": self.backend_runtime.assets} if is_deformable else {}
            if is_deformable:
                self._track_deformable_asset(
                    fixtr.name,
                    source="fixture_ref",
                    prim_path=prim_path,
                )
            self.add_asset(
                ref_cls(
                    name=fixtr.name,
                    object_type=object_type,
                    prim_path=f"{{ENV_REGEX_NS}}/Scene{prim_path}",
                    parent_asset=self.scene_assets['Scene'],
                    **reference_kwargs,
                )
            )
        for fixtr in self.fixture_refs.values():
            if isinstance(fixtr, Fixture):
                fixtr.setup_cfg(self)

    def _apply_object_placements(self, object_placements):
        if self.fix_object_pose_cfg is not None:
            for obj_name, obj_placement in self.object_placements.items():
                if obj_name in self.fix_object_pose_cfg:
                    obj_pos = obj_placement[0]
                    obj_rot = obj_placement[1]
                    if "pos" in self.fix_object_pose_cfg[obj_name]:
                        obj_pos = self.fix_object_pose_cfg[obj_name]["pos"]
                    if "rot" in self.fix_object_pose_cfg[obj_name]:
                        obj_rot = self.fix_object_pose_cfg[obj_name]["rot"]
                    self.object_placements[obj_name] = (obj_pos, obj_rot, obj_placement[2])

        for obj_pos, obj_quat, obj in object_placements.values():
            if obj.task_name in self.assets:
                obj_pose = Pose(position_xyz=obj_pos, rotation_wxyz=tuple(obj_quat))
                self.assets[obj.task_name].set_initial_pose(obj_pose)

    def setup_env_config(self, orchestrator):
        robot_to_fixture_dist = getattr(self.context, "robot_to_fixture_dist", None)
        if robot_to_fixture_dist is not None:
            self.robot_to_fixture_dist = float(robot_to_fixture_dist)

        self.scene_assets = orchestrator.scene.assets
        self.fixtures = orchestrator.scene.fixtures
        self.scene_retry_count = 0
        self.object_retry_count = 0
        self.is_replay_mode = orchestrator.scene.is_replay_mode
        self._setup_kitchen_references(orchestrator.scene)
        self._init_ref_fixtures()
        self._get_obj_cfgs()
        # self.object_init_offset \
        #     = csv_loader.load_object_offset(orchestrator.embodiment.name,
        #                                     orchestrator.scene.scene_name,
        #                                     orchestrator.task.task_name)
        self._create_objects()

        self.object_placements = EnvUtils.sample_object_placements(orchestrator)
        self._apply_object_placements(self.object_placements)

        resolved_pose = EnvUtils.resolve_robot_base_spawn_pose(type(self), context=self.context)
        if resolved_pose is not None:
            robot_pos_list, robot_ori_list = resolved_pose
            self.init_robot_base_pos = robot_pos_list
            self.init_robot_base_ori = robot_ori_list
            self.resample_robot_placement_on_reset = False
            robot_pos = np.asarray(self.init_robot_base_pos, dtype=np.float32)
            robot_ori = np.asarray(self.init_robot_base_ori, dtype=np.float32)
            if robot_ori.size == 3:
                robot_rot = Tn.convert_quat(Tn.mat2quat(Tn.euler2mat(robot_ori)), to="xyzw").tolist()
            elif robot_ori.size == 4:
                robot_rot = robot_ori.tolist()
            else:
                raise ValueError("init_robot_base_ori must have 3 or 4 values")
            orchestrator.embodiment.scene_config.robot.init_state.pos = tuple(float(v) for v in robot_pos)
            orchestrator.embodiment.scene_config.robot.init_state.rot = tuple(float(v) for v in robot_rot)
            if os.environ.get("LW_QUAT_DIAG", "").strip() == "1":
                print(
                    "[QUAT_DIAG] BaseTask.setup_env_config "
                    f"resolved_pos={self.init_robot_base_pos} "
                    f"resolved_ori={self.init_robot_base_ori} "
                    f"written_init_state_pos={orchestrator.embodiment.scene_config.robot.init_state.pos} "
                    f"written_init_state_rot={orchestrator.embodiment.scene_config.robot.init_state.rot}",
                    flush=True,
                )

        OU.register_declared_contact_sensors(self)

    def get_ep_meta(self):
        ep_meta = {}
        ep_meta["task_name"] = self.task_name
        ep_meta["object_cfgs"] = [copy_dict_for_json(cfg) for cfg in self.object_cfgs]
        # serialize np arrays to lists
        for cfg in ep_meta["object_cfgs"]:
            if cfg.get("reset_region", None) is not None:
                if isinstance(cfg["reset_region"], dict):
                    cfg["reset_region"] = [cfg["reset_region"]]
                for region in cfg["reset_region"]:
                    for (k, v) in region.items():
                        if isinstance(v, np.ndarray):
                            region[k] = list(v)

        ep_meta["lang"] = ""
        ep_meta["usd_simplify"] = self.context.usd_simplify
        ep_meta["objects_version"] = self.objects_version
        ep_meta["source"] = self.context.sources
        ep_meta["object_projects"] = self.context.object_projects
        ep_meta["seed"] = self.context.seed
        if self.init_robot_base_pos is not None:
            ep_meta["init_robot_base_pos"] = list(self.init_robot_base_pos)
        if self.init_robot_base_ori is not None:
            ep_meta["init_robot_base_ori"] = list(self.init_robot_base_ori)
        return ep_meta

    def get_mimic_env_cfg(self):
        return None

    def get_prompt(self):
        return self.get_ep_meta()["lang"]

    def _setup_scene(self, env, env_ids=None):
        pass


class TestAssetTask(LwTaskBase):
    def __init__(self):
        super().__init__()
