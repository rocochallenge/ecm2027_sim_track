"""Sequential teleoperation task for chamber service and part replacement."""

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab_arena.assets.asset import Asset
from scipy.spatial.transform import Rotation as R

from robofinals.data import robofinals_DATA_PATH
import robofinals.utils.object_utils as OU
from robofinals_tasks.lightwheel_home_tasks.home_base_task import HomeBaseTask
from robofinals.core.models.fixtures.fixture_types import FixtureType


class ChamberServiceTaskAssets(Asset):
    """Task-local rigid objects registered before scene composition."""

    SPONGE_BASE_POS = (0.37, -0.50, 0.78)
    SPONGE_BASE_ROT_XYZW = (1.0, 0.0, 0.0, 0.0)
    SPONGE_POS_JITTER = 0.05

    def __init__(self):
        super().__init__(name="ChamberServiceTaskAssets")
        self.object_cfg = {
            "spongebody": self._make_sponge_cfg(
                self.SPONGE_BASE_POS,
                self.SPONGE_BASE_ROT_XYZW,
            )
        }

    @classmethod
    def sample_sponge_init_state(cls, rng) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Sample sponge pose near the robot-facing table edge with the black face down."""
        base = np.asarray(cls.SPONGE_BASE_POS, dtype=np.float64)
        pos = base + rng.uniform(-cls.SPONGE_POS_JITTER, cls.SPONGE_POS_JITTER, size=3)
        rotation = R.from_euler("z", rng.uniform(0.0, 2.0 * np.pi)) * R.from_quat(cls.SPONGE_BASE_ROT_XYZW)
        rot_xyzw = tuple(float(value) for value in rotation.as_quat())
        return (float(pos[0]), float(pos[1]), float(pos[2])), rot_xyzw

    def resample_sponge_init_state(self, rng) -> None:
        pos, rot = self.sample_sponge_init_state(rng)
        self.object_cfg["spongebody"].init_state.pos = pos
        self.object_cfg["spongebody"].init_state.rot = rot

    @classmethod
    def _make_sponge_cfg(cls, pos, rot_xyzw) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sponge/SpongeBody",
            spawn=UsdFileCfg(
                usd_path=str(
                    robofinals_DATA_PATH
                    / "assets"
                    / "roco_chamber_service"
                    / "Sponge027"
                    / "Sponge027.usd"
                ),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=4.0,
                    linear_damping=0.5,
                    angular_damping=0.5,
                    sleep_threshold=0.1,
                    stabilization_threshold=0.02,
                    solver_position_iteration_count=48,
                    solver_velocity_iteration_count=8,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=pos,
                rot=rot_xyzw,
            ),
        )

    def get_object_cfg(self):
        return self.object_cfg


class ChamberServiceTask(HomeBaseTask):
    """Open, extract, pack, wipe, replace, and close the chamber."""

    task_name = "ChamberServiceTask"
    movable_fixtures = ["Chamber002", "Chamber002_01"]
    spongebody_contact_prim_path = "{ENV_REGEX_NS}/Sponge/SpongeBody/Sponge027"
    pairwise_contacts = (("spongebody", "chamber"),)

    LID_OPEN_THRESHOLD = 0.4
    SPONGE_CHAMBER_CONTACTS_REQUIRED = 3
    SPONGE_CHAMBER_CONTACT_DURATION_S = 0.5
    BOX_PLACEMENT_THRESHOLD = 0.15
    CHAMBER_PLACEMENT_THRESHOLD = 0.4

    success_stages = (
        "1. Lift off chamber top",
        "2. Extract large part",
        "3. Pack part into box",
        "4. Wiping of the chamber",
        "5. Unbox & place replacement",
        "6. Close up chamber",
    )
    _stage_names = success_stages

    def __init__(self):
        super().__init__()
        self._stage_flags = None
        self._process_score = None
        self._sponge_chamber_contact_count = None
        self._sponge_chamber_contact_time = None
        self._sponge_chamber_contact_qualified = None
        self._missing_asset_reported = False
        self._task_assets: ChamberServiceTaskAssets | None = None

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.old_part = self._register_fixture_by_name("old_part", "Chamber002")
        self.new_part = self._register_fixture_by_name("new_part", "Chamber002_01")
        self.chamber = self._register_fixture_by_name("chamber", "Chamber001_01", fallback_type=FixtureType.CHAMBER)
        self.empty_box = self._register_fixture_by_name("empty_box", "Box291")
        self.full_box = self._register_fixture_by_name("full_box", "Box291_01")
        self.table = self._register_fixture_by_name("table", "Table291")
        self.init_robot_base_ref = self.empty_box

    def modify_env_cfg(self, env_cfg):
        env_cfg = super().modify_env_cfg(env_cfg)
        env_cfg.episode_length_s = 300.0
        # env_cfg.viewer.eye = (2.6, -3.0, 2.1)
        # env_cfg.viewer.lookat = (0.0, 0.0, 0.55)
        return env_cfg

    def _apply_sponge_init_to_sim(self, env, env_ids=None) -> None:
        if self._task_assets is None or "spongebody" not in env.scene.keys():
            return
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        init_state = self._task_assets.object_cfg["spongebody"].init_state
        pos = torch.tensor(init_state.pos, device=env.device, dtype=torch.float32)
        quat_xyzw = torch.tensor(
            init_state.rot,
            device=env.device,
            dtype=torch.float32,
        )
        root_poses = []
        for env_id in env_ids.tolist():
            world_pos = pos + env.scene.env_origins[env_id]
            root_poses.append(torch.cat([world_pos, quat_xyzw], dim=0))
        root_pose = torch.stack(root_poses, dim=0)
        env.scene["spongebody"].write_root_pose_to_sim(root_pose, env_ids=env_ids)

    def setup_env_config(self, orchestrator):
        self._task_assets = ChamberServiceTaskAssets()
        self._task_assets.resample_sponge_init_state(self.rng)
        self.add_asset(self._task_assets)
        return super().setup_env_config(orchestrator)

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        if self._task_assets is not None:
            self._task_assets.resample_sponge_init_state(self.rng)
            self._apply_sponge_init_to_sim(env, env_ids)
        self._ensure_state_buffers(env)
        self._stage_flags.zero_()
        self._process_score.zero_()
        self._sponge_chamber_contact_count.zero_()
        self._sponge_chamber_contact_time.zero_()
        self._sponge_chamber_contact_qualified.zero_()

    def _ensure_state_buffers(self, env):
        if (
            self._stage_flags is not None
            and self._stage_flags.shape[0] == env.num_envs
            and self._stage_flags.device == torch.device(env.device)
        ):
            return
        n = env.num_envs
        device = env.device
        self._stage_flags = torch.zeros(n, len(self._stage_names), dtype=torch.bool, device=device)
        self._process_score = torch.zeros(n, dtype=torch.long, device=device)
        self._sponge_chamber_contact_count = torch.zeros(n, dtype=torch.long, device=device)
        self._sponge_chamber_contact_time = torch.zeros(n, dtype=torch.float32, device=device)
        self._sponge_chamber_contact_qualified = torch.zeros(n, dtype=torch.bool, device=device)

    def _reset_episode_state(self, env):
        reset_ids = env.episode_length_buf <= 1
        if not torch.any(reset_ids):
            return
        self._stage_flags[reset_ids] = False
        self._process_score[reset_ids] = 0
        self._sponge_chamber_contact_count[reset_ids] = 0
        self._sponge_chamber_contact_time[reset_ids] = 0.0
        self._sponge_chamber_contact_qualified[reset_ids] = False

    def _complete_stage(self, stage_idx: int, condition: torch.Tensor) -> None:
        newly_done = condition & ~self._stage_flags[:, stage_idx]
        self._stage_flags[:, stage_idx] |= condition
        self._process_score += newly_done.long()

    def _lid_open(self, env) -> torch.Tensor:
        if self.chamber is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return self.chamber.is_lid_open(env, th=self.LID_OPEN_THRESHOLD)

    def _lid_closed(self, env) -> torch.Tensor:
        if self.chamber is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return self.chamber.is_lid_closed(env)

    def _old_part_in_chamber(self, env) -> torch.Tensor:
        if self.old_part is None or self.chamber is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return OU.check_obj_in_receptacle_no_contact(
            env,
            "Chamber002",
            "Chamber001_01",
            th=self.CHAMBER_PLACEMENT_THRESHOLD,
        )

    def _old_part_extracted(self, env) -> torch.Tensor:
        return ~self._old_part_in_chamber(env)

    def _old_part_in_empty_box(self, env) -> torch.Tensor:
        if self.old_part is None or self.empty_box is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return OU.check_obj_in_receptacle_no_contact(
            env,
            "Chamber002",
            "Box291",
            th=self.BOX_PLACEMENT_THRESHOLD,
        )

    def _new_part_in_chamber(self, env) -> torch.Tensor:
        if self.new_part is None or self.chamber is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return OU.check_obj_in_receptacle_no_contact(
            env,
            "Chamber002_01",
            "Chamber001_01",
            th=self.CHAMBER_PLACEMENT_THRESHOLD,
        )

    def _update_sponge_chamber_contacts(self, env, active: torch.Tensor) -> torch.Tensor:
        if "spongebody" not in env.scene.keys():
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        contact_now = self.check_declared_contact(env, ("spongebody", "chamber")) & active
        step_dt = float(getattr(env, "step_dt", getattr(env, "physics_dt", 0.0)))
        self._sponge_chamber_contact_time = torch.where(
            contact_now,
            self._sponge_chamber_contact_time + step_dt,
            torch.zeros_like(self._sponge_chamber_contact_time),
        )
        newly_qualified = (
            contact_now
            & (self._sponge_chamber_contact_time >= self.SPONGE_CHAMBER_CONTACT_DURATION_S)
            & ~self._sponge_chamber_contact_qualified
        )
        self._sponge_chamber_contact_count += newly_qualified.long()
        self._sponge_chamber_contact_qualified = torch.where(
            contact_now,
            self._sponge_chamber_contact_qualified | newly_qualified,
            torch.zeros_like(self._sponge_chamber_contact_qualified),
        )
        return self._sponge_chamber_contact_count >= self.SPONGE_CHAMBER_CONTACTS_REQUIRED

    def _check_success(self, env):
        self._ensure_state_buffers(env)
        self._reset_episode_state(env)

        if "spongebody" not in env.scene.keys():
            if not self._missing_asset_reported:
                print("[ChamberServiceTask] missing scene asset: spongebody", flush=True)
                self._missing_asset_reported = True
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        flags = self._stage_flags

        # Step 1: lift the chamber top.
        self._complete_stage(0, self._lid_open(env))

        # Step 2: extract the large part from the chamber.
        old_part_extracted = self._old_part_extracted(env)
        self._complete_stage(1, flags[:, 0] & old_part_extracted)

        # Step 3: pack the extracted part into the empty box.
        old_part_in_empty_box = self._old_part_in_empty_box(env)
        self._complete_stage(2, flags[:, 1] & old_part_in_empty_box)

        # Step 4: wipe the chamber after the original part is packed.
        wipe_active = flags[:, 2] & ~flags[:, 3]
        chamber_wiped = self._update_sponge_chamber_contacts(env, wipe_active)
        self._complete_stage(3, flags[:, 2] & chamber_wiped)

        # Step 5: unbox the replacement and place it in the chamber.
        new_part_in_chamber = self._new_part_in_chamber(env)
        self._complete_stage(4, flags[:, 3] & new_part_in_chamber)

        # Step 6: close and lock the chamber.
        self._complete_stage(5, flags[:, 4] & self._lid_closed(env))

        success = torch.all(flags, dim=1)
        return success.reshape(env.num_envs)

    def get_process_score_metrics(self):
        """Task progress snapshot for teleop/export (not IsaacLab Arena metric recorders)."""
        if self._process_score is None or self._stage_flags is None:
            return {"process_score": 0, "completed_stages": []}
        completed = [
            stage_name
            for stage_name, completed_flag in zip(self.success_stages, self.get_success_stage_flags())
            if completed_flag
        ]
        return {
            "process_score": int(self._process_score.max().item()),
            "process_score_per_env": self._process_score.detach().cpu().tolist(),
            "completed_stages": completed,
        }

    def get_success_stage_flags(self) -> list[bool]:
        if self._stage_flags is None:
            return [False] * len(self.success_stages)
        return self._stage_flags[0].detach().cpu().tolist()

    def get_checker_results(self):
        results = super().get_checker_results()
        results["chamber_process"] = self.get_process_score_metrics()
        return results

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Lift the chamber top, extract the large part, pack it into the box, wipe "
            "the chamber, unbox and place the replacement, then close and lock the chamber."
        )
        ep_meta["success_stages"] = list(self._stage_names)
        ep_meta["object_cfgs"] = {}
        return ep_meta
