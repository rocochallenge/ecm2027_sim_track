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
from robofinals.utils.env import ExecuteMode
from robofinals.utils.place_utils.env_utils import sample_robot_base_helper


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

    # ------------------------------------------------------------------
    # Official 13-point scoring (Roco27 SGEAIC scoring table, 2026-11-19).
    # Stages are evaluated in order: a stage is only assessed after every
    # scoring point of the previous stage has been credited.
    #   1. Lift off chamber top      2 pts  (lid lifted 1 cm; lid stably open 1 s)
    #   2. Extract large part        1 pt   (part lifted 1 cm)
    #   3. Pack part into box        1 pt   (part resting inside the box, stable)
    #   4. Wiping of the chamber     6 pts  (sponge picked up; 5 faces x 1, any order)
    #   5. Unbox & place replacement 2 pts  (lifted 1 cm; stably placed)
    #   6. Close up chamber          1 pt   (lid stably closed)
    # All thresholds below are in SI units (m, s, rad).
    # ------------------------------------------------------------------
    MAX_SCORE = 13
    # Lid (hinged about its rear edge; normalized joint value 0 = closed, 1 = open).
    LID_LENGTH_M = 0.698  # hinge axis to front edge, from scene04_1.usd
    LID_LIFT_HEIGHT_M = 0.01
    LID_OPEN_THRESHOLD = 0.4
    LID_CLOSED_THRESHOLD = 0.005
    LID_STABLE_ANG_VEL = 0.1
    # "Stable" = (nearly) motionless for this long.
    STABLE_DURATION_S = 1.0
    STABLE_LIN_VEL = 0.05
    # Pick-up heights above the object's resting height.
    PART_LIFT_HEIGHT_M = 0.01
    SPONGE_LIFT_HEIGHT_M = 0.01
    # The original part's resting pose is latched this long after reset. It is
    # the centre of the chamber cavity (XY) and the replacement target pose.
    REFERENCE_SETTLE_S = 0.5
    # Empty packing box (static, fixed to the table). Its geometry is read from the
    # scene fixture; these env-local values from scene04_1.usd are only a fallback.
    BOX_CENTER_XY = (0.6979, -0.4223)
    BOX_HALF_SIZE_XY = 0.175
    BOX_TOP_Z = 0.8049
    # Replacement is "placed" when its centre is this close to the original part's resting pose.
    NEW_PART_PLACE_DISTANCE_M = 0.05
    # Chamber cavity (collision geometry), relative to the original part's resting
    # pose. Measured in simulation on 2026-11-19 by pressing the sponge on each face.
    CHAMBER_INNER_HALF_X = 0.344
    CHAMBER_INNER_HALF_Y = 0.347
    CHAMBER_FLOOR_BELOW_PART_M = 0.183
    CHAMBER_INNER_HEIGHT_M = 0.30
    # Wiping: a face is credited once after this much continuous sponge contact
    # while that face is the nearest cavity face to the sponge centre.
    SPONGE_CHAMBER_CONTACT_DURATION_S = 0.5
    # The sponge centre must be inside the cavity (the walls are thin, so a loose
    # tolerance would also accept contact on the outside of a wall).
    WIPE_FACE_MAX_DISTANCE_M = 0.08
    WIPE_FACE_OUTSIDE_TOLERANCE_M = 0.01
    WIPE_FACE_NAMES = ("wall_x_neg", "wall_x_pos", "wall_y_neg", "wall_y_pos", "floor")
    SPONGE_CHAMBER_CONTACTS_REQUIRED = 5  # number of cavity faces
    ROBOT_RESET_POS_DEV_X = 0.10
    ROBOT_RESET_POS_DEV_Y = 0.10
    # About +/- 29 degrees, large enough to be visible while keeping the
    # robot facing the same work area.
    ROBOT_RESET_YAW_DEV = 0.50

    POINT_NAMES = (
        "1a_lid_lifted_1cm",
        "1b_lid_stable_open",
        "2_part_lifted_1cm",
        "3_part_in_box_stable",
        "4a_sponge_picked_up",
        "4b_wipe_wall_x_neg",
        "4b_wipe_wall_x_pos",
        "4b_wipe_wall_y_neg",
        "4b_wipe_wall_y_pos",
        "4b_wipe_floor",
        "5a_new_part_lifted_1cm",
        "5b_new_part_placed_stable",
        "6_lid_closed_stable",
    )
    # [start, end) indices into POINT_NAMES for each of the six stages.
    STAGE_POINT_SLICES = ((0, 2), (2, 3), (3, 4), (4, 10), (10, 12), (12, 13))
    _WIPE_SLICE = (5, 10)
    MAX_FINISHED_EPISODES = 200

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
        self._point_flags = None
        self._score_steps = None
        self._hold_time = None
        self._face_time = None
        self._ref_valid = None
        self._old_part_ref = None
        self._new_part_min_z = None
        self._sponge_min_z = None
        self._prev_valid = None
        self._prev_old_pos = None
        self._prev_new_pos = None
        self._prev_lid_angle = None
        self._sponge_chamber_contact_count = None
        self._step_dt = 0.0
        self._finished_episodes: list[dict] = []
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
        # Benchmark timeout, measured in simulation seconds.
        env_cfg.episode_length_s = 15.0 * 60.0
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
        result = super().setup_env_config(orchestrator)
        # Use the same randomized reset distribution for collection and eval.
        self.resample_robot_placement_on_reset = True
        return result

    def sample_robot_base_reset_pose(self, env, env_ids, orchestrator):
        """Return a collision-free random base pose around the task anchor."""
        anchor_pos = np.asarray(
            self.init_robot_base_pos
            if self.init_robot_base_pos is not None
            else orchestrator.embodiment.init_robot_base_pos_anchor,
            dtype=np.float32,
        )
        anchor_ori = np.asarray(
            self.init_robot_base_ori
            if self.init_robot_base_ori is not None
            else orchestrator.embodiment.init_robot_base_ori_anchor,
            dtype=np.float32,
        )
        robot_pos = sample_robot_base_helper(
            env=env,
            anchor_pos=anchor_pos,
            anchor_ori=anchor_ori,
            rot_dev=self.ROBOT_RESET_YAW_DEV,
            pos_dev_x=self.ROBOT_RESET_POS_DEV_X,
            pos_dev_y=self.ROBOT_RESET_POS_DEV_Y,
            env_ids=env_ids,
            # The helper intentionally fixes EVAL poses for generic tasks. This
            # task requires eval and data collection to share the distribution.
            execute_mode=ExecuteMode.TELEOP,
        )
        robot_ori = anchor_ori.copy()
        robot_ori[2] += float(self.rng.uniform(-self.ROBOT_RESET_YAW_DEV, self.ROBOT_RESET_YAW_DEV))
        return robot_pos.tolist(), robot_ori.tolist()

    def _setup_scene(self, env, env_ids=None):
        # Runs on every environment reset. Keep the finished episode's result
        # before its scoring state is cleared.
        reason = getattr(env, "_forced_episode_end_reason", None) or "reset"
        self._snapshot_finished_episode(reason, env_ids, env=env)
        super()._setup_scene(env, env_ids)
        if self._task_assets is not None:
            self._task_assets.resample_sponge_init_state(self.rng)
            self._apply_sponge_init_to_sim(env, env_ids)
        self._ensure_state_buffers(env)
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if env_ids is None:
            mask[:] = True
        else:
            mask[torch.as_tensor(env_ids, device=env.device, dtype=torch.long)] = True
        self._clear_scoring_state(mask)

    def _ensure_state_buffers(self, env):
        if (
            self._point_flags is not None
            and self._point_flags.shape[0] == env.num_envs
            and self._point_flags.device == torch.device(env.device)
        ):
            return
        n = env.num_envs
        device = env.device
        self._point_flags = torch.zeros(n, len(self.POINT_NAMES), dtype=torch.bool, device=device)
        self._stage_flags = torch.zeros(n, len(self._stage_names), dtype=torch.bool, device=device)
        self._process_score = torch.zeros(n, dtype=torch.long, device=device)
        self._sponge_chamber_contact_count = torch.zeros(n, dtype=torch.long, device=device)
        self._score_steps = torch.zeros(n, dtype=torch.long, device=device)
        # Hold timers: 0 lid open, 1 part in box, 2 replacement placed, 3 lid closed.
        self._hold_time = torch.zeros(n, 4, dtype=torch.float32, device=device)
        self._face_time = torch.zeros(n, len(self.WIPE_FACE_NAMES), dtype=torch.float32, device=device)
        self._ref_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._old_part_ref = torch.zeros(n, 3, dtype=torch.float32, device=device)
        self._new_part_min_z = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
        self._sponge_min_z = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
        self._prev_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._prev_old_pos = torch.zeros(n, 3, dtype=torch.float32, device=device)
        self._prev_new_pos = torch.zeros(n, 3, dtype=torch.float32, device=device)
        self._prev_lid_angle = torch.zeros(n, dtype=torch.float32, device=device)

    def _clear_scoring_state(self, env_mask: torch.Tensor | None = None) -> None:
        """Zero the per-episode scoring state (all envs, or the masked ones)."""
        if self._point_flags is None:
            return
        if env_mask is None:
            env_mask = torch.ones(self._point_flags.shape[0], dtype=torch.bool, device=self._point_flags.device)
        self._point_flags[env_mask] = False
        self._stage_flags[env_mask] = False
        self._process_score[env_mask] = 0
        self._sponge_chamber_contact_count[env_mask] = 0
        self._score_steps[env_mask] = 0
        self._hold_time[env_mask] = 0.0
        self._face_time[env_mask] = 0.0
        self._ref_valid[env_mask] = False
        self._old_part_ref[env_mask] = 0.0
        self._new_part_min_z[env_mask] = float("inf")
        self._sponge_min_z[env_mask] = float("inf")
        self._prev_valid[env_mask] = False

    def _reset_episode_state(self, env):
        reset_ids = env.episode_length_buf <= 1
        if not torch.any(reset_ids):
            return
        # Normally the reset event has already archived and cleared the state;
        # this also covers resets that bypass the reset event.
        if bool((self._score_steps[reset_ids] > 0).any()):
            self._snapshot_finished_episode("reset", reset_ids)
        self._clear_scoring_state(reset_ids)

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------
    def _asset_pos_w(self, env, asset_name: str) -> torch.Tensor:
        pos = OU._scene_asset_pos_w(env, asset_name)
        if pos is None:
            pos = OU.body_com_pos_w_per_env_first_link(env.scene[asset_name].data)
        return pos.to(dtype=torch.float32)

    def _lid_state(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (normalized opening in [0, 1], opening angle in rad) of the chamber lid."""
        if self.chamber is None:
            zeros = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
            return zeros, zeros.clone()
        joint_name = self.chamber.lid_joint_names[0]
        norm = self.chamber.get_joint_state(env, [joint_name])[joint_name].to(dtype=torch.float32)
        norm = norm.reshape(env.num_envs)
        articulation = env.scene.articulations[self.chamber.name]
        joint_idx = articulation.data.joint_names.index(joint_name)
        limits = OU._asset_data_to_torch(articulation.data.joint_pos_limits)[0, joint_idx, :]
        span = float(limits[1] - limits[0])
        if not np.isfinite(span) or span <= 0.0:
            span = float(np.pi / 2.0)
        angle = torch.clamp(norm, 0.0, 1.0) * span
        return norm, angle

    def _box_geometry(self) -> tuple[float, float, float, float]:
        """(centre x, centre y, half size, rim height) of the empty box, env-local frame."""
        box = getattr(self, "empty_box", None)
        try:
            pos = np.asarray(getattr(box, "pos", None), dtype=np.float64).reshape(-1)
            size = np.asarray(getattr(box, "size", None), dtype=np.float64).reshape(-1)
            if pos.size >= 3 and size.size >= 3 and float(np.min(size[:3])) > 0.0:
                half = 0.5 * float(min(size[0], size[1]))
                return float(pos[0]), float(pos[1]), half, float(pos[2]) + 0.5 * float(size[2])
        except (TypeError, ValueError):
            pass
        return self.BOX_CENTER_XY[0], self.BOX_CENTER_XY[1], self.BOX_HALF_SIZE_XY, self.BOX_TOP_Z

    def _part_inside_box(self, env, part_pos_w: torch.Tensor) -> torch.Tensor:
        """Part centre is within the empty box footprint and below the box rim."""
        center_x, center_y, half, rim_z = self._box_geometry()
        local = part_pos_w - env.scene.env_origins.to(dtype=torch.float32)
        inside_xy = (torch.abs(local[:, 0] - center_x) <= half) & (torch.abs(local[:, 1] - center_y) <= half)
        return inside_xy & (local[:, 2] < rim_z)

    def _award(self, point_idx: int, condition: torch.Tensor) -> None:
        """Latch a scoring point; a point can be credited only once per episode."""
        self._point_flags[:, point_idx] |= condition

    def _hold(self, timer_idx: int, condition: torch.Tensor) -> torch.Tensor:
        """Accumulate how long ``condition`` has held continuously (seconds)."""
        self._hold_time[:, timer_idx] = torch.where(
            condition,
            self._hold_time[:, timer_idx] + self._step_dt,
            torch.zeros_like(self._hold_time[:, timer_idx]),
        )
        return self._hold_time[:, timer_idx]

    def _stage_points_done(self, stage_idx: int) -> torch.Tensor:
        start, end = self.STAGE_POINT_SLICES[stage_idx]
        return torch.all(self._point_flags[:, start:end], dim=1)

    def _update_wipe_faces(self, env, sponge_pos_w: torch.Tensor, active: torch.Tensor) -> None:
        """Credit each cavity face once after sustained sponge contact on that face.

        The contact sensor only reports sponge-vs-chamber-body contact, so the
        face is taken to be the cavity face nearest to the sponge centre.
        """
        contact = self.check_declared_contact(env, ("spongebody", "chamber")) & active
        rel = sponge_pos_w - self._old_part_ref
        height = rel[:, 2] + self.CHAMBER_FLOOR_BELOW_PART_M
        distances = torch.stack(
            (
                rel[:, 0] + self.CHAMBER_INNER_HALF_X,
                self.CHAMBER_INNER_HALF_X - rel[:, 0],
                rel[:, 1] + self.CHAMBER_INNER_HALF_Y,
                self.CHAMBER_INNER_HALF_Y - rel[:, 1],
                height,
            ),
            dim=1,
        )
        nearest_dist, nearest_face = torch.min(distances, dim=1)
        in_cavity = (nearest_dist >= -self.WIPE_FACE_OUTSIDE_TOLERANCE_M) & (height <= self.CHAMBER_INNER_HEIGHT_M)
        touching = contact & in_cavity & (nearest_dist <= self.WIPE_FACE_MAX_DISTANCE_M)
        face_mask = torch.nn.functional.one_hot(nearest_face, num_classes=len(self.WIPE_FACE_NAMES)).to(torch.bool)
        face_mask &= touching.unsqueeze(1)
        self._face_time = torch.where(face_mask, self._face_time + self._step_dt, torch.zeros_like(self._face_time))
        start, end = self._WIPE_SLICE
        self._point_flags[:, start:end] |= face_mask & (self._face_time >= self.SPONGE_CHAMBER_CONTACT_DURATION_S)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _check_success(self, env):
        self._ensure_state_buffers(env)
        self._reset_episode_state(env)

        if "spongebody" not in env.scene.keys():
            if not self._missing_asset_reported:
                print("[ChamberServiceTask] missing scene asset: spongebody", flush=True)
                self._missing_asset_reported = True
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        dt = float(getattr(env, "step_dt", getattr(env, "physics_dt", 0.0)))
        self._step_dt = dt
        self._score_steps += 1
        points = self._point_flags

        # --- measurements -------------------------------------------------
        lid_norm, lid_angle = self._lid_state(env)
        old_pos = self._asset_pos_w(env, "Chamber002")
        new_pos = self._asset_pos_w(env, "Chamber002_01")
        sponge_pos = self._asset_pos_w(env, "spongebody")

        # Speeds by finite differences; unknown (treated as moving) on the first sample.
        inf = torch.full((env.num_envs,), float("inf"), dtype=torch.float32, device=env.device)
        if dt > 0.0:
            old_speed = torch.where(self._prev_valid, torch.linalg.vector_norm(old_pos - self._prev_old_pos, dim=1) / dt, inf)
            new_speed = torch.where(self._prev_valid, torch.linalg.vector_norm(new_pos - self._prev_new_pos, dim=1) / dt, inf)
            lid_rate = torch.where(self._prev_valid, torch.abs(lid_angle - self._prev_lid_angle) / dt, inf)
        else:
            old_speed, new_speed, lid_rate = inf, inf.clone(), inf.clone()
        self._prev_old_pos[:] = old_pos
        self._prev_new_pos[:] = new_pos
        self._prev_lid_angle[:] = lid_angle
        self._prev_valid[:] = True

        # Resting heights of the replacement part and the sponge.
        self._new_part_min_z = torch.minimum(self._new_part_min_z, new_pos[:, 2])
        self._sponge_min_z = torch.minimum(self._sponge_min_z, sponge_pos[:, 2])

        # Latch the original part's resting pose shortly after reset.
        settle_steps = max(1, int(round(self.REFERENCE_SETTLE_S / dt))) if dt > 0.0 else 1
        capture = (~self._ref_valid) & (self._score_steps >= settle_steps)
        self._old_part_ref[capture] = old_pos[capture]
        self._ref_valid |= capture
        ref_ok = self._ref_valid

        # --- Stage 1: lift off chamber top ----------------------------------
        lid_rise = self.LID_LENGTH_M * torch.sin(lid_angle)
        self._award(0, lid_rise >= self.LID_LIFT_HEIGHT_M)
        lid_open_still = (lid_norm >= self.LID_OPEN_THRESHOLD) & (lid_rate <= self.LID_STABLE_ANG_VEL)
        held = self._hold(0, points[:, 0] & lid_open_still)
        self._award(1, points[:, 0] & (held >= self.STABLE_DURATION_S))
        stage_1 = self._stage_points_done(0)

        # --- Stage 2: extract large part -------------------------------------
        lifted = old_pos[:, 2] >= self._old_part_ref[:, 2] + self.PART_LIFT_HEIGHT_M
        self._award(2, stage_1 & ref_ok & lifted)
        stage_2 = stage_1 & self._stage_points_done(1)

        # --- Stage 3: pack part into box --------------------------------------
        in_box_still = self._part_inside_box(env, old_pos) & (old_speed <= self.STABLE_LIN_VEL)
        held = self._hold(1, stage_2 & in_box_still)
        self._award(3, stage_2 & (held >= self.STABLE_DURATION_S))
        stage_3 = stage_2 & self._stage_points_done(2)

        # --- Stage 4: wipe the chamber -----------------------------------------
        self._award(4, stage_3 & (sponge_pos[:, 2] >= self._sponge_min_z + self.SPONGE_LIFT_HEIGHT_M))
        self._update_wipe_faces(env, sponge_pos, active=stage_3 & points[:, 4] & ref_ok)
        stage_4 = stage_3 & self._stage_points_done(3)

        # --- Stage 5: unbox and place replacement -------------------------------
        self._award(10, stage_4 & (new_pos[:, 2] >= self._new_part_min_z + self.PART_LIFT_HEIGHT_M))
        at_target = torch.linalg.vector_norm(new_pos - self._old_part_ref, dim=1) <= self.NEW_PART_PLACE_DISTANCE_M
        placed_still = ref_ok & at_target & (new_speed <= self.STABLE_LIN_VEL)
        held = self._hold(2, stage_4 & points[:, 10] & placed_still)
        self._award(11, stage_4 & points[:, 10] & (held >= self.STABLE_DURATION_S))
        stage_5 = stage_4 & self._stage_points_done(4)

        # --- Stage 6: close up chamber --------------------------------------------
        lid_closed_still = (lid_norm <= self.LID_CLOSED_THRESHOLD) & (lid_rate <= self.LID_STABLE_ANG_VEL)
        held = self._hold(3, stage_5 & lid_closed_still)
        self._award(12, stage_5 & (held >= self.STABLE_DURATION_S))
        stage_6 = stage_5 & self._stage_points_done(5)

        # --- derived state ------------------------------------------------------------
        self._stage_flags[:] = torch.stack((stage_1, stage_2, stage_3, stage_4, stage_5, stage_6), dim=1)
        self._process_score[:] = points.sum(dim=1).to(torch.long)
        start, end = self._WIPE_SLICE
        self._sponge_chamber_contact_count[:] = points[:, start:end].sum(dim=1).to(torch.long)

        success = torch.all(self._stage_flags, dim=1)
        return success.reshape(env.num_envs)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _episode_metrics(self) -> dict:
        """Score breakdown of the episode currently held in the scoring buffers."""
        points = self._point_flags.detach().cpu()
        stage_flags = self._stage_flags.detach().cpu()
        env0_points = [bool(v) for v in points[0].tolist()]
        stage_scores = {}
        for stage_name, (start, end) in zip(self._stage_names, self.STAGE_POINT_SLICES):
            stage_scores[stage_name] = int(sum(env0_points[start:end]))
        start, end = self._WIPE_SLICE
        steps = int(self._score_steps.max().item())
        return {
            "process_score": int(self._process_score.max().item()),
            "process_score_per_env": [int(v) for v in self._process_score.detach().cpu().tolist()],
            "max_score": self.MAX_SCORE,
            "points": dict(zip(self.POINT_NAMES, env0_points)),
            "stage_scores": stage_scores,
            "wipe_faces_completed_per_env": [int(v) for v in points[:, start:end].sum(dim=1).tolist()],
            "completed_stages": [name for name, done in zip(self._stage_names, stage_flags[0].tolist()) if done],
            "success": bool(stage_flags[0].all().item()),
            "episode_steps": steps,
            "episode_sim_time_s": round(steps * float(self._step_dt), 3),
        }

    def _snapshot_finished_episode(self, reason: str, env_ids=None, env=None) -> None:
        """Archive only the slots being reset, before their scores are cleared."""
        if self._point_flags is None or self._score_steps is None:
            return
        if env_ids is None:
            ids = range(self._point_flags.shape[0])
        else:
            ids_tensor = torch.as_tensor(env_ids, device=self._point_flags.device)
            if ids_tensor.dtype == torch.bool:
                ids_tensor = ids_tensor.nonzero(as_tuple=False).flatten()
            ids = ids_tensor.flatten().detach().cpu().tolist()
        for env_id in ids:
            env_id = int(env_id)
            steps = int(self._score_steps[env_id].item())
            if steps < 1 and reason != "wall_clock_timeout":
                continue
            points = [bool(v) for v in self._point_flags[env_id].detach().cpu().tolist()]
            stages = [bool(v) for v in self._stage_flags[env_id].detach().cpu().tolist()]
            wipe_start, wipe_end = self._WIPE_SLICE
            terminated = truncated = False
            if env is not None and getattr(env, "reset_buf", None) is not None and bool(env.reset_buf[env_id]):
                terminated = bool(env.termination_manager.terminated[env_id])
                truncated = bool(env.termination_manager.time_outs[env_id])
            if reason == "wall_clock_timeout":
                terminated, truncated = False, True
                end_reason = "wall_clock_timeout"
            elif terminated and truncated:
                end_reason = "terminated_and_truncated"
            elif terminated:
                end_reason = "terminated"
            elif truncated:
                end_reason = "truncated"
            else:
                end_reason = reason
            metrics = {
                "env_id": env_id,
                "terminated": terminated,
                "truncated": truncated,
                "process_score": int(self._process_score[env_id].item()),
                "max_score": self.MAX_SCORE,
                "points": dict(zip(self.POINT_NAMES, points)),
                "stage_scores": {name: int(sum(points[start:end]))
                                 for name, (start, end) in zip(self._stage_names, self.STAGE_POINT_SLICES)},
                "wipe_faces_completed": int(sum(points[wipe_start:wipe_end])),
                "completed_stages": [name for name, done in zip(self._stage_names, stages) if done],
                "success": all(stages),
                "episode_steps": steps,
                "episode_sim_time_s": round(steps * float(self._step_dt), 3),
                "end_reason": end_reason,
            }
            callback = getattr(self, "_episode_finished_callback", None)
            if callback is not None:
                callback(env_id, metrics)
            self._finished_episodes.append(metrics)
        if len(self._finished_episodes) > self.MAX_FINISHED_EPISODES:
            del self._finished_episodes[: -self.MAX_FINISHED_EPISODES]

    def get_process_score_metrics(self):
        """Official score report.

        The top-level fields describe the episode in progress, or, right after
        a reset (success or timeout ends an episode and resets the env), the
        episode that has just finished. ``finished_episodes`` lists every
        finished episode since the simulator started.
        """
        if self._point_flags is None or self._stage_flags is None:
            return {"process_score": 0, "max_score": self.MAX_SCORE, "completed_stages": [], "finished_episodes": []}
        current = self._episode_metrics()
        finished = [dict(item) for item in self._finished_episodes]
        if current["episode_steps"] < 2 and finished:
            report = dict(finished[-1])
            report["reported_episode"] = "last_finished"
        else:
            report = current
            report["reported_episode"] = "in_progress"
        report["finished_episode_count"] = len(finished)
        report["finished_episodes"] = finished
        return report

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
