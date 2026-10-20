"""PhysX scene transform sampling for USD animation overlays."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any

import numpy as np

from robofinals.utils.isaac_data_compat import as_torch, sim_quat_raw_to_xyzw_np


@dataclass(frozen=True)
class _RigidTrack:
    name: str
    asset: Any
    prim_path: str


@dataclass(frozen=True)
class _ArticulationTrack:
    name: str
    asset: Any
    body_paths: dict[int, str]


class PhysXRigidAnimationRecorder:
    """Sample registered non-robot PhysX assets through IsaacLab's public data API."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger
        self.enabled = False
        self.clip_start_step: int | None = None
        self.fps = 60.0
        self._rigid_tracks: list[_RigidTrack] = []
        self._articulation_tracks: list[_ArticulationTrack] = []
        self._samples: dict[str, list[tuple[float, np.ndarray]]] = {}
        self._last_sample_step: int | None = None

    @property
    def sample_count(self) -> int:
        return max((len(samples) for samples in self._samples.values()), default=0)

    @property
    def track_count(self) -> int:
        return len(self._samples)

    def reset(self) -> None:
        self.enabled = False
        self.clip_start_step = None
        self._rigid_tracks = []
        self._articulation_tracks = []
        self._samples = {}
        self._last_sample_step = None

    def prepare(self, env: Any) -> None:
        self.reset()
        stage = getattr(getattr(env, "sim", None), "stage", None)
        scene = getattr(env, "scene", None)
        if stage is None or scene is None:
            self.logger.warning("[AnimRecording][PhysX] Live stage or scene is unavailable.")
            return

        step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        self.fps = 1.0 / step_dt if step_dt > 0.0 else 60.0

        for name, asset in dict(getattr(scene, "rigid_objects", {}) or {}).items():
            prim_path = self._resolve_rigid_body_path(stage, asset)
            if prim_path is None or self._is_robot_path(prim_path):
                self.logger.warning(
                    f"[AnimRecording][PhysX] Could not map rigid object {name!r} to a scene body."
                )
                continue
            self._rigid_tracks.append(_RigidTrack(str(name), asset, prim_path))
            self._samples.setdefault(prim_path, [])

        for name, asset in dict(getattr(scene, "articulations", {}) or {}).items():
            if str(name).lower() == "robot":
                continue
            root_path = self._resolve_asset_root_path(stage, asset)
            if root_path is None or self._is_robot_path(root_path):
                self.logger.warning(
                    f"[AnimRecording][PhysX] Could not map articulation {name!r} to the scene."
                )
                continue

            body_names = tuple(getattr(asset, "body_names", ()) or ())
            paths_by_name = self._descendant_rigid_body_paths_by_name(stage, root_path)
            body_paths = {
                index: paths_by_name[body_name]
                for index, body_name in enumerate(body_names)
                if body_name in paths_by_name
            }
            if not body_paths:
                self.logger.warning(
                    f"[AnimRecording][PhysX] Articulation {name!r} exposes no mapped body links."
                )
                continue
            missing = [body_name for body_name in body_names if body_name not in paths_by_name]
            if missing:
                self.logger.warning(
                    f"[AnimRecording][PhysX] Articulation {name!r} has unmapped links: {missing}"
                )
            self._articulation_tracks.append(_ArticulationTrack(str(name), asset, body_paths))
            for prim_path in body_paths.values():
                self._samples.setdefault(prim_path, [])

        self.enabled = bool(self._samples)
        self.logger.info(
            "[AnimRecording][PhysX] Armed runtime scene transforms: "
            f"{len(self._rigid_tracks)} rigid objects, "
            f"{len(self._articulation_tracks)} articulations, "
            f"{self.track_count} body tracks."
        )

    def mark_start(self, env: Any) -> None:
        if not self.enabled:
            return
        self.clip_start_step = int(getattr(env, "common_step_counter", 0) or 0)
        self._last_sample_step = None
        for samples in self._samples.values():
            samples.clear()

    def sample(self, env: Any) -> None:
        if not self.enabled or self.clip_start_step is None:
            return
        step = int(getattr(env, "common_step_counter", 0) or 0)
        if self._last_sample_step is not None and step <= self._last_sample_step:
            return
        time_code = float(max(0, step - self.clip_start_step))

        for track in self._rigid_tracks:
            poses = as_torch(track.asset.data.root_pose_w)
            pose = poses if poses.ndim == 1 else poses[0]
            self._append_pose(track.prim_path, time_code, pose)

        for track in self._articulation_tracks:
            poses = as_torch(track.asset.data.body_link_pose_w)
            if poses.ndim == 3:
                poses = poses[0]
            for body_index, prim_path in track.body_paths.items():
                if body_index < poses.shape[0]:
                    self._append_pose(prim_path, time_code, poses[body_index])

        self._last_sample_step = step

    def finish(self, output_dir: str) -> str | None:
        if not self.enabled or self.sample_count == 0:
            self.logger.warning(
                "[AnimRecording][PhysX] No runtime scene transform samples were captured."
            )
            return None

        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "rigid_body_over.usda")
        stage = Usd.Stage.CreateNew(output_path)
        stage.SetTimeCodesPerSecond(float(self.fps))
        stage.SetFramesPerSecond(float(self.fps))
        stage.SetStartTimeCode(0.0)

        end_time_code = 0.0
        for prim_path, samples in self._samples.items():
            if not samples:
                continue
            prim = stage.OverridePrim(prim_path)
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            prim.RemoveAPI(UsdPhysics.MassAPI)
            xformable = UsdGeom.Xformable(prim)
            xformable.SetResetXformStack(True)
            transform_op = xformable.AddTransformOp(
                UsdGeom.XformOp.PrecisionDouble,
                opSuffix="robofinalsPhysxRecording",
            )
            for time_code, pose in samples:
                # IsaacLab 3 PhysX pose buffers use xyzw; the helper also supports
                # older wxyz installations and the explicit LW_SIM_QUAT_LAYOUT override.
                quat_xyzw = sim_quat_raw_to_xyzw_np(pose[3:7])
                quat = Gf.Quatd(
                    float(quat_xyzw[3]),
                    Gf.Vec3d(
                        float(quat_xyzw[0]),
                        float(quat_xyzw[1]),
                        float(quat_xyzw[2]),
                    ),
                )
                matrix = Gf.Matrix4d(
                    Gf.Rotation(quat),
                    Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])),
                )
                transform_op.Set(matrix, Usd.TimeCode(time_code))
                end_time_code = max(end_time_code, time_code)

        stage.SetEndTimeCode(end_time_code)
        stage.GetRootLayer().Save()
        self.logger.info(
            "[AnimRecording][PhysX] Wrote runtime scene overlay: "
            f"{output_path} ({self.track_count} tracks, {self.sample_count} frames)."
        )
        return output_path

    def _append_pose(self, prim_path: str, time_code: float, pose: Any) -> None:
        value = as_torch(pose).detach().cpu().numpy().astype(np.float64, copy=True)
        if value.shape == (7,) and np.isfinite(value).all():
            self._samples[prim_path].append((time_code, value))

    @staticmethod
    def _is_robot_path(path: str) -> bool:
        return "/Robot/" in path or path.rstrip("/").endswith("/Robot")

    @classmethod
    def _resolve_rigid_body_path(cls, stage: Any, asset: Any) -> str | None:
        root_view = getattr(asset, "root_view", None)
        prim_paths = list(getattr(root_view, "prim_paths", ()) or ())
        for path in prim_paths:
            normalized = cls._normalize_env_path(str(path))
            if "/World/envs/env_0/" in normalized:
                return normalized

        root_path = cls._resolve_asset_root_path(stage, asset)
        if root_path is None:
            return None
        from pxr import UsdPhysics

        root_prim = stage.GetPrimAtPath(root_path)
        if root_prim.IsValid() and root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return root_path
        matches = cls._descendant_rigid_body_paths_by_name(stage, root_path)
        return next(iter(matches.values()), None)

    @classmethod
    def _resolve_asset_root_path(cls, stage: Any, asset: Any) -> str | None:
        cfg = getattr(asset, "cfg", None)
        expression = str(getattr(cfg, "prim_path", "") or "")
        if not expression:
            return None
        expression = expression.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        expression = cls._normalize_env_path(expression)
        direct = stage.GetPrimAtPath(expression)
        if direct.IsValid():
            return expression
        try:
            pattern = re.compile(f"^{expression}$")
        except re.error:
            return None
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if pattern.fullmatch(path):
                return path
        return None

    @staticmethod
    def _normalize_env_path(path: str) -> str:
        return re.sub(r"/World/(?:envs/)?env_\d+/", "/World/envs/env_0/", path)

    @staticmethod
    def _descendant_rigid_body_paths_by_name(stage: Any, root_path: str) -> dict[str, str]:
        from pxr import UsdPhysics

        root_prefix = root_path.rstrip("/") + "/"
        result: dict[str, str] = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if (path == root_path or path.startswith(root_prefix)) and prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                result.setdefault(prim.GetName(), path)
        return result
