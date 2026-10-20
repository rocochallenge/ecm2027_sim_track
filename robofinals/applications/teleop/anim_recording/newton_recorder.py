"""Newton animation recorder adapter.

Newton does not produce PhysX OVD data.  This controller records Newton's public
articulation and rigid-object state and writes a USD transform overlay directly.
The deformable point-sampling implementation remains backend-neutral and is
composed into the same playback entry at finalization.
"""

from __future__ import annotations

from datetime import datetime
import os
import re
from typing import Any

import numpy as np

from robofinals.utils.isaac_data_compat import as_torch

from .ovd_recorder import OvdRecorderController


class NewtonRecorderController(OvdRecorderController):
    _OUTPUT_FILES = frozenset(
        {
            "Stage.usd",
            "baked_animation_recording.usda",
            "deformable_animation_over.usda",
            "deformable_bake_debug.json",
            "flattened_playback_over.usda",
            "layer_over.usda",
            "layer_root.usda",
            "rigid_body_over.usda",
            "teleop_baked_with_deformable.usda",
        }
    )

    def reset_state(self):
        super().reset_state()
        self.output_dir: str | None = None
        self._articulation_tracks: list[dict[str, Any]] = []
        self._rigid_tracks: list[dict[str, Any]] = []
        self._samples: dict[str, list[tuple[float, np.ndarray]]] = {}
        self._last_sample_step: int | None = None
        self._last_checkpoint_step: int | None = None
        self._checkpoint_interval = max(
            1, int(getattr(self.args_cli, "anim_recording_checkpoint_interval", 600) or 600)
        )
        self._fps = 60.0

    def prepare(self, env):
        self.reset_state()
        self.deformable.reset()
        if not self.enabled:
            return

        self.output_dir = self._resolve_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        self._remove_stale_outputs()
        self._fps = self._resolve_fps(env)
        self._prepare_tracks(env)
        self.deformable.prepare(env)
        self.logger.info(
            "[AnimRecording][Newton] Armed direct USD recording: "
            f"{len(self._articulation_tracks)} articulations, "
            f"{len(self._rigid_tracks)} rigid objects, output={self.output_dir}"
        )

    def request_start(self, env):
        if not self.enabled or self.clip_started:
            return
        if self.output_dir is None:
            self.prepare(env)
        self.mark_start(env)
        self.deformable.mark_start(
            env,
            clip_start_time=self.clip_start_time,
            clip_start_step=self.clip_start_step,
            clip_start_env_time=self.clip_start_env_time,
        )
        self.sample(env)
        self.logger.info("[AnimRecording][Newton] Recording started.")

    def sample(self, env):
        if not self.enabled or not self.clip_started:
            return

        step = int(getattr(env, "common_step_counter", 0) or 0)
        if self._last_sample_step is not None and step <= self._last_sample_step:
            return
        start_step = int(self.clip_start_step or 0)
        time_code = float(max(0, step - start_step))

        for track in self._articulation_tracks:
            poses = as_torch(track["asset"].data.body_link_pose_w)[0].detach().cpu().numpy()
            for body_index, prim_path in track["body_paths"].items():
                self._samples[prim_path].append((time_code, np.asarray(poses[body_index], dtype=np.float64)))

        for track in self._rigid_tracks:
            pose = as_torch(track["asset"].data.root_pose_w)[0].detach().cpu().numpy()
            self._samples[track["prim_path"]].append((time_code, np.asarray(pose, dtype=np.float64)))

        self._last_sample_step = step
        self.deformable.sample(env)
        if (
            self._last_checkpoint_step is None
            or step - self._last_checkpoint_step >= self._checkpoint_interval
        ):
            try:
                self._write_checkpoint(env)
            except Exception as exc:
                self.logger.warning(
                    f"[AnimRecording][Newton] Checkpoint write failed: {type(exc).__name__}: {exc}"
                )

    def _write_checkpoint(self, env) -> str | None:
        if self.output_dir is None:
            return None
        stage = getattr(getattr(env, "sim", None), "stage", None)
        if stage is None:
            return None

        layer_root_path = os.path.join(self.output_dir, "layer_root.usda")
        stage_path = os.path.join(self.output_dir, "Stage.usd")
        if not os.path.exists(layer_root_path):
            self._export_stage(stage, layer_root_path)
        if not os.path.exists(stage_path):
            self._export_stage(stage, stage_path)

        self._write_transform_overlay(stage, os.path.join(self.output_dir, "layer_over.usda"))
        self._write_baked_entry(stage)
        combined_output_path = self.deformable.finish(self.output_dir)
        if combined_output_path is None:
            combined_output_path = self.playback_builder.build(
                self.output_dir,
                include_rigid_overlay=False,
            )

        self._last_checkpoint_step = self._last_sample_step
        if combined_output_path is not None:
            env._anim_recording_baked_path = os.path.join(
                self.output_dir, "baked_animation_recording.usda"
            )
            env._anim_recording_baked_delivery_entry = combined_output_path
            sample_count = max((len(values) for values in self._samples.values()), default=0)
            self.logger.info(
                "[AnimRecording][Newton] USD checkpoint saved: "
                f"{combined_output_path} ({sample_count} frames)"
            )
        return combined_output_path

    def finish(self, env):
        if self.baked or not self.enabled:
            return
        self.baked = True
        if not self.clip_started:
            self.logger.info("[AnimRecording][Newton] Teleop clip never started; no USD was written.")
            return
        if self.output_dir is None:
            self.logger.warning("[AnimRecording][Newton] Output directory was not prepared.")
            return

        self.sample(env)
        stage = getattr(getattr(env, "sim", None), "stage", None)
        if stage is None:
            self.logger.warning("[AnimRecording][Newton] Live USD stage is unavailable; cannot finalize.")
            return

        layer_root_path = os.path.join(self.output_dir, "layer_root.usda")
        stage_path = os.path.join(self.output_dir, "Stage.usd")
        self._export_stage(stage, layer_root_path)
        self._export_stage(stage, stage_path)

        layer_over_path = os.path.join(self.output_dir, "layer_over.usda")
        self._write_transform_overlay(stage, layer_over_path)
        self._write_baked_entry(stage)

        combined_output_path = self.deformable.finish(self.output_dir)
        if combined_output_path is None:
            combined_output_path = self.playback_builder.build(
                self.output_dir,
                include_rigid_overlay=False,
            )

        if combined_output_path is not None:
            env._anim_recording_baked_path = os.path.join(
                self.output_dir, "baked_animation_recording.usda"
            )
            env._anim_recording_baked_delivery_entry = combined_output_path
            sample_count = max((len(values) for values in self._samples.values()), default=0)
            self.logger.info(
                "[AnimRecording][Newton] USD recording saved: "
                f"{combined_output_path} ({sample_count} frames, {len(self._samples)} transform tracks)"
            )
        else:
            self.logger.warning(
                f"[AnimRecording][Newton] Final playback entry was not created in {self.output_dir}"
            )

    def _resolve_output_dir(self) -> str:
        configured = getattr(self.args_cli, "anim_recording_output_dir", None)
        if configured:
            return os.path.abspath(os.path.expanduser(str(configured)))
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        return os.path.abspath(os.path.join(os.getcwd(), "anim_recordings", timestamp))

    def _remove_stale_outputs(self) -> None:
        assert self.output_dir is not None
        for filename in self._OUTPUT_FILES:
            path = os.path.join(self.output_dir, filename)
            if os.path.isfile(path):
                os.remove(path)

    @staticmethod
    def _resolve_fps(env) -> float:
        step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        return 1.0 / step_dt if step_dt > 0.0 else 60.0

    def _prepare_tracks(self, env) -> None:
        stage = getattr(getattr(env, "sim", None), "stage", None)
        scene = getattr(env, "scene", None)
        if stage is None or scene is None:
            return

        for name, asset in getattr(scene, "articulations", {}).items():
            body_names = tuple(getattr(asset, "body_names", ()) or ())
            root_path = self._resolve_prim_path(stage, getattr(asset.cfg, "prim_path", ""))
            if root_path is None or not body_names:
                self.logger.warning(
                    f"[AnimRecording][Newton] Could not map articulation {name!r} to USD bodies."
                )
                continue
            name_to_path = self._descendant_paths_by_name(stage, root_path)
            body_paths = {
                index: name_to_path[body_name]
                for index, body_name in enumerate(body_names)
                if body_name in name_to_path
            }
            missing = [body_name for body_name in body_names if body_name not in name_to_path]
            if missing:
                self.logger.warning(
                    f"[AnimRecording][Newton] Articulation {name!r} has unmapped bodies: {missing}"
                )
            if body_paths:
                self._articulation_tracks.append(
                    {"name": name, "asset": asset, "body_paths": body_paths}
                )
                for prim_path in body_paths.values():
                    self._samples.setdefault(prim_path, [])

        for name, asset in getattr(scene, "rigid_objects", {}).items():
            prim_path = self._resolve_prim_path(stage, getattr(asset.cfg, "prim_path", ""))
            if prim_path is None:
                self.logger.warning(
                    f"[AnimRecording][Newton] Could not map rigid object {name!r} to a USD prim."
                )
                continue
            self._rigid_tracks.append({"name": name, "asset": asset, "prim_path": prim_path})
            self._samples.setdefault(prim_path, [])

    @staticmethod
    def _resolve_prim_path(stage, expression: str) -> str | None:
        if not expression:
            return None
        expression = expression.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
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
    def _descendant_paths_by_name(stage, root_path: str) -> dict[str, str]:
        root_prefix = root_path.rstrip("/") + "/"
        paths: dict[str, str] = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path == root_path or path.startswith(root_prefix):
                paths.setdefault(prim.GetName(), path)
        return paths

    def _write_transform_overlay(self, source_stage, output_path: str) -> None:
        from pxr import Gf, Usd, UsdGeom

        overlay = Usd.Stage.CreateNew(output_path)
        overlay.SetTimeCodesPerSecond(self._fps)
        overlay.SetFramesPerSecond(self._fps)
        overlay.SetStartTimeCode(0.0)
        end_time_code = max(
            (samples[-1][0] for samples in self._samples.values() if samples),
            default=0.0,
        )
        overlay.SetEndTimeCode(end_time_code)
        UsdGeom.SetStageUpAxis(overlay, UsdGeom.GetStageUpAxis(source_stage))

        for prim_path, samples in self._samples.items():
            if not samples:
                continue
            prim = overlay.OverridePrim(prim_path)
            xformable = UsdGeom.Xformable(prim)
            xformable.SetResetXformStack(True)
            transform_op = xformable.AddTransformOp(opSuffix="robofinalsRecording")
            for time_code, pose in samples:
                position = Gf.Vec3d(*[float(value) for value in pose[:3]])
                # Newton exposes wp.transformf storage directly: (px, py, pz, qx, qy, qz, qw).
                quat_xyzw = pose[3:7]
                quat = Gf.Quatd(
                    float(quat_xyzw[3]),
                    float(quat_xyzw[0]),
                    float(quat_xyzw[1]),
                    float(quat_xyzw[2]),
                )
                transform_op.Set(Gf.Matrix4d(Gf.Rotation(quat), position), time_code)

        overlay.GetRootLayer().Save()

    def _write_baked_entry(self, source_stage) -> None:
        from pxr import Usd, UsdGeom

        path = os.path.join(self.output_dir, "baked_animation_recording.usda")
        baked = Usd.Stage.CreateNew(path)
        baked.SetTimeCodesPerSecond(self._fps)
        baked.SetFramesPerSecond(self._fps)
        baked.SetStartTimeCode(0.0)
        baked.SetEndTimeCode(
            max((samples[-1][0] for samples in self._samples.values() if samples), default=0.0)
        )
        UsdGeom.SetStageUpAxis(baked, UsdGeom.GetStageUpAxis(source_stage))
        baked.GetRootLayer().subLayerPaths = ["layer_over.usda", "layer_root.usda"]
        baked.GetRootLayer().Save()


    @staticmethod
    def _export_stage(source_stage, output_path: str) -> None:
        from pxr import Sdf, Usd, UsdGeom
        from isaaclab.sim.utils.stage import resolve_paths

        source_layer = source_stage.GetRootLayer()
        output_layer = Sdf.Layer.CreateNew(output_path)
        Sdf.CreatePrimInLayer(output_layer, "/")
        Sdf.CopySpec(source_layer, "/", output_layer, "/")
        output_stage = Usd.Stage.Open(output_layer)
        UsdGeom.SetStageUpAxis(output_stage, UsdGeom.GetStageUpAxis(source_stage))
        UsdGeom.SetStageMetersPerUnit(
            output_stage,
            UsdGeom.GetStageMetersPerUnit(source_stage),
        )
        resolve_paths(source_layer.identifier, output_layer.identifier)
        output_layer.Save()
