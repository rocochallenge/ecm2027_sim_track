import json
import os

import numpy as np

from .deformable_mapping import (
    OMNIPHYSICS_DEFORMABLE_BODY_APIS,
    compute_deformable_world_to_local_matrix,
    deformable_tensor_to_numpy,
    get_usd_attr_time_samples,
    prepare_deformable_runtime_mapping,
    prim_has_api_schema_name,
    reconstruct_deformable_render_points,
    resolve_omniphysics_render_mesh_prim,
    find_omniphysics_simulation_mesh_prim,
)
from .deformable_runtime import (
    create_omniphysics_runtime_view,
    get_deformable_runtime_nodal_positions,
    get_first_runtime_frame_points,
    resolve_deformable_root_prim_path,
)
from .usd_playback import (
    UsdPlaybackBuilder,
    compute_deformable_time_code_from_step,
    read_layer_over_timeline,
)


class DeformableAnimationRecorder:
    def __init__(self, args_cli, logger_getter):
        self.args_cli = args_cli
        self._logger_getter = logger_getter
        self.playback_builder = UsdPlaybackBuilder(logger_getter)
        self.reset()

    @property
    def logger(self):
        return self._logger_getter()

    @property
    def enabled(self):
        return bool(getattr(self.args_cli, "anim_recording_enabled", False))

    @property
    def assets(self):
        return self.state["assets"]

    def reset(self):
        self.state = {
            "clip_start_time": None,
            "clip_start_step": None,
            "clip_start_env_time": None,
            "sample_time_source": "env_step_scaled_to_rigid_timecode",
            "time_samples": [],
            "sample_steps": [],
            "assets": {},
            "combined_output_path": None,
            "diagnostics": [],
        }

    def append_diagnostic(self, **payload):
        self.state.setdefault("diagnostics", []).append(payload)

    def write_debug(self, anim_output_dir, reason, extra=None):
        debug_payload = {
            "reason": reason,
            "clip_start_time": self.state["clip_start_time"],
            "clip_start_step": self.state.get("clip_start_step"),
            "clip_start_env_time": self.state.get("clip_start_env_time"),
            "sample_time_source": self.state.get("sample_time_source"),
            "time_sample_count": len(self.state["time_samples"]),
            "time_samples_preview": self.state["time_samples"][:10],
            "sample_steps_preview": self.state.get("sample_steps", [])[:10],
            "asset_count": len(self.state["assets"]),
            "diagnostics": self.state.get("diagnostics", []),
            "assets": {
                asset_name: {
                    "mesh_prim_path": asset_record.get("mesh_prim_path"),
                    "body_prim_path": asset_record.get("body_prim_path"),
                    "simulation_mesh_prim_path": asset_record.get("simulation_mesh_prim_path"),
                    "runtime_api": asset_record.get("runtime_api"),
                    "vertex_count": asset_record.get("vertex_count"),
                    "sample_source": asset_record.get("sample_source"),
                    "sample_source_point_count": asset_record.get("sample_source_point_count"),
                    "mapping_mode": asset_record.get("mapping_mode"),
                    "tet_mapped_count": asset_record.get("tet_mapped_count"),
                    "nearest_mapped_count": asset_record.get("nearest_mapped_count"),
                    "rest_reconstruction_max_error": asset_record.get("rest_reconstruction_max_error"),
                    "rest_reconstruction_mean_error": asset_record.get("rest_reconstruction_mean_error"),
                    "sample_count": len(asset_record.get("samples", [])),
                    "arm_world_to_local_translate": asset_record.get("arm_world_to_local_translate"),
                    "playback_world_to_local_translate": asset_record.get("playback_world_to_local_translate"),
                    "playback_world_to_local_source": asset_record.get("playback_world_to_local_source"),
                }
                for asset_name, asset_record in self.state["assets"].items()
            },
        }
        if extra:
            debug_payload["extra"] = extra

        debug_output_path = os.path.join(anim_output_dir, "deformable_bake_debug.json")
        with open(debug_output_path, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, indent=2)
        self.logger.warning(
            f"[AnimRecording][Deformable] Debug summary written to {debug_output_path} (reason: {reason})"
        )
        return debug_output_path

    def prepare(self, env):
        self.reset()

        if not self.enabled:
            self.append_diagnostic(stage="prepare", status="anim_recording_disabled")
            self.logger.info("[AnimRecording][Deformable] Skipping arm stage because anim recording is disabled.")
            return

        scene = getattr(env, "scene", None)
        deformable_objects = getattr(scene, "deformable_objects", None)
        if deformable_objects is None:
            deformable_objects = {}
        if len(deformable_objects) == 0:
            self.append_diagnostic(stage="prepare", status="no_deformable_objects")
            self.logger.info(
                "[AnimRecording][Deformable] Scene has no legacy deformable objects; scanning OmniPhysics 5.1 deformables."
            )
        else:
            self.append_diagnostic(
                stage="prepare",
                status="found_deformable_objects",
                deformable_object_names=sorted(list(deformable_objects.keys())),
            )

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self.logger.warning("[AnimRecording][Deformable] USD stage is unavailable; skipping deformable bake.")
            return

        prepared_assets = {}
        self._prepare_scene_deformables(stage, deformable_objects, prepared_assets)
        self._prepare_omniphysics_fallbacks(stage, prepared_assets)

        self.state["assets"] = prepared_assets
        if len(prepared_assets) > 0:
            summary = ", ".join(
                f"{asset_name}:{asset_record['mesh_prim_path']}" for asset_name, asset_record in prepared_assets.items()
            )
            self.logger.info(f"[AnimRecording][Deformable] Armed assets: {summary}")
        else:
            self.logger.info("[AnimRecording][Deformable] No eligible deformable meshes were armed.")

    def mark_start(self, env, clip_start_time=None, clip_start_step=None, clip_start_env_time=None):
        if not self.enabled:
            return
        if len(self.state["assets"]) == 0:
            return
        if self.state["clip_start_time"] is not None:
            return

        if clip_start_time is not None:
            self.state["clip_start_time"] = clip_start_time
            self.state["clip_start_step"] = clip_start_step
            self.state["clip_start_env_time"] = clip_start_env_time
        else:
            step_count, env_time = self._get_env_step_time(env)
            self.state["clip_start_time"] = 0.0
            self.state["clip_start_step"] = step_count
            self.state["clip_start_env_time"] = env_time

        if self.state["clip_start_time"] is None:
            self.state["clip_start_time"] = 0.0

        self.logger.info(
            "[AnimRecording][Deformable] Clip start marked at "
            f"{self.state['clip_start_time']:.2f}s "
            f"(env_step={self.state['clip_start_step']}, "
            f"env_time={self.state['clip_start_env_time']:.4f}s)."
        )

    def sample(self, env):
        if not self.enabled:
            return
        if len(self.state["assets"]) == 0:
            return

        clip_start_step = self.state.get("clip_start_step")
        if clip_start_step is None:
            return

        env_step_count, env_time = self._get_env_step_time(env)
        if env_step_count < int(clip_start_step):
            return

        time_samples = self.state["time_samples"]
        sample_steps = self.state.setdefault("sample_steps", [])
        if len(sample_steps) > 0 and env_step_count <= sample_steps[-1]:
            return

        step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        sample_time = max(0.0, (env_step_count - int(clip_start_step)) * step_dt)
        time_samples.append(sample_time)
        sample_steps.append(int(env_step_count))

        current_stage = None
        invalid_assets = []
        for asset_name, asset_record in self.state["assets"].items():
            frame_points_value = get_deformable_runtime_nodal_positions(asset_record)
            if frame_points_value is None:
                self.append_diagnostic(
                    stage="sample_asset",
                    asset_name=asset_name,
                    status="missing_runtime_frame_points",
                    runtime_api=asset_record.get("runtime_api"),
                    body_prim_path=asset_record.get("body_prim_path"),
                )
                invalid_assets.append(asset_name)
                continue

            frame_points = deformable_tensor_to_numpy(frame_points_value)
            if frame_points.ndim == 3:
                frame_points = frame_points[0]
            elif frame_points.ndim == 2:
                pass
            else:
                self.append_diagnostic(
                    stage="sample_asset",
                    asset_name=asset_name,
                    status="invalid_frame_shape",
                    frame_points_shape=tuple(frame_points.shape),
                )
                invalid_assets.append(asset_name)
                continue

            expected_point_count = asset_record["sample_source_point_count"]
            if frame_points.shape[0] != expected_point_count:
                self.logger.warning(
                    f"[AnimRecording][Deformable] Sample source vertex count mismatch for '{asset_name}' "
                    f"({frame_points.shape[0]} vs {expected_point_count}); dropping this asset from recording."
                )
                self.append_diagnostic(
                    stage="sample_asset",
                    asset_name=asset_name,
                    status="sample_source_mismatch",
                    frame_point_count=int(frame_points.shape[0]),
                    expected_point_count=int(expected_point_count),
                    sample_source=asset_record.get("sample_source"),
                    mapping_mode=asset_record.get("mapping_mode"),
                )
                invalid_assets.append(asset_name)
                continue

            if current_stage is None:
                import omni.usd

                current_stage = omni.usd.get_context().get_stage()

            world_to_local_matrix = asset_record["world_to_local_matrix"]
            if current_stage is not None:
                sampled_world_to_local_matrix = compute_deformable_world_to_local_matrix(
                    current_stage, asset_record["mesh_prim_path"], sample_time
                )
                if sampled_world_to_local_matrix is not None:
                    world_to_local_matrix = sampled_world_to_local_matrix

            asset_record["samples"].append(
                {
                    "clip_time": float(sample_time),
                    "env_step": int(env_step_count),
                    "clip_step": int(env_step_count - int(clip_start_step)),
                    "env_time": float(env_time),
                    "world_to_local_matrix": np.asarray(world_to_local_matrix, dtype=np.float32).copy(),
                    "frame_points": np.asarray(frame_points, dtype=np.float32).copy(),
                }
            )

        for asset_name in invalid_assets:
            self.state["assets"].pop(asset_name, None)

        if len(time_samples) == 1:
            self.logger.info(
                "[AnimRecording][Deformable] First sample captured for assets: "
                + ", ".join(sorted(self.state["assets"].keys()))
            )

    def finish(self, anim_output_dir):
        if len(self.state["assets"]) == 0:
            self.write_debug(anim_output_dir, "no_armed_assets")
            return None
        if len(self.state["time_samples"]) == 0:
            self.logger.info("[AnimRecording][Deformable] No deformable frames were sampled; skipping bake.")
            self.write_debug(anim_output_dir, "no_time_samples")
            return None

        baked_output_path = os.path.join(anim_output_dir, "baked_animation_recording.usda")
        if not os.path.exists(baked_output_path):
            self.logger.warning(f"[AnimRecording][Deformable] Missing baked animation file: {baked_output_path}")
            self.write_debug(anim_output_dir, "missing_baked_animation", {"baked_output_path": baked_output_path})
            return None

        from pxr import Gf, Usd, UsdGeom, Vt

        baked_stage = Usd.Stage.Open(baked_output_path)
        if baked_stage is None:
            self.logger.warning(f"[AnimRecording][Deformable] Failed to open playback stage: {baked_output_path}")
            self.write_debug(anim_output_dir, "failed_to_open_playback_stage", {"baked_output_path": baked_output_path})
            return None

        layer_over_path = os.path.join(anim_output_dir, "layer_over.usda")
        baked_fps_fallback = float(baked_stage.GetTimeCodesPerSecond() or 180.0)
        if os.path.exists(layer_over_path):
            time_codes_per_second, rigid_start_time_code, rigid_end_time_code = read_layer_over_timeline(
                layer_over_path, baked_fps_fallback
            )
        else:
            time_codes_per_second = baked_fps_fallback
            rigid_start_time_code = float(baked_stage.GetStartTimeCode())
            rigid_end_time_code = float(baked_stage.GetEndTimeCode())

        baked_start_time_code = rigid_start_time_code
        baked_end_time_code = rigid_end_time_code
        up_axis = UsdGeom.GetStageUpAxis(baked_stage)
        clip_start_step = self.state.get("clip_start_step")

        self.logger.info(
            f"[AnimRecording][Deformable] Timeline anchor: fps={time_codes_per_second} "
            f"rigid_end_timecode={rigid_end_time_code} (= "
            f"{rigid_end_time_code / time_codes_per_second if time_codes_per_second else 0.0:.2f}s)"
        )
        eligible_sample_steps = [
            sample_step
            for sample_step in self.state.get("sample_steps", [])
            if clip_start_step is None or int(sample_step) >= int(clip_start_step)
        ]
        if len(eligible_sample_steps) == 0:
            self.logger.info(
                "[AnimRecording][Deformable] No sampled frames fall inside the teleop clip window; skipping bake."
            )
            self.write_debug(
                anim_output_dir,
                "no_eligible_time_samples",
                {
                    "baked_start_time_code": baked_start_time_code,
                    "baked_end_time_code": baked_end_time_code,
                    "time_codes_per_second": time_codes_per_second,
                },
            )
            return None
        last_eligible_sample_step = max(int(sample_step) for sample_step in eligible_sample_steps)

        deformable_layer_path = os.path.join(anim_output_dir, "deformable_animation_over.usda")
        if os.path.exists(deformable_layer_path):
            os.remove(deformable_layer_path)

        deformable_stage = Usd.Stage.CreateNew(deformable_layer_path)
        UsdGeom.SetStageUpAxis(deformable_stage, up_axis)
        deformable_stage.SetTimeCodesPerSecond(time_codes_per_second)
        deformable_stage.SetStartTimeCode(baked_start_time_code)

        layer_root_path = os.path.join(anim_output_dir, "layer_root.usda")
        playback_stage = None
        if os.path.exists(layer_root_path):
            playback_stage = Usd.Stage.Open(layer_root_path)
            if playback_stage is None:
                self.logger.warning(
                    f"[AnimRecording][Deformable] Failed to open {layer_root_path} for playback xform lookup."
                )

        playback_world_to_local_by_asset = {}
        for asset_name, asset_record in self.state["assets"].items():
            playback_w2l = None
            if playback_stage is not None:
                playback_w2l = compute_deformable_world_to_local_matrix(
                    playback_stage, asset_record["mesh_prim_path"], None
                )
            playback_world_to_local_by_asset[asset_name] = playback_w2l

            arm_matrix = asset_record.get("world_to_local_matrix")
            arm_translate = None
            if arm_matrix is not None:
                arm_translate = [float(arm_matrix[3][0]), float(arm_matrix[3][1]), float(arm_matrix[3][2])]
            playback_translate = None
            if playback_w2l is not None:
                playback_translate = [float(playback_w2l[3][0]), float(playback_w2l[3][1]), float(playback_w2l[3][2])]
            asset_record["arm_world_to_local_translate"] = arm_translate
            asset_record["playback_world_to_local_translate"] = playback_translate
            asset_record["playback_world_to_local_source"] = (
                "layer_root.usda" if playback_w2l is not None else "arm_stage_fallback"
            )
            self.append_diagnostic(
                stage="playback_xform",
                asset_name=asset_name,
                status="resolved" if playback_w2l is not None else "missing_layer_root",
                arm_world_to_local_translate=arm_translate,
                playback_world_to_local_translate=playback_translate,
            )

        latest_time_code = baked_start_time_code
        written_asset_count = 0
        for asset_name, asset_record in self.state["assets"].items():
            if len(asset_record["samples"]) == 0:
                continue

            mesh_prim = deformable_stage.OverridePrim(asset_record["mesh_prim_path"])
            mesh = UsdGeom.Mesh(mesh_prim)
            points_attr = mesh.CreatePointsAttr()
            asset_sample_count = 0
            playback_w2l = playback_world_to_local_by_asset.get(asset_name)

            for sample_record in asset_record["samples"]:
                if clip_start_step is not None and int(sample_record.get("env_step", -1)) < int(clip_start_step):
                    continue
                time_code = compute_deformable_time_code_from_step(
                    sample_record,
                    clip_start_step,
                    baked_start_time_code,
                    baked_end_time_code,
                    last_eligible_sample_step,
                )
                latest_time_code = max(latest_time_code, time_code)
                world_to_local_matrix = playback_w2l if playback_w2l is not None else sample_record.get(
                    "world_to_local_matrix"
                )
                render_frame_points = reconstruct_deformable_render_points(
                    asset_record,
                    sample_record["frame_points"],
                    world_to_local_matrix=world_to_local_matrix,
                )
                points_attr.Set(
                    Vt.Vec3fArray([Gf.Vec3f(float(pt[0]), float(pt[1]), float(pt[2])) for pt in render_frame_points]),
                    time_code,
                )
                asset_sample_count += 1

            if asset_sample_count > 0:
                written_asset_count += 1

        if written_asset_count == 0:
            self.logger.info("[AnimRecording][Deformable] No deformable assets produced time samples.")
            self.write_debug(anim_output_dir, "no_written_assets")
            return None

        deformable_stage.SetEndTimeCode(max(baked_end_time_code, latest_time_code))
        deformable_stage.GetRootLayer().Save()

        combined_layer_path = self.playback_builder.build(
            anim_output_dir,
            deformable_layer_path,
            assets=self.state["assets"],
        )
        if combined_layer_path is None:
            self.write_debug(
                anim_output_dir,
                "failed_to_build_playback_stage",
                {"deformable_layer_path": deformable_layer_path},
            )
            return None

        self.state["combined_output_path"] = combined_layer_path
        self.logger.info(
            f"[AnimRecording][Deformable] Wrote {written_asset_count} deformable assets with "
            f"{len(self.state['time_samples'])} captured timestamps to {deformable_layer_path}"
        )
        self.logger.info(
            "[AnimRecording][Deformable] Timeline aligned to baked clip: "
            f"baked=[{baked_start_time_code / time_codes_per_second:.2f}, "
            f"{baked_end_time_code / time_codes_per_second:.2f}]s, "
            f"deformable_steps=[{eligible_sample_steps[0]}, {eligible_sample_steps[-1]}]"
        )
        self.logger.info(f"[AnimRecording][Deformable] Combined playback entry: {combined_layer_path}")
        self.write_debug(
            anim_output_dir,
            "success",
            {
                "deformable_layer_path": deformable_layer_path,
                "combined_layer_path": combined_layer_path,
                "written_asset_count": written_asset_count,
            },
        )
        return combined_layer_path

    def _prepare_scene_deformables(self, stage, deformable_objects, prepared_assets):
        for asset_name, deformable in deformable_objects.items():
            if deformable.num_instances < 1:
                self.append_diagnostic(stage="prepare_asset", asset_name=asset_name, status="no_live_instances")
                self.logger.warning(f"[AnimRecording][Deformable] '{asset_name}' has no live instances; skipping.")
                continue

            nodal_pos_w = deformable.data.nodal_pos_w
            if nodal_pos_w.ndim != 3 or nodal_pos_w.shape[0] < 1:
                self.append_diagnostic(
                    stage="prepare_asset",
                    asset_name=asset_name,
                    status="invalid_nodal_shape",
                    nodal_pos_w_shape=tuple(nodal_pos_w.shape),
                )
                self.logger.warning(
                    f"[AnimRecording][Deformable] '{asset_name}' nodal_pos_w has unexpected shape "
                    f"{tuple(nodal_pos_w.shape)}; skipping."
                )
                continue

            nodal_vertex_count = int(nodal_pos_w.shape[1])
            runtime_api = getattr(deformable, "runtime_api", "physx_soft_body")
            root_prim_path = resolve_deformable_root_prim_path(deformable)
            if root_prim_path is None:
                self.append_diagnostic(
                    stage="prepare_asset",
                    asset_name=asset_name,
                    status="root_prim_unresolved",
                    prim_path_expr=getattr(getattr(deformable, "cfg", None), "prim_path", None),
                )
                self.logger.warning(
                    f"[AnimRecording][Deformable] Could not resolve a concrete prim path for '{asset_name}' "
                    f"from expression '{getattr(getattr(deformable, 'cfg', None), 'prim_path', None)}'."
                )
                continue

            self.append_diagnostic(
                stage="prepare_asset",
                asset_name=asset_name,
                status="inspecting",
                nodal_pos_w_shape=tuple(nodal_pos_w.shape),
                nodal_vertex_count=nodal_vertex_count,
                root_prim_path=root_prim_path,
                runtime_api=runtime_api,
            )

            if runtime_api == "omniphysics_deformable_body":
                root_prim = stage.GetPrimAtPath(root_prim_path)
                mesh_prim_path = resolve_omniphysics_render_mesh_prim(root_prim) if root_prim.IsValid() else None
            else:
                mesh_prim_path = self._resolve_deformable_mesh_prim(
                    stage, asset_name, root_prim_path, nodal_vertex_count
                )
            if mesh_prim_path is None:
                self.append_diagnostic(
                    stage="prepare_asset",
                    asset_name=asset_name,
                    status="mesh_unresolved",
                    nodal_pos_w_shape=tuple(nodal_pos_w.shape),
                    nodal_vertex_count=nodal_vertex_count,
                    root_prim_path=root_prim_path,
                    runtime_api=runtime_api,
                )
                continue

            asset_record = self._build_asset_record_from_mesh(
                stage,
                asset_name,
                mesh_prim_path,
                nodal_vertex_count,
                deformable=deformable,
                runtime_api=runtime_api,
                root_prim_path=root_prim_path,
            )
            if asset_record is not None:
                prepared_assets[asset_name] = asset_record

    def _prepare_omniphysics_fallbacks(self, stage, prepared_assets):
        from pxr import UsdGeom

        existing_mesh_paths = {asset_record["mesh_prim_path"] for asset_record in prepared_assets.values()}
        for body_prim in stage.Traverse():
            if not prim_has_api_schema_name(body_prim, *OMNIPHYSICS_DEFORMABLE_BODY_APIS):
                continue

            body_prim_path = body_prim.GetPath().pathString
            mesh_prim_path = resolve_omniphysics_render_mesh_prim(body_prim)
            simulation_mesh_prim = find_omniphysics_simulation_mesh_prim(body_prim)
            if mesh_prim_path is None or simulation_mesh_prim is None:
                self.append_diagnostic(
                    stage="prepare_asset",
                    asset_name=body_prim.GetName(),
                    status="omniphysics_mesh_unresolved",
                    body_prim_path=body_prim_path,
                    mesh_prim_path=mesh_prim_path,
                    simulation_mesh_prim_path=(
                        simulation_mesh_prim.GetPath().pathString if simulation_mesh_prim is not None else None
                    ),
                )
                continue
            if mesh_prim_path in existing_mesh_paths:
                continue

            runtime_view, runtime_view_kind = create_omniphysics_runtime_view(body_prim_path)
            if runtime_view is None:
                self.append_diagnostic(
                    stage="prepare_asset",
                    asset_name=body_prim.GetName(),
                    status="omniphysics_runtime_view_unresolved",
                    body_prim_path=body_prim_path,
                    runtime_view_error=runtime_view_kind,
                )
                self.logger.warning(
                    f"[AnimRecording][Deformable] Could not create OmniPhysics runtime view for '{body_prim_path}' "
                    f"({runtime_view_kind}); skipping."
                )
                continue

            try:
                nodal_pos_w = get_first_runtime_frame_points(runtime_view)
            except Exception as exc:
                self.append_diagnostic(
                    stage="prepare_asset",
                    asset_name=body_prim.GetName(),
                    status="omniphysics_runtime_sample_failed",
                    body_prim_path=body_prim_path,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            if nodal_pos_w is None or nodal_pos_w.ndim != 3 or nodal_pos_w.shape[0] < 1:
                self.append_diagnostic(
                    stage="prepare_asset",
                    asset_name=body_prim.GetName(),
                    status="omniphysics_invalid_nodal_shape",
                    body_prim_path=body_prim_path,
                    nodal_pos_w_shape=tuple(nodal_pos_w.shape) if nodal_pos_w is not None else None,
                )
                continue
            nodal_vertex_count = int(nodal_pos_w.shape[1])

            mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
            mesh = UsdGeom.Mesh(mesh_prim)
            mesh_point_values = mesh.GetPointsAttr().Get()
            if mesh_point_values is None or len(mesh_point_values) == 0:
                self.append_diagnostic(
                    stage="prepare_asset",
                    asset_name=body_prim.GetName(),
                    status="omniphysics_mesh_has_no_points",
                    body_prim_path=body_prim_path,
                    mesh_prim_path=mesh_prim_path,
                )
                continue
            mesh_vertex_count = int(len(mesh_point_values))
            mapping_record = prepare_deformable_runtime_mapping(
                body_prim.GetName(),
                mesh_prim_path,
                mesh,
                mesh_vertex_count,
                runtime_point_count=nodal_vertex_count,
            )
            if mapping_record is None or mapping_record["sample_source"] is None:
                self.append_diagnostic(
                    stage="prepare_asset",
                    asset_name=body_prim.GetName(),
                    status="omniphysics_mapping_unresolved",
                    body_prim_path=body_prim_path,
                    mesh_prim_path=mesh_prim_path,
                    simulation_mesh_prim_path=simulation_mesh_prim.GetPath().pathString,
                    mesh_vertex_count=mesh_vertex_count,
                    nodal_vertex_count=nodal_vertex_count,
                )
                continue

            parent_prim = body_prim.GetParent()
            base_asset_name = (
                parent_prim.GetName()
                if parent_prim is not None and parent_prim.IsValid() and parent_prim.GetName() not in ("Scene", "World")
                else body_prim.GetName()
            )
            asset_name = base_asset_name
            suffix = 1
            while asset_name in prepared_assets:
                asset_name = f"{base_asset_name}_{suffix}"
                suffix += 1

            prepared_assets[asset_name] = self._make_prepared_asset_record(
                deformable=None,
                runtime_api="omniphysics_deformable_body",
                runtime_view=runtime_view,
                runtime_view_kind=runtime_view_kind,
                root_prim_path=body_prim_path,
                body_prim_path=body_prim_path,
                simulation_mesh_prim_path=simulation_mesh_prim.GetPath().pathString,
                mesh_prim_path=mesh_prim_path,
                mesh_vertex_count=mesh_vertex_count,
                nodal_vertex_count=nodal_vertex_count,
                mapping_record=mapping_record,
            )
            existing_mesh_paths.add(mesh_prim_path)
            self.append_diagnostic(
                stage="prepare_asset",
                asset_name=asset_name,
                status="armed_omniphysics",
                body_prim_path=body_prim_path,
                mesh_prim_path=mesh_prim_path,
                simulation_mesh_prim_path=simulation_mesh_prim.GetPath().pathString,
                runtime_view_kind=runtime_view_kind,
                nodal_pos_w_shape=tuple(nodal_pos_w.shape),
                nodal_vertex_count=nodal_vertex_count,
                mesh_vertex_count=mesh_vertex_count,
                sample_source=mapping_record["sample_source"],
                sample_source_point_count=mapping_record["sample_source_point_count"],
                mapping_mode=mapping_record["mapping_mode"],
                tet_mapped_count=mapping_record["tet_mapped_count"],
                nearest_mapped_count=mapping_record["nearest_mapped_count"],
                rest_reconstruction_max_error=mapping_record["rest_reconstruction_max_error"],
                rest_reconstruction_mean_error=mapping_record["rest_reconstruction_mean_error"],
            )

    def _build_asset_record_from_mesh(
        self,
        stage,
        asset_name,
        mesh_prim_path,
        nodal_vertex_count,
        deformable,
        runtime_api,
        root_prim_path,
    ):
        from pxr import UsdGeom

        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
        mesh = UsdGeom.Mesh(mesh_prim)
        mesh_point_values = mesh.GetPointsAttr().Get()
        if mesh_point_values is None or len(mesh_point_values) == 0:
            self.append_diagnostic(
                stage="prepare_asset",
                asset_name=asset_name,
                status="mesh_has_no_points",
                root_prim_path=root_prim_path,
                mesh_prim_path=mesh_prim_path,
            )
            return None
        mesh_vertex_count = int(len(mesh_point_values))
        mapping_record = prepare_deformable_runtime_mapping(
            asset_name,
            mesh_prim_path,
            mesh,
            mesh_vertex_count,
            runtime_point_count=nodal_vertex_count,
        )
        if mapping_record is None or mapping_record["sample_source"] is None:
            self.append_diagnostic(
                stage="prepare_asset",
                asset_name=asset_name,
                status="mapping_unresolved",
                root_prim_path=root_prim_path,
                mesh_prim_path=mesh_prim_path,
                mesh_vertex_count=mesh_vertex_count,
            )
            self.logger.warning(
                f"[AnimRecording][Deformable] Could not build a runtime mapping for '{asset_name}'; skipping."
            )
            return None

        runtime_view = getattr(deformable, "runtime_view", None)
        runtime_view_kind = getattr(deformable, "runtime_view_kind", None)
        body_prim_path = getattr(deformable, "body_prim_path", None) or mapping_record.get("body_prim_path")
        self.append_diagnostic(
            stage="prepare_asset",
            asset_name=asset_name,
            status="armed",
            nodal_vertex_count=nodal_vertex_count,
            mesh_vertex_count=mesh_vertex_count,
            root_prim_path=root_prim_path,
            body_prim_path=body_prim_path,
            mesh_prim_path=mesh_prim_path,
            runtime_api=runtime_api,
            runtime_view_kind=runtime_view_kind,
            sample_source=mapping_record["sample_source"],
            sample_source_point_count=mapping_record["sample_source_point_count"],
            mapping_mode=mapping_record["mapping_mode"],
            tet_mapped_count=mapping_record["tet_mapped_count"],
            nearest_mapped_count=mapping_record["nearest_mapped_count"],
            rest_reconstruction_max_error=mapping_record["rest_reconstruction_max_error"],
            rest_reconstruction_mean_error=mapping_record["rest_reconstruction_mean_error"],
        )
        return self._make_prepared_asset_record(
            deformable=deformable,
            runtime_api=runtime_api,
            runtime_view=runtime_view,
            runtime_view_kind=runtime_view_kind,
            root_prim_path=root_prim_path,
            body_prim_path=body_prim_path,
            simulation_mesh_prim_path=mapping_record.get("simulation_mesh_prim_path"),
            mesh_prim_path=mesh_prim_path,
            mesh_vertex_count=mesh_vertex_count,
            nodal_vertex_count=nodal_vertex_count,
            mapping_record=mapping_record,
        )

    def _make_prepared_asset_record(
        self,
        deformable,
        runtime_api,
        runtime_view,
        runtime_view_kind,
        root_prim_path,
        body_prim_path,
        simulation_mesh_prim_path,
        mesh_prim_path,
        mesh_vertex_count,
        nodal_vertex_count,
        mapping_record,
    ):
        return {
            "deformable": deformable,
            "runtime_api": runtime_api,
            "runtime_view": runtime_view,
            "runtime_view_kind": runtime_view_kind,
            "root_prim_path": root_prim_path,
            "body_prim_path": body_prim_path,
            "simulation_mesh_prim_path": simulation_mesh_prim_path,
            "mesh_prim_path": mesh_prim_path,
            "vertex_count": mesh_vertex_count,
            "nodal_vertex_count": nodal_vertex_count,
            "sample_source": mapping_record["sample_source"],
            "sample_source_point_count": mapping_record["sample_source_point_count"],
            "mapping_mode": mapping_record["mapping_mode"],
            "world_to_local_matrix": mapping_record["world_to_local_matrix"],
            "render_rest_points_local": mapping_record["render_rest_points_local"],
            "render_tet_target_indices": mapping_record["render_tet_target_indices"],
            "render_tet_vertex_indices": mapping_record["render_tet_vertex_indices"],
            "render_tet_barycentrics": mapping_record["render_tet_barycentrics"],
            "render_nearest_target_indices": mapping_record["render_nearest_target_indices"],
            "render_nearest_source_indices": mapping_record["render_nearest_source_indices"],
            "tet_mapped_count": mapping_record["tet_mapped_count"],
            "nearest_mapped_count": mapping_record["nearest_mapped_count"],
            "rest_reconstruction_max_error": mapping_record["rest_reconstruction_max_error"],
            "rest_reconstruction_mean_error": mapping_record["rest_reconstruction_mean_error"],
            "samples": [],
        }

    def _resolve_deformable_mesh_prim(self, stage, asset_name, root_prim_path, expected_vertex_count=None):
        from pxr import Usd, UsdGeom

        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            self.append_diagnostic(
                stage="resolve_mesh",
                asset_name=asset_name,
                root_prim_path=root_prim_path,
                expected_vertex_count=expected_vertex_count,
                status="invalid_root_prim",
            )
            self.logger.warning(
                f"[AnimRecording][Deformable] Root prim is invalid for '{asset_name}': {root_prim_path}"
            )
            return None

        root_name = root_prim.GetName()
        candidates = []
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            if points is None:
                continue

            mesh_prim_path = prim.GetPath().pathString
            score = 0
            if "/Visuals/" in mesh_prim_path:
                score += 4
            if mesh_prim_path.endswith(f"/{asset_name}"):
                score += 3
            if mesh_prim_path.endswith(f"/{root_name}"):
                score += 2
            if prim.GetName() == asset_name:
                score += 1
            candidates.append((score, mesh_prim_path, len(points)))

        preferred_candidates = candidates
        if expected_vertex_count is not None:
            exact_matches = [item for item in candidates if item[2] == expected_vertex_count]
            if len(exact_matches) > 0:
                preferred_candidates = exact_matches

        if len(preferred_candidates) == 0:
            self.append_diagnostic(
                stage="resolve_mesh",
                asset_name=asset_name,
                root_prim_path=root_prim_path,
                expected_vertex_count=expected_vertex_count,
                status="no_candidates",
                candidates=[
                    {"score": score, "mesh_prim_path": mesh_path, "point_count": point_count}
                    for score, mesh_path, point_count in candidates[:8]
                ],
            )
            self.logger.warning(
                f"[AnimRecording][Deformable] No mesh candidates found under '{root_prim_path}' for '{asset_name}'."
            )
            return None

        preferred_candidates.sort(key=lambda item: (-item[0], item[1]))
        if len(preferred_candidates) > 1 and preferred_candidates[0][0] == preferred_candidates[1][0]:
            self.append_diagnostic(
                stage="resolve_mesh",
                asset_name=asset_name,
                root_prim_path=root_prim_path,
                expected_vertex_count=expected_vertex_count,
                status="ambiguous_candidates",
                candidates=[
                    {"score": score, "mesh_prim_path": mesh_path, "point_count": point_count}
                    for score, mesh_path, point_count in preferred_candidates[:8]
                ],
            )
            self.logger.warning(
                f"[AnimRecording][Deformable] Ambiguous mesh candidates for '{asset_name}': "
                f"{preferred_candidates[0][1]} vs {preferred_candidates[1][1]}. Skipping this asset."
            )
            return None

        self.append_diagnostic(
            stage="resolve_mesh",
            asset_name=asset_name,
            root_prim_path=root_prim_path,
            expected_vertex_count=expected_vertex_count,
            status="resolved",
            selected_mesh_prim_path=preferred_candidates[0][1],
            selected_point_count=preferred_candidates[0][2],
            candidates=[
                {"score": score, "mesh_prim_path": mesh_path, "point_count": point_count}
                for score, mesh_path, point_count in preferred_candidates[:8]
            ],
        )
        return preferred_candidates[0][1]

    def _get_env_step_time(self, env):
        step_count = int(getattr(env, "common_step_counter", 0) or 0)
        step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        return step_count, step_count * step_dt
