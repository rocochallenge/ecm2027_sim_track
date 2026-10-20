import os

from .deformable_mapping import get_usd_attr_time_samples


def read_layer_over_timeline(layer_over_path, fallback_fps):
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(layer_over_path)
    if layer is None:
        return float(fallback_fps), 0.0, 0.0

    fps_attr = layer.timeCodesPerSecond
    fps = float(fps_attr) if fps_attr else float(fallback_fps)

    lo = float("inf")
    hi = float("-inf")
    queue = [layer.pseudoRoot]
    while queue:
        spec = queue.pop()
        for child in spec.nameChildren:
            queue.append(child)
        for prop in spec.properties:
            if not prop.name.startswith("xformOp:"):
                continue
            samples = layer.ListTimeSamplesForPath(prop.path)
            if not samples:
                continue
            if samples[0] < lo:
                lo = float(samples[0])
            if samples[-1] > hi:
                hi = float(samples[-1])

    if lo == float("inf"):
        lo, hi = 0.0, 0.0
    return fps, lo, hi


def compute_deformable_time_code_from_step(
    sample_record,
    clip_start_step,
    baked_start_time_code,
    rigid_end_time_code,
    last_sample_step,
):
    sample_step = sample_record.get("env_step")
    if sample_step is None or clip_start_step is None:
        return baked_start_time_code + float(sample_record.get("clip_time", 0.0))
    clip_step = max(0, int(sample_step) - int(clip_start_step))
    total_clip_steps = max(1, int(last_sample_step) - int(clip_start_step))
    rigid_duration = max(0.0, float(rigid_end_time_code) - float(baked_start_time_code))
    return float(baked_start_time_code) + rigid_duration * (float(clip_step) / float(total_clip_steps))


def retime_rigid_overlay_to_timeline(
    overlay_path: str,
    *,
    target_fps: float,
    target_start_time_code: float,
    target_end_time_code: float,
):
    """Retimes all transform samples and metadata onto the robot timeline.

    The end frame is authoritative because playback layers are usually clipped by
    the baked robot animation.  A rounded sample ratio keeps integer frame spacing
    for the common 2x rigid/robot capture rates.
    """

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(overlay_path)
    if stage is None:
        return None
    source_fps = float(stage.GetTimeCodesPerSecond() or target_fps)
    sample_times: list[float] = []
    attrs = []
    for prim in stage.TraverseAll():
        for attr in UsdGeom.Xformable(prim).GetOrderedXformOps():
            value_attr = attr.GetAttr()
            times = [float(value) for value in value_attr.GetTimeSamples()]
            if times:
                sample_times.extend(times)
                attrs.append((value_attr, times))
    if not sample_times:
        stage.SetTimeCodesPerSecond(float(target_fps))
        stage.GetRootLayer().Save()
        return {"source_fps": source_fps, "target_fps": float(target_fps), "sample_ratio": 1.0}

    source_start, source_end = min(sample_times), max(sample_times)
    source_span = max(1.0, source_end - source_start)
    target_span = max(1.0, float(target_end_time_code) - float(target_start_time_code))
    ratio = max(1.0, float(round(target_span / source_span)))
    anchor_start = float(target_end_time_code) - source_span * ratio
    for attr, times in attrs:
        values = [(time, attr.Get(Usd.TimeCode(time))) for time in times]
        for time in times:
            attr.ClearAtTime(Usd.TimeCode(time))
        for time, value in values:
            mapped = anchor_start + (time - source_start) * ratio
            attr.Set(value, Usd.TimeCode(float(round(mapped))))
    stage.SetTimeCodesPerSecond(float(target_fps))
    stage.SetStartTimeCode(float(target_start_time_code))
    stage.SetEndTimeCode(float(target_end_time_code))
    stage.GetRootLayer().Save()
    return {
        "source_fps": source_fps,
        "target_fps": float(target_fps),
        "sample_ratio": ratio,
        "source_start": source_start,
        "source_end": source_end,
    }


class UsdPlaybackBuilder:
    def __init__(self, logger_getter):
        self._logger_getter = logger_getter

    @property
    def logger(self):
        return self._logger_getter()

    def build(self, anim_output_dir, deformable_layer_path=None, assets=None, *, include_rigid_overlay=True):
        baked_output_path = os.path.join(anim_output_dir, "baked_animation_recording.usda")
        layer_over_path = os.path.join(anim_output_dir, "layer_over.usda")
        layer_root_path = os.path.join(anim_output_dir, "layer_root.usda")
        if not os.path.exists(baked_output_path):
            self.logger.warning(f"[AnimRecording] Missing baked animation file: {baked_output_path}")
            return None
        if not os.path.exists(layer_over_path):
            self.logger.warning(f"[AnimRecording] Missing animation over layer: {layer_over_path}")
            return None
        if not os.path.exists(layer_root_path):
            self.logger.warning(f"[AnimRecording] Missing animation root layer: {layer_root_path}")
            return None

        try:
            from pxr import PhysxSchema
        except ImportError:
            PhysxSchema = None
        from pxr import Usd, UsdGeom

        baked_stage = Usd.Stage.Open(baked_output_path)
        if baked_stage is None:
            self.logger.warning(f"[AnimRecording] Failed to open baked animation stage: {baked_output_path}")
            return None

        deformable_stage = None
        if deformable_layer_path is not None and os.path.exists(deformable_layer_path):
            deformable_stage = Usd.Stage.Open(deformable_layer_path)

        baked_fps_fallback = float(baked_stage.GetTimeCodesPerSecond() or 180.0)
        time_codes_per_second, rigid_start, rigid_end = read_layer_over_timeline(
            layer_over_path, baked_fps_fallback
        )

        start_time_code = 0.0
        end_time_code = max(rigid_end, 0.0)
        up_axis = UsdGeom.GetStageUpAxis(baked_stage)

        self.logger.info(
            f"[AnimRecording] Timeline anchor from layer_over.usda: "
            f"fps={time_codes_per_second} rigid_range=[{rigid_start}, {rigid_end}] "
            f"(= {rigid_end / time_codes_per_second if time_codes_per_second else 0.0:.2f}s)"
        )

        normalized_deformable_start_time = None
        assets = assets or {}
        if deformable_stage is not None:
            for asset_record in assets.values():
                mesh_prim = deformable_stage.GetPrimAtPath(asset_record["mesh_prim_path"])
                if not mesh_prim.IsValid():
                    continue
                time_samples = get_usd_attr_time_samples(UsdGeom.Mesh(mesh_prim).GetPointsAttr())
                if len(time_samples) == 0:
                    continue
                first_time_code = float(time_samples[0])
                if normalized_deformable_start_time is None or first_time_code < normalized_deformable_start_time:
                    normalized_deformable_start_time = first_time_code

        flattened_over_path = os.path.join(anim_output_dir, "flattened_playback_over.usda")
        if os.path.exists(flattened_over_path):
            os.remove(flattened_over_path)

        flattened_stage = Usd.Stage.CreateNew(flattened_over_path)
        flattened_stage.SetTimeCodesPerSecond(time_codes_per_second)
        flattened_stage.SetStartTimeCode(start_time_code)
        flattened_stage.SetEndTimeCode(end_time_code)
        UsdGeom.SetStageUpAxis(flattened_stage, up_axis)
        rigid_overlay_path = os.path.join(anim_output_dir, "rigid_body_over.usda")
        if include_rigid_overlay and os.path.exists(rigid_overlay_path):
            retime_rigid_overlay_to_timeline(
                rigid_overlay_path,
                target_fps=time_codes_per_second,
                target_start_time_code=start_time_code,
                target_end_time_code=end_time_code,
            )
        flattened_stage.GetRootLayer().subLayerPaths = (
            ["rigid_body_over.usda", "layer_over.usda"]
            if include_rigid_overlay and os.path.exists(rigid_overlay_path)
            else ["layer_over.usda"]
        )

        physics_scene_prim = flattened_stage.OverridePrim("/physicsScene")
        physics_scene_prim.SetActive(False)
        if PhysxSchema is not None:
            physics_scene_prim.RemoveAPI(PhysxSchema.PhysxSceneAPI)

        if deformable_stage is not None:
            for asset_record in assets.values():
                mesh_prim = deformable_stage.GetPrimAtPath(asset_record["mesh_prim_path"])
                if not mesh_prim.IsValid():
                    continue

                mesh = UsdGeom.Mesh(mesh_prim)
                points_attr = mesh.GetPointsAttr()
                time_samples = get_usd_attr_time_samples(points_attr)
                if len(time_samples) == 0:
                    continue

                target_prim = flattened_stage.OverridePrim(asset_record["mesh_prim_path"])
                self._remove_physx_api_if_available(target_prim, "PhysxCollisionAPI")
                self._remove_physx_api_if_available(target_prim, "PhysxDeformableBodyAPI")
                self._remove_physx_api_if_available(target_prim, "OmniPhysicsDeformablePoseAPI", "default")
                self._remove_physx_api_if_available(target_prim, "OmniPhysicsDeformablePoseAPI")

                body_prim_path = asset_record.get("body_prim_path")
                if body_prim_path:
                    body_prim = flattened_stage.OverridePrim(body_prim_path)
                    self._remove_physx_api_if_available(body_prim, "PhysxAutoDeformableBodyAPI")
                    self._remove_physx_api_if_available(body_prim, "PhysxAutoDeformableMeshSimplificationAPI")
                    self._remove_physx_api_if_available(body_prim, "PhysxAutoDeformableHexahedralMeshAPI")
                    self._remove_physx_api_if_available(body_prim, "PhysxBaseDeformableBodyAPI")
                    self._remove_physx_api_if_available(body_prim, "OmniPhysicsDeformableBodyAPI")

                simulation_mesh_prim_path = asset_record.get("simulation_mesh_prim_path")
                if simulation_mesh_prim_path:
                    simulation_mesh_prim = flattened_stage.OverridePrim(simulation_mesh_prim_path)
                    self._remove_physx_api_if_available(simulation_mesh_prim, "OmniPhysicsVolumeDeformableSimAPI")
                    self._remove_physx_api_if_available(
                        simulation_mesh_prim, "OmniPhysicsDeformablePoseAPI", "default"
                    )
                    self._remove_physx_api_if_available(simulation_mesh_prim, "OmniPhysicsDeformablePoseAPI")

                target_points_attr = UsdGeom.Mesh(target_prim).CreatePointsAttr()
                time_offset = normalized_deformable_start_time if normalized_deformable_start_time is not None else 0.0
                for time_code in time_samples:
                    normalized_time_code = float(time_code) - time_offset
                    target_points_attr.Set(points_attr.Get(Usd.TimeCode(float(time_code))), normalized_time_code)
                    end_time_code = max(end_time_code, normalized_time_code)

        flattened_stage.SetEndTimeCode(end_time_code)
        flattened_stage.GetRootLayer().Save()

        playback_entry_path = os.path.join(anim_output_dir, "teleop_baked_with_deformable.usda")
        if os.path.exists(playback_entry_path):
            os.remove(playback_entry_path)

        playback_entry_stage = Usd.Stage.CreateNew(playback_entry_path)
        playback_entry_stage.SetTimeCodesPerSecond(time_codes_per_second)
        playback_entry_stage.SetStartTimeCode(start_time_code)
        playback_entry_stage.SetEndTimeCode(end_time_code)
        UsdGeom.SetStageUpAxis(playback_entry_stage, up_axis)
        playback_entry_stage.GetRootLayer().subLayerPaths = [
            "flattened_playback_over.usda",
            "layer_root.usda",
        ]
        playback_entry_stage.GetRootLayer().Save()

        self.logger.info(f"[AnimRecording] Playback entry: {playback_entry_path}")
        return playback_entry_path

    def _remove_physx_api_if_available(self, prim, schema_name, instance_name=None):
        from pxr import PhysxSchema

        schema_class = getattr(PhysxSchema, schema_name, None)
        if schema_class is None:
            return
        try:
            if instance_name is None:
                prim.RemoveAPI(schema_class)
            else:
                prim.RemoveAPI(schema_class, instance_name)
        except TypeError:
            try:
                prim.RemoveAPI(schema_class)
            except Exception:
                pass
        except Exception:
            pass
