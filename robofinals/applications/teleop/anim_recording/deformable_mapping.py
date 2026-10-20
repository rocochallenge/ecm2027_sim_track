import numpy as np


OMNIPHYSICS_DEFORMABLE_BODY_APIS = (
    "OmniPhysicsDeformableBodyAPI",
    "PhysxBaseDeformableBodyAPI",
    "PhysxAutoDeformableBodyAPI",
)


def compute_deformable_world_to_local_matrix(stage, mesh_prim_path, time_code=None):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(mesh_prim_path)
    if not prim.IsValid():
        return None
    mesh = UsdGeom.Mesh(prim)
    usd_time_code = Usd.TimeCode.Default() if time_code is None else Usd.TimeCode(float(time_code))
    local_to_world_matrix = np.asarray(mesh.ComputeLocalToWorldTransform(usd_time_code), dtype=np.float32).reshape(
        4, 4
    )
    return np.linalg.inv(local_to_world_matrix).astype(np.float32)


def deformable_values_to_numpy(values, dtype=np.float32, width=None):
    if values is None:
        if width is None:
            return np.asarray([], dtype=dtype)
        return np.zeros((0, width), dtype=dtype)

    try:
        length = len(values)
    except TypeError:
        return np.asarray(values, dtype=dtype)

    if length == 0:
        if width is None:
            return np.asarray([], dtype=dtype)
        return np.zeros((0, width), dtype=dtype)

    first_value = values[0]
    if hasattr(first_value, "__iter__") and not isinstance(first_value, (str, bytes)):
        array = np.asarray([tuple(value) for value in values], dtype=dtype)
    else:
        array = np.asarray(values, dtype=dtype)

    if width is not None:
        array = array.reshape(-1, width)
    return array


def deformable_tensor_to_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def prim_has_api_schema_name(prim, *schema_names):
    api_schemas = prim.GetMetadata("apiSchemas")
    if api_schemas is None:
        return False
    api_schemas_text = str(api_schemas)
    return any(schema_name in api_schemas_text for schema_name in schema_names)


def get_usd_attr_time_samples(attr):
    if attr is None or not attr.IsValid():
        return []
    try:
        time_samples = attr.GetTimeSamples()
        return list(time_samples) if time_samples is not None else []
    except Exception:
        time_samples = []
        attr.GetTimeSamples(time_samples)
        return list(time_samples)


def deformable_transform_points(points, matrix):
    if points.shape[0] == 0:
        return points.copy()
    homogeneous = np.concatenate(
        [points.astype(np.float32, copy=False), np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    return (homogeneous @ matrix)[:, :3].astype(np.float32, copy=False)


def evaluate_deformable_mapping_local(asset_record, source_points_local):
    render_points_local = asset_record["render_rest_points_local"].copy()

    tet_target_indices = asset_record["render_tet_target_indices"]
    if tet_target_indices.size > 0:
        tet_vertex_positions = source_points_local[asset_record["render_tet_vertex_indices"]]
        tet_barycentrics = asset_record["render_tet_barycentrics"]
        if tet_barycentrics.shape[1] == 4:
            render_points_local[tet_target_indices] = np.sum(
                tet_vertex_positions * tet_barycentrics[:, :, None],
                axis=1,
            )
        else:
            render_points_local[tet_target_indices] = (
                tet_vertex_positions[:, 0]
                + tet_barycentrics[:, 0:1] * (tet_vertex_positions[:, 1] - tet_vertex_positions[:, 0])
                + tet_barycentrics[:, 1:2] * (tet_vertex_positions[:, 2] - tet_vertex_positions[:, 0])
                + tet_barycentrics[:, 2:3] * (tet_vertex_positions[:, 3] - tet_vertex_positions[:, 0])
            )

    nearest_target_indices = asset_record["render_nearest_target_indices"]
    if nearest_target_indices.size > 0:
        render_points_local[nearest_target_indices] = source_points_local[
            asset_record["render_nearest_source_indices"]
        ]

    return render_points_local


def deformable_points_to_attachment_list(points):
    points_array = deformable_values_to_numpy(points, dtype=np.float32, width=3)
    if points_array.shape[0] == 0:
        return []

    try:
        import carb

        return [carb.Float3(float(point[0]), float(point[1]), float(point[2])) for point in points_array]
    except Exception:
        from pxr import Gf

        return [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in points_array]


def deformable_indices_to_attachment_list(indices):
    indices_array = deformable_values_to_numpy(indices, dtype=np.int32)
    if indices_array.size == 0:
        return []
    return [int(index) for index in indices_array.reshape(-1)]


def build_nearest_deformable_mapping(attachment_iface, source_points, query_points):
    if len(source_points) == 0 or len(query_points) == 0:
        return None

    source_points_array = deformable_values_to_numpy(source_points, dtype=np.float32, width=3)
    query_points_array = deformable_values_to_numpy(query_points, dtype=np.float32, width=3)
    if source_points_array.shape == query_points_array.shape and np.allclose(
        source_points_array, query_points_array, rtol=1.0e-5, atol=1.0e-6
    ):
        return np.arange(source_points_array.shape[0], dtype=np.int32)

    if attachment_iface is None:
        # Newton does not expose PhysX private attachment interface. Resolve
        # the rest-pose mapping in bounded chunks to avoid an NxM allocation.
        indices = np.empty(query_points_array.shape[0], dtype=np.int32)
        source_norm = np.sum(source_points_array * source_points_array, axis=1)
        for start in range(0, query_points_array.shape[0], 256):
            query_chunk = query_points_array[start : start + 256]
            distances = (
                np.sum(query_chunk * query_chunk, axis=1)[:, None]
                + source_norm[None, :]
                - 2.0 * query_chunk @ source_points_array.T
            )
            indices[start : start + query_chunk.shape[0]] = np.argmin(distances, axis=1)
        return indices

    source_points_for_attachment = deformable_points_to_attachment_list(source_points_array)
    query_points_for_attachment = deformable_points_to_attachment_list(query_points_array)
    if len(source_points_for_attachment) == 0 or len(query_points_for_attachment) == 0:
        return None

    point_finder = attachment_iface.create_point_finder(source_points_for_attachment)
    try:
        output = attachment_iface.points_to_indices(point_finder, query_points_for_attachment)
    finally:
        attachment_iface.release_point_finder(point_finder)

    indices = deformable_values_to_numpy(output.get("indices"), dtype=np.int32)
    if indices.ndim != 1:
        indices = indices.reshape(-1)
    return indices


def find_ancestor_with_api_schema_name(prim, *schema_names):
    current_prim = prim
    while current_prim is not None and current_prim.IsValid():
        if prim_has_api_schema_name(current_prim, *schema_names):
            return current_prim
        current_prim = current_prim.GetParent()
    return None


def find_omniphysics_simulation_mesh_prim(body_prim):
    from pxr import Usd

    if body_prim is None or not body_prim.IsValid():
        return None

    for prim in Usd.PrimRange(body_prim):
        if prim == body_prim:
            continue
        rest_points_attr = prim.GetAttribute("omniphysics:restShapePoints")
        rest_indices_attr = prim.GetAttribute("omniphysics:restTetVtxIndices")
        if (
            rest_points_attr.IsValid()
            and rest_points_attr.HasAuthoredValueOpinion()
            and rest_indices_attr.IsValid()
            and rest_indices_attr.HasAuthoredValueOpinion()
        ):
            return prim
    return None


def resolve_omniphysics_render_mesh_prim(body_prim):
    from pxr import Usd, UsdGeom

    if body_prim is None or not body_prim.IsValid():
        return None

    candidates = []
    body_name = body_prim.GetName()
    for prim in Usd.PrimRange(body_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if points is None or len(points) == 0:
            continue
        prim_path = prim.GetPath().pathString
        score = 0
        if "/Visuals/" in prim_path:
            score += 8
        if prim_has_api_schema_name(prim, "OmniPhysicsDeformablePoseAPI"):
            score += 4
        if prim.GetName() == body_name:
            score += 2
        candidates.append((score, prim_path, len(points)))

    if len(candidates) == 0:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def prepare_deformable_runtime_mapping(
    asset_name,
    mesh_prim_path,
    mesh,
    mesh_vertex_count,
    runtime_point_count=None,
):
    mesh_prim = mesh.GetPrim()
    render_rest_points_value = mesh.GetPointsAttr().Get()
    if render_rest_points_value is None or len(render_rest_points_value) == 0:
        return None

    rest_points_attr = mesh_prim.GetAttribute("physxDeformable:restPoints")
    if rest_points_attr.IsValid():
        authored_rest_points = rest_points_attr.Get()
        if authored_rest_points is not None and len(authored_rest_points) == len(render_rest_points_value):
            render_rest_points_value = authored_rest_points
    pose_points_attr = mesh_prim.GetAttribute("deformablePose:default:omniphysics:points")
    if pose_points_attr.IsValid():
        authored_pose_points = pose_points_attr.Get()
        if authored_pose_points is not None and len(authored_pose_points) == len(render_rest_points_value):
            render_rest_points_value = authored_pose_points

    render_rest_points_local = deformable_values_to_numpy(render_rest_points_value, dtype=np.float32, width=3)
    world_to_local_matrix = compute_deformable_world_to_local_matrix(mesh_prim.GetStage(), mesh_prim_path, None)
    if world_to_local_matrix is None:
        return None

    attachment_iface = None
    try:
        import omni.physx

        acquire_attachment_iface = getattr(
            omni.physx, "acquire_physx_attachment_private_interface", None
        )
        if acquire_attachment_iface is not None:
            attachment_iface = acquire_attachment_iface()
    except (ImportError, RuntimeError):
        pass
    collision_rest_attr = mesh_prim.GetAttribute("physxDeformable:collisionRestPoints")
    collision_indices_attr = mesh_prim.GetAttribute("physxDeformable:collisionIndices")
    simulation_rest_attr = mesh_prim.GetAttribute("physxDeformable:simulationRestPoints")
    simulation_indices_attr = mesh_prim.GetAttribute("physxDeformable:simulationIndices")

    collision_rest_points_value = collision_rest_attr.Get() if collision_rest_attr.IsValid() else None
    collision_indices_value = collision_indices_attr.Get() if collision_indices_attr.IsValid() else None
    simulation_rest_points_value = simulation_rest_attr.Get() if simulation_rest_attr.IsValid() else None
    simulation_indices_value = simulation_indices_attr.Get() if simulation_indices_attr.IsValid() else None

    omniphysics_body_prim = find_ancestor_with_api_schema_name(mesh_prim, *OMNIPHYSICS_DEFORMABLE_BODY_APIS)
    omniphysics_sim_prim = find_omniphysics_simulation_mesh_prim(omniphysics_body_prim)
    omniphysics_rest_points_value = None
    omniphysics_indices_value = None
    if omniphysics_sim_prim is not None:
        omniphysics_rest_attr = omniphysics_sim_prim.GetAttribute("omniphysics:restShapePoints")
        omniphysics_indices_attr = omniphysics_sim_prim.GetAttribute("omniphysics:restTetVtxIndices")
        omniphysics_rest_points_value = omniphysics_rest_attr.Get() if omniphysics_rest_attr.IsValid() else None
        omniphysics_indices_value = omniphysics_indices_attr.Get() if omniphysics_indices_attr.IsValid() else None

    def new_mapping_record():
        return {
            "asset_name": asset_name,
            "mesh_prim_path": mesh_prim_path,
            "body_prim_path": omniphysics_body_prim.GetPath().pathString if omniphysics_body_prim is not None else None,
            "simulation_mesh_prim_path": (
                omniphysics_sim_prim.GetPath().pathString if omniphysics_sim_prim is not None else None
            ),
            "render_rest_points_local": render_rest_points_local,
            "render_vertex_count": mesh_vertex_count,
            "world_to_local_matrix": world_to_local_matrix,
            "render_tet_target_indices": np.asarray([], dtype=np.int32),
            "render_tet_vertex_indices": np.zeros((0, 4), dtype=np.int32),
            "render_tet_barycentrics": np.zeros((0, 4), dtype=np.float32),
            "render_nearest_target_indices": np.asarray([], dtype=np.int32),
            "render_nearest_source_indices": np.asarray([], dtype=np.int32),
            "sample_source": None,
            "sample_source_point_count": 0,
            "mapping_mode": None,
            "tet_mapped_count": 0,
            "nearest_mapped_count": 0,
            "rest_reconstruction_max_error": None,
            "rest_reconstruction_mean_error": None,
        }

    def build_mapping_candidate(source_name, source_points_value, source_indices_value=None, allow_tet_embedding=True):
        if source_points_value is None:
            return None

        source_points_local = deformable_values_to_numpy(source_points_value, dtype=np.float32, width=3)
        if source_points_local.shape[0] == 0:
            return None
        candidate = new_mapping_record()
        candidate["sample_source"] = source_name
        candidate["sample_source_point_count"] = source_points_local.shape[0]

        unresolved_target_indices = np.arange(render_rest_points_local.shape[0], dtype=np.int32)
        if attachment_iface is not None and allow_tet_embedding and source_indices_value is not None:
            source_points_for_attachment = deformable_points_to_attachment_list(source_points_local)
            render_points_for_attachment = deformable_points_to_attachment_list(render_rest_points_local)
            source_tet_indices = deformable_values_to_numpy(source_indices_value, dtype=np.int32)
            if source_tet_indices.size > 0 and source_tet_indices.size % 4 == 0:
                source_tet_indices = source_tet_indices.reshape(-1, 4)
                if source_tet_indices.shape[0] > 0:
                    source_indices_for_attachment = deformable_indices_to_attachment_list(source_tet_indices)
                    tet_finder = attachment_iface.create_tet_finder(
                        source_points_for_attachment,
                        source_indices_for_attachment,
                    )
                    try:
                        tet_output = attachment_iface.points_to_tetmesh_local(
                            tet_finder,
                            render_points_for_attachment,
                        )
                    finally:
                        attachment_iface.release_tet_finder(tet_finder)

                    tet_ids = deformable_values_to_numpy(tet_output.get("tet_ids"), dtype=np.int32)
                    if tet_ids.ndim != 1:
                        tet_ids = tet_ids.reshape(-1)
                    barycentrics = deformable_values_to_numpy(
                        tet_output.get("barycentric_coords"), dtype=np.float32
                    )
                    if barycentrics.ndim == 1 and barycentrics.size > 0:
                        if barycentrics.size == tet_ids.shape[0] * 4:
                            barycentrics = barycentrics.reshape(-1, 4)
                        else:
                            barycentrics = barycentrics.reshape(-1, 3)

                    if (
                        tet_ids.shape[0] == render_rest_points_local.shape[0]
                        and barycentrics.ndim == 2
                        and barycentrics.shape[0] == render_rest_points_local.shape[0]
                    ):
                        valid_tet_mask = (tet_ids >= 0) & (tet_ids < source_tet_indices.shape[0])
                    else:
                        valid_tet_mask = np.zeros(render_rest_points_local.shape[0], dtype=bool)

                    valid_tet_indices = np.nonzero(valid_tet_mask)[0].astype(np.int32)
                    if valid_tet_indices.size > 0:
                        candidate["mapping_mode"] = f"{source_name}_tet_embedding"
                        candidate["render_tet_target_indices"] = valid_tet_indices
                        candidate["render_tet_vertex_indices"] = source_tet_indices[tet_ids[valid_tet_indices]]
                        candidate["render_tet_barycentrics"] = barycentrics[valid_tet_indices].astype(
                            np.float32, copy=False
                        )
                        candidate["tet_mapped_count"] = int(valid_tet_indices.size)
                        unresolved_mask = np.ones(render_rest_points_local.shape[0], dtype=bool)
                        unresolved_mask[valid_tet_indices] = False
                        unresolved_target_indices = np.nonzero(unresolved_mask)[0].astype(np.int32)

        if unresolved_target_indices.size > 0:
            nearest_indices = build_nearest_deformable_mapping(
                attachment_iface,
                source_points_local,
                render_rest_points_local[unresolved_target_indices],
            )
            if nearest_indices is not None and nearest_indices.shape[0] == unresolved_target_indices.shape[0]:
                valid_nearest_mask = (nearest_indices >= 0) & (nearest_indices < source_points_local.shape[0])
                nearest_target_indices = unresolved_target_indices[valid_nearest_mask]
                if nearest_target_indices.size > 0:
                    if candidate["mapping_mode"] is None:
                        candidate["mapping_mode"] = f"{source_name}_nearest"
                    else:
                        candidate["mapping_mode"] = f"{source_name}_hybrid"
                    candidate["render_nearest_target_indices"] = nearest_target_indices
                    candidate["render_nearest_source_indices"] = nearest_indices[valid_nearest_mask].astype(
                        np.int32, copy=False
                    )
                    candidate["nearest_mapped_count"] = int(nearest_target_indices.size)

        mapped_point_count = candidate["tet_mapped_count"] + candidate["nearest_mapped_count"]
        if mapped_point_count == 0:
            return None

        reconstructed_rest_points = evaluate_deformable_mapping_local(candidate, source_points_local)
        reconstruction_error = np.linalg.norm(reconstructed_rest_points - render_rest_points_local, axis=1)
        candidate["rest_reconstruction_max_error"] = float(np.max(reconstruction_error))
        candidate["rest_reconstruction_mean_error"] = float(np.mean(reconstruction_error))
        return candidate

    simulation_candidate = build_mapping_candidate(
        "simulation",
        simulation_rest_points_value,
        simulation_indices_value,
        allow_tet_embedding=True,
    )
    if simulation_candidate is not None:
        return simulation_candidate

    omniphysics_candidate = build_mapping_candidate(
        "omniphysics_simulation",
        omniphysics_rest_points_value,
        omniphysics_indices_value,
        allow_tet_embedding=True,
    )
    if omniphysics_candidate is not None:
        return omniphysics_candidate

    collision_candidate = build_mapping_candidate(
        "collision",
        collision_rest_points_value,
        collision_indices_value,
        allow_tet_embedding=True,
    )
    if collision_candidate is not None:
        return collision_candidate

    if runtime_point_count is None or int(runtime_point_count) != render_rest_points_local.shape[0]:
        return None

    candidate = new_mapping_record()
    identity_indices = np.arange(render_rest_points_local.shape[0], dtype=np.int32)
    candidate["sample_source"] = "runtime_nodal"
    candidate["sample_source_point_count"] = int(runtime_point_count)
    candidate["mapping_mode"] = "runtime_nodal_identity"
    candidate["render_nearest_target_indices"] = identity_indices
    candidate["render_nearest_source_indices"] = identity_indices.copy()
    candidate["nearest_mapped_count"] = int(identity_indices.size)
    return candidate


def reconstruct_deformable_render_points(asset_record, frame_source_points_world, world_to_local_matrix=None):
    if frame_source_points_world.shape[0] == 0:
        return asset_record["render_rest_points_local"].copy()

    if world_to_local_matrix is None:
        world_to_local_matrix = asset_record["world_to_local_matrix"]
    source_points_local = deformable_transform_points(
        frame_source_points_world.astype(np.float32, copy=False),
        world_to_local_matrix,
    )
    return evaluate_deformable_mapping_local(asset_record, source_points_local)
