from .deformable_mapping import deformable_tensor_to_numpy


def resolve_deformable_root_prim_path(deformable):
    import isaaclab.sim as sim_utils

    prim_path_expr = getattr(getattr(deformable, "cfg", None), "prim_path", None)
    if not prim_path_expr:
        return None

    template_prim = sim_utils.find_first_matching_prim(prim_path_expr)
    if template_prim is None:
        return None

    return template_prim.GetPath().pathString


def create_omniphysics_runtime_view(body_prim_path):
    try:
        from isaacsim.core.simulation_manager import SimulationManager
    except Exception as exc:
        return None, f"missing_simulation_manager:{type(exc).__name__}"

    physics_sim_view = SimulationManager.get_physics_sim_view()
    if physics_sim_view is None:
        return None, "missing_physics_sim_view"

    pattern = body_prim_path.replace(".*", "*")
    for create_fn_name in ("create_volume_deformable_body_view", "create_surface_deformable_body_view"):
        create_fn = getattr(physics_sim_view, create_fn_name, None)
        if create_fn is None:
            continue
        try:
            runtime_view = create_fn(pattern)
        except Exception as exc:
            last_error = f"{create_fn_name}:{type(exc).__name__}"
            continue
        if getattr(runtime_view, "_backend", None) is not None and int(getattr(runtime_view, "count", 0) or 0) > 0:
            return runtime_view, create_fn_name
        last_error = f"{create_fn_name}:empty"
    return None, locals().get("last_error", "no_supported_create_fn")


def get_deformable_runtime_nodal_positions(asset_record):
    if asset_record.get("runtime_api") == "omniphysics_deformable_body":
        runtime_view = asset_record.get("runtime_view")
        if runtime_view is not None:
            get_positions = getattr(runtime_view, "get_simulation_nodal_positions", None)
            if get_positions is None:
                get_positions = getattr(runtime_view, "get_sim_nodal_positions", None)
            if get_positions is not None:
                return get_positions()

        deformable = asset_record.get("deformable")
        if deformable is not None:
            return deformable.data.nodal_pos_w
        return None

    deformable = asset_record.get("deformable")
    if deformable is None:
        return None
    return deformable.data.nodal_pos_w


def get_first_runtime_frame_points(runtime_view):
    get_positions = getattr(runtime_view, "get_simulation_nodal_positions", None)
    if get_positions is None:
        get_positions = getattr(runtime_view, "get_sim_nodal_positions", None)
    if get_positions is None:
        return None
    return deformable_tensor_to_numpy(get_positions())
