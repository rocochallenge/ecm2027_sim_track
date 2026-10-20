"""Newton manager and model-builder patches."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from robofinals.utils.warp_compat import ensure_warp_config_compat

_DEFAULT_NEWTON_NCONMAX = 4096
# Upper bound for NewtonCollisionPipelineCfg.rigid_contact_max in this repo.
_NEWTON_RIGID_CONTACT_MAX = 65535
_NEWTON_VIEWER_CAMERA_COPIES = (
    ("left_hand_camera", "newton_left_wrist_camera"),
    ("right_hand_camera", "newton_right_wrist_camera"),
)


def _resolve_newton_nconmax() -> int:
    """Return MuJoCo Warp contact buffer size (``nconmax`` / ``naconmax``).

    Override at runtime without per-scene YAML edits::

        export robofinals_NEWTON_NCONMAX=49152

    Run once, read the peak from ``Number of Newton contacts (N) exceeded MJWarp
    limit`` warnings, then set this to ``ceil(N * 1.2)`` capped at
    ``_NEWTON_RIGID_CONTACT_MAX``.
    """
    raw = os.environ.get("robofinals_NEWTON_NCONMAX", "").strip()
    if raw:
        return max(256, min(int(raw), _NEWTON_RIGID_CONTACT_MAX))
    return _DEFAULT_NEWTON_NCONMAX


def _pin_pxr_work_threads(limit: int = 1) -> None:
    """Re-apply OpenUSD single-thread mode right before USD physics import."""
    value = str(int(limit))
    os.environ["PXR_WORK_THREAD_LIMIT"] = value
    os.putenv("PXR_WORK_THREAD_LIMIT", value)
    try:
        from pxr import Work

        Work.SetConcurrencyLimit(int(limit))
    except ImportError:
        pass


def _configure_newton_warp_logging() -> None:
    """Optional Warp verbosity via ``robofinals_WARP_LOG_LEVEL`` (DEBUG/INFO/WARNING/ERROR)."""
    ensure_warp_config_compat()
    level_name = os.environ.get("robofinals_WARP_LOG_LEVEL", "").strip().upper()
    if not level_name:
        return
    try:
        import warp as wp
    except ImportError:
        return
    level_map = {
        "DEBUG": wp.LOG_DEBUG,
        "INFO": wp.LOG_INFO,
        "WARNING": wp.LOG_WARNING,
        "ERROR": wp.LOG_ERROR,
    }
    if level_name in level_map:
        wp.config.log_level = level_map[level_name]


def _newton_collision_cfg(nconmax: int) -> Any:
    """Collision pipeline capacity aligned with MuJoCo Warp contact buffer."""
    from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg

    return NewtonCollisionPipelineCfg(rigid_contact_max=_NEWTON_RIGID_CONTACT_MAX)


@lru_cache(maxsize=1)
def _multiccd_mjwarp_solver_cfg_cls() -> type:
    """Return :class:`MultiCcdMJWarpSolverCfg` (lazy import for optional Newton dep)."""
    from isaaclab.utils.configclass import configclass
    from isaaclab_newton.physics import MJWarpSolverCfg

    @configclass
    class MultiCcdMJWarpSolverCfg(MJWarpSolverCfg):
        """MJWarpSolverCfg that forwards ``enable_multiccd`` to :class:`SolverMuJoCo`."""

        enable_multiccd: bool = True

    return MultiCcdMJWarpSolverCfg


def make_default_newton_physics_cfg() -> Any:
    """Build the default :class:`~isaaclab_newton.physics.newton_manager_cfg.NewtonCfg` for robofinals."""
    from robofinals.core.backends.newton.physics_cfg import RobofinalsNewtonCfg

    MultiCcdMJWarpSolverCfg = _multiccd_mjwarp_solver_cfg_cls()
    # from isaaclab_newton.physics.newton_manager_cfg import NewtonShapeCfg
    nconmax = _resolve_newton_nconmax()
    # num_substeps=1 matches benchmark_x7s_newton.py (single physics substep per env step).
    return RobofinalsNewtonCfg(
        num_substeps=4,
        use_cuda_graph=True,
        solver_cfg=MultiCcdMJWarpSolverCfg(
            disable_contacts=False,
            use_mujoco_contacts=True,
            njmax=2048,
            cone="elliptic",
            nconmax=nconmax,
            integrator="implicitfast",
            impratio=5,
        ),
        collision_cfg=None,
        # Per-shape contact material defaults forwarded to builder.default_shape_cfg
        # via checked_apply. Tune ke/kd together: kd = 2 * zeta * sqrt(ke).
        # Newton SDK defaults: ke=2.5e3, kd=100.0, mu=1.0.
        # default_shape_cfg=NewtonShapeCfg(
        #     # margin virtually thickens thin walls (fixes gripper passing through)
        #     # AND rounds sharp edges so first-contact force spreads over an area
        #     # instead of a single point (fixes bounce-off at bowl rim).
        #     margin=0.002,  # 4mm; raise toward 0.006 if thin walls still slip through
        #     ke=2.5e3,      # keep low — high ke was the source of bounce-off
        #     kd=300.0,      # raised from 100: zeta=3 overdamped, absorbs edge-contact impulse
        #     mu=1.5,        # friction for grasp stability
        # ),
    )


def configure_newton_manager_env_cfg(cfg: Any) -> None:
    """Tune :class:`~isaaclab.envs.ManagerBasedRLEnv` for Newton + Kit.

    ``ManagerBasedRLEnv.reset()`` calls ``sim.forward()`` and may invoke RTX
    ``sim.render()`` when ``rerender_on_reset`` / ``wait_for_textures`` are set.
    Those paths share cuda:0 with Warp/Newton and can stall or poison the CUDA
    context before the first ``env.step()`` collide.  InteractiveScene-only
    benchmarks skip this reset path; robofinals must disable it for Newton.
    """
    _configure_newton_warp_logging()
    from robofinals.core.backends.newton.mesh_approximation import patch_newton_mesh_approximation

    patch_newton_mesh_approximation()
    patch_newton_model_builder_add_usd()
    patch_newton_manager_isaaclab_env_layout()
    patch_newton_manager_model_dump()
    from isaaclab_newton.physics.newton_manager import NewtonManager

    NewtonManager._robofinals_prebuild_hooks = tuple(
        getattr(cfg, "_robofinals_newton_prebuild_hooks", ())
    )
    cfg.rerender_on_reset = False
    cfg.num_rerenders_on_reset = 0
    cfg.wait_for_textures = False
    # Newton physics replicate can crash when USD assets leave None in custom-attribute
    # world columns (e.g. local scenes with bottle_* prims). USD inherit-clone is enough
    # for teleop / single-env runs.
    if hasattr(cfg, "scene") and cfg.scene is not None:
        cfg.scene.replicate_physics = False


def _isaaclab_newton_env_paths(stage) -> list[tuple[int, str]]:
    """Isaac Lab places env roots at ``/World/envs/env_N`` (not ``/World/Env_N``)."""
    import re

    env_pattern = re.compile(r"^[Ee]nv_(\d+)$")
    world_prim = stage.GetPrimAtPath("/World")
    env_paths: list[tuple[int, str]] = []
    if not world_prim or not world_prim.IsValid():
        return env_paths

    search_roots = [world_prim]
    envs_prim = world_prim.GetChild("envs")
    if envs_prim and envs_prim.IsValid():
        search_roots.append(envs_prim)

    for parent in search_roots:
        for child in parent.GetChildren():
            match = env_pattern.match(child.GetName())
            if match:
                env_paths.append((int(match.group(1)), child.GetPath().pathString))
    env_paths.sort(key=lambda item: item[0])
    return env_paths


def _newton_builder_from_env_paths(cls, stage, up_axis, env_paths: list[tuple[int, str]]) -> None:
    """Upstream env-scoped Newton builder path (shared roots + per-env prototype)."""
    import warp as wp
    from newton import ModelBuilder, solvers
    from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
    from pxr import UsdGeom

    from isaaclab_newton.physics.newton_manager import NewtonManager

    builder = ModelBuilder(up_axis=up_axis)
    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    ignore_paths = [path for _, path in env_paths]

    # Robot profiles may author last-mile USD properties before model import.
    for hook in getattr(cls, "_robofinals_prebuild_hooks", ()):
        hook(stage, env_paths)

    builder.add_usd(stage, ignore_paths=ignore_paths, schema_resolvers=schema_resolvers)

    _, proto_path = env_paths[0]
    proto = ModelBuilder(up_axis=up_axis)
    # Register MuJoCo custom attributes (gravcomp, condim, …) on the prototype builder
    # before add_usd so that mjc:* USD attributes are read into the model.
    # Mirrors newton_replicate.py:85 — without this, mjc:gravcomp written by the spawn
    # cfg is silently ignored and body_gravcomp stays empty in the compiled MuJoCo model.
    solvers.SolverMuJoCo.register_custom_attributes(proto)
    proto.add_usd(stage, root_path=proto_path, schema_resolvers=schema_resolvers)

    global_sites, proto_sites, world_sites = cls._cl_inject_sites(builder, {proto_path: proto})
    global_site_map: dict[str, tuple[int, None]] = {label: (idx, None) for label, idx in global_sites.items()}
    num_worlds = len(env_paths)
    local_site_map: dict[str, list[list[int]]] = {}
    site_entries = proto_sites.get(id(proto), {})
    world_xforms: list[wp.transform] = []

    xform_cache = UsdGeom.XformCache()
    for col, (_, env_path) in enumerate(env_paths):
        builder.begin_world()
        offset = builder.shape_count
        world_xform = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(env_path))
        translation = world_xform.ExtractTranslation()
        rotation = world_xform.ExtractRotationQuat()
        pos = (translation[0], translation[1], translation[2])
        quat = (
            rotation.GetImaginary()[0],
            rotation.GetImaginary()[1],
            rotation.GetImaginary()[2],
            rotation.GetReal(),
        )
        env_xform = wp.transform(pos, quat)
        world_xforms.append(env_xform)
        builder.add_builder(proto, xform=env_xform)
        for label, xform in world_sites.items():
            if label not in local_site_map:
                local_site_map[label] = [[] for _ in range(num_worlds)]
            site_idx = builder.add_site(body=-1, xform=wp.transform_multiply(env_xform, xform), label=label)
            local_site_map[label][col].append(site_idx)
        for label, proto_shape_indices in site_entries.items():
            if label not in local_site_map:
                local_site_map[label] = [[] for _ in range(num_worlds)]
            for proto_shape_idx in proto_shape_indices:
                local_site_map[label][col].append(offset + proto_shape_idx)
        builder.end_world()

    NewtonManager._cl_site_index_map = {
        **global_site_map,
        **{label: (None, per_world) for label, per_world in local_site_map.items()},
    }
    NewtonManager._world_xforms = world_xforms
    NewtonManager._num_envs = len(env_paths)
    cls.set_builder(builder)


def patch_newton_manager_isaaclab_env_layout() -> None:
    """Route Newton import through ``/World/envs/env_N`` instead of flat full-stage load."""
    try:
        from isaaclab.sim.utils.stage import get_current_stage
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from pxr import UsdGeom
    except ImportError:
        return

    if getattr(NewtonManager, "_robofinals_isaaclab_env_patch", False):
        return

    original_instantiate = NewtonManager.instantiate_builder_from_stage.__func__

    @classmethod
    def instantiate_builder_from_stage_patched(cls):
        stage = get_current_stage()
        env_paths = _isaaclab_newton_env_paths(stage)
        if env_paths:
            up_axis = UsdGeom.GetStageUpAxis(stage)
            _newton_builder_from_env_paths(cls, stage, up_axis, env_paths)
            return
        return original_instantiate(cls)

    NewtonManager.instantiate_builder_from_stage = instantiate_builder_from_stage_patched
    NewtonManager._robofinals_isaaclab_env_patch = True


_NEWTON_SHAPE_MATERIAL_KE = 10000.0
_NEWTON_SHAPE_MATERIAL_KD = 500.0
_NEWTON_SHAPE_GAP = 0.001


def _apply_newton_shape_contact_material_overrides(
    model,
    *,
    ke: float = _NEWTON_SHAPE_MATERIAL_KE,
    kd: float = _NEWTON_SHAPE_MATERIAL_KD,
    gap: float = _NEWTON_SHAPE_GAP,
) -> None:
    """Set all shape contact ``ke`` / ``kd`` / ``gap`` on a finalized Newton model."""
    shape_count = int(getattr(model, "shape_count", 0))
    if shape_count <= 0:
        return

    for attr_name, value in (
        # ("shape_material_ke", ke),
        # ("shape_material_kd", kd),
        ("shape_gap", gap),
    ):
        arr = getattr(model, attr_name, None)
        if arr is None:
            continue
        if hasattr(arr, "fill_"):
            arr.fill_(float(value))
            continue
        import numpy as np

        filled = np.full(shape_count, float(value), dtype=np.float32)
        arr.assign(filled)

    print(
        f"[robofinals] Newton shape materials overridden: "
        f"ke={ke:g}, kd={kd:g}, gap={gap:g}, shape_count={shape_count}"
    )


def _resolve_newton_model_dump_dir() -> Path | None:
    """Return dump output dir when ``robofinals_NEWTON_MODEL_DUMP`` is enabled."""
    flag = os.environ.get("robofinals_NEWTON_MODEL_DUMP", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return None
    raw_dir = os.environ.get("robofinals_NEWTON_MODEL_DUMP_DIR", "model_dumps").strip()
    return Path(raw_dir or "model_dumps")


def _stage_usd_path(stage) -> Path:
    root_layer = stage.GetRootLayer()
    for candidate in (getattr(root_layer, "realPath", None), getattr(root_layer, "identifier", None)):
        if candidate:
            return Path(str(candidate))
    return Path("isaaclab_live_stage.usd")


def _maybe_dump_newton_model(cls) -> None:
    dump_dir = _resolve_newton_model_dump_dir()
    if dump_dir is None or cls._model is None:
        return
    if getattr(cls, "_robofinals_model_dump_done", False) and os.environ.get(
        "robofinals_NEWTON_MODEL_DUMP_ALWAYS", ""
    ).strip().lower() not in ("1", "true", "yes", "on"):
        return

    try:
        from isaaclab.physics import PhysicsManager
        from isaaclab.sim.utils.stage import get_current_stage

        from robofinals.core.backends.newton.model_dump import dump_newton_model

        stage = get_current_stage()
        usd_path = _stage_usd_path(stage)
        solver_name = "mujoco_warp"
        physics_cfg = getattr(PhysicsManager, "_cfg", None)
        solver_cfg = getattr(physics_cfg, "solver_cfg", None)
        if solver_cfg is not None:
            solver_name = str(getattr(solver_cfg, "solver_type", solver_name))

        dump_path = dump_newton_model(
            cls._model,
            usd_path=usd_path,
            solver_name=solver_name,
            output_dir=dump_dir,
            state=getattr(cls, "_state_0", None),
            solver=getattr(cls, "_solver", None),
            extra={
                "num_envs": getattr(cls, "_num_envs", None),
                "world_count": int(getattr(cls._model, "world_count", 0)),
            },
        )
        cls._robofinals_model_dump_done = True
        print(f"[robofinals] Newton model dumped to {dump_path}")
    except Exception as exc:
        print(f"[robofinals] Newton model dump failed: {exc}")


def patch_newton_manager_model_dump() -> None:
    """Dump finalized Newton model JSON after builder.finalize() when enabled."""
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager
    except ImportError:
        return

    if getattr(NewtonManager, "_robofinals_model_dump_patch", False):
        return

    original_start = NewtonManager.start_simulation.__func__

    @classmethod
    def start_simulation_patched(cls):
        original_start(cls)
        if cls._model is not None:
            _apply_newton_shape_contact_material_overrides(cls._model)
        _maybe_dump_newton_model(cls)

    NewtonManager.start_simulation = start_simulation_patched
    NewtonManager._robofinals_model_dump_patch = True


def patch_newton_model_builder_add_usd() -> None:
    """Default Newton stage import to skip visual meshes (Kit renders via RTX)."""
    try:
        from newton import ModelBuilder
    except ImportError:
        return
    from robofinals.core.backends.newton.mesh_approximation import (
        with_newton_asset_ignore_paths,
    )

    if getattr(ModelBuilder, "_robofinals_add_usd_patch", False):
        return

    original_add_usd = ModelBuilder.add_usd

    def add_usd_patched(self, *args, **kwargs):
        _pin_pxr_work_threads(1)
        kwargs.setdefault("load_visual_shapes", False)
        kwargs.setdefault("hide_collision_shapes", True)
        kwargs["ignore_paths"] = with_newton_asset_ignore_paths(
            kwargs.get("ignore_paths")
        )
        return original_add_usd(self, *args, **kwargs)

    ModelBuilder.add_usd = add_usd_patched
    ModelBuilder.rigid_gap = 0.001
    ModelBuilder._robofinals_add_usd_patch = True
