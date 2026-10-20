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

"""Schema-aware deformable loading for the Newton backend.

The source USD is referenced without generating or rewriting a simulation
mesh. Volume bodies select authored ``collision`` or ``simulation``
tetrahedral data. Current-schema surface bodies retain their authored triangle
simulation mesh and are registered as Newton cloth.

This spawner is still responsible for two pieces of stage setup that are not
solver geometry:

* locating the source deformable body and, for volume bodies only, authoring
  the Newton tetrahedral source selector on the owned wrapper root;
* binding a Newton deformable material to the wrapper root.

The original visual mesh, including its points, topology, primvars, and
material bindings, is never modified here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import DeformableObjectCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, find_first_matching_prim, get_current_stage
from isaaclab.utils.configclass import configclass

from .deformable_material_conversion import SurfaceMaterialInputs, convert_surface_material
from robofinals.utils.isaaclab_utils.assets.deformable_schema_utils import prim_has_applied_schema

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab_newton.sim.spawners.materials import (
        NewtonDeformableBodyMaterialCfg,
        NewtonSurfaceDeformableBodyMaterialCfg,
    )


logger = logging.getLogger(__name__)

DeformableSchema = Literal["auto", "legacy", "omniphysics"]
DeformableKind = Literal["volume", "surface"]
TetMeshSource = Literal["collision", "simulation"]

_LEGACY_BODY_API = "PhysxDeformableBodyAPI"
_CURRENT_BODY_API = "OmniPhysicsDeformableBodyAPI"
_CURRENT_VOLUME_SIM_API = "OmniPhysicsVolumeDeformableSimAPI"
_CURRENT_SURFACE_SIM_API = "OmniPhysicsSurfaceDeformableSimAPI"

_LEGACY_TET_ATTRIBUTES = {
    "collision": (
        ("physxDeformable:collisionRestPoints", "physxDeformable:collisionPoints"),
        "physxDeformable:collisionIndices",
    ),
    "simulation": (
        ("physxDeformable:simulationRestPoints", "physxDeformable:simulationPoints"),
        "physxDeformable:simulationIndices",
    ),
}


def _validate_child_name(name: str, field_name: str) -> None:
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"{field_name} must be one non-empty USD prim name, got {name!r}.")


def _attribute_has_values(prim, attribute_name: str) -> bool:
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        return False
    value = attribute.Get()
    return value is not None and len(value) > 0


def _has_legacy_tet_source(prim, tetmesh_source: TetMeshSource) -> bool:
    point_names, indices_name = _LEGACY_TET_ATTRIBUTES[tetmesh_source]
    return any(_attribute_has_values(prim, name) for name in point_names) and _attribute_has_values(
        prim, indices_name
    )


def _resolve_relative_prim(stage, source_prim, configured_path: str):
    path = configured_path
    if not path.startswith("/"):
        path = f"{source_prim.GetPath()}/{path.lstrip('/')}"
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"Configured deformable root prim '{path}' does not exist.")
    return prim


def _detect_deformable_root(
    stage,
    source_prim,
    schema: DeformableSchema,
    configured_path: str | None,
):
    """Return the source schema and its deformable body prim."""
    from pxr import Usd

    if schema not in ("auto", "legacy", "omniphysics"):
        raise ValueError(f"Unsupported deformable schema selector: {schema!r}.")

    if configured_path is not None:
        candidates = [_resolve_relative_prim(stage, source_prim, configured_path)]
    else:
        candidates = list(Usd.PrimRange(source_prim))

    legacy_api_roots = [
        prim for prim in candidates if prim_has_applied_schema(prim, _LEGACY_BODY_API)
    ]
    legacy_data_roots = [
        prim
        for prim in candidates
        if _has_legacy_tet_source(prim, "collision")
        or _has_legacy_tet_source(prim, "simulation")
    ]
    # Prefer the authored body API. Cooked arrays may live on a descendant
    # Mesh, and treating both prims as body roots would create a false
    # ambiguity.
    legacy_roots = legacy_api_roots if legacy_api_roots else legacy_data_roots
    current_roots = [prim for prim in candidates if prim_has_applied_schema(prim, _CURRENT_BODY_API)]

    if schema == "legacy":
        selected_schema, roots = "legacy", legacy_roots
    elif schema == "omniphysics":
        selected_schema, roots = "omniphysics", current_roots
    else:
        if legacy_roots and current_roots:
            raise RuntimeError(
                f"Found both legacy and current deformable roots below '{source_prim.GetPath()}'. "
                "Specify schema and deformable_root_prim_path explicitly."
            )
        if current_roots:
            selected_schema, roots = "omniphysics", current_roots
        else:
            selected_schema, roots = "legacy", legacy_roots

    if len(roots) != 1:
        paths = [str(prim.GetPath()) for prim in roots]
        raise RuntimeError(
            f"Expected exactly one {selected_schema} deformable root below '{source_prim.GetPath()}', "
            f"found {len(roots)}: {paths}."
        )
    return selected_schema, roots[0]


def _get_bound_physics_material(body_root):
    from pxr import Usd, UsdShade

    for candidate in Usd.PrimRange(body_root):
        if not candidate.HasAPI(UsdShade.MaterialBindingAPI):
            continue
        targets = UsdShade.MaterialBindingAPI(candidate).GetDirectBindingRel("physics").GetTargets()
        for target in targets:
            prim = body_root.GetStage().GetPrimAtPath(target)
            if prim.IsValid() and prim.IsA(UsdShade.Material):
                return prim
    return None


def _detect_deformable_kind(schema: str, body_root) -> DeformableKind:
    """Resolve volume versus surface without inferring from arbitrary meshes."""
    from pxr import Usd

    if schema == "legacy":
        # Legacy PhysX deformables handled by this adapter expose cooked
        # tetrahedral arrays and are always volume bodies.
        return "volume"

    candidates = list(Usd.PrimRange(body_root))
    volume_sim_prims = [
        prim for prim in candidates if prim_has_applied_schema(prim, _CURRENT_VOLUME_SIM_API)
    ]
    surface_sim_prims = [
        prim for prim in candidates if prim_has_applied_schema(prim, _CURRENT_SURFACE_SIM_API)
    ]
    if volume_sim_prims and surface_sim_prims:
        raise RuntimeError(
            f"Current deformable root '{body_root.GetPath()}' contains both volume and surface "
            "simulation APIs."
        )
    if len(volume_sim_prims) > 1:
        raise RuntimeError(
            f"Expected one volume simulation mesh below '{body_root.GetPath()}', found "
            f"{[str(prim.GetPath()) for prim in volume_sim_prims]}."
        )
    if len(surface_sim_prims) > 1:
        raise RuntimeError(
            f"Expected one surface simulation mesh below '{body_root.GetPath()}', found "
            f"{[str(prim.GetPath()) for prim in surface_sim_prims]}."
        )
    if volume_sim_prims:
        return "volume"
    if surface_sim_prims:
        return "surface"
    raise RuntimeError(
        f"Current deformable root '{body_root.GetPath()}' has neither "
        f"{_CURRENT_VOLUME_SIM_API} nor {_CURRENT_SURFACE_SIM_API} below it."
    )


def _max_world_point_error(points_a, prim_a, points_b, prim_b) -> float:
    """Return the largest world-space distance between paired mesh points."""
    from pxr import Gf, UsdGeom

    xform_cache = UsdGeom.XformCache()
    transform_a = xform_cache.GetLocalToWorldTransform(prim_a)
    transform_b = xform_cache.GetLocalToWorldTransform(prim_b)
    max_error = 0.0
    for point_a, point_b in zip(points_a, points_b, strict=True):
        world_a = transform_a.Transform(
            Gf.Vec3d(float(point_a[0]), float(point_a[1]), float(point_a[2]))
        )
        world_b = transform_b.Transform(
            Gf.Vec3d(float(point_b[0]), float(point_b[1]), float(point_b[2]))
        )
        max_error = max(max_error, float((world_a - world_b).GetLength()))
    return max_error


def _flatten_triangle_indices(indices) -> list[int]:
    """Normalize USD ``int3[]`` or flat integer arrays to one flat list."""
    flattened: list[int] = []
    for item in indices:
        try:
            flattened.extend(int(value) for value in item)
        except TypeError:
            flattened.append(int(item))
    return flattened


def _validate_surface_geometry(body_root, search_root=None):
    """Validate the one-to-one surface path supported by the Newton bridge.

    Surface tetrahedral embedding is intentionally out of scope. The authored
    simulation mesh must therefore be triangular and have a separate visual
    mesh whose rest points match one-to-one in world space.
    """
    from pxr import Usd, UsdGeom

    # The runtime bridge searches below the spawned wrapper, not only below
    # the body API prim. Scan the referenced source root when available so a
    # sibling Mesh cannot later be mistaken for the visual mesh.
    search_root = body_root if search_root is None else search_root
    descendants = list(Usd.PrimRange(search_root))
    mesh_prims = [prim for prim in descendants if prim.IsA(UsdGeom.Mesh)]
    sim_prims = [
        prim for prim in mesh_prims if prim_has_applied_schema(prim, _CURRENT_SURFACE_SIM_API)
    ]
    if len(sim_prims) != 1:
        raise RuntimeError(
            f"Expected exactly one {_CURRENT_SURFACE_SIM_API} Mesh below '{search_root.GetPath()}', "
            f"found {[str(prim.GetPath()) for prim in sim_prims]}."
        )
    sim_prim = sim_prims[0]
    visual_prims = [
        prim
        for prim in mesh_prims
        if prim is not sim_prim
        and not prim_has_applied_schema(prim, _CURRENT_VOLUME_SIM_API)
        and not prim_has_applied_schema(prim, _CURRENT_SURFACE_SIM_API)
    ]
    if len(visual_prims) != 1:
        raise RuntimeError(
            f"Surface deformable '{body_root.GetPath()}' currently requires exactly one separate "
            f"visual Mesh below '{search_root.GetPath()}'; found "
            f"{[str(prim.GetPath()) for prim in visual_prims]}."
        )
    visual_prim = visual_prims[0]

    sim_mesh = UsdGeom.Mesh(sim_prim)
    sim_points = sim_mesh.GetPointsAttr().Get()
    face_counts = sim_mesh.GetFaceVertexCountsAttr().Get()
    face_indices = sim_mesh.GetFaceVertexIndicesAttr().Get()
    if sim_points is None or len(sim_points) == 0:
        raise ValueError(f"Surface simulation mesh '{sim_prim.GetPath()}' has no points.")
    if face_counts is None or len(face_counts) == 0 or any(int(count) != 3 for count in face_counts):
        raise ValueError(
            f"Surface simulation mesh '{sim_prim.GetPath()}' must be fully triangulated."
        )
    if face_indices is None or len(face_indices) != 3 * len(face_counts):
        raise ValueError(
            f"Surface simulation mesh '{sim_prim.GetPath()}' has inconsistent triangle indices."
        )
    if min(face_indices) < 0 or max(face_indices) >= len(sim_points):
        raise ValueError(f"Surface simulation mesh '{sim_prim.GetPath()}' has out-of-range indices.")

    rest_points = sim_prim.GetAttribute("omniphysics:restShapePoints").Get()
    rest_indices = sim_prim.GetAttribute("omniphysics:restTriVtxIndices").Get()
    if rest_points is None or len(rest_points) != len(sim_points):
        raise ValueError(
            f"Surface simulation mesh '{sim_prim.GetPath()}' must author "
            "omniphysics:restShapePoints with one entry per point."
        )
    if rest_indices is None or _flatten_triangle_indices(rest_indices) != list(face_indices):
        raise ValueError(
            f"Surface simulation mesh '{sim_prim.GetPath()}' must author "
            "omniphysics:restTriVtxIndices matching faceVertexIndices."
        )
    rest_error = _max_world_point_error(sim_points, sim_prim, rest_points, sim_prim)
    if rest_error > 1.0e-6:
        raise ValueError(
            f"Surface simulation mesh '{sim_prim.GetPath()}' starts with points that differ from "
            f"its rest shape by up to {rest_error:.6g} m; the current Newton bridge does not import "
            "a separate surface rest state."
        )

    visual_points = UsdGeom.Mesh(visual_prim).GetPointsAttr().Get()
    if visual_points is None or len(visual_points) != len(sim_points):
        visual_count = 0 if visual_points is None else len(visual_points)
        raise ValueError(
            f"Surface visual mesh '{visual_prim.GetPath()}' has {visual_count} points but simulation "
            f"mesh '{sim_prim.GetPath()}' has {len(sim_points)}. Surface embedding is not supported."
        )
    visual_error = _max_world_point_error(sim_points, sim_prim, visual_points, visual_prim)
    if visual_error > 1.0e-6:
        raise ValueError(
            f"Surface visual mesh '{visual_prim.GetPath()}' is not a one-to-one rest match for "
            f"'{sim_prim.GetPath()}'; maximum point error is {visual_error:.6g} m. "
            "Surface embedding is not supported."
        )
    return sim_prim, visual_prim


def _read_number(prim, *attribute_names: str):
    if prim is None:
        return None
    for name in attribute_names:
        attribute = prim.GetAttribute(name)
        if attribute.IsValid():
            value = attribute.Get()
            if value is not None:
                return float(value)
    return None


def _derive_newton_volume_material(body_root):
    """Convert supported source values into a Newton volume material cfg."""
    from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

    cfg = NewtonDeformableBodyMaterialCfg()
    material_prim = _get_bound_physics_material(body_root)
    if material_prim is None:
        return cfg

    density = _read_number(
        material_prim,
        "newton:density",
        "omniphysics:density",
        "physxDeformableBodyMaterial:density",
    )
    if density is not None and density > 0.0:
        cfg.density = density

    particle_radius = _read_number(material_prim, "newton:particleRadius")
    if particle_radius is not None and particle_radius > 0.0:
        cfg.particle_radius = particle_radius

    k_mu = _read_number(material_prim, "newton:kMu")
    k_lambda = _read_number(material_prim, "newton:kLambda")
    if k_mu is not None and k_mu >= 0.0:
        cfg.k_mu = k_mu
    if k_lambda is not None and k_lambda >= 0.0:
        cfg.k_lambda = k_lambda

    if k_mu is None or k_lambda is None:
        youngs_modulus = _read_number(
            material_prim,
            "omniphysics:youngsModulus",
            "physxDeformableBodyMaterial:youngsModulus",
        )
        poissons_ratio = _read_number(
            material_prim,
            "omniphysics:poissonsRatio",
            "physxDeformableBodyMaterial:poissonsRatio",
        )
        if youngs_modulus is not None and youngs_modulus >= 0.0 and poissons_ratio is not None:
            if not -1.0 < poissons_ratio < 0.5:
                raise ValueError(
                    f"Cannot convert Poisson's ratio {poissons_ratio} on "
                    f"'{material_prim.GetPath()}' to Lame parameters."
                )
            if k_mu is None:
                cfg.k_mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
            if k_lambda is None:
                cfg.k_lambda = youngs_modulus * poissons_ratio / (
                    (1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio)
                )

    k_damp = _read_number(material_prim, "newton:kDamp")
    if k_damp is not None and k_damp >= 0.0:
        cfg.k_damp = k_damp
    return cfg


def _derive_newton_surface_material(body_root):
    """Convert current-schema shell material values into Newton cloth values."""
    from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

    cfg = NewtonSurfaceDeformableBodyMaterialCfg()
    material_prim = _get_bound_physics_material(body_root)
    if material_prim is None:
        logger.warning(
            "Surface deformable '%s' has no bound source physics material; using Newton defaults.",
            body_root.GetPath(),
        )
        return cfg

    explicit_density = _read_number(material_prim, "newton:density")
    explicit_particle_radius = _read_number(material_prim, "newton:particleRadius")
    explicit_tri_ke = _read_number(material_prim, "newton:triKe")
    explicit_tri_ka = _read_number(material_prim, "newton:triKa")
    explicit_tri_kd = _read_number(material_prim, "newton:triKd")
    explicit_edge_ke = _read_number(material_prim, "newton:edgeKe")
    explicit_edge_kd = _read_number(material_prim, "newton:edgeKd")

    source_density = _read_number(
        material_prim,
        "omniphysics:density",
        "physics:density",
        "physxDeformableBodyMaterial:density",
    )
    if source_density is None:
        source_density = _read_number(
            body_root,
            "omniphysics:density",
            "physics:density",
            "physxDeformableBody:density",
        )

    converted = convert_surface_material(
        SurfaceMaterialInputs(
            thickness=_read_number(
                material_prim,
                "omniphysics:surfaceThickness",
                "physics:thickness",
            ),
            density=source_density,
            youngs_modulus=_read_number(
                material_prim,
                "omniphysics:youngsModulus",
                "physics:youngsModulus",
                "physxDeformableBodyMaterial:youngsModulus",
            ),
            poissons_ratio=_read_number(
                material_prim,
                "omniphysics:poissonsRatio",
                "physics:poissonsRatio",
                "physxDeformableBodyMaterial:poissonsRatio",
            ),
            stretch_stiffness=_read_number(
                material_prim,
                "omniphysics:surfaceStretchStiffness",
                "physics:stretchStiffness",
            ),
            shear_stiffness=_read_number(
                material_prim,
                "omniphysics:surfaceShearStiffness",
                "physics:shearStiffness",
            ),
            bend_stiffness=_read_number(
                material_prim,
                "omniphysics:surfaceBendStiffness",
                "physics:bendStiffness",
            ),
        )
    )

    if explicit_density is not None and explicit_density > 0.0:
        # An explicitly authored Newton surface density is already areal.
        cfg.density = explicit_density
    elif converted.density is not None:
        cfg.density = converted.density

    if explicit_particle_radius is not None and explicit_particle_radius > 0.0:
        cfg.particle_radius = explicit_particle_radius
    else:
        cfg.particle_radius = converted.particle_radius

    for explicit, converted_value, field_name in (
        (explicit_tri_ke, converted.tri_ke, "tri_ke"),
        (explicit_tri_ka, converted.tri_ka, "tri_ka"),
        (explicit_edge_ke, converted.edge_ke, "edge_ke"),
    ):
        if explicit is not None and explicit >= 0.0:
            setattr(cfg, field_name, explicit)
        elif converted_value is not None:
            setattr(cfg, field_name, converted_value)

    if explicit_tri_kd is not None and explicit_tri_kd >= 0.0:
        cfg.tri_kd = explicit_tri_kd
    if explicit_edge_kd is not None and explicit_edge_kd >= 0.0:
        cfg.edge_kd = explicit_edge_kd

    if converted.used_default_thickness:
        logger.warning(
            "Surface material '%s' has no positive thickness; assuming %.6g m for Newton "
            "mass, stiffness, and collision-radius conversion.",
            material_prim.GetPath(),
            converted.thickness,
        )
    if converted.ignored_shear_stiffness is not None:
        logger.warning(
            "Surface material '%s' authors shear stiffness %.6g, but Newton's isotropic "
            "add_cloth_mesh interface cannot represent an independent shear coefficient; "
            "the value is ignored.",
            material_prim.GetPath(),
            converted.ignored_shear_stiffness,
        )
    return cfg


def _derive_newton_material(body_root, deformable_kind: DeformableKind):
    if deformable_kind == "surface":
        return _derive_newton_surface_material(body_root)
    return _derive_newton_volume_material(body_root)


def _bind_newton_deformable_material(
    stage,
    root_prim,
    *,
    physics_material_prim_name: str,
    material_cfg,
) -> None:
    """Create and bind a Newton deformable physics material on ``root_prim``."""
    from isaaclab.sim.spawners.materials.physics_materials import spawn_deformable_body_material
    from pxr import Sdf, UsdShade

    _validate_child_name(physics_material_prim_name, "physics_material_prim_name")
    prim_path = str(root_prim.GetPath())
    material_path = f"{prim_path}/{physics_material_prim_name}"
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        spawn_deformable_body_material(material_path, material_cfg)
        material_prim = stage.GetPrimAtPath(material_path)
    material_binding_api = UsdShade.MaterialBindingAPI.Apply(root_prim)
    material_binding_api.Bind(
        UsdShade.Material(material_prim),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    material_targets = UsdShade.MaterialBindingAPI(root_prim).GetDirectBindingRel("physics").GetTargets()
    if Sdf.Path(material_path) not in material_targets:
        raise RuntimeError(f"Failed to bind Newton physics material '{material_path}' to '{prim_path}'.")


def _prepare_schema_aware_deformable_stage(
    stage,
    root_prim,
    *,
    schema: DeformableSchema,
    tetmesh_source: TetMeshSource,
    deformable_root_prim_path: str | None,
    physics_material_prim_name: str,
    physics_material,
    prim_path_label: str | None = None,
):
    """Author Newton runtime metadata on an already composed deformable prim."""
    from pxr import Sdf

    prim_path_label = prim_path_label or str(root_prim.GetPath())
    schema, body_root = _detect_deformable_root(
        stage,
        root_prim,
        schema,
        deformable_root_prim_path,
    )
    deformable_kind = _detect_deformable_kind(schema, body_root)
    surface_sim_prim = None
    surface_visual_prim = None
    if deformable_kind == "surface":
        surface_sim_prim, surface_visual_prim = _validate_surface_geometry(body_root, root_prim)
    else:
        tet_attr = root_prim.GetAttribute("newton:tetMeshSource")
        if not tet_attr.IsValid():
            tet_attr = root_prim.CreateAttribute("newton:tetMeshSource", Sdf.ValueTypeNames.String)
        tet_attr.Set(tetmesh_source)

    material_cfg = (
        physics_material
        if physics_material is not None
        else _derive_newton_material(body_root, deformable_kind)
    )
    _bind_newton_deformable_material(
        stage,
        root_prim,
        physics_material_prim_name=physics_material_prim_name,
        material_cfg=material_cfg,
    )

    if deformable_kind == "surface":
        logger.info(
            "Prepared %s Newton surface deformable '%s': source_root=%s, sim_mesh=%s, "
            "visual_mesh=%s. No source mesh or visual topology was modified.",
            schema,
            prim_path_label,
            body_root.GetPath(),
            surface_sim_prim.GetPath(),
            surface_visual_prim.GetPath(),
        )
    else:
        logger.info(
            "Prepared %s Newton volume deformable '%s': tetmesh_source=%s, source_root=%s. "
            "No USD simulation TetMesh was generated.",
            schema,
            prim_path_label,
            tetmesh_source,
            body_root.GetPath(),
        )
    return root_prim


@clone
def spawn_schema_aware_deformable_from_usd(
    prim_path: str,
    cfg: SchemaAwareDeformableUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Reference a deformable USD without generating a simulation TetMesh."""
    del kwargs
    for field_name in ("source_prim_name", "physics_material_prim_name"):
        _validate_child_name(getattr(cfg, field_name), field_name)
    if cfg.source_prim_name == cfg.physics_material_prim_name:
        raise ValueError("Source and physics material prim names must be unique.")

    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if root_prim.IsValid():
        raise ValueError(f"A prim already exists at schema-aware deformable path '{prim_path}'.")
    root_prim = create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
        scale=cfg.scale,
        stage=stage,
    )

    source_path = f"{prim_path}/{cfg.source_prim_name}"
    source_prim = spawn_from_usd(source_path, UsdFileCfg(usd_path=cfg.usd_path, scale=None))
    return _prepare_schema_aware_deformable_stage(
        stage,
        root_prim,
        schema=cfg.schema,
        tetmesh_source=cfg.tetmesh_source,
        deformable_root_prim_path=cfg.deformable_root_prim_path,
        physics_material_prim_name=cfg.physics_material_prim_name,
        physics_material=cfg.physics_material,
        prim_path_label=prim_path,
    )


@clone
def spawn_schema_aware_deformable_on_existing_prim(
    prim_path: str,
    cfg: "SchemaAwareDeformableReferenceCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> "Usd.Prim":
    """Prepare a deformable prim that already exists in the composed scene."""
    del kwargs, translation, orientation

    stage = get_current_stage()
    root_prim = find_first_matching_prim(prim_path, stage)
    if root_prim is None or not root_prim.IsValid():
        raise ValueError(f"Deformable reference prim not found for expression '{prim_path}'.")
    return _prepare_schema_aware_deformable_stage(
        stage,
        root_prim,
        schema=cfg.schema,
        tetmesh_source=cfg.tetmesh_source,
        deformable_root_prim_path=cfg.deformable_root_prim_path,
        physics_material_prim_name=cfg.physics_material_prim_name,
        physics_material=cfg.physics_material,
        prim_path_label=prim_path,
    )


@configclass
class SchemaAwareDeformableUsdFileCfg(SpawnerCfg):
    """Spawner configuration for legacy/current Newton deformables."""

    func: Callable = spawn_schema_aware_deformable_from_usd

    usd_path: str = MISSING
    """Source USD containing exactly one deformable body."""

    schema: DeformableSchema = "auto"
    """Expected source schema. ``auto`` detects it after referencing the USD."""

    tetmesh_source: TetMeshSource = "collision"
    """Tet data used to construct the Newton soft body.

    ``collision`` is the default because it generally conforms more closely to
    the visual surface. ``simulation`` selects the authored simulation tet data.
    This setting is ignored for current-schema surface deformables.
    """

    scale: tuple[float, float, float] | None = None
    """Scale applied to the wrapper root."""

    deformable_root_prim_path: str | None = None
    """Optional source-relative or absolute deformable root path."""

    source_prim_name: str = "source"
    physics_material_prim_name: str = "physicsMaterial"

    physics_material: object | None = None
    """Optional Newton material cfg; source material values are converted when omitted."""


@configclass
class SchemaAwareDeformableReferenceCfg(SpawnerCfg):
    """Spawner configuration for deformables already authored in a scene USD."""

    func: Callable = spawn_schema_aware_deformable_on_existing_prim

    schema: DeformableSchema = "auto"
    """Expected source schema. ``auto`` detects it on the referenced prim."""

    tetmesh_source: TetMeshSource = "collision"
    """Tet data used to construct a Newton volume deformable."""

    deformable_root_prim_path: str | None = None
    """Optional source-relative or absolute deformable root path."""

    physics_material_prim_name: str = "physicsMaterial"
    physics_material: object | None = None
    """Optional Newton material cfg; source material values are converted when omitted."""


def make_schema_aware_deformable_object_cfg(
    *,
    usd_path: str,
    prim_path: str,
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: tuple[float, float, float] | None = None,
    schema: DeformableSchema = "auto",
    tetmesh_source: TetMeshSource = "collision",
    deformable_root_prim_path: str | None = None,
    physics_material: NewtonDeformableBodyMaterialCfg | NewtonSurfaceDeformableBodyMaterialCfg | None = None,
    debug_vis: bool = False,
) -> DeformableObjectCfg:
    """Create a backend-dispatched cfg for one Newton deformable."""
    spawn_cfg = SchemaAwareDeformableUsdFileCfg(
        usd_path=usd_path,
        scale=scale,
        schema=schema,
        tetmesh_source=tetmesh_source,
        deformable_root_prim_path=deformable_root_prim_path,
        physics_material=physics_material,
    )
    return DeformableObjectCfg(
        prim_path=prim_path,
        spawn=spawn_cfg,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
        debug_vis=debug_vis,
    )


__all__ = [
    "DeformableKind",
    "DeformableSchema",
    "TetMeshSource",
    "SchemaAwareDeformableReferenceCfg",
    "SchemaAwareDeformableUsdFileCfg",
    "make_schema_aware_deformable_object_cfg",
    "spawn_schema_aware_deformable_from_usd",
    "spawn_schema_aware_deformable_on_existing_prim",
]
