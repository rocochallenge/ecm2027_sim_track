
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

import math
import os
from turtle import st

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdSkel

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None

import robofinals.utils.math_utils.transform_utils.numpy_impl as T


def parse_size_attr(size_value) -> np.ndarray | None:
    """Parse fixture ``size`` attribute (``\"x,y,z\"`` string) into a length-3 array.

    Returns ``None`` for missing values and invalid strings such as ``\"None, None, None\"``.
    """
    if size_value is None:
        return None
    if not isinstance(size_value, str):
        return None
    size_value = size_value.strip()
    if not size_value:
        return None
    parts = [part.strip() for part in size_value.split(",")]
    if len(parts) != 3:
        return None
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    if any(not np.isfinite(value) for value in values):
        return None
    return np.array(values, dtype=np.float64)


class OpenUsd:
    """USD utility class - encapsulates all USD operation methods"""

    def __init__(self, usd_path=None):
        self.stage = None
        if usd_path:
            self.stage = self.get_stage(usd_path)

    @staticmethod
    def get_stage(usd_path):
        """Get USD Stage"""
        stage = Usd.Stage.Open(usd_path)
        return stage

    @staticmethod
    def get_all_prims(stage, prim=None, prims_list=None):
        """Get all prims"""
        if prims_list is None:
            prims_list = []
        if prim is None:
            prim = stage.GetPseudoRoot()
        for child in prim.GetAllChildren():
            prims_list.append(child)
            OpenUsd.get_all_prims(stage, child, prims_list)
        return prims_list

    @staticmethod
    def classify_prim(prim):
        """Classify prim"""
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return "Articulation"
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return "RigidBody"
        else:
            return "Normal"

    _VISUAL_ONLY_SCOPE_NAMES = frozenset({"physics", "collisions", "collision"})
    _VISUAL_ONLY_NAME_TOKENS = ("collider", "collision", "collisions")

    @staticmethod
    def _strip_physics_apis(prim):
        """Remove rigid-body / collision / articulation APIs from a prim."""
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if PhysxSchema:
                prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            if PhysxSchema:
                prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if PhysxSchema:
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        for attr in ("physxContactReportThreshold", "physxContactReportForceThreshold"):
            if prim.HasAttribute(attr):
                prim.RemoveProperty(attr)
        if PhysxSchema and prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            prim.RemoveAPI(PhysxSchema.PhysxContactReportAPI)

    @staticmethod
    def _disable_collision_prim(prim):
        """Disable a collider mesh so Newton/PhysX no longer treat it as a shape."""
        OpenUsd._strip_physics_apis(prim)
        collision_attr = prim.GetAttribute("physics:collisionEnabled")
        if collision_attr and collision_attr.IsValid():
            collision_attr.Set(False)
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding("physics")
        physics_binding = prim.GetRelationship("material:binding:physics")
        if physics_binding and physics_binding.IsValid():
            physics_binding.SetTargets([])

    @staticmethod
    def _strip_fixture_to_visual_only(fixture_prim):
        """Strip all physics from a fixture subtree while keeping visuals.

        Payload-composed assets cannot be deleted with ``RemovePrim``; we instead
        deactivate physics scopes and disable every collider mesh so Newton does
        not try to resolve orphaned ``material:binding:physics`` targets.
        """
        OpenUsd._strip_physics_apis(fixture_prim)

        descendants = list(Usd.PrimRange(fixture_prim))
        # Deepest prims first so deactivating parent scopes does not invalidate
        # child prim handles before collider attributes are overridden.
        descendants.sort(key=lambda prim: len(str(prim.GetPath())), reverse=True)

        for prim in descendants:
            if prim == fixture_prim:
                continue
            if not prim.IsValid():
                continue

            name_l = prim.GetName().lower()
            collision_attr = prim.GetAttribute("physics:collisionEnabled")
            if collision_attr and collision_attr.IsValid() and collision_attr.Get():
                OpenUsd._disable_collision_prim(prim)

            is_physics_scope = name_l in OpenUsd._VISUAL_ONLY_SCOPE_NAMES or any(
                token in name_l for token in OpenUsd._VISUAL_ONLY_NAME_TOKENS
            )
            if is_physics_scope:
                prim.SetActive(False)
                continue

            if UsdPhysics.Joint(prim) or "joint" in name_l:
                prim.SetActive(False)
                continue

            OpenUsd._strip_physics_apis(prim)

            if prim.GetTypeName() == "PhysicsScene":
                prim.SetActive(False)

    @staticmethod
    def usd_simplify(stage, ref_fixture_names, prim=None, preserve_keywords=None):
        """Remove rigidbody/collision/articulation APIs from non-task fixtures.

        Prims whose name appears in ``ref_fixture_names`` keep full physics.
        Prims whose name contains a ``preserve_keywords`` token also keep physics;
        the default keeps floor contact only (wall/ceiling colliders are stripped).
        """
        if preserve_keywords is None:
            preserve_keywords = ("floor",)
        if prim is None:
            prim = stage.GetPseudoRoot()
        parent_is_world = prim.GetName() == "World"
        for child in prim.GetAllChildren():
            if any(name == child.GetName() for name in ref_fixture_names):
                continue
            if not child.IsValid():
                continue
            name_l = child.GetName().lower()
            if preserve_keywords and any(kw in name_l for kw in preserve_keywords):
                OpenUsd.usd_simplify(
                    stage,
                    ref_fixture_names,
                    child,
                    preserve_keywords=preserve_keywords,
                )
                continue
            if parent_is_world:
                OpenUsd._strip_fixture_to_visual_only(child)
                continue

            OpenUsd._strip_physics_apis(child)
            if "joint" in name_l or UsdPhysics.Joint(child):
                child.SetActive(False)
                continue
            if name_l in OpenUsd._VISUAL_ONLY_SCOPE_NAMES or any(
                token in name_l for token in OpenUsd._VISUAL_ONLY_NAME_TOKENS
            ):
                collision_attr = child.GetAttribute("physics:collisionEnabled")
                if collision_attr and collision_attr.IsValid():
                    OpenUsd._disable_collision_prim(child)
                child.SetActive(False)
                continue
            OpenUsd.usd_simplify(
                stage,
                ref_fixture_names,
                child,
                preserve_keywords=preserve_keywords,
            )
        return stage

    @staticmethod
    def collect_physics_fixture_prim_names(
        stage,
        layout_registry_names,
        *,
        enable_fixtures=None,
        preferred_fixture_names=None,
    ) -> list[str] | None:
        """Map task fixture requirements to top-level USD prim names that keep physics."""
        from robofinals.core.models.fixtures.fixture import FIXTURES
        from robofinals.core.models.fixtures.fixture_types import FixtureType

        keep_names: set[str] = set(enable_fixtures or [])
        keep_names.update(name for name in (preferred_fixture_names or []) if name)

        if not layout_registry_names:
            return sorted(keep_names) if keep_names else None

        wanted_types: set[FixtureType] = set()
        for entry in layout_registry_names:
            wanted_types.add(entry if isinstance(entry, FixtureType) else FixtureType(entry))

        allowed_usd_types: set[str] = set()
        for cls_name, cls in FIXTURES.items():
            if any(ft in cls.fixture_types for ft in wanted_types):
                allowed_usd_types.add(cls_name)

        root_children = stage.GetPseudoRoot().GetChildren()
        if not root_children:
            return sorted(keep_names) if keep_names else None
        root_prim = root_children[0]

        type_matched_infos = [
            info
            for info in OpenUsd.get_child_xform_infos(root_prim)
            if (info.get("type") or "") in allowed_usd_types
        ]

        preferred = [name for name in (preferred_fixture_names or []) if name]
        by_type: dict[str, list[str]] = {}
        for info in type_matched_infos:
            by_type.setdefault(info["type"], []).append(info["name"])

        selected: list[str] = []
        for usd_type in allowed_usd_types:
            candidates = by_type.get(usd_type, [])
            if not candidates:
                continue
            chosen = [name for name in candidates if name in preferred]
            if not chosen and preferred:
                for pref in preferred:
                    chosen = [
                        name
                        for name in candidates
                        if pref in name or name.startswith(pref)
                    ]
                    if chosen:
                        break
            selected.extend(chosen if chosen else candidates)

        keep_names.update(selected)
        return sorted(keep_names)

    @staticmethod
    def activate_prim(stage, name):
        import omni
        root_prim = stage.GetPseudoRoot()
        all_prims = OpenUsd.get_prim_by_prefix(root_prim, name)
        for prim in all_prims:
            prim.SetActive(True)
        return stage

    @staticmethod
    def deactivate_prim(stage, name):
        import omni
        root_prim = stage.GetPseudoRoot()
        all_prims = OpenUsd.get_prim_by_prefix(root_prim, name)
        for prim in all_prims:
            prim.SetActive(False)
        return stage

    @staticmethod
    def has_articulation_root(prim):
        """Recursively check if prim or any of its descendants has ArticulationRootAPI"""
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return True
        for child in prim.GetAllChildren():
            if OpenUsd.has_articulation_root(child):
                return True
        return False

    @staticmethod
    def is_articulation_root(prim):
        """Check if prim is articulation root"""
        return prim.HasAPI(UsdPhysics.ArticulationRootAPI)

    @staticmethod
    def fix_joint_drive_properties(
        stage,
        default_revolute_damping: float = 2.0,
        default_prismatic_damping: float = 2.0,
        default_max_force: float = 1000.0,
        min_joint_friction: float = 1.0,
        log_prefix: str = "[usd-fix]",
    ) -> dict:
        """In-place patch every revolute/prismatic joint on ``stage`` so PhysX
        sees a usable drive / friction configuration.

        Solves three independent issues that conspire to make passive scene
        articulations (drawers, cabinet doors, microwaves, ovens, fridges, …)
        slide / flop around freely under IsaacLab 3.x + PhysX 5+:

        1. **drive axis migration** — PhysX 5+ only honors a joint ``DriveAPI``
           whose instance name is ``angular`` (revolute) or ``linear``
           (prismatic). Asset packs that author ``drive:X:*`` (a non-standard
           but common convention in RoboCasa-style scenes) are silently
           ignored, leaving stiffness / damping / maxForce / targetPosition at
           zero at runtime. We copy them onto a freshly applied standard-axis
           DriveAPI. The original ``DriveAPI:X`` is left untouched.

        2. **missing DriveAPI** — joints with no DriveAPI at all (e.g. raw
           prismatic drawer joints) get a default DriveAPI applied with
           ``stiffness=0`` (no positional drive) and a small damping so they
           are not perfectly frictionless.

        3. **jointFriction floor** — ``physxJoint:jointFriction`` values
           below ``min_joint_friction`` (including ``None``) are bumped up to
           that floor. If ``PhysxJointAPI`` is not applied, it is applied.

        This relies on USD's layer cache: editing the in-memory layer
        underlying ``stage`` (typically the cached root layer for the USD
        file) is observable by anyone who later opens or references the same
        path — including IsaacLab's ``UsdFileCfg.spawn`` add_reference, which
        is how the modifications reach the live simulation.

        Args:
            stage: A ``Usd.Stage`` to patch in-place.
            default_revolute_damping: Damping injected into freshly added
                DriveAPI on revolute joints. Also used as the floor — joints
                with an existing damping below this value are NOT bumped (we
                only fill in zeros / missing values).
            default_prismatic_damping: Same, for prismatic joints (default
                is higher since prismatic drawers under gravity need more).
            default_max_force: ``maxForce`` for freshly added DriveAPI.
            min_joint_friction: Floor for ``physxJoint:jointFriction``.
            log_prefix: Prefix used in the summary print.

        Returns:
            ``dict`` with keys ``axis_migrated``, ``drive_added``,
            ``friction_bumped``, ``joints_seen`` (counts).
        """
        try:
            from pxr import UsdPhysics as _UsdPhysics
        except ImportError:
            return {"axis_migrated": 0, "drive_added": 0, "friction_bumped": 0, "joints_seen": 0}
        try:
            from pxr import PhysxSchema as _PS
        except ImportError:
            _PS = None  # type: ignore[assignment]

        type_to_axis: dict[str, tuple[str, float]] = {
            "PhysicsRevoluteJoint": ("angular", default_revolute_damping),
            "PhysicsPrismaticJoint": ("linear", default_prismatic_damping),
        }
        drive_props_to_copy = (
            ("stiffness", "CreateStiffnessAttr"),
            ("damping", "CreateDampingAttr"),
            ("maxForce", "CreateMaxForceAttr"),
            ("targetPosition", "CreateTargetPositionAttr"),
            ("targetVelocity", "CreateTargetVelocityAttr"),
        )

        counts = {"axis_migrated": 0, "drive_added": 0, "friction_bumped": 0, "joints_seen": 0}

        for prim in stage.Traverse():
            info = type_to_axis.get(str(prim.GetTypeName()))
            if info is None:
                continue
            target_axis, default_damping = info
            counts["joints_seen"] += 1

            applied = set(prim.GetAppliedSchemas())
            has_x_drive = "PhysicsDriveAPI:X" in applied
            has_target_drive = f"PhysicsDriveAPI:{target_axis}" in applied

            # (1) axis migration
            if has_x_drive and not has_target_drive:
                new_drive = _UsdPhysics.DriveAPI.Apply(prim, target_axis)
                for prop_name, create_attr_method in drive_props_to_copy:
                    src_attr = prim.GetAttribute(f"drive:X:physics:{prop_name}")
                    if not src_attr or not src_attr.IsValid():
                        continue
                    val = src_attr.Get()
                    if val is None:
                        continue
                    getattr(new_drive, create_attr_method)(val)
                type_src = prim.GetAttribute("drive:X:physics:type")
                if type_src and type_src.IsValid():
                    tv = type_src.Get()
                    if tv is not None:
                        new_drive.CreateTypeAttr(tv)
                counts["axis_migrated"] += 1
                has_target_drive = True

            # (2) default DriveAPI for joints with no drive at all
            if not has_target_drive:
                new_drive = _UsdPhysics.DriveAPI.Apply(prim, target_axis)
                new_drive.CreateStiffnessAttr(0.0)
                new_drive.CreateDampingAttr(default_damping)
                new_drive.CreateMaxForceAttr(default_max_force)
                counts["drive_added"] += 1

            # (3) jointFriction floor
            if _PS is not None:
                has_physx_joint = prim.HasAPI(_PS.PhysxJointAPI)
                fric_attr = prim.GetAttribute("physxJoint:jointFriction")
                current = fric_attr.Get() if fric_attr and fric_attr.IsValid() else None
                if (current is None or current < min_joint_friction):
                    px = _PS.PhysxJointAPI.Apply(prim) if not has_physx_joint else _PS.PhysxJointAPI(prim)
                    px.CreateJointFrictionAttr(min_joint_friction)
                    counts["friction_bumped"] += 1

        print(
            f"{log_prefix} fix_joint_drive_properties: "
            f"axis_migrated={counts['axis_migrated']}  "
            f"drive_added={counts['drive_added']}  "
            f"friction_bumped={counts['friction_bumped']}  "
            f"joints_seen={counts['joints_seen']}"
        )
        return counts

    @staticmethod
    def fix_drive_axis_naming(stage) -> int:
        """Backward-compatible alias — calls :func:`fix_joint_drive_properties`
        and returns the number of axis-migrated joints only.
        """
        return OpenUsd.fix_joint_drive_properties(stage)["axis_migrated"]

    @staticmethod
    def has_rigidbody_api(prim):
        """Recursively check if prim or any of its descendants has RigidBodyAPI"""
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        for child in prim.GetAllChildren():
            if OpenUsd.has_rigidbody_api(child):
                return True
        return False

    @staticmethod
    def has_deformable_body_api(prim):
        """Recursively check if prim or any of its descendants has deformable body API."""
        if OpenUsd.is_deformable(prim):
            return True
        deformable_body_api = getattr(UsdPhysics, "DeformableBodyAPI", None)
        if deformable_body_api is not None and prim.HasAPI(deformable_body_api):
            return True
        for child in prim.GetAllChildren():
            if OpenUsd.has_deformable_body_api(child):
                return True
        return False

    @staticmethod
    def is_rigidbody(prim):
        """Check if prim is rigidbody"""
        return prim.HasAPI(UsdPhysics.RigidBodyAPI)

    @staticmethod
    def has_api_schema_name(prim, *schema_names):
        """Check authored API schema names for schemas missing in older pxr bindings."""
        api_schemas = prim.GetMetadata("apiSchemas")
        if api_schemas is None:
            return False
        api_schemas_text = str(api_schemas)
        return any(schema_name in api_schemas_text for schema_name in schema_names)

    @staticmethod
    def is_legacy_deformable(prim):
        """Check if prim is an Isaac Sim 4.x PhysX deformable body."""
        deformable_body_api = getattr(PhysxSchema, "PhysxDeformableBodyAPI", None) if PhysxSchema else None
        if deformable_body_api is None:
            return False
        return prim.HasAPI(deformable_body_api)

    @staticmethod
    def is_omniphysics_deformable(prim):
        """Check if prim is an Isaac Sim 5.x OmniPhysics deformable body."""
        return OpenUsd.has_api_schema_name(
            prim,
            "OmniPhysicsDeformableBodyAPI",
            "PhysxBaseDeformableBodyAPI",
            "PhysxAutoDeformableBodyAPI",
        )

    @staticmethod
    def is_deformable(prim):
        """Check if prim is any supported deformable body."""
        return OpenUsd.is_legacy_deformable(prim) or OpenUsd.is_omniphysics_deformable(prim)

    @staticmethod
    def find_articulation_root(prim):
        """Find articulation root"""
        if OpenUsd.is_articulation_root(prim):
            return prim
        for child in prim.GetAllChildren():
            articulation_root = OpenUsd.find_articulation_root(child)
            if articulation_root:
                return articulation_root
        return None

    @staticmethod
    def get_all_joints(stage):
        """Get all joints"""
        joints = []

        def recurse(prim):
            # Check if it's a Joint
            if UsdPhysics.Joint(prim):
                joints.append(prim)
            for child in prim.GetAllChildren():
                recurse(child)
        recurse(stage.GetPseudoRoot())
        return joints

    @staticmethod
    def get_prim_pos_rot_in_world(prim):
        """Get prim position, rotation and scale in world coordinates"""
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return None, None, None
        matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        try:
            pos, rot, scale = UsdSkel.DecomposeTransform(matrix)
            # pos = matrix.ExtractTranslation()
            # rot = matrix.ExtractRotationQuat()
            pos_list = list(pos)
            quat_list = [rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2]]  # wxyz
            scale_list = list(scale)
            return pos_list, quat_list, scale_list
        except Exception as e:
            print(f"Error decomposing transform for {prim.GetName()}: {e}")
            return None, None, None

    @staticmethod
    def usd_quat_wxyz_to_isaac_init_rot(quat_wxyz):
        """USD / GfQuat here is (w,x,y,z); Isaac Lab ``init_state.rot`` / root pose use (x,y,z,w)."""
        w, x, y, z = quat_wxyz
        return (float(x), float(y), float(z), float(w))

    @staticmethod
    def get_prim_size(prim):
        bbox = OpenUsd.get_prim_aabb_bounding_box(prim)
        min_point = bbox.GetMin()
        max_point = bbox.GetMax()
        size = max_point - min_point
        return size

    @staticmethod
    def get_articulation_joints(articulation_prim):
        """Get joints of articulation"""
        joints = []

        def recurse(prim):
            # Check if it's a Joint
            if UsdPhysics.Joint(prim):
                joints.append(prim)
            for child in prim.GetChildren():
                recurse(child)
        recurse(articulation_prim)
        return joints

    @staticmethod
    def get_joint_type(joint_prim):
        """Get joint type"""
        joint = UsdPhysics.Joint(joint_prim)
        return joint.GetTypeName()

    @staticmethod
    def get_object_extent(prim, obj_name):
        """Get object extent"""
        obj_prims = OpenUsd.get_prim_by_name(prim, obj_name, only_xform=False)
        if len(obj_prims) == 0:
            raise ValueError(f"Object {obj_name} not found in the scene")
        obj_prim = obj_prims[0]
        size = obj_prim.GetAttribute("extent").Get()
        return size

    @staticmethod
    def scale_size(root_prim, scale_factor=(1.0, 1.0, 1.0)):
        """Scale object size"""
        xformable = UsdGeom.Xformable(OpenUsd.get_prim_by_name(root_prim, "root")[0])
        if not xformable:
            raise ValueError("prim must be xformable")
        xform_ops = xformable.GetOrderedXformOps()
        found_scale = False
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set((scale_factor[0], scale_factor[1], scale_factor[2]))
                found_scale = True
                break
        if not found_scale:
            xformable.AddScaleOp().Set((scale_factor[0], scale_factor[1], scale_factor[2]))

    @staticmethod
    def _get_asset_root_prim(root_prim):
        root_candidates = OpenUsd.get_prim_by_name(root_prim, "root")
        if root_candidates:
            return root_candidates[0]
        stage = root_prim.GetStage()
        default_prim = stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            return default_prim
        children = root_prim.GetChildren() if root_prim.IsPseudoRoot() else root_prim.GetAllChildren()
        if children:
            return children[0]
        raise ValueError(f"Cannot find asset root prim under {root_prim.GetPath()}")

    @staticmethod
    def _scale_vec3(vec, sx, sy, sz):
        return Gf.Vec3f(vec[0] * sx, vec[1] * sy, vec[2] * sz)

    @staticmethod
    def _should_skip_scale_bake_prim(prim) -> bool:
        # Sites/reg_* are placement markers, not physics geometry.
        return "/Sites/" in str(prim.GetPath())

    @staticmethod
    def _scale_mesh_geometry(mesh, sx, sy, sz):
        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get()
        if points:
            points_attr.Set([OpenUsd._scale_vec3(p, sx, sy, sz) for p in points])
        extent = mesh.GetExtentAttr().Get()
        if extent and len(extent) >= 2:
            mesh.GetExtentAttr().Set([
                OpenUsd._scale_vec3(extent[0], sx, sy, sz),
                OpenUsd._scale_vec3(extent[1], sx, sy, sz),
            ])

    @staticmethod
    def _get_prim_local_scale_vec(xformable) -> tuple[float, float, float]:
        scale = [1.0, 1.0, 1.0]
        if not xformable:
            return tuple(scale)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                value = op.Get()
                if value is not None:
                    scale = [float(value[0]), float(value[1]), float(value[2])]
        return tuple(scale)

    @staticmethod
    def _scale_prim_geometry(prim, sx, sy, sz):
        local_sx, local_sy, local_sz = OpenUsd._get_prim_local_scale_vec(UsdGeom.Xformable(prim))
        total_sx, total_sy, total_sz = sx * local_sx, sy * local_sy, sz * local_sz
        if prim.IsA(UsdGeom.Mesh):
            OpenUsd._scale_mesh_geometry(UsdGeom.Mesh(prim), total_sx, total_sy, total_sz)
            return
        if prim.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim)
            base_size = float(cube.GetSizeAttr().Get()) if cube.GetSizeAttr().HasAuthoredValue() else 2.0
            cube.GetSizeAttr().Set(base_size * total_sx)
            return
        if prim.IsA(UsdGeom.Sphere):
            sphere = UsdGeom.Sphere(prim)
            base_radius = float(sphere.GetRadiusAttr().Get()) if sphere.GetRadiusAttr().HasAuthoredValue() else 1.0
            sphere.GetRadiusAttr().Set(base_radius * total_sx)
            return
        if prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            base_radius = float(cylinder.GetRadiusAttr().Get()) if cylinder.GetRadiusAttr().HasAuthoredValue() else 1.0
            base_height = float(cylinder.GetHeightAttr().Get()) if cylinder.GetHeightAttr().HasAuthoredValue() else 2.0
            cylinder.GetRadiusAttr().Set(base_radius * total_sx)
            cylinder.GetHeightAttr().Set(base_height * total_sz)
            return
        if prim.IsA(UsdGeom.Capsule):
            capsule = UsdGeom.Capsule(prim)
            base_radius = float(capsule.GetRadiusAttr().Get()) if capsule.GetRadiusAttr().HasAuthoredValue() else 1.0
            base_height = float(capsule.GetHeightAttr().Get()) if capsule.GetHeightAttr().HasAuthoredValue() else 2.0
            capsule.GetRadiusAttr().Set(base_radius * total_sx)
            capsule.GetHeightAttr().Set(base_height * total_sz)

    @staticmethod
    def _reset_scale_ops_only(prim):
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

    @staticmethod
    def _bake_mesh_prims_in_world_space(asset_root, sx, sy, sz):
        """Bake mesh points in world space so rotated collision chunks stay aligned."""
        cache = UsdGeom.XformCache()
        root_xf = cache.GetLocalToWorldTransform(asset_root)
        root_inv = root_xf.GetInverse()

        mesh_world_data = []
        for prim in Usd.PrimRange(asset_root):
            if OpenUsd._should_skip_scale_bake_prim(prim) or not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() or []
            prim_xf = cache.GetLocalToWorldTransform(prim)
            scaled_world_points = []
            for point in points:
                world_point = prim_xf.Transform(Gf.Vec3d(point[0], point[1], point[2]))
                local_from_root = root_inv.Transform(world_point)
                scaled_root_local = Gf.Vec3d(
                    local_from_root[0] * sx,
                    local_from_root[1] * sy,
                    local_from_root[2] * sz,
                )
                scaled_world_points.append(root_xf.Transform(scaled_root_local))
            mesh_world_data.append((prim, scaled_world_points))

        for prim in Usd.PrimRange(asset_root):
            if OpenUsd._should_skip_scale_bake_prim(prim):
                continue
            OpenUsd._reset_scale_ops_only(prim)

        cache.Clear()
        for prim, scaled_world_points in mesh_world_data:
            prim_xf = cache.GetLocalToWorldTransform(prim)
            prim_inv = prim_xf.GetInverse()
            local_points = [
                Gf.Vec3f(
                    prim_inv.Transform(world_point)[0],
                    prim_inv.Transform(world_point)[1],
                    prim_inv.Transform(world_point)[2],
                )
                for world_point in scaled_world_points
            ]
            mesh = UsdGeom.Mesh(prim)
            mesh.GetPointsAttr().Set(local_points)
            if local_points:
                xs = [p[0] for p in local_points]
                ys = [p[1] for p in local_points]
                zs = [p[2] for p in local_points]
                mesh.GetExtentAttr().Set([
                    Gf.Vec3f(min(xs), min(ys), min(zs)),
                    Gf.Vec3f(max(xs), max(ys), max(zs)),
                ])
        return len(mesh_world_data)

    @staticmethod
    def bake_uniform_scale(root_prim, scale_factor=(1.0, 1.0, 1.0)):
        """Bake uniform scale into geometry and reset xform scale ops to 1.0."""
        sx, sy, sz = (float(scale_factor[0]), float(scale_factor[1]), float(scale_factor[2]))
        if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6 and abs(sz - 1.0) < 1e-6:
            return
        asset_root = OpenUsd._get_asset_root_prim(root_prim)
        mesh_count = OpenUsd._bake_mesh_prims_in_world_space(asset_root, sx, sy, sz)
        for prim in Usd.PrimRange(asset_root):
            if OpenUsd._should_skip_scale_bake_prim(prim) or prim.IsA(UsdGeom.Mesh):
                continue
            OpenUsd._scale_prim_geometry(prim, sx, sy, sz)
            OpenUsd._reset_scale_ops_only(prim)
        non_identity_scales = []
        for prim in Usd.PrimRange(asset_root):
            if OpenUsd._should_skip_scale_bake_prim(prim):
                continue
            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                continue
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    value = op.Get()
                    if (
                        abs(value[0] - 1.0) > 1e-4
                        or abs(value[1] - 1.0) > 1e-4
                        or abs(value[2] - 1.0) > 1e-4
                    ):
                        non_identity_scales.append(str(prim.GetPath()))
        print(
            f"[scale-trace] bake_uniform_scale factor=({sx},{sy},{sz}) "
            f"meshes={mesh_count} non_identity_scale_prims={len(non_identity_scales)}",
            flush=True,
        )

    @staticmethod
    def enable_contact_reporter(prim, contact_force_threshold=0.0):
        """Enable contact reporting on a rigid-body prim."""
        if not PhysxSchema or not prim or not prim.IsValid():
            return False
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return False

        if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            cr_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        else:
            cr_api = PhysxSchema.PhysxContactReportAPI.Get(
                prim.GetStage(), prim.GetPath()
            )
        cr_api.CreateThresholdAttr().Set(contact_force_threshold)
        return True

    @staticmethod
    def set_contact_force_threshold(stage, root_prim, name, contact_force_threshold=0.0):
        if not PhysxSchema:
            return
        matched_prims = OpenUsd.get_prim_by_name(root_prim, name)
        if len(matched_prims) == 0:
            print(
                f"[USD Warning] set_contact_force_threshold skip: prim '{name}' not found under {root_prim.GetPrimPath()}",
                flush=True,
            )
            return
        OpenUsd.enable_contact_reporter(matched_prims[0], contact_force_threshold)

    @staticmethod
    def set_rgb(root_prim, rgb):
        """Set object rgb"""
        xformable = UsdGeom.Xformable(OpenUsd.get_prim_by_name(root_prim, "root")[0])
        if not xformable:
            raise ValueError("prim must be xformable")

        shader_prims = OpenUsd.get_prim_by_name_and_type(root_prim, "Shader", "Shader")
        if shader_prims.__len__() == 0:
            raise ValueError("prim has no shader")

        for shader_prim in shader_prims:
            shader = UsdShade.Shader(shader_prim)
            if not shader:
                continue
            diffuse_tint_input = shader.GetInput("inputs:diffuse_tint")
            if not (diffuse_tint_input and diffuse_tint_input.GetAttr().IsValid()):
                diffuse_tint_input = shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f)
            diffuse_tint_input.Set((rgb[0], rgb[1], rgb[2]))

    @staticmethod
    def convert_asset_instanceable(asset_usd_path, save_as_path=None, create_xforms=True):
        """
            Makes all mesh/geometry prims instanceable (and optionally inserts Xform parents) while preserving materials.
            All operations are performed within the save_as_path file; no additional mesh files are generated.
        """
        import omni.usd

        working_path = save_as_path if save_as_path else asset_usd_path
        root_path = "/"

        if create_xforms:
            omni.usd.get_context().open_stage(asset_usd_path)
            stage = omni.usd.get_context().get_stage()
            prims = [stage.GetPrimAtPath(root_path)]
            edits = Sdf.BatchNamespaceEdit()

            while len(prims) > 0:
                prim = prims.pop(0)
                if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
                    new_xform = UsdGeom.Xform.Define(
                        stage, str(prim.GetPath()) + "_xform")
                    edits.Add(Sdf.NamespaceEdit.Reparent(
                        prim.GetPath(), new_xform.GetPath(), 0))
                    continue
                children_prims = prim.GetChildren()
                prims += children_prims

            stage.GetRootLayer().Apply(edits)
            if save_as_path is None:
                omni.usd.get_context().save_stage()
            else:
                omni.usd.get_context().save_as_stage(save_as_path)

        omni.usd.get_context().open_stage(working_path)
        stage = omni.usd.get_context().get_stage()
        prims = [stage.GetPrimAtPath(root_path)]

        while len(prims) > 0:
            prim = prims.pop(0)
            if prim:
                if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
                    parent_prim = prim.GetParent()
                    if parent_prim and not parent_prim.IsInstance():
                        parent_prim.SetInstanceable(True)
                        continue
                children_prims = prim.GetChildren()
                prims += children_prims

        if save_as_path is None:
            omni.usd.get_context().save_stage()
        else:
            omni.usd.get_context().save_as_stage(save_as_path)

    @staticmethod
    def export(stage, path):
        """Export prim to path"""
        stage.Export(path)

    @staticmethod
    def is_fixed_joint(prim):
        """Check if joint is fixed"""
        return prim.GetTypeName() == 'PhysicsFixedJoint'

    @staticmethod
    def is_revolute_joint(prim):
        """Check if joint is revolute"""
        return prim.GetTypeName() == 'PhysicsRevoluteJoint'

    @staticmethod
    def is_prismatic_joint(prim):
        """Check if joint is prismatic"""
        return prim.GetTypeName() == "PhysicsPrismaticJoint"

    @staticmethod
    def get_joint_name_and_qpos(joint_prim):
        """Get joint name and position"""
        joint = UsdPhysics.Joint(joint_prim)
        return joint.GetName(), joint.GetPositionAttr().Get()

    @staticmethod
    def get_all_joints_without_fixed(articulation_prim):
        """Get all non-fixed joints"""
        joints = OpenUsd.get_articulation_joints(articulation_prim)
        return [joint for joint in joints if not OpenUsd.is_fixed_joint(joint)]

    @staticmethod
    def get_prim_by_name(prim, name, only_xform=True, case_sensitive=False, only_first=False):
        """Get prim by name"""
        result = []
        if not case_sensitive:
            lower_name = name.lower()
            for child in Usd.PrimRange(prim):
                if child.GetName().lower() == lower_name:
                    if not only_xform or child.GetTypeName() == "Xform":
                        result.append(child)
                        if only_first:
                            break
                # result.extend(OpenUsd.get_prim_by_name(child, name, only_xform))
        else:
            for child in Usd.PrimRange(prim):
                if child.GetName() == name:
                    if not only_xform or child.GetTypeName() == "Xform":
                        result.append(child)
                        if only_first:
                            break

        return result

    @staticmethod
    def get_prim_by_type(prim, include_types=None, exclude_types=None):
        """Get prim by type"""
        result = []
        if (
            (include_types is None or prim.GetTypeName() in include_types)
            and (exclude_types is None or prim.GetTypeName() not in exclude_types)
        ):
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_type(child, include_types, exclude_types))
        return result

    @staticmethod
    def get_prim_by_name_and_type(prim, name, type):
        """Get prim by name and type"""
        result = []
        if prim.GetName().lower() == name.lower() and prim.GetTypeName() == type:
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_name_and_type(child, name, type))
        return result

    @staticmethod
    def get_prim_by_prefix(prim, prefix, only_xform=True):
        """Get prim by prefix"""
        result = []
        if prim.GetName().lower().startswith(prefix.lower()):
            if not only_xform or prim.GetTypeName() == "Xform":
                result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_prefix(child, prefix, only_xform))
        return result

    @staticmethod
    def get_prim_by_prefix_and_type(prim, prefix, type):
        """Get prim by prefix and type"""
        result = []
        if prim.GetName().lower().startswith(prefix.lower()) and prim.GetTypeName() == type:
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_prefix_and_type(child, prefix, type))
        return result

    @staticmethod
    def get_prim_by_suffix(prim, suffix, only_xform=True):
        """Get prim by suffix"""
        result = []
        if prim.GetName().lower().endswith(suffix.lower()):
            if not only_xform or prim.GetTypeName() == "Xform":
                result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_suffix(child, suffix, only_xform))
        return result

    @staticmethod
    def get_prim_by_suffix_and_type(prim, suffix, type):
        """Get prim by suffix and type"""
        result = []
        if prim.GetName().lower().endswith(suffix.lower()) and prim.GetTypeName() == type:
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_suffix_and_type(child, suffix, type))
        return result

    @staticmethod
    def get_prim_by_subname(prim, subname):
        """Get prim by subname"""
        result = []
        if subname in prim.GetName():
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_subname(child, subname))
        return result

    @staticmethod
    def get_prim_by_subname_and_type(prim, subname, type):
        """Get prim by subname and type"""
        result = []
        if subname in prim.GetName() and prim.GetTypeName() == type:
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_subname_and_type(child, subname, type))
        return result

    @staticmethod
    def get_prim_by_types(prim, types):
        """Get prim by types"""
        result = []
        if prim.GetTypeName() in types:
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_types(child, types))
        return result

    @staticmethod
    def has_contact_reporter(prim):
        """Check if prim has contact reporter"""
        if not PhysxSchema:
            return False
        return prim.HasAPI(PhysxSchema.PhysxContactReportAPI)

    @staticmethod
    def get_child_commonprefix_name(prim):
        """Get common prefix name of child elements"""
        child_rigidbody_prims = []
        for child in prim.GetChildren():
            if OpenUsd.is_rigidbody(child):
                child_rigidbody_prims.append(child)
        names = [prim.GetName() for prim in child_rigidbody_prims]

        if len(names) == 1:
            return names[0]

        return os.path.commonprefix(names)

    @staticmethod
    def get_child_commonprefix_name_deformable(prim):
        """Get common prefix name of deformable child elements."""
        child_deformable_prims = []
        for child in prim.GetChildren():
            if OpenUsd.is_deformable(child):
                child_deformable_prims.append(child)
        names = [prim.GetName() for prim in child_deformable_prims]
        if len(names) == 1:
            return names[0]
        return os.path.commonprefix(names)

    @staticmethod
    def get_all_child_commonprefix_name(prim):
        """Get common prefix name of child elements"""
        child_rigidbody_prims = []
        for child in prim.GetAllChildren():
            if OpenUsd.is_rigidbody(child):
                child_rigidbody_prims.append(child)
        names = [prim.GetName() for prim in child_rigidbody_prims]

        if len(names) == 1:
            return names[0]

        return os.path.commonprefix(names)

    @staticmethod
    def _get_xform_type_recursive(prim):
        """Return the first ``type`` attribute found on descendant Xform prims."""
        if prim.GetTypeName() == "Xform":
            type_attr = prim.GetAttribute("type")
            if type_attr.IsValid():
                value = type_attr.Get()
                if value is not None:
                    return value

        for child in prim.GetChildren():
            value = OpenUsd._get_xform_type_recursive(child)
            if value is not None:
                return value
        return None

    @staticmethod
    def get_child_xform_infos(prim):
        """Get child xform infos"""
        infos = []
        for child in prim.GetChildren():
            if child.GetTypeName() == "Xform":
                info = dict(
                    name=child.GetName(),
                    type=OpenUsd._get_xform_type_recursive(child),
                    prim=child,
                )
                infos.append(info)
        return infos

    @staticmethod
    def get_all_child_xform_infos(prim):
        """Get child xform infos"""
        infos = []
        for child in prim.GetAllChildren():
            if child.GetTypeName() == "Xform":
                info = dict(
                    name=child.GetName(),
                    type=child.GetAttribute("type").Get(),
                    prim=child,
                )
                infos.append(info)
        return infos

    @staticmethod
    def get_prim_aabb_bounding_box(prim, use_cache=True):
        """Get prim aabb bounding box without caching"""
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        if not use_cache:
            # Clear any existing cache to ensure fresh computation
            bbox_cache.Clear()
        return bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()

    @staticmethod
    def _xform_quat_xyzw_from_prim(prim):
        """Extract local rotation as (x, y, z, w) quaternion from USD xform ops."""
        fixture_rot = prim.GetAttribute("xformOp:rotateXYZ").Get()
        if fixture_rot:
            return T.mat2quat(
                T.euler2mat([
                    math.radians(fixture_rot[0]),
                    math.radians(fixture_rot[1]),
                    math.radians(fixture_rot[2]),
                ])
            )

        fixture_rot = prim.GetAttribute("xformOp:rotateZYX").Get()
        if fixture_rot:
            return T.mat2quat(
                T.euler2mat([
                    math.radians(fixture_rot[2]),
                    math.radians(fixture_rot[1]),
                    math.radians(fixture_rot[0]),
                ])
            )

        orient = prim.GetAttribute("xformOp:orient").Get()
        if orient is not None:
            if hasattr(orient, "GetReal"):
                imag = orient.GetImaginary()
                return np.array([
                    float(imag[0]), float(imag[1]), float(imag[2]), float(orient.GetReal()),
                ])
            # USD quatf/quatd tuple: (w, x, y, z)
            return np.array([
                float(orient[1]), float(orient[2]), float(orient[3]), float(orient[0]),
            ])

        return np.array([0.0, 0.0, 0.0, 1.0])

    @staticmethod
    def get_fixture_placements(root_prim, fixture_cfgs):
        valid_fixture_names = []
        fixture_placements = {}
        for fxr_cfg in fixture_cfgs:
            if "ProcGenFixture" not in [cls.__name__ for cls in fxr_cfg["model"].__class__.mro()] and \
               "room" not in fxr_cfg["name"].lower():  # room fixture doesnt have reg_*
                valid_fixture_names.append(fxr_cfg["name"])
        for name in valid_fixture_names:
            prim = OpenUsd.get_prim_by_name(root_prim, name)
            if len(prim) == 0:
                raise ValueError(f"Fixture {name} not found in the scene")
            prim = prim[0]
            fixture_pos = tuple(prim.GetAttribute("xformOp:translate").Get())
            fixture_quat = OpenUsd._xform_quat_xyzw_from_prim(prim)

            from robofinals.utils.place_utils.usd_object import USDObject
            usd_obj = USDObject(
                name=name,
                task_name=name,
                prim=prim,
                obj_path=None,
                object_scale=(1.0, 1.0, 1.0),
                asset_type="fixtures"
            )

            fixture_placements[name] = (fixture_pos, fixture_quat, usd_obj)

        return fixture_placements

    @staticmethod
    def replace_prim(stage, prim, new_prim_path):
        translate_attr = None
        rotateXYZ_attr = None
        scale_attr = None
        need_reorder = False
        path = prim.GetPath()
        old_translate = prim.GetAttribute("xformOp:translate").Get()
        old_rot = prim.GetAttribute("xformOp:rotateXYZ").Get()
        old_bbox = OpenUsd.get_prim_aabb_bounding_box(prim)
        old_bbox_size = np.array([old_bbox.GetSize()[0], old_bbox.GetSize()[1], old_bbox.GetSize()[2]])
        old_bbox_size = abs(T.euler2mat(np.array(old_rot) / 180 * np.pi) @ old_bbox_size)
        prim_type = prim.GetAttribute("type").Get()
        stage.RemovePrim(path)
        new_prim = stage.DefinePrim(path, "Xform")
        new_prim.GetReferences().AddReference(new_prim_path)
        new_bbox = OpenUsd.get_prim_aabb_bounding_box(new_prim)
        new_bbox_size = [new_bbox.GetSize()[0], new_bbox.GetSize()[1], new_bbox.GetSize()[2]]
        relative_scale = [float((old_bbox_size[i]) / new_bbox_size[i]) for i in range(3)]
        xformable = UsdGeom.Xformable(new_prim)
        ops = xformable.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_attr = op
                op.Set(Gf.Vec3f(relative_scale[0], relative_scale[1], relative_scale[2]))
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rotateXYZ_attr = op
                op.Set(Gf.Vec3f(old_rot[0], old_rot[1], old_rot[2]))
            elif op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_attr = op
                op.Set(Gf.Vec3f(old_translate[0], old_translate[1], old_translate[2]))
            elif op.GetOpType() in [UsdGeom.XformOp.TypeRotateZYX, UsdGeom.XformOp.TypeOrient]:
                rotateXYZ_attr = xformable.AddRotateXYZOp()
                rotateXYZ_attr.Set(Gf.Vec3f(old_rot[0], old_rot[1], old_rot[2]))
                need_reorder = True
        if need_reorder:
            xformable.SetXformOpOrder([
                translate_attr,
                rotateXYZ_attr,
                scale_attr,
            ])
        type_attr = new_prim.CreateAttribute("type", Sdf.ValueTypeNames.String)
        type_attr.Set(prim_type)
        size_attr = new_prim.CreateAttribute("size", Sdf.ValueTypeNames.String)
        size_attr.Set(f"{old_bbox_size[0]}, {old_bbox_size[1]}, {old_bbox_size[2]}")


class OpenUsdWrapper:
    def __init__(self, usd_path):
        self._usd = OpenUsd(usd_path)

    def __getattr__(self, name):
        return getattr(self._usd, name)

    @property
    def stage(self):
        return self._usd.stage

    @property
    def root_prim(self):
        return self.stage.GetPseudoRoot()

    def get_all_prims(self, prim=None, prims_list=None):
        return self._usd.get_all_prims(self.stage, prim, prims_list)

    def usd_simplify(self, ref_prim_names, prim=None):
        return self._usd.usd_simplify(self.stage, ref_prim_names, prim)

    def activate_prim(self, name):
        return self._usd.activate_prim(self.stage, name)

    def deactivate_prim(self, name):
        return self._usd.deactivate_prim(self.stage, name)

    def get_all_joints(self):
        return self._usd.get_all_joints(self.stage)

    def get_prim_pos_rot_in_world(self, prim):
        return self._usd.get_prim_pos_rot_in_world(prim)

    def get_object_extent(self, obj_name, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_object_extent(prim, obj_name)

    def find_articulation_root(self, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.find_articulation_root(prim)

    def scale_size(self, scale_factor=(1.0, 1.0, 1.0)):
        return self._usd.scale_size(self.root_prim, scale_factor)

    def bake_uniform_scale(self, scale_factor=(1.0, 1.0, 1.0)):
        return self._usd.bake_uniform_scale(self.root_prim, scale_factor)

    def set_contact_force_threshold(self, name, contact_force_threshold=0.0):
        return self._usd.set_contact_force_threshold(self.stage, self.root_prim, name, contact_force_threshold)

    def set_rgb(self, rgb=(1.0, 1.0, 1.0)):
        return self._usd.set_rgb(self.root_prim, rgb)

    def convert_asset_instanceable(self, save_as_path=None):
        return self._usd.convert_asset_instanceable(self.stage, save_as_path)

    def export(self, path):
        return self._usd.export(self.stage, path)

    def get_articulation_joints(self, articulation_prim=None):
        if articulation_prim is None:
            articulation_prim = self.root_prim
        return self._usd.get_articulation_joints(articulation_prim)

    def get_all_joints_without_fixed(self, articulation_prim=None):
        if articulation_prim is None:
            articulation_prim = self.root_prim
        return self._usd.get_all_joints_without_fixed(articulation_prim)

    def get_prim_by_name(self, name, only_xform=True, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_name(prim, name, only_xform)

    def get_prim_by_name_and_type(self, name, type, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_name_and_type(self.root_prim, name, type)

    def get_prim_by_prefix(self, prefix, only_xform=True, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_prefix(prim, prefix, only_xform)

    def get_prim_by_prefix_and_type(self, prefix, type, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_prefix_and_type(prim, prefix, type)

    def get_prim_by_suffix(self, suffix, only_xform=True, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_suffix(prim, suffix, only_xform)

    def get_prim_by_suffix_and_type(self, suffix, type, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_suffix_and_type(prim, suffix, type)

    def get_prim_by_subname(self, subname, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_subname(prim, subname)

    def get_prim_by_subname_and_type(self, subname, type, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_subname_and_type(prim, subname, type)

    def get_prim_by_types(self, types, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_types(prim, types)

    def get_all_child_commonprefix_name(self, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_all_child_commonprefix_name(prim)

    def get_all_child_xform_names(self, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_all_child_xform_names(prim)

    def get_fixture_placements(self, fixture_cfgs, fixtures, root_prim=None):
        if root_prim is None:
            root_prim = self.root_prim
        return self._usd.get_fixture_placements(root_prim, fixture_cfgs, fixtures)

    def replace_prim(self, prim, new_prim_path):
        return self._usd.replace_prim(self.stage, prim, new_prim_path)
