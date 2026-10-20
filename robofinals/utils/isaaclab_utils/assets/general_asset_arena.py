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

from typing import Any

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose

from robofinals.core.backends import make_asset_compiler, resolve_backend_kind
from robofinals.utils.usd_utils import OpenUsd as usd



class GeneralAssetArena(Object):
    """
    Arena version of GeneralAsset that recursively loads articulations and rigid bodies from a USD file.
    """

    def __init__(
        self,
        name: str,
        usd_path: str,
        object_min_z: float = 0.1,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        physics_backend: str | None = None,
        backend_assets: Any = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            prim_path=prim_path,
            object_type=ObjectType.BASE,
            usd_path=usd_path,
            scale=scale,
            initial_pose=initial_pose,
            **kwargs,
        )
        self.object_min_z = object_min_z
        self.stage = usd.get_stage(usd_path)
        self.articulation_cfgs = {}
        self.deformable_cfgs = {}
        self.rigidbody_cfgs = {}
        if backend_assets is None:
            backend_assets = make_asset_compiler(resolve_backend_kind(physics_backend))
        self.backend_assets = backend_assets
        self._parse_usd_and_create_subassets()

    def _make_articulation_cfg(self, prim):
        pos, quat_wxyz, _ = usd.get_prim_pos_rot_in_world(prim)
        if pos is None or quat_wxyz is None:
            print(f"GeneralAssetArena: {prim.GetName()} none pos or quat")
            return None
        joints = usd.get_all_joints_without_fixed(prim)
        if not joints:
            return None
        orin_prim_path = prim.GetPath().pathString
        name = orin_prim_path.split("/")[-1]
        sub_prim_path = orin_prim_path[orin_prim_path.find("/", 1) + 1:]
        prim_path = f"{{ENV_REGEX_NS}}/{self.name}/{sub_prim_path}"

        artic_cfg = ArticulationCfg(
            prim_path=prim_path,
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos,
                rot=usd.usd_quat_wxyz_to_isaac_init_rot(quat_wxyz),
            ),
            # NOTE on passive-fixture joint defaults (drawers, cabinets,
            # microwaves, fridges, …): in isaaclab 2.x ``actuators={}`` was
            # enough because the PhysX backend consumed USD
            # ``drive:*:physics:*`` and ``physxJoint:jointFriction`` directly.
            # In isaaclab 3.x sim buffers are filled exclusively from an
            # actuator group, and many scene-USD authoring tools either
            # (a) leave passive joints without any DriveAPI, or (b) author
            # ``drive:X:*`` which PhysX 5+ silently ignores (the standard axis
            # instance name is ``angular``/``linear``). Both pitfalls result
            # in stiffness=damping=friction=0 → fixtures slide freely.
            #
            # We pair this catch-all actuator group (small default damping +
            # friction so a missing DriveAPI doesn't blow up) with the
            # ``OpenUsd.fix_drive_axis_naming`` shim invoked from
            # ``_parse_usd_and_create_subassets`` (copies ``drive:X:*`` onto
            # the standard axis so PhysX actually reads it). Armature stays
            # ``None`` because PhysX correctly inherits ``physxJoint:armature``
            # from USD already.
            actuators={
                "joints_from_usd_default": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0,
                    damping=2.0,
                    friction=1.0,
                    armature=None,
                ),
            },
        )
        return name, artic_cfg

    def _make_rigidbody_cfg(self, prim):
        pos, quat_wxyz, _ = usd.get_prim_pos_rot_in_world(prim)
        if pos is None or quat_wxyz is None:
            print(f"GeneralAssetArena: {prim.GetName()} none pos or quat")
            return None
        orin_prim_path = prim.GetPath().pathString
        name = orin_prim_path.split("/")[-1]
        sub_prim_path = orin_prim_path[orin_prim_path.find("/", 1) + 1:]
        prim_path = f"{{ENV_REGEX_NS}}/{self.name}/{sub_prim_path}"

        rb_cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=pos,
                rot=usd.usd_quat_wxyz_to_isaac_init_rot(quat_wxyz),
            ),
            **self.backend_assets.rigid_object_cfg_kwargs(),
        )
        return name, rb_cfg

    def _make_deformable_cfg(self, prim):
        pos, quat_wxyz, _ = usd.get_prim_pos_rot_in_world(prim)
        if pos is None or quat_wxyz is None:
            print(f"GeneralAssetArena: {prim.GetName()} none pos or quat")
            return None
        orin_prim_path = prim.GetPath().pathString
        name = orin_prim_path.split("/")[-1]
        sub_prim_path = orin_prim_path[orin_prim_path.find("/", 1) + 1:]
        prim_path = f"{{ENV_REGEX_NS}}/{self.name}/{sub_prim_path}"

        deform_cfg = self.backend_assets.make_deformable_cfg(
            prim_path=prim_path,
            initial_pose=Pose(
                position_xyz=tuple(pos),
                rotation_wxyz=usd.usd_quat_wxyz_to_isaac_init_rot(quat_wxyz),
            ),
        )
        return name, deform_cfg

    def _parse_usd_and_create_subassets(self):
        prims = usd.get_all_prims(self.stage)
        articulation_sub_prims = list()

        for prim in prims:
            if usd.is_articulation_root(prim):
                # Newton collapses fixed-only articulations into the physics
                # scene but exposes neither an articulation nor an independent
                # rigid object view for them. Mark their descendants as owned
                # so the rigid-body pass does not register an invalid view.
                if (
                    not self.backend_assets.bind_zero_dof_articulation()
                    and not usd.get_all_joints_without_fixed(prim)
                ):
                    articulation_sub_prims.extend(usd.get_all_prims(self.stage, prim))
                    continue

                result = self._make_articulation_cfg(prim)
                if result is None:
                    continue
                name, art_cfg = result
                self.articulation_cfgs[name] = art_cfg
                articulation_sub_prims.extend(usd.get_all_prims(self.stage, prim))

        for prim in prims:
            if usd.is_legacy_deformable(prim):
                result = self._make_deformable_cfg(prim)
                if result is None:
                    continue
                name, deform_cfg = result
                self.deformable_cfgs[name] = deform_cfg
                articulation_sub_prims.extend(usd.get_all_prims(self.stage, prim))
            elif usd.is_omniphysics_deformable(prim):
                result = self._make_deformable_cfg(prim)
                if result is None:
                    continue
                name, deform_cfg = result
                self.backend_assets.configure_deformable_cfg(deform_cfg, omniphysics=True)
                self.deformable_cfgs[name] = deform_cfg
                articulation_sub_prims.extend(usd.get_all_prims(self.stage, prim))

        for prim in prims:
            if usd.is_rigidbody(prim):
                if prim in articulation_sub_prims or usd.is_deformable(prim):
                    continue
                if (
                    not self.backend_assets.bind_fixed_joint_rigid_body()
                    and usd.get_prim_by_type(prim, include_types=["PhysicsFixedJoint"])
                ):
                    continue
                result = self._make_rigidbody_cfg(prim)
                if result is None:
                    continue
                name, rb_cfg = result
                self.rigidbody_cfgs[name] = rb_cfg

    def get_object_cfg(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the base configuration and
        all discovered articulation and rigidbody configurations.
        This method is called by Scene.get_scene_cfg() to generate the scene configuration.
        """
        cfg_dict = {}
        cfg_dict[self.name] = self.object_cfg
        cfg_dict.update(self.articulation_cfgs)
        cfg_dict.update(self.deformable_cfgs)
        cfg_dict.update(self.rigidbody_cfgs)
        return cfg_dict

    def _generate_base_cfg(self):
        """
        Generate base configuration for the background USD file.
        This is called by ObjectBase._init_object_cfg().
        """
        object_cfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + self.name,
            spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg
