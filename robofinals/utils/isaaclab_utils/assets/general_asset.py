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


from isaaclab.assets.articulation import Articulation
from isaaclab.assets.asset_base import AssetBase
from isaaclab.assets.rigid_object import RigidObject

from isaaclab_arena.utils.pose import Pose

from robofinals.core.backends import make_asset_compiler, resolve_backend_kind
from robofinals.utils.usd_utils import OpenUsd as usd



class GeneralAsset(AssetBase):

    def __init__(self, cfg, env_regex_ns, *, backend_assets=None):
        super().__init__(cfg)
        self.cfg = cfg
        self.rigid_objects = list()
        self.deformable_objects = list()
        self.articulations = list()
        self.env_regex_ns = env_regex_ns
        if backend_assets is None:
            backend = resolve_backend_kind(getattr(cfg, "physics_backend", None))
            backend_assets = make_asset_compiler(backend)
        self.backend_assets = backend_assets
        self.stage = usd.get_stage(self.cfg.spawn.usd_path)
        self._parse_usd_and_create_subassets()

    def _make_articulation_cfg(self, prim):
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
        pos, quat_wxyz, _ = usd.get_prim_pos_rot_in_world(prim)
        if pos is None or quat_wxyz is None:
            print(f"GeneralAsset: {prim.GetName()} none pos or quat")
            return None
        joints = usd.get_all_joints_without_fixed(prim)
        if not joints:
            return None
        orin_prim_path = prim.GetPath().pathString
        name = orin_prim_path.split("/")[-1]
        sub_prim_path = orin_prim_path[orin_prim_path.find('/', 1) + 1:]
        prim_path = f"{{ENV_REGEX_NS}}/Scene/{sub_prim_path}"
        prim_path = prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
        # See ``general_asset_arena.py`` for the full explanation. Short
        # version: isaaclab 3.x fills sim buffers exclusively from actuator
        # groups, and many scene-USDs either omit DriveAPI or author it
        # against the non-standard ``X`` axis (which PhysX 5+ ignores). We
        # therefore provide a catch-all implicit actuator with small default
        # damping/friction so passive joints (drawers, doors, etc.) don't end
        # up at zero gains. Pair this with ``OpenUsd.fix_drive_axis_naming``
        # in ``_parse_usd_and_create_subassets`` for the ``drive:X`` fix-up.
        return ArticulationCfg(prim_path=prim_path,
                               spawn=None,
                               init_state=ArticulationCfg.InitialStateCfg(
                                   pos=pos,
                                   rot=usd.usd_quat_wxyz_to_isaac_init_rot(quat_wxyz),
                               ),
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

    def _make_rigidbody_cfg(self, prim):
        from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
        pos, quat_wxyz, _ = usd.get_prim_pos_rot_in_world(prim)
        if pos is None or quat_wxyz is None:
            print(f"GeneralAsset: {prim.GetName()} none pos or quat")
            return None
        orin_prim_path = prim.GetPath().pathString
        name = orin_prim_path.split("/")[-1]
        sub_prim_path = orin_prim_path[orin_prim_path.find('/', 1) + 1:]
        prim_path = f"{{ENV_REGEX_NS}}/Scene/{sub_prim_path}"
        prim_path = prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
        return RigidObjectCfg(prim_path=prim_path,
                              spawn=None,
                              init_state=RigidObjectCfg.InitialStateCfg(
                                  pos=pos,
                                  rot=usd.usd_quat_wxyz_to_isaac_init_rot(quat_wxyz),
                              ),
                              **self.backend_assets.rigid_object_cfg_kwargs(),
                              )

    def _make_deformable_cfg(self, prim):
        pos, quat_wxyz, _ = usd.get_prim_pos_rot_in_world(prim)
        if pos is None or quat_wxyz is None:
            print(f"GeneralAsset: {prim.GetName()} none pos or quat")
            return None
        orin_prim_path = prim.GetPath().pathString
        sub_prim_path = orin_prim_path[orin_prim_path.find('/', 1) + 1:]
        prim_path = f"{{ENV_REGEX_NS}}/Scene/{sub_prim_path}"
        prim_path = prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
        return self.backend_assets.make_deformable_cfg(
            prim_path=prim_path,
            initial_pose=Pose(
                position_xyz=tuple(pos),
                rotation_wxyz=usd.usd_quat_wxyz_to_isaac_init_rot(quat_wxyz),
            ),
        )

    def _parse_usd_and_create_subassets(self):

        prims = usd.get_all_prims(self.stage)
        articulation_sub_prims = list()
        for prim in prims:
            if usd.is_articulation_root(prim):
                # Keep Newton fixed-only articulations in the physics scene,
                # but do not create an IsaacLab articulation/rigid-object view
                # that Newton cannot bind.
                if (
                    not self.backend_assets.bind_zero_dof_articulation()
                    and not usd.get_all_joints_without_fixed(prim)
                ):
                    articulation_sub_prims.extend(usd.get_all_prims(self.stage, prim))
                    continue

                art_cfg = self._make_articulation_cfg(prim)
                if art_cfg is None:
                    continue
                self.articulations.append(Articulation(art_cfg))
                articulation_sub_prims.extend(usd.get_all_prims(self.stage, prim))
        for prim in prims:
            if usd.is_legacy_deformable(prim):
                deform_cfg = self._make_deformable_cfg(prim)
                if deform_cfg is None:
                    continue
                self.deformable_objects.append(
                    self.backend_assets.instantiate_deformable(deform_cfg)
                )
                articulation_sub_prims.extend(usd.get_all_prims(self.stage, prim))
            elif usd.is_omniphysics_deformable(prim):
                deform_cfg = self._make_deformable_cfg(prim)
                if deform_cfg is None:
                    continue
                self.backend_assets.configure_deformable_cfg(
                    deform_cfg, omniphysics=True
                )
                self.deformable_objects.append(
                    self.backend_assets.instantiate_deformable(
                        deform_cfg, omniphysics=True
                    )
                )
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
                rb_cfg = self._make_rigidbody_cfg(prim)
                if rb_cfg is None:
                    continue
                rigid_object_type = getattr(rb_cfg, "class_type", None)
                if not isinstance(rigid_object_type, type):
                    rigid_object_type = RigidObject
                self.rigid_objects.append(rigid_object_type(rb_cfg))

    def reset(self, env_ids=None):
        for art in self.articulations:
            art.reset(env_ids)
        for deform in self.deformable_objects:
            deform.reset(env_ids)
        for rb in self.rigid_objects:
            rb.reset(env_ids)

    def write_data_to_sim(self):
        for art in self.articulations:
            art.write_data_to_sim()
        for deform in self.deformable_objects:
            deform.write_data_to_sim()
        for rb in self.rigid_objects:
            rb.write_data_to_sim()

    def update(self, dt):
        for art in self.articulations:
            art.update(dt)
        for deform in self.deformable_objects:
            deform.update(dt)
        for rb in self.rigid_objects:
            rb.update(dt)

    def get_all_joint_names(self):
        names = []
        for art in self.articulations:
            names.extend(art.data.joint_names)
        return names

    def get_all_body_names(self):
        names = []
        for rb in self.rigid_objects:
            names.extend(rb.data.body_names)
        for art in self.articulations:
            names.extend(art.data.body_names)
        return names

    def _initialize_impl(self):
        pass

    @property
    def data(self):
        pass

    @property
    def num_instances(self):
        pass
