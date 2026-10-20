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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType

from robofinals.core.backends import make_asset_compiler, resolve_backend_kind



class LwObject(Object):

    def __init__(self, *args, tetmesh_source: str = "collision", backend_assets=None, **kwargs):
        if tetmesh_source not in ("collision", "simulation"):
            raise ValueError(
                f"tetmesh_source must be 'collision' or 'simulation', got {tetmesh_source!r}."
            )
        # Object.__init__ constructs object_cfg, so this selection must exist
        # before delegating to it.
        self.tetmesh_source = tetmesh_source
        if backend_assets is None:
            backend_assets = make_asset_compiler(
                resolve_backend_kind(kwargs.pop("physics_backend", None))
            )
        self.backend_assets = backend_assets
        super().__init__(*args, **kwargs)

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.5,
                    angular_damping=0.5,
                    sleep_threshold=0.1,
                    stabilization_threshold=0.02,
                    solver_position_iteration_count=48,
                    solver_velocity_iteration_count=8,
                    max_depenetration_velocity=4.0,
                ),
            ),
            **self.backend_assets.rigid_object_cfg_kwargs(),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        assert self.object_type == ObjectType.ARTICULATION
        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    sleep_threshold=0.0,
                    stabilization_threshold=0.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=False,
                ),
            ),
            # NOTE on passive-fixture joint defaults: in isaaclab 2.x
            # ``actuators={}`` was sufficient because the PhysX backend
            # consumed USD ``drive:*:physics:*`` and ``physxJoint:jointFriction``
            # directly. In isaaclab 3.x sim buffers are filled exclusively
            # from an actuator group; many object USDs either omit DriveAPI
            # altogether or author it against the non-standard ``X`` axis
            # (which PhysX 5+ silently ignores). Either way, without this
            # catch-all group, passive articulation objects (drawers, doors,
            # fridges, …) load with stiffness=damping=friction=0 and slide
            # around freely under gravity/contact perturbations.
            #
            # We therefore provide a catch-all implicit actuator with small
            # default damping/friction. Armature stays ``None`` because
            # PhysX correctly inherits ``physxJoint:armature`` from USD.
            # actuators={
            #     "joints_from_usd_default": ImplicitActuatorCfg(
            #         joint_names_expr=[".*"],
            #         stiffness=0.0,
            #         damping=2.0,
            #         friction=1.0,
            #         armature=None,
            #     ),
            # },
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg

    def _generate_deformable_cfg(self) -> DeformableObjectCfg:
        """Create one backend-dispatched deformable config for the object.

        Schema detection and legacy adaptation are deferred to the spawner so
        they run against the referenced asset in the live simulation stage.
        """
        assert self.object_type == ObjectType.DEFORMABLE
        object_cfg = self.backend_assets.make_deformable_cfg(
            prim_path=self.prim_path,
            usd_path=self.usd_path,
            scale=self.scale,
            tetmesh_source=self.tetmesh_source,
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg
