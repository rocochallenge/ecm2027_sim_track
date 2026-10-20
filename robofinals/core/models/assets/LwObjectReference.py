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

from isaaclab.assets import DeformableObjectCfg
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_reference import ObjectReference

from robofinals.core.backends import make_asset_compiler, resolve_backend_kind




class LwObjectReference(ObjectReference):
    """Scene-embedded object reference with backend-dispatched deformable support."""

    def __init__(
        self,
        *args,
        tetmesh_source: str = "collision",
        schema: str = "auto",
        backend_assets=None,
        **kwargs,
    ):
        if tetmesh_source not in ("collision", "simulation"):
            raise ValueError(
                f"tetmesh_source must be 'collision' or 'simulation', got {tetmesh_source!r}."
            )
        if schema not in ("auto", "legacy", "omniphysics"):
            raise ValueError(
                f"schema must be 'auto', 'legacy', or 'omniphysics', got {schema!r}."
            )
        self.tetmesh_source = tetmesh_source
        self.schema = schema
        if backend_assets is None:
            backend_assets = make_asset_compiler(
                resolve_backend_kind(kwargs.pop("physics_backend", None))
            )
        self.backend_assets = backend_assets
        super().__init__(*args, **kwargs)

    def _generate_deformable_cfg(self) -> DeformableObjectCfg:
        """Prepare an in-scene deformable prim for the selected backend."""
        assert self.object_type == ObjectType.DEFORMABLE
        object_cfg = self.backend_assets.make_deformable_cfg(
            prim_path=self.prim_path,
            initial_pose=self.get_initial_pose(),
            schema=self.schema,
            tetmesh_source=self.tetmesh_source,
        )
        if hasattr(object_cfg, "tet_mesh_source"):
            object_cfg.tet_mesh_source = self.tetmesh_source
        return object_cfg
