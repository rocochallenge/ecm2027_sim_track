from __future__ import annotations

from collections.abc import Sequence
import weakref

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets.asset_base import AssetBase
from isaaclab_physx.physics import PhysxManager
from pxr import UsdShade

from robofinals.utils.isaac_data_compat import as_torch, wrap_sim_buffer

from robofinals.utils.isaaclab_utils.assets.deformable_schema_utils import prim_has_applied_schema


def _get_view_method(view, *method_names: str):
    for method_name in method_names:
        method = getattr(view, method_name, None)
        if method is not None:
            return method
    return None


def _sim_buffer_as_torch(value, device: torch.device | str | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        try:
            tensor = as_torch(value)
        except Exception:
            tensor = torch.as_tensor(value)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _sim_buffer_as_view(value, device: torch.device | str | None = None):
    return wrap_sim_buffer(_sim_buffer_as_torch(value, device=device))


def _view_uses_warp(view) -> bool:
    backend = getattr(view, "_backend", None)
    backend_name = type(backend).__name__.lower() if backend is not None else ""
    frontend_name = type(getattr(view, "_frontend", None)).__name__.lower()
    return "warp" in backend_name or "warp" in frontend_name


def _torch_to_warp(value: torch.Tensor):
    value = value.detach().contiguous()
    return wp.from_torch(value)


def _indices_to_warp(indices, device: torch.device | str):
    if isinstance(indices, torch.Tensor):
        index_tensor = indices.to(device=device, dtype=torch.int32).contiguous()
    else:
        index_tensor = torch.as_tensor(indices, device=device, dtype=torch.int32).contiguous()
    return wp.from_torch(index_tensor)


class OmniPhysicsDeformableObjectData:
    def __init__(self, owner: "OmniPhysicsDeformableObject"):
        self._owner = weakref.proxy(owner)
        self._sim_timestamp = 0.0
        self._nodal_pos_w_timestamp = -1.0
        self._nodal_vel_w_timestamp = -1.0
        self._nodal_state_w_timestamp = -1.0
        self._nodal_pos_w = None
        self._nodal_vel_w = None
        self._nodal_state_w = None
        self.default_nodal_state_w = None
        self.nodal_kinematic_target = None

    def update(self, dt: float):
        self._sim_timestamp += dt

    def invalidate(self):
        self._nodal_pos_w_timestamp = -1.0
        self._nodal_vel_w_timestamp = -1.0
        self._nodal_state_w_timestamp = -1.0
        self._nodal_pos_w = None
        self._nodal_vel_w = None
        self._nodal_state_w = None

    @property
    def nodal_pos_w(self):
        if self._nodal_pos_w is None or self._nodal_pos_w_timestamp < self._sim_timestamp:
            self._nodal_pos_w = self._owner.get_simulation_nodal_positions()
            self._nodal_pos_w_timestamp = self._sim_timestamp
        return self._nodal_pos_w

    @property
    def nodal_vel_w(self):
        if self._nodal_vel_w is None or self._nodal_vel_w_timestamp < self._sim_timestamp:
            self._nodal_vel_w = self._owner.get_simulation_nodal_velocities()
            self._nodal_vel_w_timestamp = self._sim_timestamp
        return self._nodal_vel_w

    @property
    def nodal_state_w(self):
        if self._nodal_state_w is None or self._nodal_state_w_timestamp < self._sim_timestamp:
            self._nodal_state_w = torch.cat(
                (as_torch(self.nodal_pos_w), as_torch(self.nodal_vel_w)),
                dim=-1,
            )
            self._nodal_state_w_timestamp = self._sim_timestamp
        return self._nodal_state_w


class OmniPhysicsDeformableObject(AssetBase):
    runtime_api = "omniphysics_deformable_body"

    def __init__(self, cfg):
        self._physics_sim_view = None
        self._root_physx_view = None
        self._root_prim_path = None
        self._root_prim_path_expr = None
        self._runtime_view_kind = None
        self._deformable_type = None
        self._data = OmniPhysicsDeformableObjectData(self)
        self._ALL_INDICES = None
        super().__init__(cfg)

    @property
    def data(self) -> OmniPhysicsDeformableObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        if self._root_physx_view is None:
            return 0
        return int(getattr(self._root_physx_view, "count", 0) or 0)

    @property
    def num_bodies(self) -> int:
        return 1

    @property
    def root_physx_view(self):
        return self._root_physx_view

    @property
    def runtime_view(self):
        return self._root_physx_view

    @property
    def runtime_view_kind(self):
        return self._runtime_view_kind

    @property
    def root_prim_path(self):
        return self._root_prim_path

    @property
    def body_prim_path(self):
        return self._root_prim_path

    def reset(self, env_ids: Sequence[int] | None = None):
        pass

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        self._data.update(dt)

    def get_simulation_nodal_positions(self):
        get_positions = _get_view_method(
            self.root_physx_view,
            "get_simulation_nodal_positions",
            "get_sim_nodal_positions",
        )
        if get_positions is None:
            raise RuntimeError(f"OmniPhysics deformable view has no nodal position getter: {self.cfg.prim_path}")
        return _sim_buffer_as_view(get_positions(), device=self.device)

    def get_simulation_nodal_velocities(self):
        get_velocities = _get_view_method(
            self.root_physx_view,
            "get_simulation_nodal_velocities",
            "get_sim_nodal_velocities",
        )
        if get_velocities is None:
            return torch.zeros_like(as_torch(self.data.nodal_pos_w))
        return _sim_buffer_as_view(get_velocities(), device=self.device)

    def write_nodal_state_to_sim(self, nodal_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        self.write_nodal_pos_to_sim(nodal_state[..., :3], env_ids=env_ids)
        self.write_nodal_velocity_to_sim(nodal_state[..., 3:], env_ids=env_ids)

    def write_nodal_pos_to_sim(self, nodal_pos: torch.Tensor, env_ids: Sequence[int] | None = None):
        full_nodal_pos = as_torch(self.data.nodal_pos_w).clone()
        env_selector = slice(None) if env_ids is None else env_ids
        full_nodal_pos[env_selector] = nodal_pos.clone()
        self._write_nodal_tensor_to_sim(
            full_nodal_pos,
            env_ids,
            "set_simulation_nodal_positions",
            "set_sim_nodal_positions",
        )
        self._data.invalidate()

    def write_nodal_velocity_to_sim(self, nodal_vel: torch.Tensor, env_ids: Sequence[int] | None = None):
        full_nodal_vel = as_torch(self.data.nodal_vel_w).clone()
        env_selector = slice(None) if env_ids is None else env_ids
        full_nodal_vel[env_selector] = nodal_vel.clone()
        self._write_nodal_tensor_to_sim(
            full_nodal_vel,
            env_ids,
            "set_simulation_nodal_velocities",
            "set_sim_nodal_velocities",
        )
        self._data.invalidate()

    def _write_nodal_tensor_to_sim(self, value: torch.Tensor, env_ids, *setter_names: str):
        setter = _get_view_method(self.root_physx_view, *setter_names)
        if setter is None:
            return

        physx_env_ids = env_ids
        if env_ids is None:
            physx_env_ids = self._ALL_INDICES

        if _view_uses_warp(self.root_physx_view):
            setter(
                _torch_to_warp(value.to(device=self.device, dtype=torch.float32)),
                indices=_indices_to_warp(physx_env_ids, self.device),
            )
            return

        setter(value, indices=physx_env_ids)

    def _initialize_impl(self):
        self._physics_sim_view = PhysxManager.get_physics_sim_view()
        if self._physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not available for OmniPhysics deformables.")

        matches = sim_utils.resolve_matching_prims_from_source(self.cfg.prim_path)
        if not matches:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim, template_prim_path_expr = matches[0]
        template_prim_path = template_prim.GetPath().pathString
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim_has_applied_schema(prim, "OmniPhysicsDeformableBodyAPI"),
            traverse_instance_prims=False,
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find an OmniPhysicsDeformableBodyAPI prim under '{template_prim_path}'"
                f" when resolving '{self.cfg.prim_path}'."
            )
        if len(root_prims) > 1:
            matched_paths = [prim.GetPath().pathString for prim in root_prims]
            raise RuntimeError(
                f"Failed to find a single OmniPhysicsDeformableBodyAPI prim under '{template_prim_path}'"
                f" when resolving '{self.cfg.prim_path}'. Found {len(root_prims)}: {matched_paths}."
            )

        root_prim = root_prims[0]
        self._root_prim_path = root_prim.GetPath().pathString
        self._root_prim_path_expr = template_prim_path_expr + self._root_prim_path[len(template_prim_path) :]
        self._deformable_type = self._resolve_deformable_type(root_prim)
        self._root_physx_view, self._runtime_view_kind = self._create_runtime_view(
            self._root_prim_path_expr,
            self._deformable_type,
        )
        if self._root_physx_view is None:
            raise RuntimeError(
                f"Failed to create {self._deformable_type} OmniPhysics deformable body view for "
                f"'{self._root_prim_path_expr}' (resolved from '{self.cfg.prim_path}')."
            )

        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        nodal_positions = as_torch(self.get_simulation_nodal_positions())
        nodal_velocities = as_torch(self.get_simulation_nodal_velocities())
        self._data.default_nodal_state_w = torch.cat((nodal_positions, nodal_velocities), dim=-1)
        self._data.update(0.0)

    def _resolve_deformable_type(self, root_prim):
        if root_prim.HasAPI(UsdShade.MaterialBindingAPI):
            material_paths = UsdShade.MaterialBindingAPI(root_prim).GetDirectBindingRel("physics").GetTargets()
            for mat_path in material_paths:
                mat_prim = root_prim.GetStage().GetPrimAtPath(mat_path)
                if not prim_has_applied_schema(mat_prim, "OmniPhysicsDeformableMaterialAPI"):
                    continue
                if prim_has_applied_schema(mat_prim, "PhysxSurfaceDeformableMaterialAPI"):
                    return "surface"
                if prim_has_applied_schema(mat_prim, "PhysxDeformableMaterialAPI"):
                    return "volume"

        has_tetmesh = (
            len(sim_utils.get_all_matching_child_prims(root_prim.GetPath(), lambda prim: prim.GetTypeName() == "TetMesh"))
            > 0
        )
        if has_tetmesh:
            return "volume"

        has_mesh = (
            len(sim_utils.get_all_matching_child_prims(root_prim.GetPath(), lambda prim: prim.GetTypeName() == "Mesh"))
            > 0
        )
        if has_mesh:
            return "surface"

        raise RuntimeError(
            f"Failed to determine OmniPhysics deformable type for '{root_prim.GetPath().pathString}'."
            " Expected a deformable material binding, TetMesh, or Mesh under the body root."
        )

    def _create_runtime_view(self, root_prim_path_expr: str, deformable_type: str):
        pattern = root_prim_path_expr.replace(".*", "*")
        create_fn_name = (
            "create_surface_deformable_body_view"
            if deformable_type == "surface"
            else "create_volume_deformable_body_view"
        )
        create_fn = getattr(self._physics_sim_view, create_fn_name, None)
        if create_fn is None:
            return None, f"{create_fn_name}:missing"
        try:
            runtime_view = create_fn(pattern)
        except Exception as exc:
            return None, f"{create_fn_name}:{type(exc).__name__}"
        if getattr(runtime_view, "_backend", None) is not None and int(getattr(runtime_view, "count", 0) or 0) > 0:
            return runtime_view, create_fn_name
        return None, f"{create_fn_name}:empty:{pattern}"

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._root_physx_view = None
        self._runtime_view_kind = None
        self._deformable_type = None
        self._ALL_INDICES = None
        self._data.invalidate()
