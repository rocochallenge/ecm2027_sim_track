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

"""Optional garment loading support for teleop sessions.

This module intentionally avoids importing Isaac Sim, Omni, or LeHome modules at
module import time.  TeleopSession imports this module only after AppLauncher is
running, but keeping heavy/runtime imports inside methods preserves the same
startup discipline as the rest of the teleop stack.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robofinals.utils.log_utils import get_default_logger


@dataclass(frozen=True)
class GarmentSpec:
    """Normalized config for a single garment in a teleop scene."""

    enabled: bool = False
    repo_root: Path | None = None
    garment_name: str = "Top_Long_Unseen_0"
    garment_version: str = "Release"
    garment_cfg_base_path: str = "Assets/objects/Challenge_Garment"
    particle_cfg_path: Path | None = None
    prim_path: str = "/World/Garment/Top_Long_Unseen_0"
    pos: tuple[float, float, float] | None = None
    rot_euler_deg: tuple[float, float, float] | None = None
    scale: tuple[float, float, float] | None = None
    warmup_steps: int = 5
    settle_steps: int = 120
    seed: int = 42

    @classmethod
    def from_args(cls, args: Any) -> "GarmentSpec | None":
        raw_cfg = getattr(args, "garment", None)
        if raw_cfg is None:
            return None
        cfg = _as_mapping(raw_cfg)
        enabled = _as_bool(cfg.get("enabled", False))
        if not enabled:
            return cls(enabled=False)

        repo_root_value = cfg.get("repo_root")
        repo_root = _resolve_optional_path(repo_root_value, base=None)
        if repo_root is None:
            raise ValueError("garment.repo_root is required when garment loading is enabled")
        if not repo_root.exists():
            raise FileNotFoundError(f"LeHome repo root does not exist: {repo_root}")

        garment_name = str(cfg.get("garment_name", cls.garment_name))
        prim_path = str(cfg.get("prim_path") or f"/World/Garment/{garment_name}")
        particle_cfg_path = _resolve_required_path(
            cfg.get("particle_cfg_path"),
            base=repo_root,
            label="garment.particle_cfg_path",
        )

        return cls(
            enabled=True,
            repo_root=repo_root,
            garment_name=garment_name,
            garment_version=str(cfg.get("garment_version", cls.garment_version)),
            garment_cfg_base_path=str(
                cfg.get("garment_cfg_base_path", cls.garment_cfg_base_path)
            ),
            particle_cfg_path=particle_cfg_path,
            prim_path=prim_path,
            pos=_as_float_tuple(cfg.get("pos"), 3, "garment.pos"),
            rot_euler_deg=_as_float_tuple(
                cfg.get("rot_euler_deg"), 3, "garment.rot_euler_deg"
            ),
            scale=_as_scale_tuple(cfg.get("scale")),
            warmup_steps=max(0, int(cfg.get("warmup_steps", cls.warmup_steps))),
            settle_steps=max(0, int(cfg.get("settle_steps", cls.settle_steps))),
            seed=int(cfg.get("seed", cls.seed)),
        )


class GarmentManager:
    """Load, initialize, and reset one garment inside the active stage."""

    def __init__(self, spec: GarmentSpec):
        if not spec.enabled:
            raise ValueError("Cannot create GarmentManager with disabled spec")
        self.spec = spec
        self.garment = None
        self._added_to_stage = False
        self._initialized = False
        self._source_subset_face_ranges: list[dict[str, Any]] = []
        self._proxy_mesh_paths: list[str] = []
        self._proxy_topology_groups: list[dict[str, Any]] | None = None
        self._proxy_materials: list[Any] | None = None
        self._logger = get_default_logger()

    @classmethod
    def from_args(cls, args: Any) -> "GarmentManager | None":
        spec = GarmentSpec.from_args(args)
        if spec is None or not spec.enabled:
            return None
        return cls(spec)

    def add_to_stage(self) -> None:
        """Create the LeHome GarmentObject on the current USD stage."""

        if self._added_to_stage:
            return

        self._ensure_simulation_manager_compatibility()
        prims_utils = self._import_prims_utils()
        OmegaConf, ChallengeGarmentLoader, GarmentObject, np = self._import_lehome_runtime()

        self._configure_demo_like_particle_render_settings()
        self._ensure_parent_prim(prims_utils)
        particle_cfg = OmegaConf.load(str(self.spec.particle_cfg_path))
        garment_cfg = self._load_garment_cfg(ChallengeGarmentLoader)
        self._apply_pose_overrides(garment_cfg)
        self._validate_garment_asset_paths(garment_cfg)

        # LeHome GarmentObject resolves /Assets/... relative to os.getcwd(), and
        # the standalone demo sets cwd to the LeHome repo root before constructing
        # the garment. Keep the same convention here instead of rewriting config
        # paths to a different form.
        with _temporary_cwd(self.spec.repo_root), self._isaacsim_simulation_context_bridge():
            self._ensure_simulation_manager_compatibility()
            self._patch_legacy_prim_backend_utils_for_construction()
            self.garment = GarmentObject(
                prim_path=self.spec.prim_path,
                particle_config=particle_cfg,
                garment_config=garment_cfg,
                rng=np.random.RandomState(self.spec.seed),
            )
            self._sanitize_garment_source_mesh_after_construction()
            self._hide_original_mesh_before_first_render()
        self._added_to_stage = True
        self._initialized = False
        self._logger.info(
            "Loaded LeHome garment %s at %s", self.spec.garment_name, self.spec.prim_path
        )

    def initialize_after_env_reset(self, env: Any) -> None:
        """Initialize cloth after the environment reset and settle it."""

        if self.garment is None:
            self.add_to_stage()
        if self.garment is None:
            return

        self._step_env_sim(env, self.spec.warmup_steps, render=True)
        self._patch_cuda_cloth_view_indexing()
        self._prepare_garment_for_cuda_runtime(env)
        self.garment.initialize()
        self._initialized = True
        self._update_visual_from_cloth()
        self._step_env_sim(env, self.spec.settle_steps, render=True)
        self._update_visual_from_cloth()
        self._flush_renderer(env)
        self._logger.info(
            "Initialized LeHome garment %s after env reset", self.spec.garment_name
        )

    def reset_after_env_reset(self, env: Any) -> None:
        """Reset an existing cloth after the environment has reset."""

        if self.garment is None:
            self.initialize_after_env_reset(env)
            return
        if not self._initialized:
            self.initialize_after_env_reset(env)
            return

        self._patch_cuda_cloth_view_indexing()
        self._prepare_garment_for_cuda_runtime(env)
        self.garment.reset()
        self._update_visual_from_cloth()
        self._step_env_sim(env, self.spec.settle_steps, render=True)
        self._update_visual_from_cloth()
        self._flush_renderer(env)
        self._logger.info("Reset LeHome garment %s", self.spec.garment_name)

    def post_step_update(self) -> None:
        """Per-frame LeHome garment update hook for renderer visibility."""

        if self.garment is None or not self._initialized:
            return
        self._update_visual_from_cloth()

    def mark_stale(self) -> None:
        """Forget stage-bound garment handles after an env/stage remake."""

        self.garment = None
        self._added_to_stage = False
        self._initialized = False
        self._source_subset_face_ranges = []
        self._proxy_mesh_paths = []
        self._proxy_topology_groups = None
        self._proxy_materials = None

    def _update_visual_from_cloth(self) -> None:
        """Update the teleop-visible garment mesh from the cooked cloth state."""

        self._hide_original_mesh_before_first_render()
        self._repair_visual_mesh_points_from_cloth()


    def _prepare_garment_for_cuda_runtime(self, env: Any) -> None:
        """Restore CUDA runtime markers after numpy-only LeHome construction."""

        garment = self.garment
        if garment is None:
            return
        device = str(getattr(getattr(env, "sim", None), "device", None) or getattr(getattr(env, "cfg", None), "device", None) or getattr(self, "device", "cuda:0"))
        if "cuda" not in device.lower():
            return

        for obj in (garment, getattr(garment, "_cloth_prim_view", None)):
            if obj is None:
                continue
            try:
                obj._backend = "torch"
                obj._device = device
                import isaacsim.core.utils.torch as torch_utils

                obj._backend_utils = torch_utils
            except Exception:
                continue

        try:
            garment._gpu_cloth_view_initialized = False
        except Exception:
            pass

    def _sanitize_garment_source_mesh_after_construction(self) -> None:
        """Remove stale render primvars/subsets before the first teleop render sees the garment mesh."""

        garment = self.garment
        mesh_prim_path = getattr(garment, "mesh_prim_path", None)
        if garment is None or not mesh_prim_path:
            return
        try:
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import Usd, UsdGeom
        except ModuleNotFoundError:
            return
        stage = get_current_stage()
        mesh_prim = stage.GetPrimAtPath(str(mesh_prim_path))
        if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
            self._logger.warning("Cannot pre-sanitize LeHome garment mesh; invalid mesh prim: %s", mesh_prim_path)
            return
        mesh = UsdGeom.Mesh(mesh_prim)
        self._source_subset_face_ranges = self._get_source_mesh_subset_face_ranges(mesh)
        self._sanitize_original_mesh_render_data(mesh)
        self._logger.info("Pre-sanitized LeHome garment source mesh before first render: %s", mesh_prim_path)

    def _configure_demo_like_particle_render_settings(self) -> None:
        """Configure renderer settings for LeHome CUDA cloth in teleop."""

        try:
            import carb
        except ModuleNotFoundError:
            return

        settings = carb.settings.get_settings()
        settings.set_bool("/physics/updateParticlesToUsd", False)
        settings.set_bool("/physics/fabricUseGPUInterop", True)
        settings.set_bool("/physics/updateToUsd", False)
        settings.set_bool("/physics/fabricUpdateTransformations", True)
        settings.set_bool("/physics/fabricUpdateVelocities", False)
        settings.set_bool("/app/renderer/skipMaterialLoading", False)
        settings.set_bool("/app/renderer/skipTextureLoading", False)
        self._logger.info(
            "Configured LeHome CUDA cloth render settings: updateParticlesToUsd=false, fabricUseGPUInterop=true"
        )

    def _ensure_simulation_manager_compatibility(self) -> None:
        """Provide Isaac Sim legacy backend helpers after IsaacLab manager patching."""

        from isaacsim.core.simulation_manager import SimulationManager

        # LeHome constructs deprecated Isaac Sim core prim wrappers before PhysX views
        # exist. Keep those USD-authoring wrappers on numpy so pose conversion does not
        # send CUDA tensors into scipy/numpy transform code. The actual cloth view still
        # uses the IsaacLab physics device after env reset.
        SimulationManager._backend = "numpy"

        @classmethod
        def set_backend(cls, val: str) -> None:
            cls._backend = "numpy"

        @classmethod
        def get_backend(cls) -> str:
            return "numpy"

        @classmethod
        def _get_backend_utils(cls):
            import isaacsim.core.utils.numpy as np_utils

            return np_utils

        SimulationManager.set_backend = set_backend
        SimulationManager.get_backend = get_backend
        SimulationManager._get_backend_utils = _get_backend_utils

        if not hasattr(SimulationManager, "get_physics_sim_view") and hasattr(
            SimulationManager, "get_physics_simulation_view"
        ):
            @classmethod
            def get_physics_sim_view(cls):
                return cls.get_physics_simulation_view()

            SimulationManager.get_physics_sim_view = get_physics_sim_view

    def _patch_legacy_prim_backend_utils_for_construction(self) -> None:
        """Force already-imported legacy prim classes to use numpy during construction."""

        try:
            import isaacsim.core.utils.numpy as np_utils
            from isaacsim.core.prims.impl import cloth_prim, prim, single_prim_wrapper, xform_prim
        except ModuleNotFoundError:
            return

        original_prim_init = prim.Prim.__init__
        if not getattr(prim.Prim, "_robofinals_lehome_numpy_init_patch", False):
            def patched_prim_init(self, *args, **kwargs):
                original_prim_init(self, *args, **kwargs)
                self._backend = "numpy"
                self._device = "cpu"
                self._backend_utils = np_utils

            prim.Prim.__init__ = patched_prim_init
            prim.Prim._robofinals_lehome_numpy_init_patch = True

        original_cloth_init = cloth_prim.ClothPrim.__init__
        if not getattr(cloth_prim.ClothPrim, "_robofinals_lehome_numpy_init_patch", False):
            def patched_cloth_init(self, *args, **kwargs):
                original_cloth_init(self, *args, **kwargs)
                self._backend = "numpy"
                self._device = "cpu"
                self._backend_utils = np_utils

            cloth_prim.ClothPrim.__init__ = patched_cloth_init
            cloth_prim.ClothPrim._robofinals_lehome_numpy_init_patch = True


        original_set_world_pose = single_prim_wrapper._SinglePrimWrapper.set_world_pose
        if not getattr(single_prim_wrapper._SinglePrimWrapper, "_robofinals_lehome_world_pose_patch", False):
            def patched_set_world_pose(self, position=None, orientation=None):
                old_backend = getattr(self, "_backend", None)
                old_device = getattr(self, "_device", None)
                old_backend_utils = getattr(self, "_backend_utils", None)
                old_view_backend = getattr(getattr(self, "_prim_view", None), "_backend", None)
                old_view_device = getattr(getattr(self, "_prim_view", None), "_device", None)
                old_view_backend_utils = getattr(getattr(self, "_prim_view", None), "_backend_utils", None)
                try:
                    self._backend = "numpy"
                    self._device = "cpu"
                    self._backend_utils = np_utils
                    if getattr(self, "_prim_view", None) is not None:
                        self._prim_view._backend = "numpy"
                        self._prim_view._device = "cpu"
                        self._prim_view._backend_utils = np_utils
                    return original_set_world_pose(self, position=position, orientation=orientation)
                finally:
                    self._backend = old_backend
                    self._device = old_device
                    self._backend_utils = old_backend_utils
                    if getattr(self, "_prim_view", None) is not None:
                        self._prim_view._backend = old_view_backend
                        self._prim_view._device = old_view_device
                        self._prim_view._backend_utils = old_view_backend_utils

            single_prim_wrapper._SinglePrimWrapper.set_world_pose = patched_set_world_pose
            single_prim_wrapper._SinglePrimWrapper._robofinals_lehome_world_pose_patch = True

        original_set_local_scales = xform_prim.XFormPrim.set_local_scales
        if not getattr(xform_prim.XFormPrim, "_robofinals_lehome_scale_patch", False):
            def patched_set_local_scales(self, scales, indices=None):
                old_backend = getattr(self, "_backend", None)
                old_device = getattr(self, "_device", None)
                old_backend_utils = getattr(self, "_backend_utils", None)
                try:
                    self._backend = "numpy"
                    self._device = "cpu"
                    self._backend_utils = np_utils
                    return original_set_local_scales(self, scales, indices=indices)
                finally:
                    self._backend = old_backend
                    self._device = old_device
                    self._backend_utils = old_backend_utils

            xform_prim.XFormPrim.set_local_scales = patched_set_local_scales
            xform_prim.XFormPrim._robofinals_lehome_scale_patch = True

    @contextmanager
    def _isaacsim_simulation_context_bridge(self):
        """Temporarily bridge Isaac Sim core APIs to the active IsaacLab sim context."""

        from isaaclab.sim import SimulationContext as IsaacLabSimulationContext
        from isaacsim.core.api.simulation_context.simulation_context import (
            SimulationContext as IsaacSimSimulationContext,
        )
        from isaacsim.core.simulation_manager import SimulationManager

        previous_instance = IsaacSimSimulationContext._instance
        previous_initialized = IsaacSimSimulationContext._sim_context_initialized
        previous_backend = getattr(SimulationManager, "_backend", None)
        previous_get_backend = getattr(SimulationManager, "get_backend", None)
        previous_set_backend = getattr(SimulationManager, "set_backend", None)
        previous_get_backend_utils = getattr(SimulationManager, "_get_backend_utils", None)
        lab_sim = IsaacLabSimulationContext.instance()

        self._ensure_simulation_manager_compatibility()
        if lab_sim is not None:
            IsaacSimSimulationContext._instance = _IsaacSimContextBridge(lab_sim)
            IsaacSimSimulationContext._sim_context_initialized = True
        try:
            yield
        finally:
            IsaacSimSimulationContext._instance = previous_instance
            IsaacSimSimulationContext._sim_context_initialized = previous_initialized
            if previous_backend is None:
                if hasattr(SimulationManager, "_backend"):
                    delattr(SimulationManager, "_backend")
            else:
                SimulationManager._backend = previous_backend
            if previous_get_backend is not None:
                SimulationManager.get_backend = previous_get_backend
            if previous_set_backend is not None:
                SimulationManager.set_backend = previous_set_backend
            if previous_get_backend_utils is not None:
                SimulationManager._get_backend_utils = previous_get_backend_utils

    def _load_garment_cfg(self, ChallengeGarmentLoader: Any) -> Any:
        assert self.spec.repo_root is not None
        base_path = self.spec.garment_cfg_base_path
        loader_base_path = base_path
        base_path_obj = Path(base_path).expanduser()
        if base_path_obj.is_absolute():
            try:
                loader_base_path = str(base_path_obj.resolve().relative_to(self.spec.repo_root))
            except ValueError:
                # ChallengeGarmentLoader joins with cwd even for absolute paths; use a relative
                # path from cwd if possible, otherwise the existence check below will fail early.
                loader_base_path = os.path.relpath(str(base_path_obj.resolve()), str(self.spec.repo_root))

        with _temporary_cwd(self.spec.repo_root):
            loader = ChallengeGarmentLoader(base_path=loader_base_path)
            garment_cfg = loader.load_garment_config(
                self.spec.garment_name,
                self.spec.garment_version,
            )
        return garment_cfg

    def _apply_pose_overrides(self, garment_cfg: Any) -> None:
        if self.spec.pos is not None:
            pos_range = list(self.spec.pos) + list(self.spec.pos)
            garment_cfg.initial_pos_range = pos_range
            garment_cfg.soft_reset_pos_range = pos_range
        if self.spec.rot_euler_deg is not None:
            rot_range = list(self.spec.rot_euler_deg) + list(self.spec.rot_euler_deg)
            garment_cfg.initial_rot_range = rot_range
            garment_cfg.soft_reset_rot_range = rot_range
        if self.spec.scale is not None:
            garment_cfg.scale = list(self.spec.scale)

    def _validate_garment_asset_paths(self, garment_cfg: Any) -> None:
        """Validate LeHome cwd-relative asset/material paths without rewriting them."""

        assert self.spec.repo_root is not None
        asset_path = str(garment_cfg.asset_path)
        resolved_asset_path = self._resolve_lehome_cwd_path(asset_path)
        if not resolved_asset_path.exists():
            raise FileNotFoundError(
                f"LeHome garment USD does not exist for {self.spec.garment_name}: {resolved_asset_path}"
            )

        visual_paths = getattr(garment_cfg, "visual_usd_paths", None)
        if not visual_paths:
            return
        for value in visual_paths:
            if value is None:
                continue
            material_path = str(value)
            resolved_material_path = self._resolve_lehome_cwd_path(material_path)
            if not resolved_material_path.exists():
                self._logger.warning(
                    "LeHome visual material USD does not exist for %s: %s",
                    self.spec.garment_name,
                    resolved_material_path,
                )

    def _resolve_lehome_cwd_path(self, path: str) -> Path:
        """Resolve a LeHome config path the same way GarmentObject does under repo-root cwd."""

        assert self.spec.repo_root is not None
        path_obj = Path(path).expanduser()
        if path_obj.is_absolute() and not str(path).startswith("/Assets/"):
            return path_obj.resolve()
        if str(path).startswith("/"):
            return (self.spec.repo_root / str(path)[1:]).resolve()
        return (self.spec.repo_root / path_obj).resolve()

    def _ensure_parent_prim(self, prims_utils: Any) -> None:
        parent_path = str(Path(self.spec.prim_path).parent).replace(".", "/")
        if parent_path and parent_path != "/" and not prims_utils.is_prim_path_valid(parent_path):
            prims_utils.create_prim(parent_path, "Xform")

    def _hide_original_mesh_before_first_render(self) -> None:
        """Hide the original garment mesh until its cooked cloth topology has been synchronized."""

        garment = self.garment
        mesh_prim_path = getattr(garment, "mesh_prim_path", None)
        if garment is None or not mesh_prim_path:
            return

        try:
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import UsdGeom
        except ModuleNotFoundError:
            return

        stage = get_current_stage()
        mesh_prim = stage.GetPrimAtPath(str(mesh_prim_path))
        if mesh_prim.IsValid() and mesh_prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(mesh_prim).MakeInvisible()

    def _sanitize_original_mesh_render_data(self, mesh: Any) -> None:
        """Remove stale render data that is invalid after cloth cooking/topology edits."""

        try:
            from pxr import Gf, UsdGeom, Vt
        except ModuleNotFoundError:
            return

        self._remove_original_mesh_geom_subsets(mesh)
        prim = mesh.GetPrim()
        for prop in list(prim.GetProperties()):
            prop_name = prop.GetName()
            if (
                prop_name in {"normals", "primvars:displayColor", "primvars:displayOpacity"}
                or prop_name.startswith("primvars:normal")
                or prop_name.startswith("primvars:normals")
                or prop_name.startswith("primvars:displayColor:")
                or prop_name.startswith("primvars:displayOpacity:")
                or prop_name.startswith("material:binding:collection:")
                or prop_name.startswith("collection:")
            ):
                try:
                    prop.Clear()
                except Exception:
                    pass
                if prop_name != "normals":
                    prim.RemoveProperty(prop_name)
        points = mesh.GetPointsAttr().Get() or []
        if points:
            zero_normals = Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 1.0) for _ in range(len(points))])
            mesh.GetNormalsAttr().Set(zero_normals)
        else:
            mesh.GetNormalsAttr().Clear()
        try:
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        except Exception:
            pass
        try:
            primvars_api = UsdGeom.PrimvarsAPI(prim)
            st_primvar = primvars_api.GetPrimvar("st")
            if st_primvar and st_primvar.HasValue():
                st_primvar.SetInterpolation(UsdGeom.Tokens.faceVarying)
        except Exception:
            pass

    def _remove_original_mesh_geom_subsets(self, mesh: Any) -> None:
        """Remove stale GeomSubset children after filtering original mesh faces."""

        try:
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import Usd, UsdGeom
        except ModuleNotFoundError:
            return

        stage = get_current_stage()
        removed = 0
        for child in list(Usd.PrimRange(mesh.GetPrim())):
            if child == mesh.GetPrim():
                continue
            if child.IsA(UsdGeom.Subset) or child.GetTypeName() == "GeomSubset":
                stage.RemovePrim(child.GetPath())
                removed += 1
        if removed:
            self._logger.debug("Removed %d stale LeHome GeomSubset prim(s)", removed)

    @staticmethod
    def _flush_renderer(env: Any) -> None:
        """Render one frame after garment initialize/reset to flush original mesh updates."""

        sim = getattr(env, "sim", None)
        if sim is None:
            return
        try:
            if hasattr(sim, "render"):
                sim.render()
            else:
                sim.step(render=True)
        except Exception:
            return

    def _repair_visual_mesh_points_from_cloth(self) -> None:
        """Build/update a renderable visual mesh proxy from finite CUDA cloth positions."""

        garment = self.garment
        cloth_view = getattr(garment, "_cloth_prim_view", None)
        mesh_prim_path = getattr(garment, "mesh_prim_path", None)
        if garment is None or cloth_view is None or not mesh_prim_path:
            return

        try:
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
            import torch
        except ModuleNotFoundError:
            return

        try:
            positions = cloth_view.get_world_positions()
        except Exception as exc:
            self._logger.warning("Could not read LeHome cloth positions for visual mesh update: %s", exc)
            return
        if not torch.is_tensor(positions):
            positions = torch.as_tensor(positions, dtype=torch.float32)
        positions = positions.detach().float().reshape(-1, 3).cpu()
        finite_mask = torch.isfinite(positions).all(dim=1)
        if not finite_mask.all():
            invalid_count = int((~finite_mask).sum().item())
            self._logger.warning("Dropping %d non-finite LeHome cloth point(s) before visual mesh update", invalid_count)
            positions = positions[finite_mask]
        if positions.numel() == 0:
            self._logger.warning("No finite LeHome cloth positions available for visual mesh update")
            return

        local_points = positions
        min_point = local_points.min(dim=0).values
        max_point = local_points.max(dim=0).values

        stage = get_current_stage()
        source_prim = stage.GetPrimAtPath(str(mesh_prim_path))
        if source_prim.IsValid() and source_prim.IsA(UsdGeom.Imageable):
            # The PhysX cloth mesh can have a renderer-incompatible point buffer
            # after auto-particle-cloth cooking. Keep it as the physics source but
            # hide it from Hydra/RTX and render the clean proxy below instead.
            UsdGeom.Imageable(source_prim).MakeInvisible()
        if not source_prim.IsValid() or not source_prim.IsA(UsdGeom.Mesh):
            self._logger.warning("LeHome source mesh prim is not a valid UsdGeom.Mesh: %s", mesh_prim_path)
            return

        source_mesh = UsdGeom.Mesh(source_prim)
        topology_groups = self._proxy_topology_groups
        if topology_groups is None:
            topology_groups = self._build_visual_proxy_topology_groups(source_mesh, local_points)
            if not topology_groups:
                return
            self._proxy_topology_groups = topology_groups
        visual_root = f"{self.spec.prim_path}/robofinals_visual_cloth_mesh"
        UsdGeom.Xform.Define(stage, visual_root)
        loaded_materials = self._proxy_materials
        if loaded_materials is None:
            loaded_materials = self._load_visual_materials_for_proxy(stage)
            self._proxy_materials = loaded_materials
        local_points_array = local_points.numpy()
        for group in topology_groups:
            proxy_path = f"{visual_root}/{group['name']}"
            proxy_mesh = UsdGeom.Mesh.Define(stage, proxy_path)
            proxy_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(local_points_array))
            proxy_mesh.GetExtentAttr().Set(
                Vt.Vec3fArray(
                    [
                        Gf.Vec3f(float(min_point[0]), float(min_point[1]), float(min_point[2])),
                        Gf.Vec3f(float(max_point[0]), float(max_point[1]), float(max_point[2])),
                    ]
                )
            )
            if proxy_path not in self._proxy_mesh_paths:
                proxy_mesh.CreateDoubleSidedAttr().Set(True)
                proxy_mesh.CreateSubdivisionSchemeAttr().Set("none")
                proxy_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(group["face_counts"]))
                proxy_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(group["face_indices"]))
                if group["st_values"]:
                    primvars_api = UsdGeom.PrimvarsAPI(proxy_mesh.GetPrim())
                    st_primvar = primvars_api.CreatePrimvar(
                        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                    )
                    st_primvar.Set(Vt.Vec2fArray(group["st_values"]))
                if group["normals"]:
                    proxy_mesh.GetNormalsAttr().Set(Vt.Vec3fArray(group["normals"]))
                    proxy_mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

                material = None
                material_index = int(group["material_index"])
                if material_index < len(loaded_materials):
                    material = loaded_materials[material_index]
                if material is None:
                    material = self._find_or_create_visual_material(stage)
                if material is not None:
                    UsdShade.MaterialBindingAPI.Apply(proxy_mesh.GetPrim()).Bind(material)
                self._proxy_mesh_paths.append(proxy_path)
            UsdGeom.Imageable(proxy_mesh.GetPrim()).MakeVisible()

        points_prim = stage.GetPrimAtPath(f"{self.spec.prim_path}/robofinals_debug_cloth_points")
        if points_prim.IsValid() and points_prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(points_prim).MakeInvisible()

    def _build_visual_proxy_topology_groups(self, source_mesh: Any, local_points: Any):
        """Create per-submesh face-varying topology for the cooked cloth particle positions."""

        source_face_counts = list(source_mesh.GetFaceVertexCountsAttr().Get() or [])
        source_face_indices = list(source_mesh.GetFaceVertexIndicesAttr().Get() or [])
        if not source_face_counts or not source_face_indices:
            self._logger.warning("LeHome source mesh has no authored topology for visual proxy")
            return []

        source_points = source_mesh.GetPointsAttr().Get() or []
        source_count = len(source_points)
        target_count = int(local_points.shape[0])
        point_count = min(source_count, target_count)
        if point_count <= 0:
            self._logger.warning("LeHome visual proxy has no compatible points")
            return []

        source_st = None
        st_attr = source_mesh.GetPrim().GetAttribute("primvars:st")
        if st_attr:
            source_st = st_attr.Get()
        subset_face_ranges = list(self._source_subset_face_ranges)
        group_map: dict[str, dict[str, Any]] = {}
        cursor = 0
        dropped_faces = 0
        face_idx = 0
        for count in source_face_counts:
            face = source_face_indices[cursor : cursor + count]
            if len(face) == count and all(0 <= int(index) < point_count for index in face):
                group_name, material_index = self._resolve_face_group(face_idx, subset_face_ranges)
                group = group_map.setdefault(
                    group_name,
                    {
                        "name": group_name,
                        "material_index": material_index,
                        "face_counts": [],
                        "face_indices": [],
                        "st_values": [],
                        "normals": [],
                    },
                )
                group["face_counts"].append(int(count))
                group["face_indices"].extend(int(index) for index in face)
                if source_st is not None and len(source_st) >= cursor + count:
                    group["st_values"].extend(source_st[cursor : cursor + count])
            else:
                dropped_faces += 1
            cursor += count
            face_idx += 1

        if dropped_faces:
            self._logger.info(
                "Dropped %d LeHome source face(s) whose indices exceed cooked cloth point count %d/%d",
                dropped_faces,
                target_count,
                source_count,
            )
        groups = [group for group in group_map.values() if group["face_counts"]]
        if not groups:
            self._logger.warning("LeHome visual proxy topology became empty after filtering")
            return []
        return groups

    def _get_source_mesh_subset_face_ranges(self, source_mesh: Any) -> list[dict[str, Any]]:
        """Read source GeomSubset face assignments before we stop rendering the original mesh."""

        try:
            from pxr import UsdGeom
        except ModuleNotFoundError:
            return []

        subsets = []
        for child in source_mesh.GetPrim().GetChildren():
            if not (child.IsA(UsdGeom.Subset) or child.GetTypeName() == "GeomSubset"):
                continue
            indices_attr = child.GetAttribute("indices")
            indices = list(indices_attr.Get() or []) if indices_attr else []
            if not indices:
                continue
            name = child.GetName()
            material_index = 0
            if name.startswith("mesh"):
                suffix = name.replace("mesh", "")
                if suffix:
                    try:
                        material_index = int(suffix)
                    except ValueError:
                        material_index = 0
            subsets.append(
                {
                    "name": name,
                    "material_index": material_index,
                    "face_indices": {int(index) for index in indices},
                }
            )
        return subsets

    @staticmethod
    def _resolve_face_group(face_index: int, subset_face_ranges: list[dict[str, Any]]) -> tuple[str, int]:
        for subset in subset_face_ranges:
            if face_index in subset["face_indices"]:
                return str(subset["name"]), int(subset["material_index"])
        return "mesh", 0

    def _find_or_create_visual_material(self, stage: Any):
        """Return an existing LeHome material if possible, otherwise create a preview fallback."""

        from pxr import Gf, Sdf, Usd, UsdShade

        root_prim = stage.GetPrimAtPath(self.spec.prim_path)
        if root_prim.IsValid():
            for prim in Usd.PrimRange(root_prim):
                if prim.IsA(UsdShade.Material) and not prim.GetPath().pathString.startswith(
                    f"{self.spec.prim_path}/robofinals_"
                ):
                    return UsdShade.Material(prim)

        material_path = f"{self.spec.prim_path}/robofinals_visual_proxy_material"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.2, 0.08))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _load_visual_materials_for_proxy(self, stage: Any) -> list[Any]:
        """Load LeHome visual materials for proxy sub-mesh binding."""

        garment = self.garment
        visual_paths = getattr(garment, "visual_usd_paths", None)
        if garment is None or not visual_paths:
            return []

        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
            from isaacsim.core.utils.string import find_unique_string_name
            from pxr import Usd, UsdShade
        except ModuleNotFoundError:
            return []

        materials = []
        for idx, material_ref_path in enumerate(visual_paths):
            resolved_material_ref_path = str(self._resolve_lehome_cwd_path(str(material_ref_path)))
            material_prim_path = find_unique_string_name(
                f"{self.spec.prim_path}/robofinals_visual_material_{idx}",
                is_unique_fn=lambda path: not stage.GetPrimAtPath(path).IsValid(),
            )
            try:
                add_reference_to_stage(usd_path=resolved_material_ref_path, prim_path=material_prim_path)
                material_root = stage.GetPrimAtPath(material_prim_path)
                material = None
                for prim in Usd.PrimRange(material_root):
                    if prim.IsA(UsdShade.Material):
                        material = UsdShade.Material(prim)
                        break
                if material is None:
                    self._logger.warning(
                        "No material prim found for LeHome proxy material %s under %s",
                        resolved_material_ref_path,
                        material_prim_path,
                    )
                materials.append(material)
            except Exception as exc:
                self._logger.warning("Could not load LeHome proxy material %s: %s", material_ref_path, exc)
                materials.append(None)
        return materials

    @staticmethod
    def _patch_cuda_cloth_view_indexing() -> None:
        """Patch deprecated Isaac Sim cloth APIs that index Warp arrays with torch tensors."""

        try:
            from isaacsim.core.prims.impl.cloth_prim import ClothPrim
        except ModuleNotFoundError:
            return
        if getattr(ClothPrim, "_robofinals_warp_index_patch", False):
            return

        original_get_world_positions = ClothPrim.get_world_positions
        original_set_world_positions = ClothPrim.set_world_positions

        def _is_warp_cloth_view(self) -> bool:
            if str(getattr(self, "_backend", "")).lower() == "warp":
                return True
            try:
                import warp as wp

                return isinstance(self._physics_view.get_positions(), wp.array)
            except Exception:
                return False

        def _to_warp_indices(self, indices):
            import warp as wp

            if isinstance(indices, wp.array):
                return indices.to(getattr(self, "_device", None))
            if indices is None:
                count = int(getattr(self, "count", 0))
                return wp.array(list(range(count)), dtype=wp.int32, device=getattr(self, "_device", None))
            if hasattr(indices, "detach"):
                indices = indices.detach().cpu().tolist()
            elif hasattr(indices, "tolist"):
                indices = indices.tolist()
            return wp.array(indices, dtype=wp.int32, device=getattr(self, "_device", None))

        def _to_torch_index(indices):
            import torch
            import warp as wp

            if isinstance(indices, wp.array):
                return wp.to_torch(indices).to(dtype=torch.long).cpu()
            if torch.is_tensor(indices):
                return indices.to(dtype=torch.long).cpu()
            return torch.as_tensor(indices, dtype=torch.long)

        def _warp_get_world_positions(self, indices=None, clone=True):
            import warp as wp

            warp_indices = _to_warp_indices(self, indices)
            if not self._physics_view:
                return original_get_world_positions(self, indices=warp_indices, clone=clone)
            positions = self._physics_view.get_positions()
            torch_positions = wp.to_torch(positions)
            torch_indices = _to_torch_index(warp_indices)
            result = torch_positions[torch_indices].reshape(len(torch_indices), -1, 3)
            if clone:
                result = result.clone()
            return result.to(device=str(getattr(self, "_device", "cpu")))

        def _warp_set_world_positions(self, positions, indices=None):
            import torch
            import warp as wp

            warp_indices = _to_warp_indices(self, indices)
            if not self._physics_view:
                return original_set_world_positions(self, positions=positions, indices=warp_indices)
            current_positions = wp.to_torch(self._physics_view.get_positions()).clone()
            torch_indices = _to_torch_index(warp_indices)
            if not torch.is_tensor(positions):
                new_positions = torch.as_tensor(positions, dtype=current_positions.dtype)
            else:
                new_positions = positions.detach().to(dtype=current_positions.dtype)
            new_positions = new_positions.reshape(len(torch_indices), -1).to(device=current_positions.device)
            current_positions[torch_indices] = new_positions
            self._physics_view.set_positions(wp.from_torch(current_positions), warp_indices)
            return None

        def patched_get_world_positions(self, indices=None, clone=True):
            if _is_warp_cloth_view(self):
                return _warp_get_world_positions(self, indices=indices, clone=clone)
            return original_get_world_positions(self, indices=indices, clone=clone)

        def patched_set_world_positions(self, positions, indices=None):
            if _is_warp_cloth_view(self):
                return _warp_set_world_positions(self, positions=positions, indices=indices)
            return original_set_world_positions(self, positions=positions, indices=indices)

        ClothPrim.get_world_positions = patched_get_world_positions
        ClothPrim.set_world_positions = patched_set_world_positions
        ClothPrim._robofinals_warp_index_patch = True

    @staticmethod
    def _import_prims_utils() -> Any:
        try:
            from omni.isaac.core.utils import prims as prims_utils
        except ModuleNotFoundError:
            from isaacsim.core.utils import prims as prims_utils
        return prims_utils

    @staticmethod
    def _import_lehome_runtime() -> tuple[Any, Any, Any, Any]:
        from omegaconf import OmegaConf
        import numpy as np
        from robofinals.core.models.garments.garment_object import GarmentObject
        from robofinals.core.models.garments.challenge_garment_loader import (
            ChallengeGarmentLoader,
        )

        return OmegaConf, ChallengeGarmentLoader, GarmentObject, np

    @staticmethod
    def _step_env_sim(env: Any, steps: int, *, render: bool) -> None:
        if steps <= 0:
            return
        for _ in range(steps):
            env.sim.step(render=render)
            if hasattr(env, "scene") and hasattr(env.scene, "update"):
                physics_dt = getattr(env, "physics_dt", None)
                if physics_dt is None:
                    physics_dt = getattr(getattr(env, "sim", None), "physics_dt", None)
                if physics_dt is not None:
                    env.scene.update(dt=physics_dt)


class _IsaacSimContextBridge:
    """Small adapter for Isaac Sim core APIs that expect its SimulationContext."""

    def __init__(self, lab_sim: Any):
        self._lab_sim = lab_sim

    @property
    def backend(self) -> str:
        # Legacy Isaac Sim material/prim wrappers should use numpy for USD authoring.
        return "numpy"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def backend_utils(self) -> Any:
        from isaacsim.core.simulation_manager import SimulationManager

        return SimulationManager._get_backend_utils()

    def get_physics_context(self) -> Any:
        return _NoOpPhysicsContext()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lab_sim, name)


class _NoOpPhysicsContext:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


@contextmanager
def _temporary_cwd(path: Path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"Expected mapping-like garment config, got {type(value).__name__}")


def _resolve_optional_path(value: Any, *, base: Path | None) -> Path | None:
    if _is_empty(value):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _resolve_required_path(value: Any, *, base: Path | None, label: str) -> Path:
    path = _resolve_optional_path(value, base=base)
    if path is None:
        raise ValueError(f"{label} is required")
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _as_float_tuple(value: Any, length: int, label: str) -> tuple[float, ...] | None:
    if _is_empty(value):
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be a list/tuple of {length} numbers")
    if len(value) != length:
        raise ValueError(f"{label} must contain {length} numbers, got {len(value)}")
    return tuple(float(v) for v in value)


def _as_scale_tuple(value: Any) -> tuple[float, float, float] | None:
    if _is_empty(value):
        return None
    if isinstance(value, (int, float)):
        scale = float(value)
        return (scale, scale, scale)
    parsed = _as_float_tuple(value, 3, "garment.scale")
    if parsed is None:
        return None
    return parsed  # type: ignore[return-value]
