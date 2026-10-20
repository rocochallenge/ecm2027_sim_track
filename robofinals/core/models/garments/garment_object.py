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

from typing import List, Optional, Sequence

import os
import random

import numpy as np
import omni.kit.commands
import torch
from omegaconf import DictConfig, ListConfig
from pxr import Vt

import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem, SingleXFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name

from robofinals.utils.log_utils import get_default_logger

logger = get_default_logger()


class GarmentObject(SingleClothPrim):
    """Local copy of the LeHome garment runtime used by teleop garment tasks."""

    def __init__(
        self,
        prim_path: str,
        particle_config: DictConfig,
        garment_config: DictConfig,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.rng = rng
        self._gpu_cloth_view_initialized = False

        self.usd_prim_path = prim_path
        self.prim_name = prim_path.split("/")[-1]
        self.prim_parent_path = os.path.dirname(self.usd_prim_path)
        self.mesh_prim_path = find_unique_string_name(
            self.usd_prim_path + "/mesh",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        self.particle_system_path = find_unique_string_name(
            self.usd_prim_path + "/particle_system",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        self.particle_material_path = find_unique_string_name(
            self.usd_prim_path + "/particle_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        self._validate_configs(particle_config, garment_config)

        self.particle_config = particle_config
        self.garment_config = garment_config
        self.objects_config = particle_config.objects

        if "visual_usd_paths" in garment_config and garment_config.visual_usd_paths:
            filtered_paths = [
                path for path in garment_config.visual_usd_paths if path is not None
            ]
            self.visual_usd_paths = filtered_paths if filtered_paths else []
        else:
            self.visual_usd_paths = []

        if "check_point" in garment_config and garment_config.check_point is not None:
            self.check_points = list[int](garment_config.check_point)
        else:
            self.check_points = []

        if (
            "success_distance" in garment_config
            and garment_config.success_distance is not None
        ):
            self.success_distance = list[int](garment_config.success_distance)
        else:
            self.success_distance = []

        self.world_prim = SingleXFormPrim(self.usd_prim_path)
        self.init_pos, self.init_ori, self.init_scale = self._get_initial_pose()

        add_reference_to_stage(usd_path=self._get_usd_path(), prim_path=prim_path)
        self.particle_system = SingleParticleSystem(
            prim_path=self.particle_system_path,
            particle_system_enabled=self.objects_config.particle_system.get(
                "particle_system_enabled", None
            ),
            enable_ccd=self.objects_config.particle_system.get("enable_ccd", None),
            solver_position_iteration_count=self.objects_config.particle_system.get(
                "solver_position_iteration_count", None
            ),
            max_depenetration_velocity=self.objects_config.particle_system.get(
                "max_depenetration_velocity", None
            ),
            global_self_collision_enabled=self.objects_config.particle_system.get(
                "global_self_collision_enabled", None
            ),
            non_particle_collision_enabled=self.objects_config.particle_system.get(
                "non_particle_collision_enabled", None
            ),
            contact_offset=self.objects_config.particle_system.get(
                "contact_offset", None
            ),
            rest_offset=self.objects_config.particle_system.get("rest_offset", None),
            particle_contact_offset=self.objects_config.particle_system.get(
                "particle_contact_offset", None
            ),
            fluid_rest_offset=self.objects_config.particle_system.get(
                "fluid_rest_offset", None
            ),
            solid_rest_offset=self.objects_config.particle_system.get(
                "solid_rest_offset", None
            ),
            wind=self.objects_config.particle_system.get("wind", None),
            max_neighborhood=self.objects_config.particle_system.get(
                "max_neighborhood", None
            ),
            max_velocity=self.objects_config.particle_system.get("max_velocity", None),
        )
        self.particle_material = ParticleMaterial(
            prim_path=self.particle_material_path,
            adhesion=self.objects_config.particle_material.get("adhesion", None),
            adhesion_offset_scale=self.objects_config.particle_material.get(
                "adhesion_offset_scale", None
            ),
            cohesion=self.objects_config.particle_material.get("cohesion", None),
            particle_adhesion_scale=self.objects_config.particle_material.get(
                "particle_adhesion_scale", None
            ),
            particle_friction_scale=self.objects_config.particle_material.get(
                "particle_friction_scale", None
            ),
            drag=self.objects_config.particle_material.get("drag", None),
            lift=self.objects_config.particle_material.get("lift", None),
            friction=self.objects_config.particle_material.get("friction", None),
            damping=self.objects_config.particle_material.get("damping", None),
            gravity_scale=self.objects_config.particle_material.get(
                "gravity_scale", None
            ),
            viscosity=self.objects_config.particle_material.get("viscosity", None),
            vorticity_confinement=self.objects_config.particle_material.get(
                "vorticity_confinement", None
            ),
            surface_tension=self.objects_config.particle_material.get(
                "surface_tension", None
            ),
        )
        self.num_count = 0
        super().__init__(
            name=self.usd_prim_path,
            scale=self.init_scale,
            prim_path=self.mesh_prim_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
            particle_mass=self.objects_config.garment_config.get("particle_mass", None),
            self_collision=self.objects_config.garment_config.get(
                "self_collision", None
            ),
            self_collision_filter=self.objects_config.garment_config.get(
                "self_collision_filter", None
            ),
            stretch_stiffness=self.objects_config.garment_config.get(
                "stretch_stiffness", None
            ),
            bend_stiffness=self.objects_config.garment_config.get(
                "bend_stiffness", None
            ),
            shear_stiffness=self.objects_config.garment_config.get(
                "shear_stiffness", None
            ),
            spring_damping=self.objects_config.garment_config.get(
                "spring_damping", None
            ),
        )

        if self.visual_usd_paths:
            self._apply_visual_material(self.visual_usd_paths)

        self.set_world_pose(position=self.init_pos, orientation=self.init_ori)

    def _device_string(self) -> str:
        return str(getattr(self, "_device", "cpu")).lower()

    def _is_cpu_device(self) -> bool:
        return self._device_string().startswith("cpu")

    def _is_cuda_device(self) -> bool:
        return "cuda" in self._device_string()

    def _ensure_cuda_cloth_view_initialized(self):
        if not hasattr(self, "_cloth_prim_view") or self._cloth_prim_view is None:
            raise RuntimeError(
                "CUDA garment path requires _cloth_prim_view, but it was not created. "
                "Check that SingleClothPrim initialized successfully."
            )
        if self._gpu_cloth_view_initialized:
            return

        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        if self.physics_sim_view is None:
            raise RuntimeError(
                "CUDA garment initialization requires a valid physics sim view. "
                "Call sim.reset() before garment.initialize()."
            )
        self._cloth_prim_view.initialize(self.physics_sim_view)
        self._gpu_cloth_view_initialized = True

    def _prepare_cuda_world_positions(self, positions):
        if positions is None:
            raise RuntimeError("CUDA cloth world positions are None")
        if not torch.is_tensor(positions):
            positions = torch.as_tensor(positions, dtype=torch.float32)
        else:
            positions = positions.detach().clone()

        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise RuntimeError(
                f"CUDA cloth positions must have shape (1, N, 3) or (N, 3), got {tuple(positions.shape)}"
            )
        if not positions.dtype.is_floating_point:
            positions = positions.float()

        device_str = self._device_string()
        if "cuda" in device_str and torch.cuda.is_available():
            target_device = torch.device(
                device_str if device_str != "cuda" else "cuda:0"
            )
            positions = positions.to(device=target_device, dtype=torch.float32)
        else:
            positions = positions.to(dtype=torch.float32)

        if not torch.isfinite(positions).all():
            raise RuntimeError("CUDA cloth positions contain NaN or Inf values")
        return positions

    def _get_cuda_world_positions(self):
        self._ensure_cuda_cloth_view_initialized()
        positions = self._cloth_prim_view.get_world_positions()
        return self._prepare_cuda_world_positions(positions)

    def _log_point_stats(self, label: str, positions) -> None:
        if torch.is_tensor(positions):
            points = positions.detach().cpu()
            finite = bool(torch.isfinite(points).all())
            points_np = points.reshape(-1, 3).numpy()
            shape = tuple(positions.shape)
        else:
            points_np = np.asarray(positions).reshape(-1, 3)
            finite = bool(np.isfinite(points_np).all())
            shape = tuple(np.asarray(positions).shape)
        if points_np.size == 0:
            logger.warning(f"[GarmentObject] {label}: empty point array")
            return
        logger.info(
            f"[GarmentObject] {label}: shape={shape}, "
            f"finite={finite}, centroid={points_np.mean(axis=0).tolist()}, "
            f"aabb_min={points_np.min(axis=0).tolist()}, aabb_max={points_np.max(axis=0).tolist()}"
        )

    def _validate_configs(
        self,
        particle_config: DictConfig,
        garment_config: DictConfig,
    ):
        if garment_config is None:
            raise ValueError("garment_config cannot be None")
        if particle_config is None:
            raise ValueError("particle_config cannot be None")
        if not garment_config.get("asset_path"):
            raise ValueError("garment_config must contain a valid 'asset_path'")

        list_fields = ["visual_usd_paths", "check_point", "success_distance"]
        for field in list_fields:
            val = garment_config.get(field)
            if val is not None and not isinstance(val, (list, ListConfig)):
                raise ValueError(f"garment_config.{field} must be a list")

        if "objects" not in particle_config:
            raise ValueError("particle_config must contain 'objects' key")

        objects_cfg = particle_config.objects
        required_sections = [
            "common",
            "particle_system",
            "particle_material",
            "garment_config",
        ]
        missing = [section for section in required_sections if section not in objects_cfg]
        if missing:
            raise ValueError(f"particle_config.objects missing sections: {missing}")

        hybrid_params = [
            ("initial_pos_range", 6),
            ("initial_rot_range", 6),
            ("soft_reset_pos_range", 6),
            ("soft_reset_rot_range", 6),
            ("scale", 3),
        ]
        for param, expected_len in hybrid_params:
            value = garment_config.get(param) or objects_cfg.common.get(param)
            if value is None:
                raise ValueError(
                    f"Parameter '{param}' missing in garment_config and objects.common"
                )
            if hasattr(value, "__len__") and len(value) != expected_len:
                raise ValueError(
                    f"Invalid length for '{param}': expected {expected_len}, got {len(value)}"
                )

    def _get_usd_path(self) -> str:
        usd_path = self.garment_config.asset_path
        if usd_path.startswith("/"):
            usd_path = os.path.join(os.getcwd(), usd_path[1:])
        return usd_path

    def _get_config_value(
        self,
        field_name: str,
        default_source: str = "common",
    ) -> tuple:
        if (
            field_name in self.garment_config
            and self.garment_config[field_name] is not None
        ):
            return self.garment_config[field_name], "garment_config"
        if default_source == "common":
            value = self.objects_config.common.get(field_name, None)
        else:
            value = self.objects_config.get(default_source, {}).get(field_name, None)
        return value, f"objects_config.{default_source}"

    def initialize(self):
        self.set_world_pose(position=self.init_pos, orientation=self.init_ori)

        if self._is_cuda_device():
            self._ensure_cuda_cloth_view_initialized()

        self._get_initial_info()

        if self._is_cpu_device():
            self._prim.GetAttribute("points").Set(
                Vt.Vec3fArray.FromNumpy(self._get_points_pose().detach().cpu().numpy())
            )

    def reset(self):
        if self._is_cpu_device():
            self._prim.GetAttribute("points").Set(
                Vt.Vec3fArray.FromNumpy(self.initial_points_positions)
            )
        else:
            self._ensure_cuda_cloth_view_initialized()
            self._cloth_prim_view.set_world_positions(
                self._prepare_cuda_world_positions(self.initial_points_positions)
            )

        pos_reset_range, _ = self._get_config_value("soft_reset_pos_range", "common")
        rot_reset_range, _ = self._get_config_value("soft_reset_rot_range", "common")

        if self.rng is not None:
            pos = [
                self.rng.uniform(pos_reset_range[0], pos_reset_range[3]),
                self.rng.uniform(pos_reset_range[1], pos_reset_range[4]),
                self.rng.uniform(pos_reset_range[2], pos_reset_range[5]),
            ]
            ori = [
                self.rng.uniform(rot_reset_range[0], rot_reset_range[3]),
                self.rng.uniform(rot_reset_range[1], rot_reset_range[4]),
                self.rng.uniform(rot_reset_range[2], rot_reset_range[5]),
            ]
        else:
            pos = [
                random.uniform(pos_reset_range[0], pos_reset_range[3]),
                random.uniform(pos_reset_range[1], pos_reset_range[4]),
                random.uniform(pos_reset_range[2], pos_reset_range[5]),
            ]
            ori = [
                random.uniform(rot_reset_range[0], rot_reset_range[3]),
                random.uniform(rot_reset_range[1], rot_reset_range[4]),
                random.uniform(rot_reset_range[2], rot_reset_range[5]),
            ]

        self.set_world_pose(pos, euler_angles_to_quat(ori, degrees=True))
        self.reset_pose = np.concatenate(
            [np.array(pos, dtype=np.float32), np.array(ori, dtype=np.float32)]
        )

    def get_current_mesh_points(
        self,
        visualize: bool = False,
        save: bool = False,
        save_path: str = "./pointcloud.ply",
    ):
        if self._is_cpu_device():
            pos_world, ori_world = self.get_world_pose()
            scale_world = self.get_world_scale()
            mesh_points = self._get_points_pose().detach().cpu().numpy()
            transformed_mesh_points = self.transform_points(
                mesh_points,
                pos_world.detach().cpu().numpy(),
                ori_world.detach().cpu().numpy(),
                scale_world.detach().cpu().numpy(),
            )
        else:
            positions = self._get_cuda_world_positions()
            mesh_points = positions.squeeze(0).detach().cpu().numpy()
            transformed_mesh_points = mesh_points
            pos_world = None
            ori_world = None

        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed_mesh_points)
        if visualize:
            o3d.visualization.draw_geometries([pcd])
        if save:
            o3d.io.write_point_cloud(save_path, pcd)
        return transformed_mesh_points, mesh_points, pos_world, ori_world

    def _apply_visual_material(self, material_paths: Sequence[str]):
        loaded_materials = []

        for i, path in enumerate(material_paths):
            if path.startswith("/"):
                path = os.path.join(os.getcwd(), path[1:])

            mat_prim_path = find_unique_string_name(
                f"{self.usd_prim_path}/visual_material_{i}",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            add_reference_to_stage(usd_path=path, prim_path=mat_prim_path)

            vis_prim = prims_utils.get_prim_at_path(mat_prim_path)
            children = prims_utils.get_prim_children(vis_prim)
            loaded_materials.append(children[0].GetPath() if children else None)

        self.mesh_prim = prims_utils.get_prim_at_path(self.mesh_prim_path)
        self.garment_submesh = prims_utils.get_prim_children(self.mesh_prim)

        if len(self.garment_submesh) == 0:
            if loaded_materials and loaded_materials[0] is not None:
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=self.mesh_prim_path,
                    material_path=loaded_materials[0],
                )
            return

        for prim in self.garment_submesh:
            name = prim.GetName()
            target_idx = 0
            if name.startswith("mesh") and name != "mesh":
                try:
                    target_idx = int(name.replace("mesh", ""))
                except ValueError:
                    target_idx = 0

            if target_idx < len(loaded_materials) and loaded_materials[target_idx] is not None:
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=prim.GetPath(),
                    material_path=loaded_materials[target_idx],
                )

    def _get_initial_pose(self):
        pos_init_range, _ = self._get_config_value("initial_pos_range", "common")
        rot_init_range, _ = self._get_config_value("initial_rot_range", "common")

        if self.rng is not None:
            pos = [
                self.rng.uniform(pos_init_range[0], pos_init_range[3]),
                self.rng.uniform(pos_init_range[1], pos_init_range[4]),
                self.rng.uniform(pos_init_range[2], pos_init_range[5]),
            ]
            ori = [
                self.rng.uniform(rot_init_range[0], rot_init_range[3]),
                self.rng.uniform(rot_init_range[1], rot_init_range[4]),
                self.rng.uniform(rot_init_range[2], rot_init_range[5]),
            ]
        else:
            pos = [
                random.uniform(pos_init_range[0], pos_init_range[3]),
                random.uniform(pos_init_range[1], pos_init_range[4]),
                random.uniform(pos_init_range[2], pos_init_range[5]),
            ]
            ori = [
                random.uniform(rot_init_range[0], rot_init_range[3]),
                random.uniform(rot_init_range[1], rot_init_range[4]),
                random.uniform(rot_init_range[2], rot_init_range[5]),
            ]

        scale, _ = self._get_config_value("scale", "common")
        self.reset_pose = np.concatenate(
            [np.array(pos, dtype=np.float32), np.array(ori, dtype=np.float32)]
        )
        return pos, euler_angles_to_quat(ori, degrees=True), scale

    def _get_initial_info(self):
        if self._is_cpu_device():
            self.initial_points_positions = (
                self._get_points_pose().detach().cpu().numpy()
            )
            return
        self.initial_points_positions = self._get_cuda_world_positions().detach().clone()
        self._log_point_stats("Initial CUDA cloth points", self.initial_points_positions)

    def transform_points(self, points, pos, ori, scale):
        ori_matrix = quat_to_rot_matrix(ori)
        return (points * scale) @ ori_matrix.T + pos

    def inverse_transform_points(self, transformed_points, pos, ori, scale):
        ori_matrix = quat_to_rot_matrix(ori)
        shifted_points = transformed_points - pos
        rotated_points = shifted_points @ ori_matrix
        return rotated_points / scale

    def get_all_pose(self):
        return {"Garment": self.reset_pose}

    def set_all_pose(self, pose_dict: dict):
        if "Garment" not in pose_dict:
            return
        pose = pose_dict["Garment"]
        pos = pose[:3]
        ori = pose[3:]
        self.set_world_pose(
            [0.0, 0.0, 0.0],
            euler_angles_to_quat([0.0, 0.0, 0.0], degrees=True),
        )
        if self._is_cpu_device():
            self._prim.GetAttribute("points").Set(
                Vt.Vec3fArray.FromNumpy(self.initial_points_positions)
            )
        else:
            self._ensure_cuda_cloth_view_initialized()
            self._cloth_prim_view.set_world_positions(
                self._prepare_cuda_world_positions(self.initial_points_positions)
            )
        self.set_world_pose(pos, euler_angles_to_quat(ori, degrees=True))
        self.reset_pose = np.array(pose, dtype=np.float32)


def set_single_prim_visible(prim_path, visible=True):
    set_prim_visibility(prim_path, visible)


def set_single_prim_material(
    prim_path,
    color: Optional[np.ndarray] = None,
    roughness: float = 0.5,
    metallic: float = 0.0,
    specular: float = 0.5,
):
    preview_surface_path = find_unique_string_name(
        initial_name=prim_path + "/preview_surface",
        is_unique_fn=lambda x: not is_prim_path_valid(x),
    )
    preview_surface = PreviewSurface(
        prim_path=preview_surface_path,
        color=np.array(color) if color is not None else None,
        roughness=roughness,
        metallic=metallic,
        specular=specular,
    )
    prim = prims_utils.get_prim_at_path(prim_path)
    children = prims_utils.get_prim_children(prim)
    targets = children if children else [prim]
    for target in targets:
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=target.GetPath(),
            material_path=preview_surface_path,
        )
