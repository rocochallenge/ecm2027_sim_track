# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import math as math_utils

from isaaclab_arena.assets.asset import Asset


class ObjectType(Enum):
    BASE = "base"
    RIGID = "rigid"
    ARTICULATION = "articulation"
    SPAWNER = "spawner"
    DEFORMABLE = "deformable"


class ObjectBase(Asset, ABC):
    """Parent class for (spawnable) Object and ObjectReference."""

    def __init__(
        self,
        name: str,
        prim_path: str | None = None,
        object_type: ObjectType = ObjectType.BASE,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if prim_path is None:
            prim_path = "{ENV_REGEX_NS}/" + self.name
        self.prim_path = prim_path
        self.object_type = object_type

    def set_prim_path(self, prim_path: str) -> None:
        self.prim_path = prim_path

    def get_prim_path(self) -> str:
        return self.prim_path

    def get_object_cfg(self) -> dict[str, Any]:
        return {self.name: self.object_cfg}

    def _init_object_cfg(self) -> RigidObjectCfg | ArticulationCfg | AssetBaseCfg | DeformableObjectCfg:
        if self.object_type == ObjectType.RIGID:
            object_cfg = self._generate_rigid_cfg()
        elif self.object_type == ObjectType.ARTICULATION:
            object_cfg = self._generate_articulation_cfg()
        elif self.object_type == ObjectType.BASE:
            object_cfg = self._generate_base_cfg()
        elif self.object_type == ObjectType.SPAWNER:
            object_cfg = self._generate_spawner_cfg()
        elif self.object_type == ObjectType.DEFORMABLE:
            object_cfg = self._generate_deformable_cfg()
        else:
            raise ValueError(f"Invalid object type: {self.object_type}")
        return object_cfg

    def get_object_pose(self, env: ManagerBasedEnv, is_relative: bool = True) -> torch.Tensor:
        """Get the pose of the object in the environment.

        .. warning::
            A deformable object has no single, backend-independent root orientation after it
            starts deforming. Use :meth:`get_object_position` when the caller only needs a
            representative position. Returning an identity or authored quaternion here would
            look convenient, but it would silently report a rotation that is no longer defined
            by the simulated state.

        Args:
            env: The environment.
            is_relative: Whether to return the pose in the relative frame of the environment.

        Returns:
            The pose of the object in each environment. The shape is (num_envs, 7).
            The order is (x, y, z, qw, qx, qy, qz).
        """
        # We require that the asset has been added to the scene under its name.
        assert self.name in env.scene.keys(), f"Asset {self.name} not found in scene"
        if (self.object_type == ObjectType.RIGID) or (self.object_type == ObjectType.ARTICULATION):
            object_pose = env.scene[self.name].data.root_pose_w.clone()
        elif self.object_type == ObjectType.BASE:
            object_pose = torch.cat(env.scene[self.name].get_world_poses(), dim=-1)
        else:
            raise ValueError(f"Function not implemented for object type: {self.object_type}")
        if is_relative:
            object_pose[:, :3] -= env.scene.env_origins
        return object_pose

    def get_object_position(self, env: ManagerBasedEnv, is_relative: bool = True) -> torch.Tensor:
        """Get a representative position for any supported object type.

        For rigid objects and articulations this is the simulated root position. For a base
        object it is the USD prim position. For a deformable object it is the static root Xform
        position authored by ``DeformableObjectCfg.init_state`` when the asset was spawned.

        .. warning::
            The deformable position is only a stable asset anchor. The solver updates TetMesh
            nodes rather than this root Xform, so the returned value does not follow translation,
            deformation, falling, or calls to :meth:`write_object_pose_to_sim`. Use nodal data
            explicitly when a task genuinely needs the current simulated soft-body location.

        .. note::
            Reading IsaacLab's deformable ``data.root_pos_w`` would launch a backend kernel that
            reduces all simulation nodes once per simulation timestamp. Returning the static
            Xform anchor avoids that ``O(num_envs * num_nodes)`` work in observation loops.

        Args:
            env: The environment containing this object under :attr:`name`.
            is_relative: If true, subtract each environment origin from the world position.

        Returns:
            Object positions with shape ``(num_envs, 3)``.
        """
        assert self.name in env.scene.keys(), f"Asset {self.name} not found in scene"
        asset = env.scene[self.name]

        if self.object_type in (ObjectType.RIGID, ObjectType.ARTICULATION):
            object_position = asset.data.root_pos_w.torch.clone()
        elif self.object_type == ObjectType.DEFORMABLE:
            # The spawned root Xform is initialized from cfg.init_state. Reconstructing this
            # immutable anchor from the cfg is cheaper and safer than querying USD every frame,
            # which may introduce CPU/GPU synchronization without providing simulated motion.
            initial_position = torch.as_tensor(
                asset.cfg.init_state.pos,
                dtype=env.scene.env_origins.dtype,
                device=env.scene.env_origins.device,
            )
            if is_relative:
                return initial_position.expand(asset.num_instances, -1).clone()
            object_position = env.scene.env_origins.clone()
            object_position += initial_position
        elif self.object_type == ObjectType.BASE:
            object_position = asset.get_world_poses()[0].torch.clone()
        else:
            raise ValueError(f"Function not implemented for object type: {self.object_type}")

        if is_relative:
            object_position -= env.scene.env_origins
        return object_position

    def write_object_pose_to_sim(
        self,
        env: ManagerBasedEnv,
        object_pose: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        is_relative: bool = True,
    ) -> None:
        """Reset an object's simulated state at a requested root pose.

        This provides one Arena entry point for rigid objects, articulations, and deformable
        objects. Runtime poses are passed through in the current IsaacLab backend convention
        ``(x, y, z, qx, qy, qz, qw)``. Relative positions are converted to world positions by
        adding ``scene.env_origins``; quaternions are always expressed in the world-aligned
        environment frame.

        A deformable has no writable rigid root in the solver. Its implementation therefore
        takes ``data.default_nodal_state_w``, expresses those default nodes relative to the
        configured initial root pose, and rigidly transforms every node to ``object_pose``.
        This keeps the authored root-pose convention used by ``DeformableObjectCfg.init_state``.

        .. warning::
            This method is reset/re-placement functionality, not a continuous soft-body
            controller. For deformables it deliberately restores the undeformed default shape
            and zeros all nodal velocities. Any deformation, momentum, or externally written
            nodal state accumulated before this call is lost.

        .. warning::
            :meth:`get_object_position` returns the static configured root Xform anchor. This
            method only writes solver state and intentionally does not move that Xform, so a
            subsequent position read will still return the original configured anchor rather
            than ``object_pose``.

        .. warning::
            Kinematically driven deformable nodes are rejected. Their target positions require
            a separate, policy-dependent transformation; leaving old targets in place can snap
            the object back on the next simulation step. Visual-surface updates are also owned
            by the selected backend and may not become visible until a subsequent update/render.

        Args:
            env: The environment containing this object under :attr:`name`.
            object_pose: Target poses with shape ``(len(env_ids), 7)`` or ``(7,)`` for one
                environment. Quaternion order is the backend's ``xyzw``.
            env_ids: Environment indices to update. ``None`` updates every instance.
            is_relative: If true, target positions are relative to each environment origin.

        Raises:
            ValueError: If the pose shape, selected instance count, or quaternion is invalid.
            RuntimeError: If a deformable default state is unavailable or it has kinematic nodes.
        """
        assert self.name in env.scene.keys(), f"Asset {self.name} not found in scene"
        asset = env.scene[self.name]
        device = torch.device(asset.device)

        if env_ids is None:
            resolved_env_ids = torch.arange(asset.num_instances, dtype=torch.long, device=device)
        else:
            resolved_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=device)
            if resolved_env_ids.ndim != 1:
                raise ValueError(f"env_ids must be one-dimensional, got shape {tuple(resolved_env_ids.shape)}")

        target_pose = torch.as_tensor(object_pose, dtype=torch.float32, device=device)
        if target_pose.ndim == 1:
            target_pose = target_pose.unsqueeze(0)
        expected_shape = (resolved_env_ids.numel(), 7)
        if tuple(target_pose.shape) != expected_shape:
            raise ValueError(f"object_pose must have shape {expected_shape}, got {tuple(target_pose.shape)}")
        target_pose = target_pose.clone()

        quat_norm = torch.linalg.vector_norm(target_pose[:, 3:7], dim=-1, keepdim=True)
        if torch.any(quat_norm < 1.0e-8):
            raise ValueError("object_pose contains a zero-length quaternion")
        # Normalize here so a slightly noisy placement quaternion cannot scale or shear the
        # deformable's node cloud through quaternion-vector arithmetic.
        target_pose[:, 3:7] /= quat_norm

        if is_relative:
            target_pose[:, :3] += env.scene.env_origins.index_select(0, resolved_env_ids).to(device)

        if self.object_type in (ObjectType.RIGID, ObjectType.ARTICULATION):
            asset.write_root_pose_to_sim_index(root_pose=target_pose, env_ids=resolved_env_ids)
            # Re-placement is treated as a reset for every dynamic object type. Keeping the old
            # velocity after teleporting can create large impulses and makes rigid/deformable
            # behavior inconsistent at task reset.
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((resolved_env_ids.numel(), 6), device=device),
                env_ids=resolved_env_ids,
            )
            return

        if self.object_type != ObjectType.DEFORMABLE:
            raise ValueError(f"Function not implemented for object type: {self.object_type}")

        default_nodal_state = asset.data.default_nodal_state_w
        if default_nodal_state is None:
            raise RuntimeError(
                f"Deformable asset {self.name} has no default nodal state; "
                "the physics backend may not have finished initialization"
            )

        kinematic_target = asset.data.nodal_kinematic_target
        if kinematic_target is not None:
            selected_targets = kinematic_target.torch.index_select(0, resolved_env_ids)
            if torch.any(selected_targets[..., 3] < 0.5):
                raise RuntimeError(
                    f"Deformable asset {self.name} has kinematically driven nodes. "
                    "Their targets must be transformed together with the nodal state."
                )

        nodal_state = default_nodal_state.torch.index_select(0, resolved_env_ids).clone()
        num_nodes = nodal_state.shape[1]

        # ``cfg.init_state`` and all runtime IsaacLab poses in this branch use xyzw. The default
        # nodes are already in world space, so first remove the configured initial root transform
        # and then apply the requested root transform. Do not center around the node mean here:
        # doing so would change the meaning of the target from USD root pose to TetMesh centroid.
        source_position = torch.tensor(asset.cfg.init_state.pos, dtype=nodal_state.dtype, device=device)
        source_position = source_position.unsqueeze(0) + env.scene.env_origins.index_select(0, resolved_env_ids).to(
            device
        )
        source_quaternion = torch.tensor(asset.cfg.init_state.rot, dtype=nodal_state.dtype, device=device)
        source_quaternion = source_quaternion.unsqueeze(0).expand(resolved_env_ids.numel(), -1)
        source_quaternion = source_quaternion / torch.linalg.vector_norm(
            source_quaternion, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)

        default_offset = nodal_state[..., :3] - source_position.unsqueeze(1)
        source_quaternion_per_node = source_quaternion.unsqueeze(1).expand(-1, num_nodes, -1)
        local_nodal_position = math_utils.quat_apply_inverse(source_quaternion_per_node, default_offset)
        target_quaternion_per_node = target_pose[:, 3:7].unsqueeze(1).expand(-1, num_nodes, -1)
        nodal_state[..., :3] = math_utils.quat_apply(target_quaternion_per_node, local_nodal_position) + target_pose[
            :, :3
        ].unsqueeze(1)
        nodal_state[..., 3:] = 0.0

        asset.write_nodal_state_to_sim_index(nodal_state, env_ids=resolved_env_ids)

    def get_contact_sensor_cfg(self, contact_against_prim_paths: list[str] | None = None) -> ContactSensorCfg:
        assert self.object_type == ObjectType.RIGID, "Contact sensor is only supported for rigid objects"
        if contact_against_prim_paths is None:
            contact_against_prim_paths = []
        return ContactSensorCfg(
            prim_path=self.prim_path,
            filter_prim_paths_expr=contact_against_prim_paths,
        )

    @abstractmethod
    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        # Subclasses must implement this method
        pass

    @abstractmethod
    def _generate_articulation_cfg(self) -> ArticulationCfg:
        # Subclasses must implement this method
        pass

    @abstractmethod
    def _generate_base_cfg(self) -> AssetBaseCfg:
        # Subclasses must implement this method
        pass

    def _generate_spawner_cfg(self) -> AssetBaseCfg:
        # Object Subclasses must implement this method
        pass

    @abstractmethod
    def _generate_deformable_cfg(self) -> DeformableObjectCfg:
        # Object Subclasses must implement this method
        pass
