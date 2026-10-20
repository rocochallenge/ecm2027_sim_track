# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from robofinals.utils.isaac_data_compat import as_torch

SELECTED_MESH_VERTICES_JSON = (
    Path(__file__).resolve().parents[4] / "robofinals_tasks" / "selected_mesh_vertices.json"
)

PAPER_MESH_001 = "paper_plane_2_001"
PAPER_MESH_03 = "paper_plane_2_03"
PAPER_MESH_KEYS = (PAPER_MESH_001, PAPER_MESH_03)

# (mesh_a, index_a, mesh_b, index_b) — all distances must be <= threshold.
PAPER_AIRPLANE_CHECKPOINT_PAIRS: tuple[tuple[str, int, str, int], ...] = (
    (PAPER_MESH_001, 52, PAPER_MESH_03, 169),
    (PAPER_MESH_03, 56, PAPER_MESH_03, 113),
    (PAPER_MESH_001, 257, PAPER_MESH_001, 139),
    (PAPER_MESH_03, 54, PAPER_MESH_03, 6),
    (PAPER_MESH_001, 243, PAPER_MESH_001, 202),
)

DEFAULT_FOLD_DISTANCE_THRESHOLD_M = 0.025


def _load_mesh_vertex_counts(json_path: Path = SELECTED_MESH_VERTICES_JSON) -> dict[str, int]:
    if not json_path.is_file():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    meshes = payload.get("meshes", {})
    return {
        mesh_name: int(mesh_info.get("vertex_count", 0))
        for mesh_name, mesh_info in meshes.items()
        if isinstance(mesh_info, dict)
    }


MESH_VERTEX_COUNTS = _load_mesh_vertex_counts()


def _pair_distance(positions: dict[str, torch.Tensor], mesh_a: str, idx_a: int, mesh_b: str, idx_b: int) -> torch.Tensor:
    return torch.linalg.norm(positions[mesh_a][:, idx_a, :] - positions[mesh_b][:, idx_b, :], dim=-1)


def check_paper_airplane_fold(
    positions: dict[str, torch.Tensor],
    threshold_m: float = DEFAULT_FOLD_DISTANCE_THRESHOLD_M,
) -> torch.Tensor:
    """Return a bool tensor of shape (num_envs,) for paper-airplane fold success."""
    if not positions:
        raise ValueError("positions must contain deformable mesh nodal positions")

    num_envs = next(iter(positions.values())).shape[0]
    success = torch.ones(num_envs, dtype=torch.bool, device=next(iter(positions.values())).device)
    for mesh_a, idx_a, mesh_b, idx_b in PAPER_AIRPLANE_CHECKPOINT_PAIRS:
        success &= _pair_distance(positions, mesh_a, idx_a, mesh_b, idx_b) <= threshold_m
    return success


class PaperFoldModel:
    """Read OmniPhysics deformable nodal positions and evaluate airplane fold checkpoints."""

    def __init__(
        self,
        env: Any,
        distance_threshold_m: float = DEFAULT_FOLD_DISTANCE_THRESHOLD_M,
        mesh_keys: tuple[str, ...] = PAPER_MESH_KEYS,
    ):
        self.env = env
        self.distance_threshold_m = distance_threshold_m
        self.mesh_keys = mesh_keys

    def get_mesh_nodal_positions(self) -> dict[str, torch.Tensor] | None:
        deformables = getattr(self.env.scene, "deformable_objects", {})
        positions: dict[str, torch.Tensor] = {}
        for mesh_key in self.mesh_keys:
            deformable = deformables.get(mesh_key)
            if deformable is None:
                return None
            nodal_pos = as_torch(deformable.data.nodal_pos_w)
            if nodal_pos.ndim != 3 or nodal_pos.shape[-1] != 3:
                return None
            expected_count = MESH_VERTEX_COUNTS.get(mesh_key)
            if expected_count is not None and nodal_pos.shape[1] < expected_count:
                return None
            max_index = max(
                idx
                for mesh_a, idx_a, mesh_b, idx_b in PAPER_AIRPLANE_CHECKPOINT_PAIRS
                for mesh, idx in ((mesh_a, idx_a), (mesh_b, idx_b))
                if mesh == mesh_key
            )
            if nodal_pos.shape[1] <= max_index:
                return None
            if not torch.isfinite(nodal_pos).all():
                return None
            positions[mesh_key] = nodal_pos
        return positions

    def check_success(self) -> torch.Tensor:
        positions = self.get_mesh_nodal_positions()
        if positions is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        return check_paper_airplane_fold(positions, threshold_m=self.distance_threshold_m)
