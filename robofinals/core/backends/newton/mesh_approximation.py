# Copyright 2025 Lightwheel Team
#
# SPDX-License-Identifier: Apache-2.0

"""Robofinals-specific Newton mesh-approximation overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _MeshApproximationOverride:
    """One targeted replacement for a Newton mesh-approximation request."""

    source_method: str
    label_suffix: str
    target_method: str
    kwargs: dict[str, Any]


_ASSET_IGNORE_PATTERNS = (
    r".*/Trashcan108/Sites/reg_main$",
    r".*/Trashcan108/Sites/reg_int1$",
)
_MESH_APPROXIMATION_OVERRIDES = (
    _MeshApproximationOverride(
        source_method="coacd",
        label_suffix="/Trashcan108/Visuals/Trashcan108",
        target_method="coacd",
        kwargs={
            "threshold": 0.03,
            "preprocess_resolution": 100,
            "resolution": 4000,
            "mcts_nodes": 20,
            "mcts_iterations": 10,
            "mcts_max_depth": 1,
            "merge": False,
            "max_convex_hull": 128,
        },
    ),
    _MeshApproximationOverride(
        source_method="convex_hull",
        label_suffix="/Trashcan108/Visuals/Trashcan108_Clear",
        target_method="coacd",
        kwargs={
            "threshold": 0.05,
            "preprocess_resolution": 50,
            "resolution": 2000,
            "mcts_nodes": 20,
            "mcts_iterations": 5,
            "mcts_max_depth": 1,
            "merge": True,
            "max_convex_hull": 64,
        },
    ),
)


def with_newton_asset_ignore_paths(
    ignore_paths: list[str] | None,
) -> list[str]:
    """Return Newton ignore paths extended by targeted asset overrides.

    Asset-specific patterns stay private to this module so the generic Newton
    backend does not need to know which assets require import-time exclusions.
    """
    resolved_paths = list(ignore_paths or ())
    for pattern in _ASSET_IGNORE_PATTERNS:
        if pattern not in resolved_paths:
            resolved_paths.append(pattern)
    return resolved_paths


def patch_newton_mesh_approximation() -> None:
    """Apply targeted mesh-approximation overrides registered by label.

    Newton 1.4 only exposes the global CoACD concavity threshold through
    ``ModelBuilder.default_mesh_approximation_cfg``. The remaining CoACD
    controls are keyword arguments to ``ModelBuilder.approximate_meshes`` and
    are not configurable through USD. The override table above keeps those
    targeted settings declarative while all unmatched meshes retain Newton's
    defaults.
    """
    try:
        from newton import GeoType, ModelBuilder, ShapeFlags
    except ImportError:
        return

    if getattr(ModelBuilder, "_robofinals_mesh_approximation_patch", False):
        return

    original_approximate_meshes = getattr(ModelBuilder, "approximate_meshes", None)
    if original_approximate_meshes is None:
        return

    def approximate_meshes_patched(
        self,
        method="convex_hull",
        shape_indices=None,
        raise_on_failure=False,
        keep_visual_shapes=False,
        **remeshing_kwargs,
    ):
        if shape_indices is None:
            # Match Newton's default selection exactly: only colliding meshes.
            # Passing every label back as an explicit index would also remesh
            # visual-only meshes, because the explicit-index path intentionally
            # does not apply the collision-flag filter.
            candidate_indices = [
                index
                for index, shape_type in enumerate(self.shape_type)
                if shape_type == GeoType.MESH
                and self.shape_flags[index] & ShapeFlags.COLLIDE_SHAPES
            ]
        else:
            # Newton accepts convex meshes here for explicit re-approximation.
            candidate_indices = [
                index
                for index in shape_indices
                if self.shape_type[index] in (GeoType.MESH, GeoType.CONVEX_MESH)
            ]
        matched_indices: set[int] = set()
        resolved_overrides: list[tuple[_MeshApproximationOverride, list[int]]] = []
        for override in _MESH_APPROXIMATION_OVERRIDES:
            if override.source_method != method:
                continue
            override_indices = [
                index
                for index in candidate_indices
                if index not in matched_indices
                and 0 <= index < len(self.shape_label)
                and str(self.shape_label[index]).endswith(override.label_suffix)
            ]
            if not override_indices:
                continue
            matched_indices.update(override_indices)
            resolved_overrides.append((override, override_indices))

        if not resolved_overrides:
            return original_approximate_meshes(
                self,
                method=method,
                shape_indices=shape_indices,
                raise_on_failure=raise_on_failure,
                keep_visual_shapes=keep_visual_shapes,
                **remeshing_kwargs,
            )

        default_indices = [
            index for index in candidate_indices if index not in matched_indices
        ]
        approximated: set[int] = set()
        if default_indices:
            approximated.update(
                original_approximate_meshes(
                    self,
                    method=method,
                    shape_indices=default_indices,
                    raise_on_failure=raise_on_failure,
                    keep_visual_shapes=keep_visual_shapes,
                    **remeshing_kwargs,
                )
            )

        for override, override_indices in resolved_overrides:
            override_kwargs = (
                dict(remeshing_kwargs)
                if override.target_method == method
                else {}
            )
            override_kwargs.update(override.kwargs)
            labels = [str(self.shape_label[index]) for index in override_indices]
            print(
                "[robofinals] Using targeted mesh approximation "
                f"method={override.target_method} for {labels}: {override.kwargs}",
                flush=True,
            )
            approximated.update(
                original_approximate_meshes(
                    self,
                    method=override.target_method,
                    shape_indices=override_indices,
                    raise_on_failure=raise_on_failure,
                    keep_visual_shapes=keep_visual_shapes,
                    **override_kwargs,
                )
            )
        return approximated

    ModelBuilder.approximate_meshes = approximate_meshes_patched
    ModelBuilder._robofinals_mesh_approximation_patch = True


__all__ = [
    "patch_newton_mesh_approximation",
    "with_newton_asset_ignore_paths",
]
