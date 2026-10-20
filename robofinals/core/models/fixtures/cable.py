import numpy as np
import torch

from robofinals.utils.isaac_data_compat import as_torch
from robofinals.utils.usd_utils import OpenUsd as usd

from .fixture import Fixture
from .fixture_types import FixtureType


class Cable(Fixture):
    fixture_types = [FixtureType.CABLE]
    plug_names = ("Plug002_2", "Plug002")
    default_plug_bbox_size = np.array([0.012, 0.012, 0.006], dtype=np.float64)

    @property
    def reference_prim(self):
        """Rigid plug prim used for runtime state queries."""
        return self._find_reference_prim()

    def scene_names(self):
        """Possible Isaac Lab scene asset names for this cable fixture."""
        names = []
        for candidate in (getattr(self, "name", None), "cable", "RJ45_cable"):
            if candidate:
                names.append(str(candidate))
        return list(dict.fromkeys(names))

    def scene_name(self, env) -> str | None:
        """Registered Isaac Lab scene asset name for this cable fixture."""
        for name in self.scene_names():
            if name in env.scene.rigid_objects or name in env.scene.articulations:
                return name
        return None

    def positions(self, env) -> torch.Tensor | None:
        """Plug/root positions from Isaac Lab scene state."""
        for name in self.scene_names():
            if name in env.scene.rigid_objects:
                return env.scene.rigid_objects[name].data.root_pos_w

            if name in env.scene.articulations:
                data = env.scene.articulations[name].data
                body_pos = getattr(data, "body_pos_w", None)
                body_names = getattr(data, "body_names", [])
                if body_pos is not None and body_names:
                    for body_id, body_name in enumerate(body_names):
                        if "Plug002" in body_name or "plug" in body_name.lower():
                            return body_pos[:, body_id, :]
                return data.root_pos_w

        return None

    @staticmethod
    def to_torch_positions(positions, device: torch.device | str) -> torch.Tensor:
        """Convert cable position buffers to ``(num_envs, 3)`` torch tensors."""
        positions_t = as_torch(positions).to(device=device)
        if positions_t.ndim == 3:
            positions_t = positions_t[:, 0, :]
        if positions_t.ndim != 2 or positions_t.shape[-1] != 3:
            msg = f"Expected cable positions with shape (num_envs, 3), got {tuple(positions_t.shape)}."
            raise RuntimeError(msg)
        return positions_t

    def plug_bbox_size(self) -> np.ndarray:
        """Cached plug AABB size from USD authoring data."""
        if hasattr(self, "_plug_bbox_size"):
            return self._plug_bbox_size

        size = None
        for root_prim in self._candidate_bbox_roots():
            for plug_name in self.plug_names:
                plugs = usd.get_prim_by_name(root_prim, plug_name, only_xform=False, only_first=True)
                if plugs:
                    size = np.asarray(usd.get_prim_size(plugs[0]), dtype=np.float64)
                    break
            if size is not None:
                break

        if size is None or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            size = self.default_plug_bbox_size.copy()

        self._plug_bbox_size = size
        return self._plug_bbox_size

    @staticmethod
    def site_for_env(site: np.ndarray, env_id: int, env_origin: np.ndarray) -> np.ndarray:
        """Fixture site point in world coordinates for one env."""
        site = np.asarray(site)
        if site.ndim == 2:
            site = site[env_id]
        return site + env_origin

    @classmethod
    def region_bounds(cls, region_sites, env_id: int, env_origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-space AABB min/max for one fixture interior region."""
        points = np.asarray([cls.site_for_env(site, env_id, env_origin) for site in region_sites])
        return np.min(points, axis=0), np.max(points, axis=0)

    @staticmethod
    def aabb_overlap_ratio(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
        """Overlap volume divided by the first AABB volume."""
        overlap_min = np.maximum(min_a, min_b)
        overlap_max = np.minimum(max_a, max_b)
        overlap_size = np.maximum(overlap_max - overlap_min, 0.0)
        overlap_volume = float(np.prod(overlap_size))
        a_volume = float(np.prod(np.maximum(max_a - min_a, 1e-8)))
        return overlap_volume / a_volume

    def plug_region_overlap(
        self,
        plug_position: np.ndarray,
        region_sites,
        env_id: int,
        env_origin: np.ndarray,
        position_offset: np.ndarray,
        threshold: float,
    ) -> bool:
        """Whether enough of the plug AABB overlaps an insertion region AABB."""
        overlap_position = plug_position + position_offset
        half_size = self.plug_bbox_size() / 2.0
        plug_min = overlap_position - half_size
        plug_max = overlap_position + half_size
        region_min, region_max = self.region_bounds(region_sites, env_id, env_origin)
        return self.aabb_overlap_ratio(plug_min, plug_max, region_min, region_max) >= threshold

    def plug_inserted(
        self,
        plug_position: np.ndarray,
        int_regions,
        env_id: int,
        env_origin: np.ndarray,
        position_offset: np.ndarray,
        threshold: float,
    ) -> bool:
        """True when the plug bbox overlaps any socket insertion region."""
        return any(
            self.plug_region_overlap(plug_position, sites, env_id, env_origin, position_offset, threshold)
            for sites in int_regions.values()
        )

    def _find_reference_prim(self):
        def _is_rigid_body_reference(prim):
            return usd.is_rigidbody(prim) and not usd.is_articulation_root(prim)

        for plug_name in self.plug_names:
            plugs = usd.get_prim_by_name(self.prim, plug_name, only_xform=False, only_first=True)
            if plugs and _is_rigid_body_reference(plugs[0]):
                return plugs[0]

        def _find_rigid_body_prim(prim):
            for child in prim.GetAllChildren():
                if _is_rigid_body_reference(child):
                    return child
                nested = _find_rigid_body_prim(child)
                if nested is not None:
                    return nested
            return None

        rigid_prim = _find_rigid_body_prim(self.prim)
        if rigid_prim is not None:
            return rigid_prim

        if _is_rigid_body_reference(self.prim):
            return self.prim

        return self.prim

    def _candidate_bbox_roots(self):
        roots = []
        if self.prim is not None:
            roots.append(self.prim)
            world = self.prim.GetStage().GetPrimAtPath("/World")
            if world.IsValid():
                roots.append(world)
        return roots
