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

from __future__ import annotations

import torch

import robofinals.utils.object_utils as OU
from robofinals.core.models.fixtures import FixtureType
from robofinals.modules.tasks.base import LwTaskBase


class HomeBaseTask(LwTaskBase):
    """Shared helpers for Lightwheel home-scene manipulation tasks.

    Convention:
    - ``_setup_kitchen_references``: register scene *fixtures* only (table, fridge, ...).
    - ``_get_obj_cfgs``: spawn every object the robot may manipulate or use as receptacle.
    - ``_check_success``: vectorized semantic success on spawned objects / fixtures.

    Per-embodiment robot pose / object offsets are configured on :class:`LwTaskBase`.
    """

    resample_objects_placement_on_reset: bool = True
    table_preferred_name: str | None = None

    @staticmethod
    def biyam_table_pos(x: float, y: float, z: float = 0.524) -> list[float]:
        """Default BiYam table-mount base height for home tabletop tasks."""
        return [float(x), float(y), float(z)]

    def _register_fixture_by_name(
        self,
        ref_name: str,
        preferred_name: str,
        *,
        size: tuple[float, float] = (0.8, 0.4),
        fallback_type: FixtureType | None = None,
        ref=None,
    ):
        """Resolve a scene fixture: exact name -> substring -> fallback type."""
        for full_name_check in (True, False):
            fn_kwargs = dict(id=preferred_name, full_name_check=full_name_check, size=size)
            if ref is not None:
                fn_kwargs["ref"] = ref
            fixture = self.register_fixture_ref(ref_name, fn_kwargs)
            if fixture is not None:
                return fixture

        if fallback_type is None:
            return None

        fallback_kwargs = dict(id=fallback_type, size=size)
        if ref is not None:
            fallback_kwargs["ref"] = ref
        return self.register_fixture_ref(ref_name, fallback_kwargs)

    def _register_table(
        self,
        preferred_name: str | None = None,
        *,
        ref_name: str = "table",
        size: tuple[float, float] = (1.0, 0.6),
    ):
        """Register the work surface: preferred table name, then any TABLE, then COUNTER."""
        preferred_name = preferred_name or self.table_preferred_name
        table = None
        if preferred_name:
            table = self._register_fixture_by_name(
                ref_name,
                preferred_name,
                size=size,
                fallback_type=None,
            )
        if table is None:
            table = self.register_fixture_ref(ref_name, dict(id=FixtureType.TABLE, size=size))
        if table is None:
            table = self.register_fixture_ref(ref_name, dict(id=FixtureType.COUNTER, size=size))
        self.table = table
        if table is not None:
            self.init_robot_base_ref = table
        return table

    @staticmethod
    def _spawn_names_from_items(items: list[tuple[str, str]]) -> list[str]:
        """Derive unique spawn names from ``(obj_groups, asset_name)`` item tuples."""
        counts: dict[str, int] = {}
        names: list[str] = []
        for obj_group, *_ in items:
            sub_idx = counts.get(obj_group, 0)
            names.append(f"{obj_group}_{sub_idx}")
            counts[obj_group] = sub_idx + 1
        return names

    def _object_cfg(
        self,
        name: str,
        obj_groups: str,
        asset_name: str,
        fixture,
        pos: tuple[float, float],
        *,
        size: tuple[float, float] = (0.25, 0.25),
        init_robot_here: bool = False,
        graspable: bool = True,
        object_scale: float | tuple[float, float, float] | None = None,
        rotation: float | tuple[float, float] | list | None = None,
        **extra,
    ) -> dict:
        """Build one spawned object entry for ``_get_obj_cfgs``."""
        placement = dict(
            fixture=fixture,
            size=size,
            pos=pos,
            margin=0.02,
        )
        if rotation is not None:
            placement["rotation"] = rotation
        cfg = dict(
            name=name,
            obj_groups=obj_groups,
            graspable=graspable,
            washable=True,
            asset_name=asset_name,
            init_robot_here=init_robot_here,
            placement=placement,
        )
        if object_scale is not None:
            cfg["object_scale"] = object_scale
        cfg.update(extra)
        return cfg

    def _receptacle_cfg(
        self,
        name: str,
        obj_groups: str,
        asset_name: str,
        fixture,
        pos: tuple[float, float],
        *,
        size: tuple[float, float] = (0.25, 0.25),
        object_scale: float | tuple[float, float, float] | None = None,
        rotation: float | tuple[float, float] | list | None = None,
        **extra,
    ) -> dict:
        """Spawn a non-graspable receptacle object (basket, rack, container base)."""
        return self._object_cfg(
            name,
            obj_groups,
            asset_name,
            fixture,
            pos,
            size=size,
            graspable=False,
            object_scale=object_scale,
            rotation=rotation,
            **extra,
        )

    def _all_grippers_far(self, env, obj_names: list[str]) -> torch.Tensor:
        ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for obj_name in obj_names:
            ok = ok & OU.gripper_obj_far(env, obj_name)
        return ok

    def _all_in_receptacle(self, env, obj_names: list[str], receptacle_name: str) -> torch.Tensor:
        ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for obj_name in obj_names:
            ok = ok & OU.check_obj_in_receptacle_no_contact(env, obj_name, receptacle_name)
        return ok

    def _all_stable(self, env, obj_names: list[str]) -> torch.Tensor:
        ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for obj_name in obj_names:
            ok = ok & OU.check_object_stable(env, obj_name)
        return ok

    def _check_placed_in_receptacle(
        self,
        env,
        obj_names: list[str],
        receptacle_name: str,
        *,
        require_stable: bool = True,
    ) -> torch.Tensor:
        """Libero-style place success: in receptacle, gripper away, optionally stable."""
        placed = self._all_in_receptacle(env, obj_names, receptacle_name)
        released = self._all_grippers_far(env, obj_names)
        if not require_stable:
            return placed & released
        return placed & released & self._all_stable(env, obj_names)
