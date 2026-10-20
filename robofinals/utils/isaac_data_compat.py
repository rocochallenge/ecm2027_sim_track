# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2-0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Isaac Sim 6 / Isaac Lab 3: sim buffers may be ``torch.Tensor`` or Warp ``wp.array``.

Use :func:`as_torch` and frame helpers before tensor math. Kept small to avoid importing ``object_utils``.

Isaac Lab 3+ stores articulation / rigid-body **runtime** quaternions as **(x, y, z, w)** to match
PhysX / Warp. Isaac Lab 2.x used **(w, x, y, z)** in those buffers. Set ``LW_SIM_QUAT_LAYOUT=xyzw``
or ``LW_SIM_QUAT_LAYOUT=wxyz`` to override auto-detection from the installed ``isaaclab`` version.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

try:
    import warp as wp
except Exception:  # pragma: no cover
    wp = None  # type: ignore[assignment]


def as_torch(x: Any) -> torch.Tensor:
    """View sim buffers as ``torch.Tensor`` (no-op if already a tensor)."""
    if isinstance(x, torch.Tensor):
        return x
    torch_view = getattr(x, "torch", None)
    if torch_view is not None:
        return torch_view
    if wp is None:
        raise RuntimeError("warp is required to convert non-tensor sim buffers to torch")
    try:
        return wp.to_torch(x)  # type: ignore[no-any-return]
    except AttributeError as exc:
        if "is_cpu" not in str(exc):
            raise
        device = getattr(x, "device", None)
        if isinstance(device, torch.device):
            if device.type == "cpu":
                return torch.as_tensor(np.asarray(x))
            return torch.as_tensor(x, device=device)
        raise


class SimBufferView:
    """Expose Isaac Lab sim buffers with a ``.torch`` view over an underlying tensor."""

    __slots__ = ("_tensor",)

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    @property
    def torch(self) -> torch.Tensor:
        return self._tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(t, (torch.Tensor, cls)) for t in types):
            return NotImplemented

        kwargs = kwargs or {}

        def _unwrap(value: Any) -> Any:
            if isinstance(value, cls):
                return value._tensor
            return value

        return func(
            *(_unwrap(arg) for arg in args),
            **{key: _unwrap(val) for key, val in kwargs.items()},
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tensor, name)

    def __getitem__(self, key: Any) -> Any:
        return self._tensor[key]

    def __repr__(self) -> str:
        return f"SimBufferView({self._tensor!r})"


def wrap_sim_buffer(tensor: torch.Tensor) -> SimBufferView:
    """Wrap a tensor for Isaac Lab APIs that read ``buffer.torch``."""
    return SimBufferView(tensor)


def physx_index_tensor(indices: Any, device: torch.device) -> torch.Tensor | None:
    """``env_ids`` / ``joint_ids`` for Articulation ``write_*_to_sim`` (Isaac Lab 3+ Warp kernels expect int32)."""
    if indices is None:
        return None
    return torch.as_tensor(indices, device=device, dtype=torch.int32).reshape(-1)


def isaaclab_asset_quat_buffers_xyzw() -> bool:
    """True if Isaac Lab stores ``ArticulationData`` / rigid-body world quats as (x,y,z,w)."""
    override = os.environ.get("LW_SIM_QUAT_LAYOUT", "").strip().lower()
    if override in ("xyzw", "wxyz"):
        return override == "xyzw"
    try:
        from importlib.metadata import version

        major = int(version("isaaclab").split(".")[0])
        return major >= 3
    except Exception:
        return False


def sim_quat_raw_to_xyzw_np(q: np.ndarray) -> np.ndarray:
    """World-frame quaternion from sim ``*.data.*quat*`` buffers → scipy ``(x,y,z,w)`` numpy."""
    v = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    if isaaclab_asset_quat_buffers_xyzw():
        out = v
    else:
        out = v[:, [1, 2, 3, 0]]
    if q.ndim == 1 and q.shape[0] == 4:
        return out[0]
    return out.reshape(q.shape)


def sim_quat_raw_to_xyzw_torch(q: torch.Tensor) -> torch.Tensor:
    """World-frame quaternion from sim buffers → ``(x,y,z,w)`` tensor (same leading dims as ``q``)."""
    if isaaclab_asset_quat_buffers_xyzw():
        return q
    return q[..., [1, 2, 3, 0]]


def isaaclab_chain_quat_to_xyzw_torch(q: torch.Tensor) -> torch.Tensor:
    """Quaternion from ``isaaclab.utils.math`` frame ops (same layout as sim buffers for that IL version)."""
    return sim_quat_raw_to_xyzw_torch(q)


def first_target_pos_w(data: Any) -> torch.Tensor:
    """``FrameTransformerData``: first target point (N, 3) in world frame."""
    pos = as_torch(data.target_pos_w)
    if pos.ndim >= 3:
        return pos[:, 0, :]
    return pos


def first_target_quat_w(data: Any) -> torch.Tensor:
    """``FrameTransformerData``: first target orientation (N, 4) in world frame."""
    quat = as_torch(data.target_quat_w)
    if quat.ndim >= 3:
        return quat[:, 0, :]
    return quat


def fingertip_targets_pos_w(
    data: Any, num_envs: int, device: torch.device, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Fingertip rows after index 0: ``(num_envs, n-1, 3)`` or empty second dim if unavailable."""
    p = as_torch(data.target_pos_w)
    if p.ndim < 3 or p.shape[1] <= 1:
        if dtype is None:
            dtype = torch.float32
        return torch.zeros((num_envs, 0, 3), device=device, dtype=dtype)
    return p[:, 1:, :]


def target_pos_w_slice(data: Any, start: int, end: int | None = None) -> torch.Tensor:
    """Slice ``target_pos_w`` along the target index dimension after :func:`as_torch`."""
    p = as_torch(data.target_pos_w)
    if p.ndim < 3:
        if start == 0 and end in (None, 1):
            return p
        return p
    if end is None:
        return p[:, start, :]
    return p[:, start:end, :]


def body_com_pose_env0(articulation_data: Any) -> torch.Tensor:
    """``body_com_pose_w`` for env 0, as ``torch`` (may be 2D for single-body / warp layouts)."""
    pose_wp = articulation_data.body_com_pose_w
    t = as_torch(pose_wp)
    if t.ndim == 2:
        return t
    return t[0]


def resolve_frame_view_prim_paths(env: Any, view: Any) -> list[str] | None:
    """Resolve prim paths from an Isaac Lab ``FrameView`` (PhysX or Newton backend).

    PhysX ``UsdFrameView`` / ``FabricFrameView`` expose a public ``prim_paths`` property
    with resolved stage paths. Newton ``NewtonSiteFrameView`` only stores path expressions
    in ``_prim_paths`` and must be expanded against the live USD stage.
    """
    prim_paths = getattr(view, "prim_paths", None)
    if prim_paths:
        return list(prim_paths)

    path_exprs = getattr(view, "_prim_paths", None)
    if not path_exprs:
        return None

    import isaaclab.sim as sim_utils

    stage = getattr(getattr(env, "sim", None), "stage", None)
    resolved: list[str] = []
    for expr in path_exprs:
        resolved.extend(sim_utils.find_matching_prim_paths(expr, stage))
    return resolved or None


def resolve_fixture_prim_paths(env: Any, fixture_name: str, fixture_prim: Any = None) -> list[str] | None:
    """Resolve per-env USD prim paths for a fixture (PhysX or Newton).

    Newton articulations may not expose ``prim_paths`` on the root view; fall back to
    spawn ``prim_path`` expressions or the fixture's placement prim from scene init.
    """
    scene = env.scene
    if fixture_name in scene.extras.keys():
        paths = resolve_frame_view_prim_paths(env, scene.extras[fixture_name])
        if paths:
            return paths

    if fixture_name in scene.articulations.keys():
        art = scene.articulations[fixture_name]
        root_view = (
            getattr(art, "root_physx_view", None)
            or getattr(art, "_root_physx_view", None)
            or getattr(art, "root_view", None)
        )
        if root_view is not None:
            paths = resolve_frame_view_prim_paths(env, root_view)
            if paths:
                return paths

        cfg = getattr(art, "cfg", None)
        spawn_path = getattr(getattr(cfg, "spawn", None), "prim_path", None)
        if spawn_path:
            import isaaclab.sim as sim_utils

            stage = getattr(getattr(env, "sim", None), "stage", None)
            if stage is not None:
                paths = list(sim_utils.find_matching_prim_paths(spawn_path, stage))
                if paths:
                    return paths

    if fixture_prim is not None and fixture_prim.IsValid():
        stage = getattr(getattr(env, "sim", None), "stage", None)
        if stage is not None:
            path = str(fixture_prim.GetPath())
            live_prim = stage.GetObjectAtPath(path)
            if live_prim and live_prim.IsValid():
                return [path]
    return None
