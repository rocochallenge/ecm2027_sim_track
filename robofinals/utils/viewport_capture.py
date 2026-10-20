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

"""Capture Kit viewport frames to CPU numpy (Newton-safe LeRobot images).

Newton + RTX ``Camera`` sensor GPU readback can poison the shared Warp cuda:0
context.  Viewport capture delivers LDR pixels on CPU via ``capture_viewport_to_buffer``,
bypassing ``observation_manager`` and ``sensor.data.output``.
"""

from __future__ import annotations

import ctypes
from typing import Any, Sequence

import numpy as np


def _flat_u8_from_capture_buffer(buffer: Any, buffer_size: int) -> np.ndarray:
    """Decode Kit viewport capture buffer (PyCapsule / bytes / ndarray) to flat uint8."""
    if buffer is None or buffer_size <= 0:
        raise ValueError(f"invalid viewport buffer size={buffer_size}")

    if isinstance(buffer, (bytes, bytearray, memoryview)):
        return np.frombuffer(buffer, dtype=np.uint8, count=buffer_size).copy()

    if isinstance(buffer, np.ndarray):
        return np.asarray(buffer, dtype=np.uint8).reshape(-1)[:buffer_size].copy()

    if hasattr(buffer, "__array_interface__"):
        return np.asarray(buffer, dtype=np.uint8).reshape(-1)[:buffer_size].copy()

    # Kit ``capture_viewport_to_buffer`` delivers a PyCapsule pointing at CPU pixels.
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
    if not ptr:
        raise ValueError("PyCapsule_GetPointer returned null for viewport capture buffer")
    c_arr = (ctypes.c_ubyte * int(buffer_size)).from_address(ptr)
    return np.frombuffer(c_arr, dtype=np.uint8).copy()


def _buffer_to_rgb_hwc(
    buffer: Any,
    buffer_size: int,
    width: int,
    height: int,
) -> np.ndarray:
    if buffer is None or buffer_size <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid viewport capture buffer size={buffer_size} {width}x{height}")
    flat = _flat_u8_from_capture_buffer(buffer, int(buffer_size))
    n_pix = int(width) * int(height)
    if flat.size >= n_pix * 4:
        rgba = flat[: n_pix * 4].reshape(int(height), int(width), 4)
        return np.ascontiguousarray(rgba[:, :, :3])
    if flat.size >= n_pix * 3:
        return np.ascontiguousarray(flat[: n_pix * 3].reshape(int(height), int(width), 3))
    raise ValueError(f"unexpected viewport buffer layout: size={flat.size} for {width}x{height}")


def _viewport_capture_max_updates(*, after_sim_render: bool = False) -> int:
    import os

    if after_sim_render:
        try:
            return max(
                1,
                int(os.environ.get("LW_BENCHHUB_VIEWPORT_CAPTURE_MAX_UPDATES_AFTER_RENDER", "2")),
            )
        except ValueError:
            return 2
    try:
        return max(8, int(os.environ.get("LW_BENCHHUB_VIEWPORT_CAPTURE_MAX_UPDATES", "32")))
    except ValueError:
        return 32


def capture_viewport_rgb_sync(
    viewport_api: Any,
    *,
    max_app_updates: int | None = None,
) -> np.ndarray:
    """Capture one LDR RGB frame from a Kit ``ViewportAPI`` (blocking, CPU output)."""
    import omni.kit.app
    from omni.kit.viewport.utility import capture_viewport_to_buffer

    holder: dict[str, Any] = {}

    def _on_capture(buffer, buffer_size, width, height, _img_format):
        try:
            holder["rgb"] = _buffer_to_rgb_hwc(buffer, int(buffer_size), int(width), int(height))
        except Exception as exc:
            holder["error"] = exc

    capture_viewport_to_buffer(viewport_api, _on_capture)
    app = omni.kit.app.get_app()
    updates = (
        int(max_app_updates)
        if max_app_updates is not None
        else _viewport_capture_max_updates(after_sim_render=False)
    )
    for _ in range(updates):
        app.update()
        if "rgb" in holder or "error" in holder:
            break
    if "error" in holder:
        raise holder["error"]
    if "rgb" not in holder:
        raise TimeoutError("viewport capture timed out (Kit did not deliver CPU buffer)")
    return holder["rgb"]


def _resolve_viewport_api(viewport_obj: Any) -> Any:
    if viewport_obj is None:
        return None
    api = getattr(viewport_obj, "viewport_api", None)
    return api if api is not None else viewport_obj


# Wrist cameras that share USD prims with Kit docked viewports (skip RTX TiledCamera on these).
TELEOP_VIEWPORT_WRIST_CAMERAS = frozenset({"left_hand_camera", "right_hand_camera"})

# policy obs key -> docked viewport dict key (None => main Kit viewport)
_POLICY_VIEWPORT_KEYS: dict[str, str | None] = {
    "first_person_camera_rgb": None,
    "left_hand_camera_rgb": "left_hand",
    "right_hand_camera_rgb": "right_hand",
}

# Viewport texture sizes (W, H). Must match ``ui_utils.setup_cameras`` / dock viewports.
FIRST_PERSON_VIEWPORT_SIZE = (1280, 720)
WRIST_VIEWPORT_SIZE = (640, 360)

# LeRobot capture target (W, H) -> numpy HWC (H, W, 3).
_POLICY_CAPTURE_TEXTURE_SIZE: dict[str, tuple[int, int]] = {
    "first_person_camera_rgb": FIRST_PERSON_VIEWPORT_SIZE,
    "left_hand_camera_rgb": WRIST_VIEWPORT_SIZE,
    "right_hand_camera_rgb": WRIST_VIEWPORT_SIZE,
}


def canonical_camera_hwc_shape(policy_key: str) -> tuple[int, int, int]:
    """Stable LeRobot image shape for a policy camera key."""
    if policy_key not in _POLICY_CAPTURE_TEXTURE_SIZE:
        raise KeyError(f"unknown policy camera key for canonical shape: {policy_key!r}")
    width, height = _POLICY_CAPTURE_TEXTURE_SIZE[policy_key]
    return (int(height), int(width), 3)


def canonical_camera_shapes(camera_policy_keys: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    return {key: canonical_camera_hwc_shape(key) for key in camera_policy_keys}


def _resize_rgb_hwc(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    th, tw = int(target_hw[0]), int(target_hw[1])
    h, w = int(img.shape[0]), int(img.shape[1])
    if (h, w) == (th, tw):
        return np.ascontiguousarray(img)
    try:
        import cv2

        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        pil = Image.fromarray(img)
        pil = pil.resize((tw, th), Image.Resampling.BILINEAR)
        return np.asarray(pil)


def _main_viewport_api() -> Any:
    """Main window viewport for first_person (never use get_active_viewport)."""
    import omni.kit.viewport.utility as vp_utils

    return vp_utils.get_viewport_from_window_name()


def _apply_capture_texture_size(viewport_api: Any, policy_key: str) -> None:
    texture_size = _POLICY_CAPTURE_TEXTURE_SIZE.get(policy_key)
    if texture_size is None:
        return
    setter = getattr(viewport_api, "set_texture_resolution", None)
    if callable(setter):
        setter(texture_size)


class AsyncPolicyViewportCapture:
    """Single-flight, non-blocking Kit viewport capture pipeline."""

    def __init__(self) -> None:
        self._pending: dict[str, dict[str, Any]] | None = None

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def discard_pending(self) -> None:
        # Callbacks retain request-local state; late delivery is safely ignored.
        self._pending = None

    def submit(
        self,
        viewports: dict[str, Any] | None,
        camera_policy_keys: Sequence[str],
    ) -> None:
        """Submit all camera readbacks and return without pumping the Kit app loop."""
        if self._pending is not None:
            raise RuntimeError("viewport capture request is already pending")

        from omni.kit.viewport.utility import capture_viewport_to_buffer

        docked = viewports or {}
        viewport_apis: dict[str, Any] = {}
        for policy_key in camera_policy_keys:
            dock_key = _POLICY_VIEWPORT_KEYS.get(policy_key, policy_key.replace("_rgb", ""))
            vp_api = (
                _main_viewport_api()
                if dock_key is None
                else _resolve_viewport_api(docked.get(dock_key))
            )
            if vp_api is None:
                raise KeyError(
                    f"viewport for policy key {policy_key!r} not found "
                    f"(dock_key={dock_key!r}, available={list(docked.keys())})"
                )
            _apply_capture_texture_size(vp_api, policy_key)
            viewport_apis[policy_key] = vp_api

        request = {key: {} for key in camera_policy_keys}
        self._pending = request

        def _make_cb(key: str):
            def _on_capture(buffer, buffer_size, width, height, _img_format):
                try:
                    request[key]["rgb"] = _buffer_to_rgb_hwc(
                        buffer, int(buffer_size), int(width), int(height)
                    )
                except Exception as exc:
                    request[key]["error"] = exc

            return _on_capture

        try:
            for policy_key, vp_api in viewport_apis.items():
                capture_viewport_to_buffer(vp_api, _make_cb(policy_key))
        except Exception:
            self._pending = None
            raise

    def take_completed(self) -> dict[str, np.ndarray] | None:
        """Return a completed request, or None without blocking."""
        request = self._pending
        if request is None:
            return None
        if not all(("rgb" in state or "error" in state) for state in request.values()):
            return None

        self._pending = None
        images: dict[str, np.ndarray] = {}
        for policy_key, state in request.items():
            if "error" in state:
                raise state["error"]
            target_hw = canonical_camera_hwc_shape(policy_key)[:2]
            images[policy_key] = _resize_rgb_hwc(state["rgb"], target_hw)
        return images

    def flush(self, *, max_app_updates: int | None = None) -> dict[str, np.ndarray] | None:
        """Synchronously finish the last request during recording shutdown."""
        if self._pending is None:
            return None

        import omni.kit.app

        app = omni.kit.app.get_app()
        updates = (
            int(max_app_updates)
            if max_app_updates is not None
            else _viewport_capture_max_updates(after_sim_render=False)
        )
        for _ in range(updates):
            images = self.take_completed()
            if images is not None:
                return images
            app.update()

        images = self.take_completed()
        if images is not None:
            return images
        self.discard_pending()
        raise TimeoutError(
            "viewport capture timed out while flushing the final recording request "
            f"(Kit did not deliver CPU buffers within {updates} updates)"
        )


def capture_policy_cameras_from_viewports(
    env: Any,
    viewports: dict[str, Any] | None,
    camera_policy_keys: Sequence[str],
    *,
    skip_sim_render: bool = False,
) -> dict[str, np.ndarray]:
    """Synchronous compatibility wrapper for non-teleop callers."""
    if not skip_sim_render and hasattr(env, "sim") and env.sim is not None:
        env.sim.render()
    pipeline = AsyncPolicyViewportCapture()
    pipeline.submit(viewports, camera_policy_keys)
    images = pipeline.flush(
        max_app_updates=_viewport_capture_max_updates(after_sim_render=skip_sim_render)
    )
    return images or {}
