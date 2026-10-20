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

"""Helpers to bake Isaac Lab OmniPVD (OVD) captures to USD on shutdown."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Optional


def attach_bake_on_env_close(
    env: Any,
    *,
    enabled: bool = True,
    delivery_dir: str | None = None,
) -> None:
    """Bake OVD to USD when ``env.close()`` runs, before PhysX tears down the recorder."""
    if not enabled or env is None or getattr(env, "_anim_recording_bake_on_close", False):
        return

    original_close: Callable[..., Any] = env.close

    def close_with_bake(*args, **kwargs):
        finish_ovd_animation_recording(
            env,
            enabled=True,
            delivery_dir=delivery_dir or getattr(env, "_anim_recording_delivery_dir", None),
        )
        return original_close(*args, **kwargs)

    env.close = close_with_bake
    env._anim_recording_bake_on_close = True
    if delivery_dir is not None:
        env._anim_recording_delivery_dir = os.path.abspath(os.path.expanduser(delivery_dir))


def finish_ovd_animation_recording(
    env: Any = None,
    *,
    enabled: bool = True,
    delivery_dir: str | None = None,
) -> Optional[str]:
    """Bake an in-progress OVD capture to ``baked_animation_recording.usda``.

    Isaac Lab records physics to ``*.ovd`` during simulation; USD is produced only
    when ``AnimationRecorder._finish()`` runs (see Isaac Lab animation recording docs).
    """
    if not enabled:
        return None

    existing = getattr(env, "_anim_recording_baked_delivery_entry", None) if env is not None else None
    if existing and os.path.isfile(existing):
        print(f"[anim_recording] Already baked; playback entry: {existing}")
        return existing

    try:
        from isaaclab_physx.physics.physx_manager import PhysxManager
    except ImportError:
        print("[anim_recording] isaaclab_physx not available; skip OVD bake.")
        return None

    recorder = PhysxManager._anim_recorder
    if recorder is None or not recorder.enabled:
        baked_path = getattr(env, "_anim_recording_baked_path", None) if env is not None else None
        if baked_path and os.path.isfile(baked_path):
            print(f"[anim_recording] Recorder already finished; baked USD: {baked_path}")
            return baked_path
        print("[anim_recording] No active OVD recorder; skip bake.")
        return None

    start_time = float(getattr(recorder, "_start_time", 0.0) or 0.0)
    stop_time = _resolve_stop_time(env, recorder, start_time)
    recorder._start_time = start_time
    recorder._stop_time = stop_time

    raw_output_dir = os.path.abspath(recorder._output_dir)
    print(
        f"[anim_recording] Baking OVD -> USD [{start_time:.3f}, {stop_time:.3f}]s "
        f"(output: {raw_output_dir})..."
    )
    print("[anim_recording] This may take a few minutes.")

    t0 = time.time()
    try:
        recorder._finish()
    except Exception as exc:
        print(f"[anim_recording] Bake failed: {exc}")
        raise
    finally:
        recorder._enabled = False

    baked_path = os.path.join(raw_output_dir, "baked_animation_recording.usda")
    if not os.path.isfile(baked_path):
        print(f"[anim_recording] Bake finished but {baked_path} was not found.")
        return None

    print(f"[anim_recording] Saved {baked_path} ({time.time() - t0:.1f}s)")
    if env is not None:
        env._anim_recording_baked_path = baked_path

    delivery_entry = postprocess_baked_recording(
        env,
        raw_output_dir,
        delivery_dir=delivery_dir or getattr(env, "_anim_recording_delivery_dir", None),
    )
    if env is not None and delivery_entry is not None:
        env._anim_recording_baked_delivery_entry = delivery_entry
    return delivery_entry or baked_path


def postprocess_baked_recording(
    env: Any,
    anim_output_dir: str,
    *,
    delivery_dir: str | None = None,
) -> Optional[str]:
    """Repair scene refs, build playback entry, and package a relocatable delivery folder."""
    anim_output_dir = os.path.abspath(anim_output_dir)
    repair_scene_references(anim_output_dir, env)

    from robofinals.applications.teleop.anim_recording.usd_playback import UsdPlaybackBuilder

    playback_entry = UsdPlaybackBuilder(_stdout_logger).build(anim_output_dir)
    if playback_entry is None or not os.path.isfile(playback_entry):
        print(
            "[anim_recording] Playback entry was not created; open the raw bake folder instead:\n"
            f"  {anim_output_dir}/baked_animation_recording.usda"
        )
        return None

    print(f"[anim_recording] Playback entry: {playback_entry}")

    from robofinals.applications.teleop.anim_recording.package_bake_delivery import package_bake_delivery

    if delivery_dir is None:
        delivery_dir = anim_output_dir.rstrip(os.sep) + "_delivery"
    delivery_dir = os.path.abspath(os.path.expanduser(delivery_dir))
    package_bake_delivery(anim_output_dir, delivery_dir, logger=_StdoutLogger())
    delivery_entry = os.path.join(delivery_dir, "teleop_baked_with_deformable.usda")

    print("[anim_recording] ===== Animation recording complete =====")
    print(f"[anim_recording] Raw bake folder : {anim_output_dir}")
    print(f"[anim_recording] Delivery folder   : {delivery_dir}")
    print(f"[anim_recording] Open this file    : {delivery_entry}")
    print("[anim_recording] ========================================")
    return delivery_entry


def repair_scene_references(anim_output_dir: str, env: Any) -> str:
    """Point ``layer_root.usda`` Scene reference at the absolute layout USD path."""
    layer_root_path = os.path.join(anim_output_dir, "layer_root.usda")
    if not os.path.exists(layer_root_path):
        return anim_output_dir

    layout_path = _resolve_layout_path(env)
    if layout_path is None:
        return anim_output_dir

    with open(layer_root_path, "r", encoding="utf-8") as f:
        content = f.read()

    fixed_layout_path = layout_path.replace("\\", "/")
    updated_content, replacements = re.subn(
        r'(def Xform "Scene" \(\s*prepend references = @)([^@]+)(@)',
        rf"\1{fixed_layout_path}\3",
        content,
        count=1,
        flags=re.S,
    )
    if replacements == 0 or updated_content == content:
        return anim_output_dir

    with open(layer_root_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(f"[anim_recording] Repaired Scene reference -> {fixed_layout_path}")
    return anim_output_dir


def _resolve_layout_path(env: Any) -> Optional[str]:
    cfg = getattr(env, "cfg", None)
    if cfg is not None:
        path = getattr(cfg, "usd_path", None)
        if path:
            path = os.path.abspath(os.path.expanduser(str(path)))
            if os.path.isfile(path):
                return path

    return None


def _resolve_stop_time(env: Any, recorder: Any, start_time: float) -> float:
    """Prefer simulation elapsed time; fall back to wall-clock recording duration."""
    if env is not None:
        step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        steps = int(getattr(env, "common_step_counter", 0) or 0)
        if step_dt > 0.0 and steps > 0:
            return max(start_time + 1e-3, steps * step_dt)

    started_at = getattr(recorder, "_started_at", None)
    if started_at is not None:
        return max(start_time + 1e-3, time.time() - started_at)

    configured = float(getattr(recorder, "_stop_time", 0.0) or 0.0)
    if configured > start_time:
        return configured
    return start_time + 1e-3


class _StdoutLogger:
    def info(self, msg: str) -> None:
        print(msg, flush=True)


def _stdout_logger():
    return _StdoutLogger()
