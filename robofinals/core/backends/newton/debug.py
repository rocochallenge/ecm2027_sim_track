# Copyright 2025 Lightwheel Team
#
# SPDX-License-Identifier: Apache-2.0

"""Optional bisect flags for Newton teleop debugging.

Production Newton teleop applies CUDA-safe defaults in :class:`~robofinals.applications.teleop.legacy_session.TeleopSession`
without any environment variables.  Set ``TELEOP_NEWTON_DEBUG`` only when bisecting regressions.

Comma-separated flags (only effective when ``physics_backend=newton``):

* ``reset_physx`` — call ``reset_physx()`` before ``env.reset()`` (default: skip on Newton)
* ``no_immediate_step`` — skip the post-reset zero-action ``env.step()``
* ``vr_auto_start`` — VR ``teleoperation_active=True`` immediately (default: wait for B)
* ``skip_step_render`` — no ``sim.render()`` inside monkey_patch decimation loop
* ``skip_ui_setup`` — skip task-description UI / viewport tweaks before main loop
* ``spawn_vis_helper`` — spawn robot vis helpers on record start (default: skip on Newton)
* ``vr_sideband`` — enable VR rollback check and ``send_robot_state`` (default: skip on Newton)
* ``verbose_step`` — log each ``env.step()`` begin/end

Example (launch.json env)::

    "TELEOP_NEWTON_DEBUG": "verbose_step"
    "TELEOP_NEWTON_DEBUG": "reset_physx,no_immediate_step"
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class NewtonTeleopDebug:
    """Per-flag Newton teleop bisect overrides."""

    reset_physx: bool = False
    no_immediate_step: bool = False
    vr_auto_start: bool = False
    skip_step_render: bool = False
    skip_ui_setup: bool = False
    spawn_vis_helper: bool = False
    vr_sideband: bool = False
    verbose_step: bool = False

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.reset_physx,
                self.no_immediate_step,
                self.vr_auto_start,
                self.skip_step_render,
                self.skip_ui_setup,
                self.spawn_vis_helper,
                self.vr_sideband,
                self.verbose_step,
            )
        )

    @classmethod
    def from_env(cls) -> NewtonTeleopDebug:
        raw = os.environ.get("TELEOP_NEWTON_DEBUG", "").strip()
        if not raw or raw.lower() in {"0", "false", "no", "off"}:
            return cls()

        tokens = {part.strip().lower() for part in raw.replace(";", ",").split(",") if part.strip()}
        return cls(
            reset_physx="reset_physx" in tokens,
            no_immediate_step="no_immediate_step" in tokens,
            vr_auto_start="vr_auto_start" in tokens,
            skip_step_render="skip_step_render" in tokens,
            skip_ui_setup="skip_ui_setup" in tokens,
            spawn_vis_helper="spawn_vis_helper" in tokens,
            vr_sideband="vr_sideband" in tokens,
            verbose_step="verbose_step" in tokens,
        )

    def active_flag_names(self) -> list[str]:
        names: list[str] = []
        if self.reset_physx:
            names.append("reset_physx")
        if self.no_immediate_step:
            names.append("no_immediate_step")
        if self.vr_auto_start:
            names.append("vr_auto_start")
        if self.skip_step_render:
            names.append("skip_step_render")
        if self.skip_ui_setup:
            names.append("skip_ui_setup")
        if self.spawn_vis_helper:
            names.append("spawn_vis_helper")
        if self.vr_sideband:
            names.append("vr_sideband")
        if self.verbose_step:
            names.append("verbose_step")
        return names

    def print_banner(self) -> None:
        if not self.enabled:
            return
        flags = ", ".join(self.active_flag_names())
        print(f"[newton_debug] TELEOP_NEWTON_DEBUG bisect overrides: {flags}", flush=True)


_DEBUG: NewtonTeleopDebug | None = None


def get_newton_teleop_debug() -> NewtonTeleopDebug:
    """Return cached bisect flags parsed from the environment."""
    global _DEBUG
    if _DEBUG is None:
        _DEBUG = NewtonTeleopDebug.from_env()
    return _DEBUG
