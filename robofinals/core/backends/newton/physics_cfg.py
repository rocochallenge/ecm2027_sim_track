# Copyright 2025-2026 Lightwheel Team
#
# SPDX-License-Identifier: Apache-2.0

"""Robofinals-specific composition helpers for Isaac Lab Newton physics."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_contrib.deformable import NewtonModelCfg
from isaaclab_newton.physics import NewtonCfg, NewtonSolverCfg


@configclass
class RobofinalsNewtonCfg(NewtonCfg):
    """Newton configuration extended with deformable model contact parameters."""

    model_cfg: NewtonModelCfg | None = None
    """Global Newton deformable and rigid-shape contact parameters."""


def set_newton_solver_cfg(physics_cfg: NewtonCfg, solver_cfg: NewtonSolverCfg) -> None:
    """Replace the active Newton solver and keep manager dispatch synchronized.

    ``NewtonCfg.__post_init__`` derives ``class_type`` from ``solver_cfg`` only
    once. Robofinals composes robot and deformable physics after construction,
    so every solver replacement must update both fields atomically.
    """
    physics_cfg.solver_cfg = solver_cfg
    physics_cfg.class_type = solver_cfg.class_type
