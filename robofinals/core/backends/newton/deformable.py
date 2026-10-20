"""Newton deformable solver composition."""

from typing import Any, Literal

def configure_newton_deformable_solver(
    cfg: Any,
    *,
    enabled: bool,
    execution: Literal["sequential", "coupled"] = "sequential",
    coupling_style: Literal["custom", "proxy", "admm"] = "custom",
    vbd_params: dict[str, Any] | None = None,
    coupler_params: dict[str, Any] | None = None,
) -> None:
    """Compose the configured Newton rigid solver with a parameterized VBD solver.

    This runs after embodiment-specific physics customization so the composed
    solver retains all robot-specific MJWarp settings. Robofinals only selects
    the execution and coupling semantics here; omitted numerical values come
    from the corresponding IsaacLab-Newton configuration defaults.

    Args:
        cfg: Final composed Isaac Lab environment configuration.
        enabled: Whether the scene contains at least one deformable asset.
        execution: ``"sequential"`` for rigid-to-soft one-way execution or
            ``"coupled"`` for two-way rigid-soft coupling.
        coupling_style: Coupler implementation used by coupled execution.
        vbd_params: Optional keyword overrides forwarded to
            :class:`VBDSolverCfg`.
        coupler_params: Optional keyword overrides forwarded to the selected
            proxy or ADMM coupler config. Sequential execution and the custom
            coupled implementation ignore coupler-specific parameters.
    """
    if not enabled:
        return
    if execution not in ("sequential", "coupled"):
        raise ValueError(
            f"Unknown Newton deformable execution {execution!r}; expected 'sequential' or 'coupled'."
        )
    if execution == "coupled" and coupling_style not in ("custom", "proxy", "admm"):
        raise ValueError(
            f"Unknown Newton deformable coupling style {coupling_style!r}; "
            "expected 'custom', 'proxy', or 'admm'."
        )
    physics_cfg = getattr(getattr(cfg, "sim", None), "physics", None)
    if physics_cfg is None:
        raise RuntimeError("A deformable task requires Newton physics, but cfg.sim.physics is not configured")

    from isaaclab_contrib.deformable.newton_manager_cfg import (
        ADMMCouplingCfg,
        CoupledMJWarpVBDSolverCfg,
        ProxyCouplingCfg,
        SequentialMJWarpVBDSolverCfg,
        VBDSolverCfg,
    )
    from isaaclab_newton.physics import MJWarpSolverCfg
    from robofinals.core.backends.newton.physics_cfg import set_newton_solver_cfg

    current_solver_cfg = physics_cfg.solver_cfg
    if isinstance(
        current_solver_cfg,
        (SequentialMJWarpVBDSolverCfg, CoupledMJWarpVBDSolverCfg),
    ):
        # Reconfiguration must not wrap an already-composed solver recursively.
        rigid_solver_cfg = current_solver_cfg.rigid_solver_cfg
    else:
        rigid_solver_cfg = current_solver_cfg
    if not isinstance(rigid_solver_cfg, MJWarpSolverCfg):
        raise TypeError(
            "Newton deformables currently require an MJWarp rigid solver, "
            f"but received {type(rigid_solver_cfg).__name__}"
        )

    soft_solver_kwargs = dict(vbd_params or {})
    # Both sequential execution and the custom coupled path run MJWarp as a
    # separate rigid solver. Native proxy/ADMM managers override this to False
    # when they create their VBD view.
    soft_solver_kwargs["integrate_with_external_rigid_solver"] = True
    soft_solver_cfg = VBDSolverCfg(**soft_solver_kwargs)

    resolved_coupling_style: str | None
    if execution == "sequential":
        solver_cfg = SequentialMJWarpVBDSolverCfg(
            rigid_solver_cfg=rigid_solver_cfg,
            soft_solver_cfg=soft_solver_cfg,
        )
        resolved_coupling_style = None
    else:
        coupled_solver_kwargs: dict[str, Any] = {}
        if coupling_style == "proxy":
            coupled_solver_kwargs["proxy_cfg"] = ProxyCouplingCfg(**dict(coupler_params or {}))
        elif coupling_style == "admm":
            coupled_solver_kwargs["admm_cfg"] = ADMMCouplingCfg(**dict(coupler_params or {}))
        solver_cfg = CoupledMJWarpVBDSolverCfg(
            rigid_solver_cfg=rigid_solver_cfg,
            soft_solver_cfg=soft_solver_cfg,
            coupling_style=coupling_style,
            **coupled_solver_kwargs,
        )
        resolved_coupling_style = coupling_style

    set_newton_solver_cfg(physics_cfg, solver_cfg)
    print(
        "[robofinals] Newton deformables enabled: "
        f"rigid={type(rigid_solver_cfg).__name__}, soft=VBDSolverCfg, "
        f"execution={execution}, coupling_style={resolved_coupling_style or 'none'}",
        flush=True,
    )
