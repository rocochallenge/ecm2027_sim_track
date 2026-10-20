"""Newton simulation capability implementation."""

from typing import Any

from robofinals.core.backends.newton.manager_patches import (
    configure_newton_manager_env_cfg,
    make_default_newton_physics_cfg,
)


class NewtonSimulationCompiler:
    def build_sim_config(self, spec: Any) -> Any:
        spec.sim.physics = make_default_newton_physics_cfg()
        return spec

    def configure_task_env(self, env_cfg: Any) -> Any:
        env_cfg.rerender_on_reset = False
        return env_cfg

    def configure_before_env_make(self, owner: Any, env_cfg: Any) -> Any:
        env_cfg.sim.render_interval = env_cfg.decimation
        configure_newton_manager_env_cfg(env_cfg)
        return env_cfg

    def install_compat_patches(self) -> None:
        from .rendering import patch_newton_camera_obs_dim_probe

        patch_newton_camera_obs_dim_probe()

    def set_scene_params(self, sim_cfg: Any, **kwargs: float | int) -> None:
        del sim_cfg, kwargs

    def configure_deformable_solver(
        self, env_cfg: Any, *, enabled: bool, solver_cfg: dict[str, Any] | None
    ) -> None:
        if solver_cfg is None:
            return
        from .deformable import configure_newton_deformable_solver

        configure_newton_deformable_solver(env_cfg, enabled=enabled, **solver_cfg)

    def kit_launch_overrides(self, headless: bool) -> tuple[str, ...]:
        if headless:
            return ()
        return (
            "--/physics/updateToUsd=false",
            "--/physics/fabricUpdateTransformations=true",
            "--/physics/fabricUseGPUInterop=false",
        )
