"""PhysX simulation capability implementation."""

from typing import Any


class PhysXSimulationCompiler:
    def build_sim_config(self, spec: Any) -> Any:
        spec.sim.physics = None
        return spec

    def configure_task_env(self, env_cfg: Any) -> Any:
        env_cfg.rerender_on_reset = True
        return env_cfg

    def configure_before_env_make(self, owner: Any, env_cfg: Any) -> Any:
        del owner
        return env_cfg

    def install_compat_patches(self) -> None:
        return None

    def set_scene_params(self, sim_cfg: Any, **kwargs: float | int) -> None:
        if hasattr(sim_cfg, "physx"):
            target = sim_cfg.physx
        else:
            from isaaclab_physx.physics import PhysxCfg

            if sim_cfg.physics is None:
                sim_cfg.physics = PhysxCfg()
            target = sim_cfg.physics
        for name, value in kwargs.items():
            if hasattr(target, name):
                setattr(target, name, value)

    def configure_deformable_solver(
        self, env_cfg: Any, *, enabled: bool, solver_cfg: dict[str, Any] | None
    ) -> None:
        del env_cfg, enabled, solver_cfg

    def kit_launch_overrides(self, headless: bool) -> tuple[str, ...]:
        if headless:
            return ()
        return (
            "--/physics/updateToUsd=true",
            "--/physics/fabricUpdateTransformations=true",
            "--/physics/fabricUpdateVelocities=true",
        )
