"""Newton state synchronization capability."""

from typing import Any


class NewtonStateSynchronizer:
    def reset(self, env: Any, env_ids: Any = None) -> None:
        del env_ids
        env.sim.reset(soft=False)

    def reset_sim(self, env: Any) -> None:
        env.sim.reset(soft=False)

    def sync_after_state_reset(self, env: Any) -> None:
        from isaaclab_newton.physics import NewtonManager

        NewtonManager._mark_state_dirty()
        NewtonManager.sync_transforms_to_usd()
        NewtonManager.sync_particles_to_usd()
        env.scene.update(dt=0.0)
        print("[NewtonReset] state synchronized to USD/Fabric", flush=True)

    def write_robot_startup_state(self, robot: Any, position: Any, rotation_xyzw: Any, env_ids: Any) -> None:
        setter = getattr(robot, "set_root_startup_state", None)
        if not callable(setter):
            return
        try:
            setter(position=position, rotation_xyzw=rotation_xyzw, env_ids=env_ids)
        except TypeError:
            import torch

            setter(torch.concat([position, rotation_xyzw], dim=-1), env_ids=env_ids)
