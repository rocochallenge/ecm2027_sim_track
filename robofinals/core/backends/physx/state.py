"""PhysX state synchronization capability."""

from typing import Any


class PhysXStateSynchronizer:
    def reset(self, env: Any, env_ids: Any = None) -> None:
        del env_ids
        env.sim.reset(soft=False)

    def reset_sim(self, env: Any) -> None:
        env.sim.reset(soft=False)

    def sync_after_state_reset(self, env: Any) -> None:
        env.scene.update(dt=0.0)

    def write_robot_startup_state(self, robot: Any, position: Any, rotation_xyzw: Any, env_ids: Any) -> None:
        del robot, position, rotation_xyzw, env_ids
