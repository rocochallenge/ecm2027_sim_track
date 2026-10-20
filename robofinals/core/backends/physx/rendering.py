"""PhysX renderer capability implementation."""

from typing import Any

from robofinals.core.backends.contract import RenderCapabilities, RendererProvider


class PhysXRenderer:
    capabilities = RenderCapabilities()

    def render(self, env: Any) -> Any:
        render = getattr(env, "render", None)
        if callable(render):
            return render()
        sim_render = getattr(getattr(env, "sim", None), "render", None)
        return sim_render() if callable(sim_render) else None

    def render_idle(self, env: Any) -> Any:
        # ManagerBasedEnv.render() intentionally skips sim.render() when RTX
        # sensors are present because a normal env.step() already rendered.
        # During teleop's pre-start idle loop there is no env.step(), so use
        # SimulationContext directly to keep Kit's viewport and keyboard event
        # loop alive.
        sim_render = getattr(getattr(env, "sim", None), "render", None)
        if callable(sim_render):
            return sim_render()
        return self.render(env)

    def should_skip_step_render(self, env: Any) -> bool:
        del env
        return False

    def camera_name_candidates(self, side: str) -> tuple[str, ...]:
        return (f"{side}_hand_camera",)

__all__ = ["RendererProvider", "PhysXRenderer"]
