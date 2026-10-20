"""Newton renderer capability implementation."""

from typing import Any

import torch

from robofinals.core.backends.contract import RenderCapabilities, RendererProvider


class NewtonRenderer:
    capabilities = RenderCapabilities(allow_reset_render=False, allow_texture_wait=False)

    def render(self, env: Any) -> Any:
        # IsaacLab env.render() intentionally skips sim.render() when RTX
        # sensors exist. Newton still needs an explicit simulation render to
        # publish reset transforms to the Kit/VR viewport.
        sim_render = getattr(getattr(env, "sim", None), "render", None)
        if callable(sim_render):
            return sim_render()
        render = getattr(env, "render", None)
        return render() if callable(render) else None

    def render_idle(self, env: Any) -> Any:
        del env
        return None

    def should_skip_step_render(self, env: Any) -> bool:
        from .debug import get_newton_teleop_debug

        debug = get_newton_teleop_debug()
        pending_capture = bool(getattr(env, "_robofinals_viewport_capture_pending", False))
        if pending_capture and not getattr(env, "_robofinals_capture_render_logged", False):
            print("[NewtonRecorderDebug] pending viewport capture; allowing Newton sim.render to drive Kit callback", flush=True)
            setattr(env, "_robofinals_capture_render_logged", True)
        if not pending_capture:
            setattr(env, "_robofinals_capture_render_logged", False)
        warmup = int(getattr(env, "_robofinals_camera_warmup_steps", 0) or 0) > 0
        if pending_capture:
            # A submitted viewport capture must be serviced by the next render.
            return warmup
        # Deferring camera observation readback must not freeze Kit/VR viewports.
        # Rendering and copying camera tensors back into observations are separate
        # operations; only explicit debug and warmup policies suppress rendering.
        return bool(getattr(debug, "skip_step_render", False) or warmup)

    def camera_name_candidates(self, side: str) -> tuple[str, ...]:
        return (f"newton_{side}_wrist_camera", f"{side}_hand_camera")


def _infer_camera_image_obs_shape(env, params: dict) -> tuple[int, ...]:
    """Infer ``mdp.image`` output shape from camera cfg without RTX readback."""
    sensor_cfg = params.get("sensor_cfg")
    if sensor_cfg is None:
        raise ValueError("camera image observation is missing sensor_cfg")
    sensor_name = getattr(sensor_cfg, "name", None)
    if not sensor_name:
        raise ValueError("camera image observation is missing sensor_cfg.name")
    sensor = env.scene.sensors[sensor_name]
    height = int(sensor.cfg.height)
    width = int(sensor.cfg.width)
    data_type = params.get("data_type", "rgb")
    if data_type == "rgba":
        channels = 4
    elif data_type == "rgb":
        channels = 3
    else:
        channels = 1
    return (int(env.num_envs), height, width, channels)


def patch_newton_camera_obs_dim_probe():
    """Avoid probing camera RGB via GPU readback while ObservationManager initializes.

    ``ObservationManager._prepare_terms`` eagerly calls ``mdp.image()`` during
    ``gym.make()``. On Newton + large scenes that shares cuda:0 with Warp, the
    first RTX/Warp D2H can fail even when teleop without ``record`` works.
    """
    from isaaclab.envs import mdp as env_mdp
    from isaaclab.envs.mdp import observations as mdp_observations
    from isaaclab.managers.observation_manager import ObservationManager

    if getattr(ObservationManager._prepare_terms, "_robofinals_camera_probe_patched", False):
        return

    _orig_prepare_terms = ObservationManager._prepare_terms
    _orig_image = mdp_observations.image

    def _prepare_terms_patched(self):
        self._env._robofinals_infer_camera_obs_dims = True
        try:
            _orig_prepare_terms(self)
        finally:
            self._env._robofinals_infer_camera_obs_dims = False

    # Signature must match ``mdp.image`` exactly — ManagerBase validates params via inspect.
    from isaaclab.managers import SceneEntityCfg

    def image_with_cfg_shape_probe(
        env,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        normalize: bool = True,
    ):
        if getattr(env, "_robofinals_infer_camera_obs_dims", False) or getattr(
            env, "_robofinals_defer_camera_read", False
        ):
            params = {
                "sensor_cfg": sensor_cfg,
                "data_type": data_type,
                "convert_perspective_to_orthogonal": convert_perspective_to_orthogonal,
                "normalize": normalize,
            }
            shape = _infer_camera_image_obs_shape(env, params)
            return torch.zeros(shape, device=env.device, dtype=torch.uint8)
        return _orig_image(
            env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=normalize,
        )

    ObservationManager._prepare_terms = _prepare_terms_patched
    ObservationManager._prepare_terms._robofinals_camera_probe_patched = True
    mdp_observations.image = image_with_cfg_shape_probe
    env_mdp.image = image_with_cfg_shape_probe


__all__ = ["RendererProvider", "NewtonRenderer"]
