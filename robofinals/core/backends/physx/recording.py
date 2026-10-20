"""PhysX recording capability."""

from typing import Any


class PhysXRecordingPolicy:
    def prepare_value(self, value: Any) -> Any:
        return value

    def should_defer_observation(self, env: Any) -> bool:
        del env
        return False

    def storage_device(self, env: Any) -> Any:
        return getattr(env, "device", None)

    def should_defer_camera_capture(self, args: Any) -> bool:
        # PhysX camera sensors can be read directly from IsaacLab observations.
        # Deferred Kit viewport capture is only required by Newton, where RTX
        # readback competes with Warp/CUDA state.
        del args
        return False

    def supports_animation_recording(self) -> bool:
        return True

    def attach_animation_on_close(self, env: Any, *, delivery_dir: str | None = None) -> None:
        from .animation_recording import attach_bake_on_env_close

        attach_bake_on_env_close(env, enabled=True, delivery_dir=delivery_dir)

    def finish_animation(self, env: Any, *, delivery_dir: str | None = None) -> str | None:
        from .animation_recording import finish_ovd_animation_recording

        return finish_ovd_animation_recording(
            env, enabled=True, delivery_dir=delivery_dir
        )
