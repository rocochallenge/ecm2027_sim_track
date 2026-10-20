"""Newton recording capability."""

from typing import Any


class NewtonRecordingPolicy:
    def prepare_value(self, value: Any) -> Any:
        if getattr(value, "is_cuda", False):
            return value.detach().cpu()
        return value

    def should_defer_observation(self, env: Any) -> bool:
        return bool(getattr(env, "_robofinals_defer_camera_read", False))

    def storage_device(self, env: Any) -> Any:
        del env
        import torch

        return torch.device("cpu")

    def should_defer_camera_capture(self, args: Any) -> bool:
        return bool(getattr(args, "enable_cameras", False) and not getattr(args, "headless", False))

    def supports_animation_recording(self) -> bool:
        return True

    def attach_animation_on_close(self, env: Any, *, delivery_dir: str | None = None) -> None:
        del env, delivery_dir

    def finish_animation(self, env: Any, *, delivery_dir: str | None = None) -> str | None:
        del env, delivery_dir
        return None
