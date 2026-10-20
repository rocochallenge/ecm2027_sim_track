from robofinals.core.backends import resolve_backend_kind


class TeleopAnimationRecorder:
    def __init__(self, args_cli, logger_getter):
        backend = resolve_backend_kind(getattr(args_cli, "physics_backend", None)).value
        if backend == "newton":
            from .newton_recorder import NewtonRecorderController

            self._controller = NewtonRecorderController(args_cli, logger_getter)
        else:
            from .physx_recorder import PhysxRecorderController

            self._controller = PhysxRecorderController(args_cli, logger_getter)

    @property
    def clip_started(self):
        return self._controller.clip_started

    def prepare(self, env):
        self._controller.prepare(env)

    def request_start(self, env):
        self._controller.request_start(env)

    def maybe_start_after_warmup(self, env):
        self._controller.maybe_start_after_warmup(env)

    def sample(self, env):
        self._controller.sample(env)

    def finish(self, env):
        self._controller.finish(env)

    def reset_deformable(self):
        self._controller.reset_deformable()

    def clear_pending_if_not_started(self):
        self._controller.clear_pending_if_not_started()
