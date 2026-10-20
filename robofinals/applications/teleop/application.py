"""Teleoperation application composition root."""

from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Any

from robofinals.core.session import SessionObserver, SessionRunner, SessionRuntime, StepCommand
from robofinals.core.runtime import EnvironmentLifecycle
from robofinals.utils.profile_utils import RollingPerfLogger

from .commands import CommandMailbox
from .driver import ActionSource, TeleopDriver
from .driver import LegacyDeviceActionSource
from .spec import TeleopSpec


class TeleopApplication:
    """Run a device against any compatible runtime through the common runner."""

    def __init__(
        self,
        runtime: SessionRuntime | None = None,
        source: ActionSource | None = None,
        *,
        mailbox: CommandMailbox | None = None,
        observers: Iterable[SessionObserver] = (),
        command_handlers: dict[str, Any] | None = None,
        close_runtime: bool = True,
        spec: TeleopSpec | None = None,
    ) -> None:
        if runtime is None or source is None:
            raise TypeError("runtime and source are required for the command-driven teleop application")
        self.spec = spec
        self.driver = TeleopDriver(source, mailbox)
        self.runner = SessionRunner(
            runtime,
            self.driver,
            observers=observers,
            command_handlers=command_handlers,
            close_runtime=close_runtime,
        )

    def run(self, *, max_cycles: int | None = None):
        print("[teleop] SessionRunner.run begin", flush=True)
        return self.runner.run(max_cycles=max_cycles)

    def setup(self):
        return self

    @property
    def env(self):
        runtime = getattr(self.runner, "runtime", None)
        return getattr(runtime, "env", None)

    @classmethod
    def from_bootstrap(cls, args: Any, app_launcher: Any, simulation_app: Any) -> "TeleopApplication":
        """Build the production runner from the existing Isaac bootstrap.

        The bootstrap owns Kit-specific setup and device construction only. It
        never owns the simulation loop; ``SessionRunner`` does.
        """

        from robofinals.applications.teleop.legacy_session import TeleopSession

        spec = TeleopSpec.from_args(args)
        bootstrap = TeleopSession(args, app_launcher, simulation_app, spec=spec)
        bootstrap.setup()
        backend = bootstrap.env.backend_runtime
        runtime = EnvironmentLifecycle(bootstrap.env, state_synchronizer=backend.state)
        runtime.start()
        print("[teleop] runner runtime started", flush=True)
        bootstrap.initialize_for_runner()
        print("[teleop] runner startup reset complete", flush=True)
        mailbox = bootstrap.command_mailbox
        recorder_observer = _TeleopRecorderObserver(
            bootstrap,
            auto_commit=bool(getattr(args, "teleop_auto_commit", False)),
        )

        def handle_command(_context: Any, command: Any) -> Any:
            if command.name in {"reset_keep_placement", "reset_random"}:
                bootstrap.teleoperation_active = False
                result = runtime.reset(seed=getattr(args, "seed", None))
                # Newton skips rerendering inside env.reset; manual R/X resets
                # must still refresh the interactive viewport before B.
                renderer = getattr(bootstrap.env.backend_runtime, "renderer", None)
                render = getattr(renderer, "render", None)
                if callable(render):
                    render(bootstrap.env)
                print("[teleop] reset complete; viewport refreshed; waiting for B", flush=True)
                return result
            if command.name == "start_teleoperation":
                print("[RecorderDebug] start_teleoperation command dispatched", flush=True)
                bootstrap.teleoperation_active = True
                anim_recorder = getattr(bootstrap, "anim_recorder", None)
                if anim_recorder is not None:
                    anim_recorder.request_start(bootstrap.env)
                return None
            if command.name == "stop_teleoperation":
                bootstrap.teleoperation_active = False
                return None
            if command.name == "flush_recorder":
                recorder_observer.commit_episode()
                return None
            if command.name in {"remake_random", "remake_fixed"}:
                remake = getattr(bootstrap, "_remake_env", None)
                if callable(remake):
                    remake()
                    runtime.env = bootstrap.env
                return None
            # Recorder/checkpoint/UI commands are handled by their existing
            # collaborators; the runner only needs a deterministic transition.
            return None

        hold_action_getter = None
        if getattr(args, "teleop_auto_start", False):
            def hold_action_getter() -> Any:
                device_env = getattr(bootstrap.teleop_interface, "env", None)
                action_manager = getattr(device_env or bootstrap.env, "action_manager", None)
                return getattr(action_manager, "action", None)

        source = LegacyDeviceActionSource(
            bootstrap.teleop_interface,
            simulation_app,
            active_getter=lambda: bootstrap.teleoperation_active,
            hold_action_getter=hold_action_getter,
        )
        bootstrap_adapter = _TeleopBootstrapAdapter(bootstrap)
        application = cls(
            runtime=runtime,
            source=source,
            mailbox=mailbox,
            observers=(
                _BootstrapObserver(bootstrap_adapter),
                _TaskStageObserver(bootstrap),
                recorder_observer,
            ),
            command_handlers={
                "reset_keep_placement": handle_command,
                "remake_random": handle_command,
                "remake_fixed": handle_command,
                "flush_recorder": handle_command,
                "rollback_checkpoint": handle_command,
                "save_checkpoint": handle_command,
                "start_teleoperation": handle_command,
                "stop_teleoperation": handle_command,
            },
            spec=spec,
        )
        application._bootstrap_adapter = bootstrap_adapter
        print("[teleop] runner constructed", flush=True)
        return application


class _TeleopBootstrapAdapter:
    """Resource adapter for the old setup code; it has no run loop."""

    def __init__(self, bootstrap: Any) -> None:
        self.bootstrap = bootstrap

    def close(self) -> None:
        env = self.bootstrap.env
        for name in ("anim_recorder", "lerobot_recorder", "mcap_controller"):
            resource = getattr(self.bootstrap, name, None)
            if resource is None:
                continue
            try:
                if name == "anim_recorder":
                    resource.finish(env)
                elif name == "lerobot_recorder":
                    # The recorder observer commits the active Isaac episode
                    # before this adapter closes the LeRobot writer. Calling
                    # on_stop_recording here first would discard that episode.
                    resource.close()
                else:
                    resource.close()
            except Exception as exc:
                print(
                    f"[teleop] failed to close {name}: {type(exc).__name__}: {exc}",
                    flush=True,
                )


class _BootstrapObserver:
    def __init__(self, adapter: _TeleopBootstrapAdapter) -> None:
        self.adapter = adapter

    def on_session_start(self, _context: Any) -> None:
        return None

    def on_transition(self, _context: Any, _transition: Any) -> None:
        return None

    def on_session_close(self, _context: Any, _error: BaseException | None) -> None:
        self.adapter.close()


class _TaskStageObserver:
    """Refresh task progress labels after each runtime transition."""

    def __init__(self, bootstrap: Any) -> None:
        self.bootstrap = bootstrap

    def on_session_start(self, _context: Any) -> None:
        return None

    def on_transition(self, _context: Any, _transition: Any) -> None:
        from robofinals.utils.ui_utils import update_success_stage_status

        try:
            update_success_stage_status(self.bootstrap.env)
        except Exception as exc:
            print(f"[TaskStageUI] update failed: {type(exc).__name__}: {exc}", flush=True)

    def on_session_close(self, _context: Any, _error: BaseException | None) -> None:
        return None


class _TeleopRecorderObserver:
    """Bridge runner transitions to the existing recorder collaborators.

    Recorder implementations remain outside ``SessionRunner``.  This observer
    owns the production teleop-specific callbacks and can therefore be removed
    or replaced without changing the common runtime contract.
    """

    def __init__(self, bootstrap: Any, *, auto_commit: bool = False) -> None:
        self.bootstrap = bootstrap
        self.auto_commit = auto_commit
        self.recording_started = False
        self.committed = False
        self._step_log_count = 0
        self._perf = RollingPerfLogger("lerobot_step")

    def on_session_start(self, _context: Any) -> None:
        return None

    def on_transition(self, _context: Any, transition: Any) -> None:
        if not isinstance(transition.command, StepCommand):
            return
        anim_recorder = getattr(self.bootstrap, "anim_recorder", None)
        if anim_recorder is not None:
            try:
                anim_recorder.sample(self.bootstrap.env)
            except Exception as exc:
                print(f"[AnimRecording] skip frame: {exc}", flush=True)

        recorder = getattr(self.bootstrap, "lerobot_recorder", None)
        if not self.recording_started:
            if recorder is not None:
                recorder.on_recording_started()
            self.recording_started = True
            print("[RecorderDebug] first StepCommand received; recording started", flush=True)
        result = transition.result
        observation = getattr(result, "observation", result)
        try:
            if recorder is not None:
                recorder_index = int(getattr(recorder, "_frame_in_episode", 0) or 0)
                will_capture_at_index = getattr(recorder, "will_capture_at_index", None)
                will_capture = (
                    bool(will_capture_at_index(recorder_index))
                    if callable(will_capture_at_index)
                    else True
                )
                recorder_started = time.perf_counter() if self._perf.enabled else 0.0
                recorder.on_recording_step(
                    self.bootstrap.env,
                    observation,
                    transition.command.action,
                    skip_sim_render=False,
                )
                if self._perf.enabled:
                    recorder_done = time.perf_counter()
                    self._perf.record(
                        total=recorder_done - recorder_started,
                        capture_count=float(will_capture),
                        deferred_capture_count=float(
                            will_capture and bool(getattr(recorder, "defer_camera_capture", False))
                        ),
                    )
            if self._step_log_count < 5:
                print(
                    f"[RecorderDebug] step frame={transition.frame_index} "
                    f"internal_frame={getattr(recorder, '_frame_in_episode', None)} "
                    f"frames_since_flush={getattr(recorder, '_frames_since_flush', None)} "
                    f"dataset_ready={getattr(recorder, '_dataset', None) is not None} "
                    f"defer_camera={getattr(recorder, 'defer_camera_capture', None)} "
                    f"buffered={len(getattr(getattr(recorder, '_episode_buffer', None), 'frames', ())) }",
                    flush=True,
                )
                self._step_log_count += 1
        except Exception as exc:
            print(f"[LeRobot] skip frame: {exc}", flush=True)
        if self.auto_commit and transition.frame_index >= 10 and not self.committed:
            self._commit_episode()

    def on_session_close(self, _context: Any, _error: BaseException | None) -> None:
        # A normal teleop stop is a completed recording attempt. Commit here so
        # HDF5 and LeRobot are finalized even when the user does not press T.
        interrupted = isinstance(_error, (KeyboardInterrupt, SystemExit))
        manager = getattr(getattr(self.bootstrap, "env", None), "recorder_manager", None)
        hdf5_enabled = getattr(manager, "_dataset_file_handler", None) is not None
        if self.recording_started and not self.committed and (_error is None or interrupted or hdf5_enabled):
            print(f"[RecorderDebug] session close commit error={_error!r}", flush=True)
            self.commit_episode()

    def commit_episode(self) -> None:
        """Commit the active Isaac recorder episode and trigger LeRobot export."""
        self._commit_episode()

    def _commit_episode(self) -> None:
        if self.committed:
            return
        recorder = getattr(self.bootstrap, "lerobot_recorder", None)
        env = getattr(self.bootstrap, "env", None)
        recorder_manager = getattr(env, "recorder_manager", None)
        active_terms = getattr(recorder_manager, "active_terms", None) if recorder_manager is not None else None
        h5_handler = getattr(recorder_manager, "_dataset_file_handler", None) if recorder_manager is not None else None
        h5_path = getattr(h5_handler, "_file_path", None) or getattr(h5_handler, "file_path", None) or getattr(h5_handler, "_hdf5_file_path", None)
        if h5_path is None and recorder_manager is not None:
            cfg = getattr(recorder_manager, "cfg", None)
            export_dir = getattr(cfg, "dataset_export_dir_path", None)
            filename = getattr(cfg, "dataset_filename", None)
            if export_dir and filename:
                h5_path = str(__import__("os").path.join(export_dir, filename))
        print(f"[RecorderDebug] hdf5 handler={type(h5_handler).__name__ if h5_handler is not None else None} path={h5_path!r} open={getattr(h5_handler, "_hdf5_file_stream", None) is not None}", flush=True)
        episodes = getattr(recorder_manager, "_episodes", {}) if recorder_manager is not None else {}
        non_empty = [eid for eid, episode in episodes.items() if not episode.is_empty()]
        print(f"[RecorderDebug] commit manager={recorder_manager is not None} active_terms={list(active_terms or ())} episodes={len(episodes)} non_empty={non_empty} lerobot_dataset={getattr(recorder, '_dataset', None) is not None if recorder is not None else False} lerobot_frames={getattr(recorder, '_frames_since_flush', None) if recorder is not None else None} deferred_frames={len(getattr(getattr(recorder, '_episode_buffer', None), 'frames', ())) if recorder is not None else None}", flush=True)
        if recorder_manager is None or not active_terms:
            print("[RecorderDebug] commit skipped: recorder manager missing or active_terms empty", flush=True)
            return
        if not self._has_recorded_frames():
            print("[RecorderDebug] commit skipped: Isaac recorder episodes empty", flush=True)
            return
        try:
            import torch

            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
            successful_before = int(recorder_manager.exported_successful_episode_count)
            recorder_manager.set_success_to_episodes(
                env_ids,
                torch.ones(env.num_envs, device=env.device, dtype=torch.bool),
            )
            print(f"[RecorderDebug] exporting env_ids={env_ids.tolist()} successful_before={successful_before}", flush=True)
            recorder_manager.export_episodes(env_ids)
            print(f"[RecorderDebug] hdf5 export call returned path={h5_path!r}", flush=True)
            successful_after = int(recorder_manager.exported_successful_episode_count)
            print(f"[RecorderDebug] export returned successful_after={successful_after}", flush=True)
            if successful_after <= successful_before:
                print("[RecorderDebug] export produced no successful episode", flush=True)
                return
            self.committed = True
            if self.auto_commit and recorder is not None:
                # The validation path may be interrupted while Kit is alive;
                # close the writer here so parquet metadata/footer is durable.
                recorder.close()
            print("[teleop] auto-committed recorder episode", flush=True)
        except Exception as exc:
            print(f"[Recorder] auto-commit failed: {exc}", flush=True)

    def _has_recorded_frames(self) -> bool:
        env = getattr(self.bootstrap, "env", None)
        recorder_manager = getattr(env, "recorder_manager", None)
        episodes = getattr(recorder_manager, "_episodes", {})
        return any(not episode.is_empty() for episode in episodes.values())
