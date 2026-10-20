"""Command-driven runner shared by interactive and automated applications."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Iterable
from typing import Any

from robofinals.utils.profile_utils import RollingPerfLogger

from .commands import (
    ApplicationCommand,
    IdleCommand,
    ResetCommand,
    ResetToCommand,
    SessionCommand,
    StepCommand,
    StopCommand,
)
from .contracts import SessionContext, SessionDriver, SessionObserver, SessionRuntime, SessionTransition


class SessionRunnerError(RuntimeError):
    """Raised for an unsupported command or invalid runner configuration."""


CommandHandler = Callable[[SessionContext, ApplicationCommand], Any]


class SessionRunner:
    """Own the common session loop; delegate business decisions to plugins."""

    def __init__(
        self,
        runtime: SessionRuntime,
        driver: SessionDriver,
        *,
        observers: Iterable[SessionObserver] = (),
        command_handlers: dict[str, CommandHandler] | None = None,
        close_runtime: bool = True,
    ) -> None:
        self.runtime = runtime
        self.driver = driver
        self.observers = tuple(observers)
        self.command_handlers = dict(command_handlers or {})
        self.close_runtime = close_runtime
        self._perf = RollingPerfLogger("session")
        if self._perf.enabled:
            print(
                f"[TeleopPerf] enabled interval={self._perf.interval}; wall-clock timings, no CUDA synchronize",
                flush=True,
            )
        self._manual_context: SessionContext | None = None
        self._manual_observers_started: list[SessionObserver] = []

    def dispatch(self, command: SessionCommand) -> SessionTransition:
        """Dispatch one command for externally-shaped Gym loops.

        The normal application path should use :meth:`run`; this method keeps
        legacy framework adapters on the same runtime and transition contract.
        """
        if self._manual_context is None:
            self._manual_context = SessionContext(runtime=self.runtime)
            self.runtime.start()
            self.driver.open(self._manual_context)
            for observer in self.observers:
                observer.on_session_start(self._manual_context)
                self._manual_observers_started.append(observer)
        context = self._manual_context
        if isinstance(command, StopCommand):
            context.metadata["stop_reason"] = command.reason
            return SessionTransition(command, None, context.frame_index, context.cycle_index)
        if not isinstance(command, (StepCommand, ResetCommand, ResetToCommand, IdleCommand, ApplicationCommand)):
            raise SessionRunnerError(f"driver returned unsupported command: {type(command).__name__}")
        context.cycle_index += 1
        result = self._dispatch(context, command)
        if isinstance(command, StepCommand):
            context.frame_index += 1
        context.latest_result = result
        transition = SessionTransition(command, result, context.frame_index, context.cycle_index)
        self.driver.on_transition(context, transition)
        for observer in self.observers:
            observer.on_transition(context, transition)
        return transition

    def close_session(self, error: BaseException | None = None) -> None:
        if self._manual_context is None:
            if self.close_runtime:
                self.runtime.close()
            return
        context = self._manual_context
        for observer in reversed(self._manual_observers_started):
            observer.on_session_close(context, error)
        self.driver.close(context)
        if self.close_runtime:
            self.runtime.close()
        self._manual_context = None
        self._manual_observers_started = []

    def run(self, *, max_cycles: int | None = None) -> SessionContext:
        if max_cycles is not None and max_cycles < 1:
            raise ValueError("max_cycles must be positive when provided")
        context = SessionContext(runtime=self.runtime)
        error: BaseException | None = None
        opened = False
        started_observers: list[SessionObserver] = []
        try:
            self.runtime.start()
            self.driver.open(context)
            opened = True
            for observer in self.observers:
                observer.on_session_start(context)
                started_observers.append(observer)

            while max_cycles is None or context.cycle_index < max_cycles:
                cycle_start = time.perf_counter()
                command = self.driver.next_command(context)
                driver_done = time.perf_counter()
                if not isinstance(
                    command,
                    (StepCommand, ResetCommand, ResetToCommand, IdleCommand, StopCommand, ApplicationCommand),
                ):
                    raise SessionRunnerError(f"driver returned unsupported command: {type(command).__name__}")
                if isinstance(command, StopCommand):
                    context.metadata["stop_reason"] = command.reason
                    print(f"[teleop] SessionRunner received StopCommand reason={command.reason!r}", flush=True)
                    break

                context.cycle_index += 1
                try:
                    result = self._dispatch(context, command)
                except ReferenceError as exc:
                    if str(exc) != "weakly-referenced object no longer exists":
                        raise
                    context.metadata["stop_reason"] = "runtime_released"
                    print(
                        "[teleop] SessionRunner stop: simulation runtime was released during step",
                        flush=True,
                    )
                    break
                dispatch_done = time.perf_counter()
                if isinstance(command, StepCommand):
                    context.frame_index += 1
                context.latest_result = result
                transition = SessionTransition(
                    command=command,
                    result=result,
                    frame_index=context.frame_index,
                    cycle_index=context.cycle_index,
                )
                self.driver.on_transition(context, transition)
                driver_transition_done = time.perf_counter()
                for observer in self.observers:
                    observer.on_transition(context, transition)
                observers_done = time.perf_counter()
                if isinstance(command, StepCommand):
                    self._perf.record(
                        total=observers_done - cycle_start,
                        device=driver_done - cycle_start,
                        env_step=dispatch_done - driver_done,
                        driver_transition=driver_transition_done - dispatch_done,
                        observers=observers_done - driver_transition_done,
                    )
            return context
        except BaseException as exc:
            error = exc
            print(
                f"[teleop] SessionRunner exception: {type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            raise
        finally:
            print(
                f"[teleop] SessionRunner closing error={error!r} "
                f"stop_reason={context.metadata.get('stop_reason')!r} "
                f"cycles={context.cycle_index} frames={context.frame_index}",
                flush=True,
            )
            for observer in reversed(started_observers):
                observer_name = type(observer).__name__
                print(f"[teleop] closing observer {observer_name}", flush=True)
                observer.on_session_close(context, error)
                print(f"[teleop] closed observer {observer_name}", flush=True)
            if opened:
                print("[teleop] closing device driver", flush=True)
                self.driver.close(context)
                print("[teleop] closed device driver", flush=True)
            if self.close_runtime:
                print("[teleop] closing environment runtime", flush=True)
                self.runtime.close()
                print("[teleop] closed environment runtime", flush=True)

    def _dispatch(self, context: SessionContext, command: SessionCommand) -> Any:
        if isinstance(command, StepCommand):
            return self.runtime.step(command.action)
        if isinstance(command, ResetCommand):
            return self.runtime.reset(command.env_ids, seed=command.seed)
        if isinstance(command, ResetToCommand):
            return self.runtime.reset_to(
                command.state,
                command.env_ids,
                seed=command.seed,
                is_relative=command.is_relative,
            )
        if isinstance(command, IdleCommand):
            return self.runtime.idle(command.reason)
        if isinstance(command, ApplicationCommand):
            try:
                handler = self.command_handlers[command.name]
            except KeyError as exc:
                raise SessionRunnerError(f"no handler registered for application command {command.name!r}") from exc
            return handler(context, command)
        raise SessionRunnerError(f"unsupported command: {type(command).__name__}")
